from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any

import httpx
from tqdm import tqdm

from scripts.extractor.pipeline.schema import read_jsonl, resolve_from_root
from scripts.extractor.synthesis.failure_ledger import (
    load_failure_ledger,
    upsert_failed_dialogue,
    write_failure_ledger,
)
from scripts.interviewer.pipeline.dialogue import (
    build_split_map,
    index_extractor_rows,
    recent_context_before,
    split_dialogue_sessions,
)
from scripts.interviewer.pipeline.schema import (
    TEACHER_SYSTEM_PROMPT,
    answerable_target_candidates,
    build_teacher_prompt,
    canonicalize_teacher_response,
    extract_json_object,
    make_unusable_decision,
    select_teacher_question,
    validate_teacher_candidates,
)
from scripts.interviewer.pipeline.state_projection import project_interviewer_state
from scripts.extractor.pipeline.state_manager import empty_patient_state, merge_facts_into_state


THREAD_LOCAL = local()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def shard_name(dialogue_id: str) -> str:
    return dialogue_id.replace("/", "__").replace("\\", "__") + ".json"


class TeacherClient:
    def __init__(self, args: argparse.Namespace) -> None:
        verify = not args.no_ssl_verify
        transport = httpx.Client(verify=verify)
        self.api_format = args.api_format
        if args.api_format == "anthropic":
            from anthropic import Anthropic

            self.client = Anthropic(
                api_key=args.api_key,
                base_url=args.base_url,
                http_client=transport,
            )
        else:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=args.api_key,
                base_url=args.base_url,
                http_client=transport,
            )
        self.model = args.model
        self.timeout = args.timeout
        self.max_output_tokens = args.max_output_tokens

    def call(self, user_prompt: str) -> str:
        if self.api_format == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                system=TEACHER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0,
                max_tokens=self.max_output_tokens,
                timeout=self.timeout,
            )
            content = "".join(
                str(block.text)
                for block in response.content
                if getattr(block, "type", None) == "text"
            )
            if not content.strip():
                raise ValueError("teacher returned empty Anthropic text content")
            return content
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=self.max_output_tokens,
            timeout=self.timeout,
        )
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise ValueError("teacher returned empty OpenAI message content")
        return content


def thread_client(args: argparse.Namespace) -> TeacherClient:
    client = getattr(THREAD_LOCAL, "teacher_client", None)
    if client is None:
        client = TeacherClient(args)
        THREAD_LOCAL.teacher_client = client
    return client


def call_and_validate(
    args: argparse.Namespace,
    row: dict[str, Any],
) -> tuple[dict[str, Any], str, int]:
    last_error: Exception | None = None
    last_raw = ""
    base_prompt = build_teacher_prompt(row)
    correction = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            last_raw = thread_client(args).call(base_prompt + correction)
            decision = canonicalize_teacher_response(extract_json_object(last_raw))
            errors = validate_teacher_candidates(decision, row)
            if errors:
                correction = (
                    "\n\n上一次输出未通过校验："
                    + last_raw[:2000]
                    + "\n校验错误："
                    + "; ".join(errors)
                    + "\n请只依据原医生回复、当前状态和上述校验错误修正，不得猜测后续患者回答。重新只输出一个合法JSON。"
                )
                raise ValueError("; ".join(errors))
            return decision, last_raw, attempt
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                delay = args.retry_sleep * (2 ** (attempt - 1)) + random.uniform(0, args.retry_jitter)
                time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}; raw={last_raw[:1000]}") from last_error


def synthesize_dialogue(
    dialogue: dict[str, Any],
    extractor_index: dict[str, dict[int, dict[str, Any]]],
    split_map: dict[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    source_dialogue_id = str(dialogue.get("dialogue_id") or "")
    rows_by_turn = extractor_index.get(source_dialogue_id, {})
    sessions = split_dialogue_sessions(dialogue, rows_by_turn)
    output: list[dict[str, Any]] = []
    for session in sessions:
        session_id = str(session[0]["session_id"])
        session_start_turn = int(session[0]["patient_turn_ids"][0])
        local_state = empty_patient_state(session_id)
        states: list[dict[str, Any]] = []
        for block in session:
            for turn_id in block["patient_turn_ids"]:
                extractor_row = rows_by_turn.get(turn_id)
                if extractor_row is None:
                    raise ValueError(f"missing v3.1 extractor row for patient turn {turn_id}")
                facts = (extractor_row.get("parsed_output") or {}).get("facts") or []
                local_state = merge_facts_into_state(local_state, facts, turn_id)
            states.append(project_interviewer_state(local_state))

        asked_targets: list[dict[str, Any]] = []
        for block_index, block in enumerate(session):
            state = states[block_index]
            next_state = states[block_index + 1] if block_index + 1 < len(states) else {}
            row = {
                "dialogue_id": session_id,
                "source_dialogue_id": source_dialogue_id,
                "session_index": block["session_index"],
                "split": split_map[source_dialogue_id],
                "block_id": block["block_id"],
                "source_block_id": block["source_block_id"],
                "patient_turn_ids": block["patient_turn_ids"],
                "doctor_turn_ids": block["doctor_turn_ids"],
                "patient_block": block["patient_block"],
                "patient_state": state,
                "recent_context": recent_context_before(
                    dialogue.get("messages") or [],
                    block["patient_turn_ids"][0],
                    args.max_context_messages,
                    args.max_context_chars,
                    start_turn_id=session_start_turn,
                ),
                "asked_targets_before": [dict(target) for target in asked_targets],
                "original_doctor_block": block["original_doctor_block"],
                "next_patient_block": block["next_patient_block"],
                "next_patient_state": next_state,
                "answerability_reference": answerable_target_candidates(state, next_state),
                "is_terminal": block["is_terminal"],
                "session_split_reason": block.get("session_split_reason"),
                "session_boundary_warning": block.get("session_boundary_warning"),
            }
            if block["is_terminal"]:
                teacher_response = None
                selected_candidate = None
                raw_response = ""
                teacher_attempts = 0
                decision = {
                    "usable": True,
                    "action": "end",
                    "next_question_target": None,
                    "utterance": "",
                }
            elif not block["original_doctor_block"]:
                teacher_response = None
                selected_candidate = None
                raw_response = ""
                teacher_attempts = 0
                decision = make_unusable_decision(
                    "NO_SAFE_DOCTOR_QUESTION",
                    "original doctor block is empty",
                )
            else:
                teacher_response, raw_response, teacher_attempts = call_and_validate(args, row)
                decision, selected_candidate = select_teacher_question(teacher_response, row)
            row["raw_response"] = raw_response
            row["teacher_response"] = teacher_response
            row["teacher_question_candidates"] = (
                teacher_response.get("questions") if isinstance(teacher_response, dict) else []
            )
            row["selected_teacher_candidate"] = selected_candidate
            row["answerability_method"] = (
                selected_candidate.get("answerability_method")
                if isinstance(selected_candidate, dict)
                else None
            )
            row["teacher_attempts"] = teacher_attempts
            row["parsed_output"] = decision
            if decision.get("usable", True) and decision.get("action") == "ask":
                asked_targets.append(dict(decision["next_question_target"]))
            row["asked_targets_after"] = [dict(target) for target in asked_targets]
            output.append(row)
    return output


def rebuild_outputs(
    dialogues: list[dict[str, Any]],
    work_dir: Path,
    output: Path,
    checkpoint: Path,
) -> tuple[int, int]:
    rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    completed = 0
    for dialogue in dialogues:
        dialogue_id = str(dialogue.get("dialogue_id") or "")
        shard = work_dir / shard_name(dialogue_id)
        if not shard.exists():
            continue
        dialogue_rows = json.loads(shard.read_text(encoding="utf-8"))
        rows.extend(dialogue_rows)
        checkpoints.append(
            {
                "dialogue_id": dialogue_id,
                "finished": True,
                "blocks": len(dialogue_rows),
                "samples": sum(1 for row in dialogue_rows if (row.get("parsed_output") or {}).get("usable", True)),
            }
        )
        completed += 1
    write_jsonl_atomic(output, rows)
    write_jsonl_atomic(checkpoint, checkpoints)
    return completed, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize interviewer decisions from MedDG and extractor v3.1 states.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--extractor-validated", type=Path, required=True)
    parser.add_argument("--extractor-train", type=Path, required=True)
    parser.add_argument("--extractor-valid", type=Path, required=True)
    parser.add_argument("--extractor-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dialogue-work-dir", type=Path, required=True)
    parser.add_argument("--api-format", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--retry-jitter", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-dialogues", type=int, default=None)
    parser.add_argument("--max-context-messages", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=1000)
    parser.add_argument("--repair-failed-dialogues", action="store_true")
    parser.add_argument("--no-ssl-verify", action="store_true")
    args = parser.parse_args()

    for name in (
        "source",
        "extractor_validated",
        "extractor_train",
        "extractor_valid",
        "extractor_test",
        "output",
        "failed",
        "checkpoint",
        "report",
        "dialogue_work_dir",
    ):
        setattr(args, name, resolve_from_root(getattr(args, name)))
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    dialogues = read_jsonl(args.source)
    extractor_rows = read_jsonl(args.extractor_validated)
    split_rows = {
        "train": read_jsonl(args.extractor_train),
        "valid": read_jsonl(args.extractor_valid),
        "test": read_jsonl(args.extractor_test),
    }
    split_map = build_split_map(split_rows)
    extractor_index = index_extractor_rows(extractor_rows)
    dialogues = [dialogue for dialogue in dialogues if str(dialogue.get("dialogue_id") or "") in split_map]
    if args.max_dialogues is not None:
        dialogues = dialogues[: args.max_dialogues]

    args.dialogue_work_dir.mkdir(parents=True, exist_ok=True)
    ledger = load_failure_ledger(args.failed)
    if args.repair_failed_dialogues:
        wanted = set(ledger)
        selected = [dialogue for dialogue in dialogues if str(dialogue.get("dialogue_id") or "") in wanted]
    else:
        selected = [
            dialogue
            for dialogue in dialogues
            if not (args.dialogue_work_dir / shard_name(str(dialogue.get("dialogue_id") or ""))).exists()
        ]

    newly_succeeded = 0
    newly_failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(synthesize_dialogue, dialogue, extractor_index, split_map, args): dialogue
            for dialogue in selected
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Interviewer synthesis", unit="dialogue", dynamic_ncols=True):
            dialogue = futures[future]
            dialogue_id = str(dialogue.get("dialogue_id") or "")
            try:
                rows = future.result()
                write_json_atomic(args.dialogue_work_dir / shard_name(dialogue_id), rows)
                ledger.pop(dialogue_id, None)
                newly_succeeded += 1
                print(f"dialogue-ok {dialogue_id}", flush=True)
            except Exception as exc:
                newly_failed += 1
                upsert_failed_dialogue(
                    ledger,
                    dialogue_id=dialogue_id,
                    dialogue=dialogue,
                    failure_type="interviewer_synthesis_error",
                    failed_at="synthesis",
                    error=f"{type(exc).__name__}: {exc}",
                    errors=[str(exc)],
                    model=args.model,
                    api_format=args.api_format,
                )
                print(f"dialogue-failed {dialogue_id}", flush=True)
            write_failure_ledger(args.failed, ledger)

    completed, row_count = rebuild_outputs(dialogues, args.dialogue_work_dir, args.output, args.checkpoint)
    output_rows = list(read_jsonl(args.output))
    session_ids = {str(row.get("dialogue_id") or "") for row in output_rows}
    selected_ids = {str(dialogue.get("dialogue_id") or "") for dialogue in dialogues}
    remaining_failed = sum(dialogue_id in selected_ids for dialogue_id in ledger)
    report = {
        "selected_parent_dialogues": len(dialogues),
        "pending_or_repair_dialogues": len(selected),
        "newly_succeeded_parent_dialogues": newly_succeeded,
        "newly_failed_parent_dialogues": newly_failed,
        "completed_parent_dialogues": completed,
        "remaining_failed_parent_dialogues": remaining_failed,
        "rows": row_count,
        "sessions": len(session_ids),
        "session_splits": max(0, len(session_ids) - completed),
        "ambiguous_session_boundaries": sum(bool(row.get("session_boundary_warning")) for row in output_rows),
        "teacher_api_attempts": sum(int(row.get("teacher_attempts") or 0) for row in output_rows),
        "teacher_api_retries": sum(max(0, int(row.get("teacher_attempts") or 0) - 1) for row in output_rows),
        "repair_mode": bool(args.repair_failed_dialogues),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
