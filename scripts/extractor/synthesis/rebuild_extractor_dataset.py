from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.extractor.pipeline.fact_pipeline import FactPipeline, FactPipelineError
from scripts.extractor.pipeline.schema import (
    DEFAULT_SOURCE,
    DEFAULT_WORK_DIR,
    clean_extraction_for_training,
    nearest_previous_assistant,
    read_jsonl,
    recent_context,
    resolve_from_root,
    select_source_dialogues,
    weak_labels_from,
    write_jsonl,
)
from scripts.extractor.pipeline.state_manager import compact_patient_state, empty_patient_state, prompt_patient_state
from scripts.extractor.synthesis.failure_ledger import load_failure_ledger
from scripts.extractor.synthesis.dialogue_work import load_dialogue_shards


def group_rows_by_dialogue(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return {}
    for row in read_jsonl(path):
        source = row.get("input") or {}
        dialogue_id = source.get("dialogue_id")
        if dialogue_id:
            grouped[str(dialogue_id)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int((item.get("input") or {}).get("turn_id") or -1))
    return dict(grouped)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp, rows)
    os.replace(tmp, path)


def user_turn_ids(dialog: dict[str, Any]) -> list[int]:
    return [
        idx
        for idx, message in enumerate(dialog.get("messages") or [])
        if message.get("role") == "user"
    ]


def rebuild_dialogue_rows(
    dialog: dict[str, Any],
    rows: list[dict[str, Any]],
    fact_pipeline: FactPipeline,
    max_context_messages: int,
    max_context_chars: int,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    dialogue_id = str(dialog.get("dialogue_id"))
    messages = dialog.get("messages") or []
    row_by_turn = {
        int((row.get("input") or {}).get("turn_id")): row
        for row in rows
        if isinstance((row.get("input") or {}).get("turn_id"), int)
    }
    expected_turns = user_turn_ids(dialog)
    missing = [turn_id for turn_id in expected_turns if turn_id not in row_by_turn]
    if missing:
        return None, {"dialogue_id": dialogue_id, "reason": "missing_turns", "turn_ids": missing}

    state = empty_patient_state(dialogue_id)
    rebuilt_rows: list[dict[str, Any]] = []
    for idx in expected_turns:
        message = messages[idx]
        original = row_by_turn[idx]
        input_row = dict(original.get("input") or {})
        input_row.update(
            {
                "dialogue_id": dialogue_id,
                "turn_id": idx,
                "previous_doctor_question": nearest_previous_assistant(messages, idx),
                "patient_utterance": str(message.get("content") or "").strip(),
                "patient_state_before_turn": prompt_patient_state(state),
                "recent_context": recent_context(
                    messages,
                    idx,
                    max_messages=max_context_messages,
                    max_chars=max_context_chars,
                ),
                "meddg_weak_labels": weak_labels_from(message),
            }
        )
        training_source = clean_extraction_for_training(original.get("parsed_output") or {"facts": []})
        try:
            processed = fact_pipeline.process(
                training_source,
                patient_utterance=input_row["patient_utterance"],
                state=state,
                turn_id=idx,
                previous_doctor_question=input_row.get("previous_doctor_question"),
                recent_context=input_row.get("recent_context") or [],
            )
        except FactPipelineError as exc:
            return None, {
                "dialogue_id": dialogue_id,
                "reason": exc.stage,
                "turn_id": idx,
                "errors": exc.errors,
            }
        state = processed.state_after
        rebuilt_rows.append(
            {
                "input": input_row,
                "raw_response": original.get("raw_response") or "",
                "parsed_output": processed.runtime_extraction,
                "patient_state_after_turn": compact_patient_state(state),
                "normalization_stats": processed.normalization_stats,
                "model": original.get("model"),
                "api_format": original.get("api_format"),
            }
        )
    return rebuilt_rows, {"dialogue_id": dialogue_id, "patient_state": compact_patient_state(state)}


def checkpoint_row(dialogue_id: str, rows: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    completed_turn_id = rows[-1]["input"]["turn_id"] if rows else None
    return {
        "dialogue_id": dialogue_id,
        "completed_turn_id": completed_turn_id,
        "patient_state": compact_patient_state(state),
        "finished": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild synthesized extractor data by source dialogue order.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--synthesized", type=Path, required=True)
    parser.add_argument("--repair-succeeded", type=Path, default=None)
    parser.add_argument("--failed-dialogues", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "rebuilt_synthesized.jsonl")
    parser.add_argument("--final-states", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dialogue-work-dir", type=Path, default=None)
    parser.add_argument("--terminology-db", type=Path, default=None)
    parser.add_argument("--max-context-messages", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=600)
    parser.add_argument("--max-dialogues", type=int, default=None)
    parser.add_argument("--max-user-turns", type=int, default=None)
    args = parser.parse_args()

    args.source = resolve_from_root(args.source)
    args.synthesized = resolve_from_root(args.synthesized)
    if args.repair_succeeded is not None:
        args.repair_succeeded = resolve_from_root(args.repair_succeeded)
    args.failed_dialogues = resolve_from_root(args.failed_dialogues)
    args.output = resolve_from_root(args.output)
    args.final_states = resolve_from_root(args.final_states)
    args.checkpoint = resolve_from_root(args.checkpoint)
    args.report = resolve_from_root(args.report)
    if args.dialogue_work_dir is not None:
        args.dialogue_work_dir = resolve_from_root(args.dialogue_work_dir)
    if args.terminology_db is not None:
        args.terminology_db = resolve_from_root(args.terminology_db)

    fact_pipeline = FactPipeline(args.terminology_db)

    grouped = group_rows_by_dialogue(args.synthesized)
    original_dialogue_count = len(grouped)
    repair_grouped = group_rows_by_dialogue(args.repair_succeeded) if args.repair_succeeded else {}
    grouped.update(repair_grouped)
    work_shards = load_dialogue_shards(args.dialogue_work_dir) if args.dialogue_work_dir else {}
    work_grouped = {
        dialogue_id: payload["rows"]
        for dialogue_id, payload in work_shards.items()
    }
    grouped.update(work_grouped)
    failed_ids = set(load_failure_ledger(args.failed_dialogues))

    rebuilt: list[dict[str, Any]] = []
    final_states: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    skipped_failed: list[str] = []
    skipped_incomplete: list[dict[str, Any]] = []
    included_dialogues = 0

    selected_dialogues, selected_user_turns = select_source_dialogues(
        args.source,
        max_dialogues=args.max_dialogues,
        max_user_turns=args.max_user_turns,
    )
    for dialog in selected_dialogues:
        dialogue_id = str(dialog.get("dialogue_id"))
        if dialogue_id in failed_ids:
            skipped_failed.append(dialogue_id)
            continue
        rows = grouped.get(dialogue_id)
        if not rows:
            continue
        rebuilt_rows, final_state = rebuild_dialogue_rows(
            dialog,
            rows,
            fact_pipeline,
            args.max_context_messages,
            args.max_context_chars,
        )
        if rebuilt_rows is None or final_state is None:
            skipped_incomplete.append(final_state or {"dialogue_id": dialogue_id})
            continue
        rebuilt.extend(rebuilt_rows)
        final_states.append(final_state)
        checkpoints.append(checkpoint_row(dialogue_id, rebuilt_rows, final_state["patient_state"]))
        included_dialogues += 1

    write_jsonl_atomic(args.output, rebuilt)
    write_jsonl_atomic(args.final_states, final_states)
    write_jsonl_atomic(args.checkpoint, checkpoints)
    report = {
        "source": str(args.source),
        "synthesized": str(args.synthesized),
        "repair_succeeded": str(args.repair_succeeded) if args.repair_succeeded else None,
        "output": str(args.output),
        "original_dialogues": original_dialogue_count,
        "repair_dialogues": len(repair_grouped),
        "work_dialogues": len(work_grouped),
        "selected_dialogues": len(selected_dialogues),
        "selected_user_turns": selected_user_turns,
        "included_dialogues": included_dialogues,
        "output_rows": len(rebuilt),
        "skipped_failed_dialogues": skipped_failed,
        "skipped_incomplete_dialogues": skipped_incomplete,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
