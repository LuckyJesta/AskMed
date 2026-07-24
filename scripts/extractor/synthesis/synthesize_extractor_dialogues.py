from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from scripts.extractor.pipeline.fact_pipeline import FactPipeline, FactPipelineError
from scripts.extractor.pipeline.schema import (
    DEFAULT_SOURCE,
    DEFAULT_WORK_DIR,
    SYSTEM_PROMPT,
    append_jsonl,
    extract_json_object,
    format_context,
    is_short_context_answer,
    is_pure_non_medical_chatter,
    nearest_previous_assistant,
    normalize_extraction,
    read_jsonl,
    recent_context,
    resolve_from_root,
    select_source_dialogues,
    weak_labels_from,
)
from scripts.extractor.pipeline.state_manager import compact_patient_state, empty_patient_state, prompt_patient_state
from scripts.extractor.synthesis.failure_ledger import (
    load_failure_ledger,
    remove_failed_dialogue,
    select_failure_dialogue_ids,
    update_failed_dialogue,
)
from scripts.extractor.synthesis.dialogue_work import (
    dialogue_shard_path,
    execute_dialogue_tasks,
    load_dialogue_shards,
    write_dialogue_shard,
)


def call_openai_compatible(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
        timeout=timeout,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI-compatible API returned an empty message content")
    return content


def call_anthropic_messages(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    max_output_tokens: int,
) -> str:
    system_parts = [message["content"] for message in messages if message.get("role") == "system"]
    anthropic_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") in {"user", "assistant"}
    ]
    response = client.messages.create(
        model=model,
        system="\n\n".join(system_parts) if system_parts else None,
        messages=anthropic_messages,
        temperature=0,
        max_tokens=max_output_tokens,
        timeout=timeout,
    )
    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    content = "".join(text_parts)
    if not content:
        raise ValueError("Anthropic API returned an empty message content")
    return content


def call_teacher_model(
    client: Any,
    api_format: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    max_output_tokens: int,
) -> str:
    if api_format == "openai":
        return call_openai_compatible(client, model, messages, timeout)
    if api_format == "anthropic":
        return call_anthropic_messages(client, model, messages, timeout, max_output_tokens)
    raise ValueError(f"Unsupported api_format: {api_format}")


def create_teacher_client(api_format: str, api_key: str, base_url: str, ssl_verify: bool) -> Any:
    http_client = httpx.Client(verify=ssl_verify)
    if api_format == "openai":
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    if api_format == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise SystemExit("Missing dependency: install the 'anthropic' Python package.") from exc
        return Anthropic(api_key=api_key, base_url=base_url, http_client=http_client)
    raise ValueError(f"Unsupported api_format: {api_format}")


def handled_exception_types(api_format: str) -> tuple[type[BaseException], ...]:
    common_errors: tuple[type[BaseException], ...] = (
        json.JSONDecodeError,
        KeyError,
        ValueError,
    )
    if api_format == "openai":
        return (
            APIConnectionError,
            APIError,
            APITimeoutError,
            RateLimitError,
            *common_errors,
        )
    if api_format == "anthropic":
        try:
            from anthropic import APIConnectionError as AnthropicAPIConnectionError
            from anthropic import APIError as AnthropicAPIError
            from anthropic import APITimeoutError as AnthropicAPITimeoutError
            from anthropic import RateLimitError as AnthropicRateLimitError
        except ImportError as exc:
            raise SystemExit("Missing dependency: install the 'anthropic' Python package.") from exc
        return (
            AnthropicAPIConnectionError,
            AnthropicAPIError,
            AnthropicAPITimeoutError,
            AnthropicRateLimitError,
            *common_errors,
        )
    raise ValueError(f"Unsupported api_format: {api_format}")


def build_user_prompt(row: dict) -> str:
    patient_utterance = str(row.get("patient_utterance") or "")
    include_context = is_short_context_answer(patient_utterance)
    if include_context:
        auxiliary = (
            "【辅助理解信息：只能用于理解短回答和绑定目标，不能单独建立事实，也不能作为evidence】\n"
            f"上一轮医生问题：{row.get('previous_doctor_question') or '无'}\n"
            "当前问诊状态（只由当前轮之前的信息形成，不包含当前发言）：\n"
            f"{json.dumps(row.get('patient_state_before_turn') or {}, ensure_ascii=False, separators=(',', ':'))}\n"
            "最近对话上下文：省略，避免旧患者发言污染本轮evidence。\n"
            "短回答特别规则：如果患者当前发言是明确的肯定、否定、程度或属性回答，结合上一轮医生问题确定医学对象并直接填写简洁的name；evidence必须使用患者当前发言原文。如果无法明确绑定回答对象，输出空facts。\n"
        )
    else:
        auxiliary = (
            "【辅助理解信息：不能直接作为事实或evidence】\n"
            f"上一轮医生问题：{row.get('previous_doctor_question') or '无'}\n"
            "当前患者发言不是短回答，本轮不要抽取问诊状态或最近上下文中的旧事实。\n"
        )
    return (
        "请完成一次医学事实抽取。严格只输出JSON。\n\n"
        f"{auxiliary}"
        "MedDG弱标签（只供合成参考，可能不完整或有误）："
        f"{json.dumps(row.get('meddg_weak_labels') or {}, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "【唯一抽取对象】\n"
        f"患者当前发言：{patient_utterance}\n\n"
        "【输出要求】\n"
        "1. 只抽取患者当前发言中明确存在或明确不存在的医学事实，status只能使用present或absent。\n"
        "2. 患者咨询、猜测、担忧、请求建议、诊断咨询、用药咨询、检查咨询不作为facts。\n"
        "3. 一句话同时包含明确事实和咨询时，只抽取明确事实部分。\n"
        "4. evidence必须来自患者当前发言的连续原文片段。\n"
        "4a. name直接填写当前问答明确支持的简洁医学事实名称，允许语义等价概括和上下文指代消解，不要求是原文片段；不得诊断、推测或补充本轮未确认的事实。type=attribute时，name填写属性类别，目标和值分别写入attribute.target和对应属性key。\n"
        "4b. 如果输出type=attribute，attribute必须同时包含一个明确target和至少一个具体属性key；目标优先取当前发言明确对象，其次取上一轮医生问题对象，再次取问诊状态中最近且唯一相关对象。存在多个合理目标、无法唯一确定时不要输出该attribute fact；同一evidence和同一属性key不得绑定不同target。\n"
        "5. 当前问诊状态和最近对话上下文中的既有事实不要重复输出，除非患者当前发言重新明确表达了该事实。\n"
        "6. 输出前逐条自检：如果某个fact的evidence不是患者当前发言的连续子串，必须删除该fact。\n"
        "7. 如果当前发言只是寒暄、确认、感谢或告别且没有新医学信息，输出 {\"facts\":[]}。\n"
        "8. 如果当前发言是短回答，结合上一轮医生问题和当前问诊状态判断目标；无法判断目标时输出 {\"facts\":[]}。\n"
        "9. 如果当前发言中症状本体和属性同时出现，把属性合并进该症状fact，不要重复输出attribute fact。\n"
        "10. 每个fact只能包含name, normalized_name, type, status, subject, evidence, standard_code, terminology, attribute。\n"
        "11. 时间、部位、剂量、结果、频率、程度、性质等补充信息都写入attribute。\n"
        "12. attribute优先使用time, body_part, duration, episode_duration, frequency, character, severity, trigger, aggravating_factor, relieving_factor, color, amount, appetite, sleep, effect, dose, result, route, side_effect, target；确实无法匹配时可以创建清晰、简短、语义明确的新key。\n"
        "13. 如果含义能被推荐key表达，必须使用推荐key，不要创建推荐key的同义key，例如用severity表示程度，不要用degree；用time表示发生时间，不要用onset_time。\n"
        "14. 不要把name, normalized_name, type, status, subject, evidence, standard_code, terminology重复放入attribute。\n"
        "15. 再次确认：患者当前发言必须明确支持最终facts；上一轮医生问题和问诊状态只能辅助理解本轮回答，不能凭辅助信息单独建立事实。\n"
        "16. medicine fact只表示患者明确已经使用、正在使用、曾经使用或明确没有使用某药；仅表示家中有药、持有或购买了药，或者询问能否服用，不输出medicine fact。\n"
        "17. 没有用药、没有吃药、没有检查等没有具体对象的泛化否定不输出facts；没吃奥美拉唑、没做胃镜等具体否定仍抽取。\n"
        "18. 如果当前发言只补充状态中已有事实的性质、程度、时间、频率、部位或诱因，输出type=attribute并复用状态中的原名称作为target，不要新建重复症状。\n"
        "19. 请只输出紧凑JSON，不要换行、缩进或多余空格。"
    )


def synthesize_turn(
    row: dict[str, Any],
    state: dict[str, Any],
    client: Any,
    handled_errors: tuple[type[BaseException], ...],
    fact_pipeline: FactPipeline,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw_response = ""
    model_name = args.model
    if is_pure_non_medical_chatter(str(row.get("patient_utterance") or "")):
        parsed = {"facts": []}
        raw_response = json.dumps(parsed, ensure_ascii=False)
        model_name = "rule:pure_non_medical_chatter"
        try:
            processed = _process_extraction(parsed, row, state, fact_pipeline)
        except FactPipelineError as exc:
            return None, _pipeline_failure(row, raw_response, exc, args)
    else:
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ]
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                raw_response = call_teacher_model(
                    client,
                    args.api_format,
                    args.model,
                    prompt_messages,
                    args.timeout,
                    args.max_output_tokens,
                )
                parsed = normalize_extraction(extract_json_object(raw_response))
            except handled_errors as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    _sleep_before_retry(args.sleep, attempt)
                    continue
                else:
                    return None, {
                        "input": row,
                        "raw_response": raw_response,
                        "error": last_error,
                        "model": args.model,
                        "api_format": args.api_format,
                    }
            try:
                processed = _process_extraction(parsed, row, state, fact_pipeline)
                break
            except FactPipelineError as exc:
                if exc.stage == "raw_validation_error" and attempt < args.max_retries:
                    prompt_messages.extend(
                        [
                            {"role": "assistant", "content": raw_response},
                            {
                                "role": "user",
                                "content": build_validation_retry_prompt(exc.errors),
                            },
                        ]
                    )
                    _sleep_before_retry(args.sleep, attempt)
                    continue
                return None, _pipeline_failure(row, raw_response, exc, args)
    return (
        {
            "input": row,
            "raw_response": raw_response,
            "parsed_output": processed.runtime_extraction,
            "patient_state_after_turn": compact_patient_state(processed.state_after),
            "normalization_stats": processed.normalization_stats,
            "model": model_name,
            "api_format": args.api_format if model_name == args.model else "rule",
        },
        None,
    )


def _process_extraction(
    extraction: dict[str, Any],
    row: dict[str, Any],
    state: dict[str, Any],
    fact_pipeline: FactPipeline,
) -> Any:
    return fact_pipeline.process(
        extraction,
        patient_utterance=str(row.get("patient_utterance") or ""),
        state=state,
        turn_id=row.get("turn_id"),
        previous_doctor_question=row.get("previous_doctor_question"),
        recent_context=row.get("recent_context") or [],
    )


def build_validation_retry_prompt(errors: list[str]) -> str:
    details = "\n".join(f"- {error}" for error in errors)
    return (
        "你上一次输出未通过确定性校验：\n"
        f"{details}\n"
        "请根据最初提供的患者当前发言和上下文重新输出完整JSON，并修正以上全部错误。"
        "evidence必须是患者当前发言中的连续原文片段；不要把core字段放入attribute；"
        "没有具体医学对象的泛化用药或检查否定必须删除。只输出JSON，不要解释。"
    )


def _sleep_before_retry(sleep_seconds: float, attempt: int) -> None:
    retry_base = max(sleep_seconds, 0.5)
    time.sleep(retry_base * (2 ** (attempt - 1)) + random.uniform(0, retry_base))


def _pipeline_failure(
    row: dict[str, Any],
    raw_response: str,
    exc: FactPipelineError,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "input": row,
        "raw_response": raw_response,
        "error": str(exc),
        "errors": exc.errors,
        "failure_type": exc.stage,
        "model": args.model,
        "api_format": args.api_format,
    }


def synthesize_dialogue(
    dialog: dict[str, Any],
    client: Any,
    handled_errors: tuple[type[BaseException], ...],
    fact_pipeline: FactPipeline,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, dict[str, Any], int]:
    dialogue_id = dialog.get("dialogue_id")
    messages = dialog.get("messages") or []
    state = empty_patient_state(dialogue_id)
    rows: list[dict[str, Any]] = []
    user_turns = 0

    for idx, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        user_turns += 1
        row = {
            "dialogue_id": dialogue_id,
            "turn_id": idx,
            "previous_doctor_question": nearest_previous_assistant(messages, idx),
            "patient_utterance": str(message.get("content") or "").strip(),
            "patient_state_before_turn": prompt_patient_state(state),
            "recent_context": recent_context(
                messages,
                idx,
                max_messages=args.max_context_messages,
                max_chars=args.max_context_chars,
            ),
            "meddg_weak_labels": weak_labels_from(message),
        }
        output_row, failure = synthesize_turn(row, state, client, handled_errors, fact_pipeline, args)
        if failure is not None:
            failure.update(
                {
                    "dialogue_id": dialogue_id,
                    "failed_turn_id": idx,
                    "dialogue": dialog,
                }
            )
            return None, failure, state, user_turns
        if output_row is None:
            return None, {
                "dialogue_id": dialogue_id,
                "failed_turn_id": idx,
                "dialogue": dialog,
                "input": row,
                "raw_response": "",
                "error": "synthesis returned neither output nor failure",
                "model": args.model,
                "api_format": args.api_format,
            }, state, user_turns
        state = output_row["patient_state_after_turn"]
        rows.append(output_row)
    return rows, None, state, user_turns


def load_checkpoints(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    checkpoints: dict[str, dict] = {}
    for row in read_jsonl(path):
        dialogue_id = row.get("dialogue_id")
        if dialogue_id:
            checkpoints[str(dialogue_id)] = row
    return checkpoints


def append_checkpoint(path: Path, dialogue_id: str | None, completed_turn_id: int | None, state: dict, finished: bool) -> None:
    append_jsonl(
        path,
        {
            "dialogue_id": dialogue_id,
            "completed_turn_id": completed_turn_id,
            "patient_state": compact_patient_state(state),
            "finished": finished,
        },
    )


_worker_local = threading.local()


def worker_resources(args: argparse.Namespace) -> tuple[Any, tuple[type[BaseException], ...], FactPipeline]:
    resources = getattr(_worker_local, "resources", None)
    if resources is None:
        resources = (
            create_teacher_client(args.api_format, args.api_key, args.base_url, ssl_verify=not args.no_ssl_verify),
            handled_exception_types(args.api_format),
            FactPipeline(args.terminology_db),
        )
        _worker_local.resources = resources
    return resources


def synthesize_dialogue_task(
    dialogue: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, dict[str, Any], int]:
    client, handled_errors, fact_pipeline = worker_resources(args)
    result = synthesize_dialogue(dialogue, client, handled_errors, fact_pipeline, args)
    if args.sleep > 0:
        time.sleep(args.sleep)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize extractor data dialogue-by-dialogue while rolling patient_state forward."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "dialogue_synthesized.jsonl")
    parser.add_argument("--failed", type=Path, default=DEFAULT_WORK_DIR / "dialogue_failed.jsonl")
    parser.add_argument("--failed-dialogues", type=Path, default=None)
    parser.add_argument("--repair-succeeded", type=Path, default=None)
    parser.add_argument("--final-states", type=Path, default=DEFAULT_WORK_DIR / "dialogue_final_states.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WORK_DIR / "dialogue_checkpoints.jsonl")
    parser.add_argument("--dialogue-work-dir", type=Path, default=None)
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Retry rows from a previous synthesis_failed JSONL instead of reading dialogue source.",
    )
    parser.add_argument(
        "--retry-failed-input",
        type=Path,
        default=None,
        help="Previous synthesis_failed JSONL to retry. Defaults to --failed when omitted.",
    )
    parser.add_argument(
        "--repair-failed-dialogues",
        action="store_true",
        help="Retry whole dialogues from --failed-dialogues and update the ledger in place.",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-format", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--terminology-db",
        type=Path,
        default=None,
        help="Optional local terminology SQLite database. Runtime state uses codes; training output keeps codes null.",
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable TLS certificate verification for API requests. Use only for trusted API gateways.",
    )
    parser.add_argument(
        "--max-dialogues",
        type=int,
        default=None,
        help="Total source-prefix dialogue limit, including dialogues already completed in checkpoints.",
    )
    parser.add_argument(
        "--max-user-turns",
        type=int,
        default=None,
        help="Select a stable source prefix reaching this many patient turns; the final dialogue stays intact.",
    )
    parser.add_argument("--max-context-messages", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of complete dialogues synthesized concurrently. Dialogue turns remain sequential.",
    )
    args = parser.parse_args()

    if args.max_dialogues is not None and args.max_dialogues < 0:
        raise SystemExit("--max-dialogues must be non-negative.")
    if args.max_user_turns is not None and args.max_user_turns < 0:
        raise SystemExit("--max-user-turns must be non-negative.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    args.source = resolve_from_root(args.source)
    args.output = resolve_from_root(args.output)
    args.failed = resolve_from_root(args.failed)
    if args.failed_dialogues is None:
        args.failed_dialogues = args.failed
    args.failed_dialogues = resolve_from_root(args.failed_dialogues)
    if args.repair_succeeded is not None:
        args.repair_succeeded = resolve_from_root(args.repair_succeeded)
    args.final_states = resolve_from_root(args.final_states)
    args.checkpoint = resolve_from_root(args.checkpoint)
    if args.dialogue_work_dir is None:
        args.dialogue_work_dir = args.output.parent / f"_{args.output.stem}_dialogue_work"
    args.dialogue_work_dir = resolve_from_root(args.dialogue_work_dir)
    if args.retry_failed_input is not None:
        args.retry_failed_input = resolve_from_root(args.retry_failed_input)
    if args.terminology_db is not None:
        args.terminology_db = resolve_from_root(args.terminology_db)

    # Validate global dependencies before starting threads. Each worker owns its own clients and caches.
    handled_exception_types(args.api_format)
    FactPipeline(args.terminology_db)

    if args.retry_failed_only:
        raise SystemExit("--retry-failed-only is deprecated. Use --repair-failed-dialogues.")

    if args.repair_failed_dialogues:
        ledger = load_failure_ledger(args.failed_dialogues)
        if not ledger:
            return
        repair_ids = select_failure_dialogue_ids(
            ledger,
            args.source,
            max_dialogues=args.max_dialogues,
            max_user_turns=args.max_user_turns,
        )
        print(
            f"repair ledger={len(ledger)} selected={len(repair_ids)} "
            f"workers={args.workers}"
        )
        if not repair_ids:
            return
        repair_output = args.repair_succeeded or args.output
        repair_dialogues: list[dict[str, Any]] = []
        for dialogue_id in repair_ids:
            record = ledger[dialogue_id]
            dialog = record.get("dialogue")
            if not isinstance(dialog, dict):
                update_failed_dialogue(
                    args.failed_dialogues,
                    dialogue_id=dialogue_id,
                    dialogue={},
                    failure_type=str(record.get("failure_type") or "synthesis_error"),
                    failed_at="repair",
                    error="failed ledger record does not contain a dialogue object",
                    model=args.model,
                    api_format=args.api_format,
                )
                print(f"dialogue-failed {dialogue_id}#unknown")
                continue
            shard = dialogue_shard_path(args.dialogue_work_dir, dialogue_id)
            if shard.exists():
                shard.unlink()
            repair_dialogues.append(dialog)

        task = lambda dialogue: synthesize_dialogue_task(dialogue, args)
        for dialog, result in execute_dialogue_tasks(repair_dialogues, args.workers, task):
            dialogue_id = str(dialog.get("dialogue_id"))
            rows, failure, state, user_turns = result
            if failure is not None or rows is None:
                failure = failure or {}
                update_failed_dialogue(
                    args.failed_dialogues,
                    dialogue_id=dialogue_id,
                    dialogue=dialog,
                    failure_type=str(failure.get("failure_type") or "synthesis_error"),
                    failed_at="repair",
                    failed_turn_id=failure.get("failed_turn_id"),
                    input_row=failure.get("input"),
                    raw_response=str(failure.get("raw_response") or ""),
                    error=str(failure.get("error") or ""),
                    errors=failure.get("errors") or [],
                    model=args.model,
                    api_format=args.api_format,
                )
                print(f"dialogue-failed {dialogue_id}#{failure.get('failed_turn_id')}")
                continue

            write_dialogue_shard(args.dialogue_work_dir, dialogue_id, rows, state, user_turns)
            for output_row in rows:
                append_jsonl(repair_output, output_row)
            append_jsonl(args.final_states, {"dialogue_id": dialogue_id, "patient_state": state})
            append_checkpoint(
                args.checkpoint,
                dialogue_id,
                rows[-1]["input"]["turn_id"] if rows else None,
                state,
                finished=True,
            )
            remove_failed_dialogue(args.failed_dialogues, dialogue_id)
            print(f"dialogue-ok {dialogue_id} turns={len(rows)}")
        return

    checkpoints = load_checkpoints(args.checkpoint)
    shards = load_dialogue_shards(args.dialogue_work_dir)
    selected, selected_user_turns = select_source_dialogues(
        args.source,
        max_dialogues=args.max_dialogues,
        max_user_turns=args.max_user_turns,
    )
    pending = [
        dialog
        for dialog in selected
        if not (checkpoints.get(str(dialog.get("dialogue_id"))) or {}).get("finished")
        and str(dialog.get("dialogue_id")) not in shards
    ]
    print(
        f"selection dialogues={len(selected)} user_turns={selected_user_turns} "
        f"pending={len(pending)} workers={args.workers}"
    )

    task = lambda dialogue: synthesize_dialogue_task(dialogue, args)
    for dialog, result in execute_dialogue_tasks(pending, args.workers, task):
        dialogue_id = str(dialog.get("dialogue_id"))
        rows, failure, state, user_turns = result
        if failure is not None or rows is None:
            failure = failure or {}
            update_failed_dialogue(
                args.failed_dialogues,
                dialogue_id=str(dialogue_id),
                dialogue=dialog,
                failure_type=str(failure.get("failure_type") or "synthesis_error"),
                failed_at="synthesis",
                failed_turn_id=failure.get("failed_turn_id"),
                input_row=failure.get("input"),
                raw_response=str(failure.get("raw_response") or ""),
                error=str(failure.get("error") or ""),
                errors=failure.get("errors") or [],
                model=args.model,
                api_format=args.api_format,
            )
            print(f"dialogue-failed {dialogue_id}#{failure.get('failed_turn_id')}")
            continue

        write_dialogue_shard(args.dialogue_work_dir, dialogue_id, rows, state, user_turns)
        for output_row in rows:
            append_jsonl(args.output, output_row)
        append_jsonl(args.final_states, {"dialogue_id": dialogue_id, "patient_state": state})
        append_checkpoint(
            args.checkpoint,
            dialogue_id,
            rows[-1]["input"]["turn_id"] if rows else None,
            state,
            finished=True,
        )
        remove_failed_dialogue(args.failed_dialogues, dialogue_id)
        print(f"dialogue-ok {dialogue_id} turns={len(rows)}")


if __name__ == "__main__":
    main()
