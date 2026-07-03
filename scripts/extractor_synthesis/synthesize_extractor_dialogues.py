from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor_common import (
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
    weak_labels_from,
)
from state_manager import compact_patient_state, empty_patient_state, merge_facts_into_state, prompt_patient_state
from terminology_normalizer import TerminologyNormalizer


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
            "【辅助理解信息：只能用于理解短回答和绑定目标，不能直接作为事实或evidence】\n"
            f"上一轮医生问题：{row.get('previous_doctor_question') or '无'}\n"
            "当前问诊状态（只由当前轮之前的信息形成，不包含当前发言）：\n"
            f"{json.dumps(row.get('patient_state_before_turn') or {}, ensure_ascii=False, separators=(',', ':'))}\n"
            "最近对话上下文：省略，避免旧患者发言污染本轮evidence。\n"
            "短回答特别规则：如果患者当前发言是“没有/没/无/不”等否定回答，被询问对象应为absent，evidence必须使用患者当前发言原文，不能复制医生问题。\n"
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
        "1. 只抽取患者当前发言中明确存在或明确不存在的医学事实，status只能是present或absent，不要输出uncertain。\n"
        "2. 患者咨询、猜测、担忧、请求建议、诊断咨询、用药咨询、检查咨询不作为facts。\n"
        "3. 一句话同时包含明确事实和咨询时，只抽取明确事实部分。\n"
        "4. evidence必须来自患者当前发言的连续原文片段。\n"
        "5. 当前问诊状态和最近对话上下文中的既有事实不要重复输出，除非患者当前发言重新明确表达了该事实。\n"
        "6. 输出前逐条自检：如果某个fact的evidence不是患者当前发言的连续子串，必须删除该fact。\n"
        "7. 如果当前发言只是寒暄、确认、感谢或告别且没有新医学信息，输出 {\"facts\":[]}。\n"
        "8. 如果当前发言是短回答，结合上一轮医生问题和当前问诊状态判断目标；无法判断目标时输出 {\"facts\":[]}。\n"
        "9. 如果当前发言中症状本体和属性同时出现，把属性合并进该症状fact，不要重复输出attribute fact。\n"
        "10. 再次确认：最终facts只能由【唯一抽取对象】中的患者当前发言支持，不能抽取辅助理解信息中的旧事实。\n"
        "11. 请只输出紧凑JSON，不要换行、缩进或多余空格。"
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize extractor data dialogue-by-dialogue while rolling patient_state forward."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "dialogue_synthesized.jsonl")
    parser.add_argument("--failed", type=Path, default=DEFAULT_WORK_DIR / "dialogue_failed.jsonl")
    parser.add_argument("--final-states", type=Path, default=DEFAULT_WORK_DIR / "dialogue_final_states.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WORK_DIR / "dialogue_checkpoints.jsonl")
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
    parser.add_argument("--max-dialogues", type=int, default=None)
    parser.add_argument("--max-user-turns", type=int, default=None)
    parser.add_argument("--max-context-messages", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    args.source = resolve_from_root(args.source)
    args.output = resolve_from_root(args.output)
    args.failed = resolve_from_root(args.failed)
    args.final_states = resolve_from_root(args.final_states)
    args.checkpoint = resolve_from_root(args.checkpoint)
    if args.terminology_db is not None:
        args.terminology_db = resolve_from_root(args.terminology_db)

    client = create_teacher_client(args.api_format, args.api_key, args.base_url, ssl_verify=not args.no_ssl_verify)
    handled_errors = handled_exception_types(args.api_format)
    normalizer = TerminologyNormalizer(args.terminology_db)
    if args.terminology_db is not None and not normalizer.enabled:
        raise SystemExit(f"Terminology database not found: {args.terminology_db}")

    checkpoints = load_checkpoints(args.checkpoint)
    processed_dialogues = 0
    processed_turns = 0

    for dialog in read_jsonl(args.source):
        if args.max_dialogues is not None and processed_dialogues >= args.max_dialogues:
            break
        dialogue_id = dialog.get("dialogue_id")
        messages = dialog.get("messages") or []
        state = empty_patient_state(dialogue_id)
        completed_turn_id: int | None = None
        checkpoint = checkpoints.get(str(dialogue_id))
        if checkpoint:
            if checkpoint.get("finished"):
                continue
            checkpoint_state = checkpoint.get("patient_state")
            if isinstance(checkpoint_state, dict):
                state = checkpoint_state
            raw_completed = checkpoint.get("completed_turn_id")
            if isinstance(raw_completed, int):
                completed_turn_id = raw_completed
        processed_dialogues += 1

        for idx, message in enumerate(messages):
            if args.max_user_turns is not None and processed_turns >= args.max_user_turns:
                append_jsonl(args.final_states, {"dialogue_id": dialogue_id, "patient_state": state})
                return
            if message.get("role") != "user":
                continue
            if completed_turn_id is not None and idx <= completed_turn_id:
                continue

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

            parsed = None
            raw_response = ""
            model_name = args.model
            if is_pure_non_medical_chatter(row["patient_utterance"]):
                parsed = {"facts": []}
                raw_response = json.dumps(parsed, ensure_ascii=False)
                model_name = "rule:pure_non_medical_chatter"
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
                        break
                    except handled_errors as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt < args.max_retries:
                            time.sleep(args.sleep * attempt)
                        else:
                            append_jsonl(
                                args.failed,
                                {
                                    "input": row,
                                    "raw_response": raw_response,
                                    "error": last_error,
                                    "model": args.model,
                                    "api_format": args.api_format,
                                },
                            )
                            print(f"failed {dialogue_id}#{idx}: {last_error}")
                if parsed is None:
                    processed_turns += 1
                    continue

            runtime_parsed = normalizer.normalize_extraction_for_runtime(parsed)
            training_parsed = normalizer.project_extraction_for_training(runtime_parsed)
            state_after = merge_facts_into_state(state, runtime_parsed.get("facts") or [], idx)
            append_jsonl(
                args.output,
                {
                    "input": row,
                    "raw_response": raw_response,
                    "parsed_output": training_parsed,
                    "patient_state_after_turn": compact_patient_state(state_after),
                    "model": model_name,
                    "api_format": args.api_format if model_name == args.model else "rule",
                },
            )
            state = state_after
            processed_turns += 1
            completed_turn_id = idx
            append_checkpoint(args.checkpoint, dialogue_id, idx, state, finished=False)
            print(f"ok {dialogue_id}#{idx}" if model_name == args.model else f"rule-empty {dialogue_id}#{idx}")
            time.sleep(args.sleep)

        append_jsonl(args.final_states, {"dialogue_id": dialogue_id, "patient_state": state})
        append_checkpoint(args.checkpoint, dialogue_id, completed_turn_id, state, finished=True)


if __name__ == "__main__":
    main()
