from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from extractor_common import (
    DEFAULT_WORK_DIR,
    append_jsonl,
    extract_json_object,
    format_context,
    is_pure_non_medical_chatter,
    normalize_extraction,
    read_jsonl,
    sample_id,
)


SYSTEM_PROMPT = """你是一个严谨的中文医学事实抽取器。
你的任务：只根据【患者当前发言】抽取医学事实；上一轮医生问题和最近对话上下文只用于理解短回答、消解指代、绑定属性目标。
禁止诊断，禁止扩写，禁止补充患者未提及的信息，禁止把医生问题或历史上下文直接当作当前患者事实。
MedDG弱标签可能不完整或有误，只能作为参考。
输出必须是纯JSON，不要Markdown，不要解释。
顶层格式必须是 {"facts":[...]}。
每个fact必须包含字段：name, normalized_name, type, status, subject, time, body_part, attribute, evidence, standard_code, terminology。
type只能是 symptom/disease/medicine/examination/attribute/history/lifestyle/other。
status只能是 present/absent/uncertain。
subject只能是 patient/family/other/unknown。
attribute必须是JSON对象；没有补充属性时填 {}，不能填空字符串。
time和body_part没有信息时填null，不能填空字符串。
evidence必须是患者当前发言中的连续原文片段。
第一版不做标准编码，standard_code和terminology必须为null。
如果没有医学事实，输出 {"facts":[]}。
如果患者当前发言只是问候、感谢、确认、寒暄或告别，例如“好的”“谢谢”“嗯嗯”“知道了”“再见”，且没有包含新的医学事实、用药计划、检查计划或症状变化，则输出 {"facts":[]}。"""


def build_user_prompt(row: dict[str, Any]) -> str:
    return (
        "请抽取患者当前发言中的医学事实。\n\n"
        f"上一轮医生问题：{row.get('previous_doctor_question') or '无'}\n"
        f"患者当前发言：{row.get('patient_utterance') or ''}\n"
        f"最近对话上下文：\n{format_context(row.get('recent_context') or [])}\n"
        f"MedDG弱标签（仅供参考，可能不完整或有误）："
        f"{json.dumps(row.get('meddg_weak_labels') or {}, ensure_ascii=False)}\n\n"
        "请只输出JSON。"
    )


def load_done_ids(*paths: Path) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            source = row.get("input") or row
            if "dialogue_id" in source and "turn_id" in source:
                done.add(sample_id(source))
    return done


def call_deepseek(
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
        raise ValueError("DeepSeek returned an empty message content")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize extractor labels with the DeepSeek API.")
    parser.add_argument("--input", type=Path, default=DEFAULT_WORK_DIR / "extractor_inputs_sample.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "deepseek_synthesized.jsonl")
    parser.add_argument("--failed", type=Path, default=DEFAULT_WORK_DIR / "deepseek_failed.jsonl")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY environment variable.")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    done = load_done_ids(args.output, args.failed)
    processed = 0
    for row in read_jsonl(args.input):
        if args.limit is not None and processed >= args.limit:
            break
        sid = sample_id(row)
        if sid in done:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ]
        if is_pure_non_medical_chatter(row.get("patient_utterance") or ""):
            parsed = {"facts": []}
            append_jsonl(
                args.output,
                {
                    "input": row,
                    "raw_response": json.dumps(parsed, ensure_ascii=False),
                    "parsed_output": parsed,
                    "model": "rule:pure_non_medical_chatter",
                },
            )
            processed += 1
            done.add(sid)
            print(f"rule-empty {sid}")
            continue
        raw_response = ""
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                raw_response = call_deepseek(client, args.model, messages, args.timeout)
                parsed = normalize_extraction(extract_json_object(raw_response))
                append_jsonl(
                    args.output,
                    {
                        "input": row,
                        "raw_response": raw_response,
                        "parsed_output": parsed,
                        "model": args.model,
                    },
                )
                processed += 1
                done.add(sid)
                print(f"ok {sid}")
                break
            except (
                APIConnectionError,
                APIError,
                APITimeoutError,
                RateLimitError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as exc:
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
                        },
                    )
                    processed += 1
                    done.add(sid)
                    print(f"failed {sid}: {last_error}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
