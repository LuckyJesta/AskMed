from __future__ import annotations

import argparse
import json
from pathlib import Path

from extractor_common import (
    DEFAULT_WORK_DIR,
    TRAINING_SYSTEM_PROMPT,
    format_alpaca_input,
    read_jsonl,
    resolve_from_root,
    write_jsonl,
)


INSTRUCTION = "请根据上一轮医生问题、患者当前发言、当前问诊状态和最近对话上下文，抽取患者当前发言中的医学事实。"


def build_alpaca_row(row: dict) -> dict[str, str]:
    source = row.get("input") or {}
    parsed_output = row.get("parsed_output") or {"facts": []}
    return {
        "system": TRAINING_SYSTEM_PROMPT,
        "instruction": INSTRUCTION,
        "input": format_alpaca_input(source),
        "output": json.dumps(parsed_output, ensure_ascii=False, separators=(",", ":")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert validated extractor data to LLaMA-Factory Alpaca format.")
    parser.add_argument("--input", type=Path, default=DEFAULT_WORK_DIR / "validated.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "alpaca_extractor.jsonl")
    args = parser.parse_args()

    args.input = resolve_from_root(args.input)
    args.output = resolve_from_root(args.output)

    # Intentionally exclude MedDG weak labels from the final fine-tuning input.
    rows = [build_alpaca_row(row) for row in read_jsonl(args.input)]
    count = write_jsonl(args.output, rows)
    print(f"Wrote {count} Alpaca rows to {args.output}")


if __name__ == "__main__":
    main()
