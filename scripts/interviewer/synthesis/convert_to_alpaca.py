from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.extractor.pipeline.schema import read_jsonl, resolve_from_root
from scripts.interviewer.pipeline.schema import INTERVIEWER_SYSTEM_PROMPT, build_interviewer_input


def build_alpaca_row(row: dict[str, Any]) -> dict[str, str]:
    decision = row.get("parsed_output") or {}
    output = {
        "action": decision.get("action"),
        "next_question_target": decision.get("next_question_target"),
        "utterance": decision.get("utterance") or "",
    }
    return {
        "system": INTERVIEWER_SYSTEM_PROMPT,
        "instruction": "决定下一步问诊动作。",
        "input": build_interviewer_input(
            row.get("patient_block") or [],
            row.get("patient_state") or {},
            row.get("recent_context") or [],
            row.get("asked_targets_before") or [],
        ),
        "output": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
    }


def convert(rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    all_rows: list[dict[str, str]] = []
    splits: dict[str, list[dict[str, str]]] = {"train": [], "valid": [], "test": []}
    for row in rows:
        decision = row.get("parsed_output") or {}
        if not decision.get("usable", True):
            continue
        alpaca = build_alpaca_row(row)
        all_rows.append(alpaca)
        split = str(row.get("split") or "")
        if split not in splits:
            raise ValueError(f"invalid inherited split: {split!r}")
        splits[split].append(alpaca)
    return all_rows, splits


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert validated interviewer rows to inherited Alpaca splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    args.input = resolve_from_root(args.input)
    args.output_prefix = resolve_from_root(args.output_prefix)
    all_rows, splits = convert(read_jsonl(args.input))
    outputs = {"all": args.output_prefix.with_name(args.output_prefix.name + "_alpaca.jsonl")}
    write_rows(outputs["all"], all_rows)
    for split, split_rows in splits.items():
        path = args.output_prefix.with_name(args.output_prefix.name + f"_{split}_alpaca.jsonl")
        outputs[split] = path
        write_rows(path, split_rows)
    report = {
        "all": len(all_rows),
        "train": len(splits["train"]),
        "valid": len(splits["valid"]),
        "test": len(splits["test"]),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    report_path = args.output_prefix.with_name(args.output_prefix.name + "_conversion_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
