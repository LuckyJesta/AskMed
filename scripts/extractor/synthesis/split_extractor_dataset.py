from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.extractor.pipeline.schema import (
    ASKMED_ROOT,
    DEFAULT_WORK_DIR,
    is_short_context_answer,
    read_jsonl,
    resolve_from_root,
    write_jsonl,
)
from scripts.extractor.synthesis.convert_to_alpaca import build_alpaca_row


DEFAULT_INPUT = DEFAULT_WORK_DIR / "MedDG_extractor_15k" / "MedDG_extractor_15k_validated.jsonl"
DEFAULT_OUTPUT_PREFIX = DEFAULT_WORK_DIR / "MedDG_extractor_15k"
SPLITS = ("train", "valid", "test")


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ASKMED_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def dialogue_id_of(row: dict[str, Any]) -> str:
    source = row.get("input") or {}
    dialogue_id = source.get("dialogue_id")
    if not dialogue_id:
        raise ValueError("Missing input.dialogue_id in validated row")
    return str(dialogue_id)


def split_dialogues(
    dialogue_ids: list[str],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, set[str]]:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if any(ratio < 0 for ratio in (train_ratio, valid_ratio, test_ratio)) or ratio_sum <= 0:
        raise ValueError("Split ratios must be non-negative and sum to a positive value")

    train_ratio /= ratio_sum
    valid_ratio /= ratio_sum
    test_ratio /= ratio_sum

    shuffled = list(dialogue_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = round(total * train_ratio)
    valid_end = train_end + round(total * valid_ratio)

    return {
        "train": set(shuffled[:train_end]),
        "valid": set(shuffled[train_end:valid_end]),
        "test": set(shuffled[valid_end:]),
    }


def summarize(rows: list[dict[str, Any]], dialogue_ids: set[str]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    facts_total = 0
    context_dependent_rows = 0

    for row in rows:
        source = row.get("input") or {}
        patient_utterance = source.get("patient_utterance") or ""
        if source.get("is_context_dependent") or is_short_context_answer(patient_utterance):
            context_dependent_rows += 1

        facts = (row.get("parsed_output") or {}).get("facts") or []
        facts_total += len(facts)
        for fact in facts:
            if isinstance(fact, dict):
                type_counts[str(fact.get("type"))] += 1
                status_counts[str(fact.get("status"))] += 1
                subject_counts[str(fact.get("subject"))] += 1

    return {
        "rows": len(rows),
        "dialogues": len(dialogue_ids),
        "facts_total": facts_total,
        "context_dependent_rows": context_dependent_rows,
        "type_counts": dict(type_counts),
        "status_counts": dict(status_counts),
        "subject_counts": dict(subject_counts),
    }


def assert_no_dialogue_overlap(split_ids: dict[str, set[str]]) -> None:
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(f"Dialogue overlap between {left} and {right}: {sample}")


def assert_no_weak_label_leak(rows: list[dict[str, str]], split: str) -> None:
    for idx, row in enumerate(rows):
        if "meddg_weak_labels" in row.get("input", ""):
            raise ValueError(f"meddg_weak_labels leaked into {split} Alpaca row {idx}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split validated extractor data by dialogue_id and write Alpaca files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.input = resolve_from_root(args.input)
    args.output_prefix = resolve_from_root(args.output_prefix)

    rows = list(read_jsonl(args.input))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[dialogue_id_of(row)].append(row)

    split_ids = split_dialogues(
        sorted(grouped),
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
        args.seed,
    )
    assert_no_dialogue_overlap(split_ids)

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for dialogue_id in sorted(split_ids[split]):
            split_rows[split].extend(grouped[dialogue_id])

    total_written = sum(len(split_rows[split]) for split in SPLITS)
    if total_written != len(rows):
        raise ValueError(f"Split row total mismatch: {total_written} != {len(rows)}")

    outputs: dict[str, dict[str, str]] = {}
    report = {
        "input": display_path(args.input),
        "output_prefix": display_path(args.output_prefix),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "valid": args.valid_ratio,
            "test": args.test_ratio,
        },
        "total_rows": len(rows),
        "total_dialogues": len(grouped),
        "splits": {},
    }

    for split in SPLITS:
        validated_path = args.output_prefix.parent / f"{args.output_prefix.name}_{split}_validated.jsonl"
        alpaca_path = args.output_prefix.parent / f"{args.output_prefix.name}_{split}_alpaca.jsonl"
        write_jsonl(validated_path, split_rows[split])

        alpaca_rows = [build_alpaca_row(row) for row in split_rows[split]]
        assert_no_weak_label_leak(alpaca_rows, split)
        write_jsonl(alpaca_path, alpaca_rows)

        outputs[split] = {
            "validated": display_path(validated_path),
            "alpaca": display_path(alpaca_path),
        }
        report["splits"][split] = summarize(split_rows[split], split_ids[split])

    report["outputs"] = outputs
    report_path = args.output_prefix.parent / f"{args.output_prefix.name}_split_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
