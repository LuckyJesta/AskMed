from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from extractor_common import (
    DEFAULT_WORK_DIR,
    append_jsonl,
    is_short_context_answer,
    normalize_extraction,
    read_jsonl,
    resolve_from_root,
    validate_extraction,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthesized extractor JSONL data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_WORK_DIR / "deepseek_synthesized.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "validated.jsonl")
    parser.add_argument("--failed", type=Path, default=DEFAULT_WORK_DIR / "validation_failed.jsonl")
    parser.add_argument("--report", type=Path, default=DEFAULT_WORK_DIR / "validation_report.json")
    args = parser.parse_args()

    args.input = resolve_from_root(args.input)
    args.output = resolve_from_root(args.output)
    args.failed = resolve_from_root(args.failed)
    args.report = resolve_from_root(args.report)

    valid_rows: list[dict] = []
    total = 0
    invalid = 0
    facts_total = 0
    context_dependent = 0
    type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()

    if args.failed.exists():
        args.failed.unlink()

    for row in read_jsonl(args.input):
        total += 1
        source = row.get("input") or {}
        extraction = normalize_extraction(row.get("parsed_output") or {})
        row["parsed_output"] = extraction
        patient_utterance = source.get("patient_utterance") or ""
        if source.get("is_context_dependent") or is_short_context_answer(patient_utterance):
            context_dependent += 1
        ok, errors = validate_extraction(extraction or {}, patient_utterance)
        if not ok:
            invalid += 1
            append_jsonl(args.failed, {"row": row, "errors": errors})
            continue
        facts = extraction.get("facts") or []
        facts_total += len(facts)
        for fact in facts:
            type_counts[fact.get("type")] += 1
            status_counts[fact.get("status")] += 1
            subject_counts[fact.get("subject")] += 1
        valid_rows.append(row)

    written = write_jsonl(args.output, valid_rows)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "total_rows": total,
        "valid_rows": written,
        "invalid_rows": invalid,
        "valid_rate": written / total if total else 0,
        "context_dependent_rows": context_dependent,
        "facts_total": facts_total,
        "average_facts_per_valid_row": facts_total / written if written else 0,
        "type_counts": dict(type_counts),
        "status_counts": dict(status_counts),
        "subject_counts": dict(subject_counts),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
