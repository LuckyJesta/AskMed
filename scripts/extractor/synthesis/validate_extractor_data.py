from __future__ import annotations

import argparse
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path

from scripts.extractor.pipeline.fact_validator import (
    validate_runtime_extraction,
    validate_state_transition,
    validate_training_projection,
)
from scripts.extractor.pipeline.schema import (
    DEFAULT_WORK_DIR,
    clean_extraction_for_training,
    is_short_context_answer,
    normalize_extraction,
    read_jsonl,
    resolve_from_root,
    write_jsonl,
)
from scripts.extractor.pipeline.state_manager import (
    compact_patient_state,
    empty_patient_state,
    merge_facts_into_state,
    prompt_patient_state,
    resolve_active_target,
)
from scripts.extractor.pipeline.terminology_normalizer import TerminologyNormalizer
from scripts.extractor.synthesis.failure_ledger import (
    load_failure_ledger,
    upsert_failed_dialogue,
    write_failure_ledger,
)


def load_source_dialogues(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    dialogues: dict[str, dict] = {}
    for dialog in read_jsonl(path):
        dialogue_id = dialog.get("dialogue_id")
        if dialogue_id:
            dialogues[str(dialogue_id)] = dialog
    return dialogues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthesized extractor JSONL data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_WORK_DIR / "deepseek_synthesized.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "validated.jsonl")
    parser.add_argument("--failed", type=Path, default=DEFAULT_WORK_DIR / "validation_failed.jsonl")
    parser.add_argument("--failed-dialogues", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_WORK_DIR / "validation_report.json")
    parser.add_argument("--terminology-db", type=Path, default=None)
    args = parser.parse_args()

    args.input = resolve_from_root(args.input)
    args.output = resolve_from_root(args.output)
    args.failed = resolve_from_root(args.failed)
    if args.failed_dialogues is None:
        args.failed_dialogues = args.failed
    args.failed_dialogues = resolve_from_root(args.failed_dialogues)
    if args.source is not None:
        args.source = resolve_from_root(args.source)
    args.report = resolve_from_root(args.report)
    if args.terminology_db is not None:
        args.terminology_db = resolve_from_root(args.terminology_db)
    normalizer = TerminologyNormalizer(args.terminology_db)
    if args.terminology_db is not None:
        normalizer.ensure_database_readable()

    valid_rows: list[dict] = []
    total = 0
    invalid = 0
    validation_failed_dialogues = 0
    facts_total = 0
    context_dependent = 0
    type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()

    source_dialogues = load_source_dialogues(args.source)
    failure_records = load_failure_ledger(args.failed_dialogues)
    grouped: dict[str, list[dict]] = defaultdict(list)
    if args.input.exists():
        for row in read_jsonl(args.input):
            total += 1
            source = row.get("input") or {}
            dialogue_id = str(source.get("dialogue_id") or "")
            grouped[dialogue_id].append(row)

    for dialogue_id, rows in grouped.items():
        rows.sort(key=lambda item: int((item.get("input") or {}).get("turn_id") or -1))
        dialogue_errors: list[str] = []
        dialogue_failure_type = "raw_validation_error"
        first_bad_row: dict | None = None
        normalized_rows: list[dict] = []
        state = empty_patient_state(dialogue_id)
        for row in rows:
            source = row.get("input") or {}
            extraction = normalize_extraction(row.get("parsed_output") or {})
            row["parsed_output"] = extraction
            patient_utterance = source.get("patient_utterance") or ""
            row_context_dependent = bool(
                source.get("is_context_dependent") or is_short_context_answer(patient_utterance)
            )
            if row_context_dependent:
                context_dependent += 1
            errors = validate_runtime_extraction(extraction or {}, patient_utterance, normalizer)
            training_projection = clean_extraction_for_training(extraction)
            projection_errors = validate_training_projection(extraction, training_projection)
            if projection_errors:
                dialogue_failure_type = "projection_error"
                errors.extend(projection_errors)

            is_new_pipeline_row = "normalization_stats" in row
            if is_new_pipeline_row and source.get("patient_state_before_turn") != prompt_patient_state(state):
                dialogue_failure_type = "state_error"
                errors.append("patient_state_before_turn does not match the previous turn state")
            if not errors:
                state_after = merge_facts_into_state(
                    state,
                    extraction.get("facts") or [],
                    source.get("turn_id"),
                    active_target_name=resolve_active_target(state, source.get("previous_doctor_question")),
                )
                state_errors = validate_state_transition(state, state_after)
                if state_errors:
                    dialogue_failure_type = "state_error"
                    errors.extend(state_errors)
                elif is_new_pipeline_row and row.get("patient_state_after_turn") != compact_patient_state(state_after):
                    dialogue_failure_type = "state_error"
                    errors.append("patient_state_after_turn does not match replayed state")
                else:
                    state = state_after

            if errors:
                invalid += 1
                if first_bad_row is None:
                    first_bad_row = row
                turn_id = source.get("turn_id")
                dialogue_errors.extend([f"turn {turn_id}: {error}" for error in errors])
            normalized_rows.append(row)

        if dialogue_errors:
            validation_failed_dialogues += 1
            bad_input = (first_bad_row or {}).get("input") or {}
            upsert_failed_dialogue(
                failure_records,
                dialogue_id=dialogue_id,
                dialogue=source_dialogues.get(dialogue_id) or {"dialogue_id": dialogue_id},
                failure_type=dialogue_failure_type,
                failed_at="validation",
                failed_turn_id=bad_input.get("turn_id"),
                input_row=bad_input,
                raw_response=str((first_bad_row or {}).get("raw_response") or ""),
                errors=dialogue_errors,
                model=(first_bad_row or {}).get("model"),
                api_format=(first_bad_row or {}).get("api_format"),
            )
            continue

        failure_records.pop(dialogue_id, None)
        for row in normalized_rows:
            facts = row.get("parsed_output", {}).get("facts") or []
            facts_total += len(facts)
            for fact in facts:
                type_counts[fact.get("type")] += 1
                status_counts[fact.get("status")] += 1
                subject_counts[fact.get("subject")] += 1
            valid_rows.append(row)

    write_failure_ledger(args.failed_dialogues, failure_records)
    written = write_jsonl(args.output, valid_rows)
    failure_type_counts = Counter(
        str(record.get("failure_type") or "unknown")
        for record in failure_records.values()
    )
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "total_rows": total,
        "valid_rows": written,
        "invalid_rows": invalid,
        "failed_dialogues": len(failure_records),
        "validation_failed_dialogues": validation_failed_dialogues,
        "failure_type_counts": dict(failure_type_counts),
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
