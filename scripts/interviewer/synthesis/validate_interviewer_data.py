from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.extractor.pipeline.schema import read_jsonl, resolve_from_root
from scripts.extractor.synthesis.failure_ledger import (
    load_failure_ledger,
    upsert_failed_dialogue,
    write_failure_ledger,
)
from scripts.interviewer.pipeline.schema import (
    GENERIC_TARGET_NAMES,
    SKIP_REASON_CODES,
    answerability_match_method,
    is_compound_question,
    validate_decision,
    validate_teacher_candidates,
    validate_teacher_answerability,
)


HIDDEN_STATE_KEYS = {
    "dialogue_id",
    "turn_id",
    "standard_code",
    "terminology",
    "aliases",
    "evidence",
    "first_turn_id",
    "last_turn_id",
}


def contains_hidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in HIDDEN_STATE_KEYS or contains_hidden_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_hidden_key(item) for item in value)
    return False


def validate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parent_id = str(row.get("source_dialogue_id") or row.get("dialogue_id") or "")
        by_dialogue[parent_id].append(row)

    valid: list[dict[str, Any]] = []
    failed: dict[str, list[str]] = {}
    actions: Counter[str] = Counter()
    usable_count = 0
    answerable_ask = 0
    ask_count = 0
    sessions: set[str] = set()
    split_sessions = 0
    skip_reasons: Counter[str] = Counter()
    target_types: Counter[str] = Counter()
    target_attributes: Counter[str] = Counter()
    generic_targets = 0
    specific_targets = 0
    source_traceable = 0
    compound_questions = 0
    repeated_targets = 0
    known_targets = 0
    teacher_candidate_count = 0
    answerability_methods: Counter[str] = Counter()
    rejected_unrelated_same_type_deltas = 0
    split_reasons: Counter[str] = Counter()
    ambiguous_boundaries = 0
    for dialogue_id, dialogue_rows in by_dialogue.items():
        dialogue_rows.sort(
            key=lambda row: (
                int(row.get("session_index") or 0),
                int(row.get("block_id") or 0),
            )
        )
        errors: list[str] = []
        rows_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in dialogue_rows:
            session_id = str(row.get("dialogue_id") or "")
            rows_by_session[session_id].append(row)
        split_sessions += max(0, len(rows_by_session) - 1)
        for session_id, session_rows in rows_by_session.items():
            sessions.add(session_id)
            session_rows.sort(key=lambda row: int(row.get("block_id") or 0))
            expected_targets: list[dict[str, Any]] = []
            previous_block = -1
            for row in session_rows:
                block_id = int(row.get("block_id") or 0)
                prefix = f"{session_id} block {block_id}"
                if block_id <= previous_block:
                    errors.append(f"{prefix}: block order is not strictly increasing")
                previous_block = block_id
                if row.get("asked_targets_before") != expected_targets:
                    errors.append(f"{prefix}: asked_targets_before breaks trajectory continuity")
                state = row.get("patient_state") or {}
                if row.get("session_split_reason"):
                    split_reasons[str(row["session_split_reason"])] += 1
                if row.get("session_boundary_warning"):
                    ambiguous_boundaries += 1
                if contains_hidden_key(state):
                    errors.append(f"{prefix}: projected state contains internal fields")
                teacher_response = row.get("teacher_response")
                if isinstance(teacher_response, dict):
                    teacher_candidate_count += len(teacher_response.get("questions") or [])
                    teacher_errors = validate_teacher_candidates(teacher_response, row)
                    errors.extend(f"{prefix}: {error}" for error in teacher_errors)
                decision = row.get("parsed_output") or {}
                decision_errors = validate_decision(
                    decision,
                    state=state,
                    asked_targets=expected_targets,
                    teacher_mode=True,
                    is_terminal=bool(row.get("is_terminal")),
                )
                decision_errors.extend(validate_teacher_answerability(decision, row))
                repeated_targets += sum("already been asked" in error for error in decision_errors)
                known_targets += sum("already known" in error for error in decision_errors)
                errors.extend(f"{prefix}: {error}" for error in decision_errors)
                usable = decision.get("usable", True)
                if usable:
                    usable_count += 1
                    action = str(decision.get("action") or "")
                    actions[action] += 1
                    if action == "ask":
                        ask_count += 1
                        target = decision.get("next_question_target")
                        selected = row.get("selected_teacher_candidate") or {}
                        source_text = str(selected.get("source_text") or "")
                        if source_text and any(
                            source_text in str(message or "")
                            for message in row.get("original_doctor_block") or []
                        ):
                            source_traceable += 1
                        if isinstance(target, dict):
                            target_types[str(target.get("type") or "")] += 1
                            target_attributes[str(target.get("attribute") or "null")] += 1
                            if str(target.get("name") or "") in GENERIC_TARGET_NAMES:
                                generic_targets += 1
                            else:
                                specific_targets += 1
                            if is_compound_question(str(decision.get("utterance") or ""), target):
                                compound_questions += 1
                            selected = row.get("selected_teacher_candidate") or {}
                            method = answerability_match_method(
                                target,
                                state,
                                row.get("next_patient_state") or {},
                                next_patient_block=row.get("next_patient_block") or [],
                                question_text=str(selected.get("source_text") or "")
                                + str(decision.get("utterance") or ""),
                            )
                            if method is not None:
                                answerable_ask += 1
                                answerability_methods[method] += 1
                                if row.get("answerability_method") != method:
                                    errors.append(f"{prefix}: answerability_method does not match validation")
                            expected_targets.append(dict(target))
                else:
                    code = str(decision.get("skip_reason_code") or "")
                    skip_reasons[code] += 1
                    if decision.get("skip_reason_detail") == "unrelated same-type state delta rejected":
                        rejected_unrelated_same_type_deltas += 1
                    if code not in SKIP_REASON_CODES:
                        errors.append(f"{prefix}: invalid skip_reason_code")
                if row.get("asked_targets_after") != expected_targets:
                    errors.append(f"{prefix}: asked_targets_after breaks trajectory continuity")
        if errors:
            failed[dialogue_id] = errors
        else:
            valid.extend(dialogue_rows)

    report = {
        "rows": len(rows),
        "parent_dialogues": len(by_dialogue),
        "dialogues": len(by_dialogue),
        "sessions": len(sessions),
        "session_splits": split_sessions,
        "session_split_reasons": dict(split_reasons),
        "ambiguous_session_boundaries": ambiguous_boundaries,
        "valid_rows": len(valid),
        "valid_parent_dialogues": len(by_dialogue) - len(failed),
        "valid_dialogues": len(by_dialogue) - len(failed),
        "failed_parent_dialogues": len(failed),
        "failed_dialogues": len(failed),
        "usable_samples": usable_count,
        "actions": dict(actions),
        "skip_reasons": dict(skip_reasons),
        "teacher_question_candidates": teacher_candidate_count,
        "teacher_reference_answerability": answerable_ask / ask_count if ask_count else 0.0,
        "source_question_traceability": source_traceable / ask_count if ask_count else 0.0,
        "compound_questions": compound_questions,
        "repeated_targets": repeated_targets,
        "known_targets": known_targets,
        "generic_targets": generic_targets,
        "specific_targets": specific_targets,
        "target_types": dict(target_types),
        "target_attributes": dict(target_attributes),
        "answerability_methods": dict(answerability_methods),
        "lexical_matches": sum(
            answerability_methods[method]
            for method in ("lexical_state_delta", "lexical_answer_match")
        ),
        "short_answer_matches": sum(
            answerability_methods[method]
            for method in ("explicit_short_answer", "short_attribute_answer")
        ),
        "target_not_resolved": skip_reasons["TARGET_NOT_RESOLVED"],
        "rejected_unrelated_same_type_deltas": rejected_unrelated_same_type_deltas,
    }
    return valid, failed, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate interviewer synthesis by complete dialogue.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for name in ("input", "output", "failed", "source", "report"):
        setattr(args, name, resolve_from_root(getattr(args, name)))

    rows = list(read_jsonl(args.input))
    valid_rows, failed, report = validate_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in valid_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    source_map = {str(row.get("dialogue_id") or ""): row for row in read_jsonl(args.source)}
    ledger = load_failure_ledger(args.failed)
    for dialogue_id, errors in failed.items():
        upsert_failed_dialogue(
            ledger,
            dialogue_id=dialogue_id,
            dialogue=source_map.get(dialogue_id, {}),
            failure_type="interviewer_validation_error",
            failed_at="validation",
            error="; ".join(errors),
            errors=errors,
        )
    for dialogue_id in set(str(row.get("source_dialogue_id") or row.get("dialogue_id") or "") for row in valid_rows):
        ledger.pop(dialogue_id, None)
    write_failure_ledger(args.failed, ledger)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
