from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.extractor.pipeline.schema import read_jsonl
from scripts.interviewer.pipeline.schema import (
    UNSAFE_PATTERNS,
    extract_json_object,
    target_is_known,
    target_key,
    validate_decision,
)


def parse_input_section(text: str, label: str, next_label: str | None) -> Any:
    start = text.find(label)
    if start < 0:
        return None
    start += len(label)
    end = text.find(next_label, start) if next_label else len(text)
    raw = text[start : end if end >= 0 else len(text)].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_runtime_input(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = parse_input_section(text, "当前问诊状态：", "\n最近对话上下文：")
    asked = parse_input_section(text, "已问目标：", None)
    return state if isinstance(state, dict) else {}, asked if isinstance(asked, list) else []


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def prf(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = ratio(tp, predicted)
    recall = ratio(tp, gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "predicted": predicted, "gold": gold, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structured AskMed interviewer predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dataset", type=Path)
    parser.add_argument("--audit-split", default="test")
    args = parser.parse_args()

    predictions = list(read_jsonl(args.predictions))
    dataset = list(read_jsonl(args.dataset))
    if len(predictions) != len(dataset):
        raise SystemExit(f"prediction count {len(predictions)} != dataset count {len(dataset)}")
    audit_rows = list(read_jsonl(args.audit_dataset)) if args.audit_dataset else []
    usable_audit = [
        row
        for row in audit_rows
        if str(row.get("split") or "") == args.audit_split
        and (row.get("parsed_output") or {}).get("usable", True)
    ]
    if usable_audit and len(usable_audit) != len(dataset):
        raise SystemExit(f"usable audit count {len(usable_audit)} != dataset count {len(dataset)}")

    parsed = schema_valid = action_correct = exact = 0
    predicted_targets = gold_targets = matched_targets = 0
    predicted_end = gold_end = matched_end = 0
    ask_predictions = one_question = unsafe = known_reasks = repeated_targets = 0
    target_gain = target_answerable = 0

    for index, (prediction_row, sample) in enumerate(zip(predictions, dataset)):
        gold = extract_json_object(str(prediction_row.get("label") or sample.get("output") or "{}"))
        if gold.get("action") == "end":
            gold_end += 1
        if target_key(gold.get("next_question_target")) is not None:
            gold_targets += 1
        try:
            predicted = extract_json_object(str(prediction_row.get("predict") or ""))
        except (json.JSONDecodeError, ValueError):
            continue
        parsed += 1
        state, asked = parse_runtime_input(str(sample.get("input") or ""))
        errors = validate_decision(predicted, state=state, asked_targets=asked)
        if not errors:
            schema_valid += 1
        if predicted.get("action") == gold.get("action"):
            action_correct += 1
        if predicted == gold:
            exact += 1

        predicted_key = target_key(predicted.get("next_question_target"))
        gold_key = target_key(gold.get("next_question_target"))
        if predicted_key is not None:
            predicted_targets += 1
        if predicted_key is not None and predicted_key == gold_key:
            matched_targets += 1
        if predicted.get("action") == "end":
            predicted_end += 1
            if gold.get("action") == "end":
                matched_end += 1
        if predicted.get("action") == "ask":
            ask_predictions += 1
            utterance = str(predicted.get("utterance") or "")
            if utterance.count("？") + utterance.count("?") == 1:
                one_question += 1
            if any(pattern in utterance for pattern in UNSAFE_PATTERNS):
                unsafe += 1
            target = predicted.get("next_question_target")
            if isinstance(target, dict) and target_is_known(target, state):
                known_reasks += 1
            asked_keys = {key for item in asked if (key := target_key(item)) is not None}
            if predicted_key is not None and predicted_key in asked_keys:
                repeated_targets += 1

        if usable_audit:
            audit = usable_audit[index]
            gold_target = gold.get("next_question_target")
            if isinstance(gold_target, dict) and audit.get("next_patient_block"):
                target_answerable += 1
                before = audit.get("patient_state") or {}
                after = audit.get("next_patient_state") or {}
                if not target_is_known(gold_target, before) and target_is_known(gold_target, after):
                    target_gain += 1

    metrics = {
        "samples": len(dataset),
        "json_parse_rate": ratio(parsed, len(dataset)),
        "schema_valid_rate": ratio(schema_valid, len(dataset)),
        "exact_match": ratio(exact, len(dataset)),
        "action_accuracy": ratio(action_correct, len(dataset)),
        "target": prf(matched_targets, predicted_targets, gold_targets),
        "end": prf(matched_end, predicted_end, gold_end),
        "one_question_rate": ratio(one_question, ask_predictions),
        "unsafe_advice_rate": ratio(unsafe, ask_predictions),
        "known_fact_reask_rate": ratio(known_reasks, ask_predictions),
        "asked_target_repeat_rate": ratio(repeated_targets, ask_predictions),
        "teacher_reference_answerability": ratio(target_answerable, gold_targets) if usable_audit else None,
        "next_state_target_gain_rate": ratio(target_gain, gold_targets) if usable_audit else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
