from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terminology_normalizer import TerminologyNormalizer


FACT_TYPES = {
    "symptom",
    "disease",
    "medicine",
    "examination",
    "attribute",
    "history",
    "lifestyle",
    "other",
}
STATUSES = {"present", "absent", "uncertain"}
SUBJECTS = {"patient", "family", "other", "unknown"}
REQUIRED_FACT_FIELDS = {
    "name",
    "normalized_name",
    "type",
    "status",
    "subject",
    "time",
    "body_part",
    "attribute",
    "evidence",
    "standard_code",
    "terminology",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSONL: {exc}") from exc
    return rows


def parse_json_object(text: str) -> dict[str, Any]:
    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("top-level value is not an object")
    return value


def validate_schema(value: dict[str, Any]) -> bool:
    facts = value.get("facts")
    if not isinstance(facts, list):
        return False
    for fact in facts:
        if not isinstance(fact, dict) or REQUIRED_FACT_FIELDS - set(fact):
            return False
        if fact.get("type") not in FACT_TYPES:
            return False
        if fact.get("status") not in STATUSES:
            return False
        if fact.get("subject") not in SUBJECTS:
            return False
        if not isinstance(fact.get("attribute"), dict):
            return False
        if not isinstance(fact.get("evidence"), str) or not fact["evidence"]:
            return False
        if fact.get("standard_code") is not None or fact.get("terminology") is not None:
            return False
    return True


def canonical_fact(fact: dict[str, Any]) -> str:
    return json.dumps(fact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def fact_names(fact: dict[str, Any]) -> set[str]:
    return {
        name
        for name in {
            normalize_text(fact.get("name")),
            normalize_text(fact.get("normalized_name")),
        }
        if name
    }


def names_match(predicted: dict[str, Any], gold: dict[str, Any]) -> bool:
    return bool(fact_names(predicted) & fact_names(gold))


def core_fact_match(predicted: dict[str, Any], gold: dict[str, Any]) -> bool:
    if not names_match(predicted, gold):
        return False
    return all(
        normalize_text(predicted.get(field)) == normalize_text(gold.get(field))
        for field in ("type", "status", "subject")
    )


def greedy_core_match_pairs(
    predicted_facts: list[dict[str, Any]], gold_facts: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    matched_gold: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for predicted_idx, predicted in enumerate(predicted_facts):
        for gold_idx, gold in enumerate(gold_facts):
            if gold_idx in matched_gold:
                continue
            if core_fact_match(predicted, gold):
                matched_gold.add(gold_idx)
                pairs.append((predicted_idx, gold_idx))
                break
    return pairs


def core_match_components(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, bool]:
    return {
        "name": names_match(predicted, gold),
        "type": values_match(predicted.get("type"), gold.get("type")),
        "status": values_match(predicted.get("status"), gold.get("status")),
        "subject": values_match(predicted.get("subject"), gold.get("subject")),
    }


def close_candidate_score(components: dict[str, bool]) -> int:
    return sum(1 for matched in components.values() if matched)


def best_core_failure_components(
    fact: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, bool] | None:
    best_components: dict[str, bool] | None = None
    best_score = -1
    for candidate in candidates:
        components = core_match_components(fact, candidate)
        score = close_candidate_score(components)
        if score > best_score:
            best_components = components
            best_score = score
    if best_score <= 0:
        return None
    return best_components


def update_core_error_impact(counter: Counter[str], components: dict[str, bool] | None) -> None:
    if components is None:
        counter["no_close_candidate"] += 1
        return
    mismatches = [field for field, matched in components.items() if not matched]
    if not mismatches:
        counter["unmatched_duplicate_or_pairing_conflict"] += 1
        return
    if len(mismatches) == 1:
        counter[f"single_{mismatches[0]}"] += 1
    else:
        counter["multiple_core_fields"] += 1
    for field in mismatches:
        counter[f"any_{field}"] += 1


def values_match(left: Any, right: Any) -> bool:
    return normalize_text(left) == normalize_text(right)


def dict_values_match(left: Any, right: Any) -> bool:
    if not isinstance(left, dict):
        left = {}
    if not isinstance(right, dict):
        right = {}
    return json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def evidence_overlaps(left: Any, right: Any) -> bool:
    left_text = normalize_text(left)
    right_text = normalize_text(right)
    if not left_text or not right_text:
        return False
    return left_text in right_text or right_text in left_text


def attribute_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def canonical_output(value: dict[str, Any]) -> str:
    facts = value.get("facts") or []
    return json.dumps(
        {"facts": sorted((json.loads(canonical_fact(fact)) for fact in facts), key=canonical_fact)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def patient_utterance(sample_input: str) -> str:
    match = re.search(r"患者当前发言：(.*?)\n当前问诊状态：", sample_input or "", flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def prf(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structured AskMed extractor predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--terminology-db",
        type=Path,
        default=None,
        help="If set, normalize predicted facts with this terminology DB before evaluation.",
    )
    parser.add_argument(
        "--standardized-predictions-output",
        type=Path,
        default=None,
        help="Optional JSONL path for predictions after terminology post-processing.",
    )
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    dataset = read_jsonl(args.dataset)
    if len(predictions) != len(dataset):
        raise SystemExit(f"prediction count {len(predictions)} != dataset count {len(dataset)}")
    normalizer = TerminologyNormalizer(args.terminology_db) if args.terminology_db else None
    if args.terminology_db and not normalizer.enabled:
        raise SystemExit(f"terminology DB not found: {args.terminology_db}")
    standardized_prediction_rows: list[dict[str, Any]] = []

    parsed_count = 0
    schema_valid_count = 0
    exact_count = 0
    non_empty_gold = 0
    non_empty_exact_count = 0
    empty_gold = 0
    empty_correct = 0
    empty_predicted = 0
    false_positive_on_empty = 0
    false_negative_all = 0
    traceable_evidence = 0
    predicted_evidence = 0
    predicted_facts = 0
    gold_facts = 0
    matched_facts = 0
    core_matched_facts = 0
    type_pred: Counter[str] = Counter()
    type_gold: Counter[str] = Counter()
    type_tp: Counter[str] = Counter()
    status_pred: Counter[str] = Counter()
    status_gold: Counter[str] = Counter()
    status_tp: Counter[str] = Counter()
    core_type_pred: Counter[str] = Counter()
    core_type_gold: Counter[str] = Counter()
    core_type_tp: Counter[str] = Counter()
    core_status_pred: Counter[str] = Counter()
    core_status_gold: Counter[str] = Counter()
    core_status_tp: Counter[str] = Counter()
    field_totals: Counter[str] = Counter()
    field_correct: Counter[str] = Counter()
    core_predicted_error_impact: Counter[str] = Counter()
    core_gold_error_impact: Counter[str] = Counter()

    for prediction_row, sample in zip(predictions, dataset):
        label_text = prediction_row.get("label") or sample.get("output") or '{"facts":[]}'
        gold = parse_json_object(label_text)
        gold_list = gold.get("facts") if isinstance(gold.get("facts"), list) else []
        gold_dicts = [fact for fact in gold_list if isinstance(fact, dict)]
        gold_set = Counter(canonical_fact(fact) for fact in gold_list if isinstance(fact, dict))
        gold_facts += sum(gold_set.values())
        for fact in gold_list:
            if isinstance(fact, dict):
                type_gold[str(fact.get("type"))] += 1
                status_gold[str(fact.get("status"))] += 1

        if not gold_list:
            empty_gold += 1
        else:
            non_empty_gold += 1

        try:
            predicted = parse_json_object(str(prediction_row.get("predict") or ""))
        except (json.JSONDecodeError, ValueError):
            if args.standardized_predictions_output is not None:
                standardized_prediction_rows.append(dict(prediction_row))
            continue

        if normalizer is not None:
            runtime_predicted = normalizer.normalize_extraction_for_runtime(predicted)
            predicted = normalizer.project_extraction_for_training(runtime_predicted)

        if args.standardized_predictions_output is not None:
            standardized_row = dict(prediction_row)
            standardized_row["predict"] = json.dumps(predicted, ensure_ascii=False, separators=(",", ":"))
            standardized_prediction_rows.append(standardized_row)

        parsed_count += 1
        if validate_schema(predicted):
            schema_valid_count += 1

        predicted_list = predicted.get("facts") if isinstance(predicted.get("facts"), list) else []
        predicted_dicts = [fact for fact in predicted_list if isinstance(fact, dict)]
        predicted_set = Counter(canonical_fact(fact) for fact in predicted_dicts)
        predicted_facts += sum(predicted_set.values())
        if not predicted_dicts:
            empty_predicted += 1
        if not gold_dicts and predicted_dicts:
            false_positive_on_empty += 1
        if gold_dicts and not predicted_dicts:
            false_negative_all += 1

        matched = predicted_set & gold_set
        matched_facts += sum(matched.values())
        core_pairs = greedy_core_match_pairs(predicted_dicts, gold_dicts)
        core_matched_facts += len(core_pairs)
        matched_predicted_core = {predicted_idx for predicted_idx, _ in core_pairs}
        matched_gold_core = {gold_idx for _, gold_idx in core_pairs}

        unmatched_gold_dicts = [
            fact for gold_idx, fact in enumerate(gold_dicts) if gold_idx not in matched_gold_core
        ]
        unmatched_predicted_dicts = [
            fact
            for predicted_idx, fact in enumerate(predicted_dicts)
            if predicted_idx not in matched_predicted_core
        ]
        for fact in unmatched_predicted_dicts:
            update_core_error_impact(
                core_predicted_error_impact,
                best_core_failure_components(fact, unmatched_gold_dicts),
            )
        for fact in unmatched_gold_dicts:
            update_core_error_impact(
                core_gold_error_impact,
                best_core_failure_components(fact, unmatched_predicted_dicts),
            )

        for fact in predicted_dicts:
            core_type_pred[str(fact.get("type"))] += 1
            core_status_pred[str(fact.get("status"))] += 1
        for fact in gold_dicts:
            core_type_gold[str(fact.get("type"))] += 1
            core_status_gold[str(fact.get("status"))] += 1
        for predicted_idx, gold_idx in core_pairs:
            predicted_fact = predicted_dicts[predicted_idx]
            gold_fact = gold_dicts[gold_idx]
            core_type_tp[str(gold_fact.get("type"))] += 1
            core_status_tp[str(gold_fact.get("status"))] += 1

            for field in ("normalized_name", "evidence", "time", "body_part"):
                field_totals[field] += 1
                if values_match(predicted_fact.get(field), gold_fact.get(field)):
                    field_correct[field] += 1

            field_totals["evidence_overlap"] += 1
            if evidence_overlaps(predicted_fact.get("evidence"), gold_fact.get("evidence")):
                field_correct["evidence_overlap"] += 1

            field_totals["attribute"] += 1
            if dict_values_match(predicted_fact.get("attribute"), gold_fact.get("attribute")):
                field_correct["attribute"] += 1

            predicted_attr = predicted_fact.get("attribute")
            gold_attr = gold_fact.get("attribute")
            for field, key in (
                ("attribute_target", "target"),
                ("attribute_value", "value"),
            ):
                field_totals[field] += 1
                if values_match(attribute_value(predicted_attr, key), attribute_value(gold_attr, key)):
                    field_correct[field] += 1

        matched_by_value = Counter(matched)
        for fact in predicted_list:
            if not isinstance(fact, dict):
                continue
            fact_type = str(fact.get("type"))
            fact_status = str(fact.get("status"))
            type_pred[fact_type] += 1
            status_pred[fact_status] += 1
            key = canonical_fact(fact)
            if matched_by_value[key] > 0:
                type_tp[fact_type] += 1
                status_tp[fact_status] += 1
                matched_by_value[key] -= 1

            evidence = fact.get("evidence")
            if isinstance(evidence, str) and evidence:
                predicted_evidence += 1
                utterance = patient_utterance(str(sample.get("input") or ""))
                if utterance and evidence in utterance:
                    traceable_evidence += 1

        if canonical_output(predicted) == canonical_output(gold):
            exact_count += 1
            if gold_list:
                non_empty_exact_count += 1
        if not gold_list and not predicted_list:
            empty_correct += 1

    total = len(dataset)
    field_level = {
        key: {
            "correct": field_correct[key],
            "total": field_totals[key],
            "accuracy": field_correct[key] / field_totals[key] if field_totals[key] else 0.0,
        }
        for key in sorted(field_totals)
    }
    core_error_impact = {
        "unmatched_predicted": sum(core_predicted_error_impact.values())
        - sum(
            value
            for key, value in core_predicted_error_impact.items()
            if key.startswith("any_")
        ),
        "unmatched_gold": sum(core_gold_error_impact.values())
        - sum(value for key, value in core_gold_error_impact.items() if key.startswith("any_")),
        "predicted_side": dict(sorted(core_predicted_error_impact.items())),
        "gold_side": dict(sorted(core_gold_error_impact.items())),
    }
    report = {
        "samples": total,
        "prediction_postprocessing": {
            "terminology_db": str(args.terminology_db) if args.terminology_db else None,
            "standardized_predictions_output": str(args.standardized_predictions_output)
            if args.standardized_predictions_output
            else None,
        },
        "json_parse_rate": parsed_count / total if total else 0.0,
        "schema_valid_rate": schema_valid_count / total if total else 0.0,
        "exact_match": exact_count / total if total else 0.0,
        "non_empty_exact_match": non_empty_exact_count / non_empty_gold if non_empty_gold else 0.0,
        "strict_fact": prf(matched_facts, predicted_facts, gold_facts),
        "core_fact": prf(core_matched_facts, predicted_facts, gold_facts),
        "core_error_impact": core_error_impact,
        "field_level_on_core_matches": field_level,
        "empty_facts_accuracy": empty_correct / empty_gold if empty_gold else 0.0,
        "empty_facts_samples": empty_gold,
        "empty_detection": {
            "gold_empty": empty_gold,
            "predicted_empty": empty_predicted,
            "empty_correct": empty_correct,
            "false_positive_on_empty": false_positive_on_empty,
            "false_negative_all": false_negative_all,
        },
        "evidence_traceability": traceable_evidence / predicted_evidence if predicted_evidence else 0.0,
        "predicted_evidence_count": predicted_evidence,
        "by_type": {
            key: prf(type_tp[key], type_pred[key], type_gold[key])
            for key in sorted(set(type_pred) | set(type_gold))
        },
        "by_status": {
            key: prf(status_tp[key], status_pred[key], status_gold[key])
            for key in sorted(set(status_pred) | set(status_gold))
        },
        "by_type_core": {
            key: prf(core_type_tp[key], core_type_pred[key], core_type_gold[key])
            for key in sorted(set(core_type_pred) | set(core_type_gold))
        },
        "by_status_core": {
            key: prf(core_status_tp[key], core_status_pred[key], core_status_gold[key])
            for key in sorted(set(core_status_pred) | set(core_status_gold))
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.standardized_predictions_output is not None:
        args.standardized_predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.standardized_predictions_output.open("w", encoding="utf-8", newline="\n") as f:
            for row in standardized_prediction_rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
