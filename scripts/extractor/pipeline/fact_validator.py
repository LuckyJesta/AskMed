from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schema import CORE_FACT_FIELDS, validate_extraction


GENERIC_NEGATIVE_NAMES = {
    "用药",
    "吃药",
    "服药",
    "药物",
    "检查",
    "检验",
    "化验",
}
IMMUTABLE_AFTER_STANDARDIZATION = {
    "name",
    "type",
    "status",
    "subject",
    "evidence",
    "attribute",
}


class FactValidationError(ValueError):
    def __init__(self, stage: str, errors: list[str]) -> None:
        self.stage = stage
        self.errors = errors
        super().__init__(f"{stage}: {'; '.join(errors)}")


def validate_raw_extraction(extraction: dict[str, Any], patient_utterance: str) -> list[str]:
    _, errors = validate_extraction(extraction, patient_utterance, strict_standard_null=True)
    facts = extraction.get("facts")
    if not isinstance(facts, list):
        return errors

    for idx, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        prefix = f"facts[{idx}]"
        normalized_name = fact.get("normalized_name")
        if not isinstance(normalized_name, str) or not normalized_name.strip():
            errors.append(f"{prefix}.normalized_name must be a non-empty string")
        attributes = fact.get("attribute")
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                if isinstance(value, dict):
                    errors.append(f"{prefix}.attribute.{key} must not be a nested object")
                elif isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
                    errors.append(f"{prefix}.attribute.{key} list values must be scalar")
        if _is_generic_negative(fact):
            errors.append(f"{prefix} is a generic medicine/examination negative without a concrete object")
    return errors


def validate_standardized_extraction(
    raw: dict[str, Any],
    runtime: dict[str, Any],
    normalizer: Any,
) -> list[str]:
    errors: list[str] = []
    raw_facts = raw.get("facts")
    runtime_facts = runtime.get("facts")
    if not isinstance(raw_facts, list) or not isinstance(runtime_facts, list):
        return ["raw and runtime facts must both be lists"]
    if len(raw_facts) != len(runtime_facts):
        return ["standardization changed the number of facts"]

    for idx, (before, after) in enumerate(zip(raw_facts, runtime_facts)):
        if not isinstance(before, dict) or not isinstance(after, dict):
            errors.append(f"facts[{idx}] must remain an object after standardization")
            continue
        if set(after) != CORE_FACT_FIELDS:
            errors.append(f"facts[{idx}] field set changed after standardization")
        for field in IMMUTABLE_AFTER_STANDARDIZATION:
            if before.get(field) != after.get(field):
                errors.append(f"facts[{idx}].{field} changed during standardization")
        normalized_name = after.get("normalized_name")
        if not isinstance(normalized_name, str) or not normalized_name.strip():
            errors.append(f"facts[{idx}].normalized_name must be non-empty after standardization")
        code = after.get("standard_code")
        terminology = after.get("terminology")
        if bool(code) != bool(terminology):
            errors.append(f"facts[{idx}] standard_code and terminology must be set together")
        elif code and not normalizer.verify_coded_fact(after):
            errors.append(f"facts[{idx}] code cannot be verified against the terminology database")
    return errors


def validate_runtime_extraction(
    extraction: dict[str, Any],
    patient_utterance: str,
    normalizer: Any,
) -> list[str]:
    _, errors = validate_extraction(extraction, patient_utterance, strict_standard_null=False)
    facts = extraction.get("facts")
    if not isinstance(facts, list):
        return errors
    for idx, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        prefix = f"facts[{idx}]"
        normalized_name = fact.get("normalized_name")
        if not isinstance(normalized_name, str) or not normalized_name.strip():
            errors.append(f"{prefix}.normalized_name must be a non-empty string")
        attributes = fact.get("attribute")
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                if isinstance(value, dict):
                    errors.append(f"{prefix}.attribute.{key} must not be a nested object")
                elif isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
                    errors.append(f"{prefix}.attribute.{key} list values must be scalar")
        if _is_generic_negative(fact):
            errors.append(f"{prefix} is a generic medicine/examination negative without a concrete object")
        code = fact.get("standard_code")
        terminology = fact.get("terminology")
        if bool(code) != bool(terminology):
            errors.append(f"{prefix} standard_code and terminology must be set together")
        elif code and not normalizer.verify_coded_fact(fact):
            errors.append(f"{prefix} code cannot be verified against the terminology database")
    return errors


def validate_training_projection(runtime: dict[str, Any], training: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    runtime_facts = runtime.get("facts")
    training_facts = training.get("facts")
    if not isinstance(runtime_facts, list) or not isinstance(training_facts, list):
        return ["runtime and training facts must both be lists"]
    if len(runtime_facts) != len(training_facts):
        return ["training projection changed the number of facts"]
    for idx, (runtime_fact, training_fact) in enumerate(zip(runtime_facts, training_facts)):
        if not isinstance(runtime_fact, dict) or not isinstance(training_fact, dict):
            errors.append(f"facts[{idx}] must remain an object in training projection")
            continue
        if training_fact.get("standard_code") is not None or training_fact.get("terminology") is not None:
            errors.append(f"facts[{idx}] training projection must clear standard_code and terminology")
        for field in CORE_FACT_FIELDS - {"standard_code", "terminology"}:
            if deepcopy(runtime_fact.get(field)) != deepcopy(training_fact.get(field)):
                errors.append(f"facts[{idx}].{field} changed in training projection")
    return errors


def validate_state_transition(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if before.get("dialogue_id") != after.get("dialogue_id"):
        errors.append("state dialogue_id changed")
    seen_codes: set[tuple[str, str, str, str]] = set()
    for bucket in (
        "problems",
        "negative_findings",
        "medications",
        "examinations",
        "histories",
        "lifestyle",
        "other_facts",
    ):
        for idx, item in enumerate(after.get(bucket) or []):
            if not isinstance(item, dict):
                errors.append(f"state.{bucket}[{idx}] must be an object")
                continue
            code = item.get("standard_code")
            terminology = item.get("terminology")
            if code and terminology:
                identity = (
                    str(terminology),
                    str(code),
                    str(item.get("type")),
                    str(item.get("subject")),
                )
                if identity in seen_codes:
                    errors.append(f"duplicate coded state entity: {identity}")
                seen_codes.add(identity)
            for key, value in (item.get("attributes") or {}).items():
                if isinstance(value, (dict, list)):
                    errors.append(f"state.{bucket}[{idx}].attributes.{key} must be a scalar string")
                if isinstance(value, str):
                    parts = [part.strip() for part in value.split("；") if part.strip()]
                    if len(parts) != len(dict.fromkeys(parts)):
                        errors.append(f"state.{bucket}[{idx}].attributes.{key} contains duplicate values")
    return errors


def _is_generic_negative(fact: dict[str, Any]) -> bool:
    if fact.get("status") != "absent" or fact.get("type") not in {"medicine", "examination"}:
        return False
    names = {
        str(fact.get("name") or "").strip(),
        str(fact.get("normalized_name") or "").strip(),
    }
    return bool(names & GENERIC_NEGATIVE_NAMES)
