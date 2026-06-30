from __future__ import annotations

from copy import deepcopy
from typing import Any


UNKNOWN_TARGETS = {"", "unknown", "未知", "不明确", "当前主要症状"}

ATTRIBUTE_ALIASES = {
    "病程": "course_duration",
    "总病程": "course_duration",
    "发病时长": "course_duration",
    "起病时间": "course_duration",
    "患病时间": "course_duration",
    "持续时间": "course_duration",
    "时长": "course_duration",
    "单次持续时间": "episode_duration",
    "每次持续时间": "episode_duration",
    "单次发作持续时间": "episode_duration",
    "发作持续时间": "episode_duration",
    "缓解时间": "episode_duration",
    "一次持续时间": "episode_duration",
    "时间": "time",
    "部位": "body_part",
    "位置": "body_part",
    "性质": "character",
    "疼痛性质": "character",
    "程度": "severity",
    "频率": "frequency",
    "次数": "frequency",
    "颜色": "color",
    "痰色": "color",
    "诱因": "trigger",
    "加重因素": "aggravating_factor",
    "缓解因素": "relieving_factor",
    "伴随症状": "associated_symptom",
}


def empty_patient_state(dialogue_id: str | None = None) -> dict[str, Any]:
    return {
        "dialogue_id": dialogue_id,
        "chief_complaint": None,
        "problems": [],
        "negative_findings": [],
        "uncertain_findings": [],
        "medications": [],
        "examinations": [],
        "histories": [],
        "lifestyle": [],
        "other_facts": [],
    }


def compact_patient_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return the stable state shape passed to the model."""
    return deepcopy(state)


def prompt_patient_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight state for model prompts while keeping full checkpoints intact."""
    return {
        "dialogue_id": state.get("dialogue_id"),
        "chief_complaint": state.get("chief_complaint"),
        "problems": [_prompt_item(item) for item in state.get("problems") or []],
        "negative_findings": [_prompt_brief_item(item) for item in state.get("negative_findings") or []],
        "uncertain_findings": [_prompt_brief_item(item) for item in state.get("uncertain_findings") or []],
        "medications": [_prompt_item(item) for item in state.get("medications") or []],
        "examinations": [_prompt_item(item) for item in state.get("examinations") or []],
        "histories": [_prompt_item(item) for item in state.get("histories") or []],
        "lifestyle": [_prompt_item(item) for item in state.get("lifestyle") or []],
        "other_facts": [_prompt_brief_item(item) for item in state.get("other_facts") or []],
    }


def merge_facts_into_state(
    state: dict[str, Any],
    facts: list[dict[str, Any]],
    turn_id: int | None,
) -> dict[str, Any]:
    state = deepcopy(state)
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        normalized = str(fact.get("normalized_name") or fact.get("name") or "").strip()
        if not normalized:
            continue
        fact_type = fact.get("type")
        status = fact.get("status")
        if fact_type == "attribute":
            if _merge_attribute_fact(state, fact, turn_id):
                continue
            _append_unique(state["other_facts"], _state_item_from_fact(fact, turn_id))
            continue

        if fact_type in {"symptom", "disease"} and status == "present":
            item = _upsert_problem(state, fact, turn_id)
            if state.get("chief_complaint") is None and fact_type == "symptom":
                state["chief_complaint"] = item["normalized_name"]
            continue

        if status == "absent":
            _append_unique(state["negative_findings"], _state_item_from_fact(fact, turn_id))
            continue

        if status == "uncertain" and fact_type in {"symptom", "disease", "other"}:
            _append_unique(state["uncertain_findings"], _state_item_from_fact(fact, turn_id))
            continue

        bucket = _bucket_for_type(fact_type)
        _append_unique(state[bucket], _state_item_from_fact(fact, turn_id))
    return state


def _prompt_item(item: dict[str, Any]) -> dict[str, Any]:
    prompt_item: dict[str, Any] = {
        "name": item.get("name"),
        "normalized_name": item.get("normalized_name"),
        "type": item.get("type"),
        "status": item.get("status"),
        "subject": item.get("subject"),
    }
    if item.get("time") is not None:
        prompt_item["time"] = item.get("time")
    if item.get("body_part") is not None:
        prompt_item["body_part"] = item.get("body_part")
    attributes = item.get("attributes")
    if attributes:
        prompt_item["attributes"] = deepcopy(attributes)
    return prompt_item


def _prompt_brief_item(item: dict[str, Any]) -> dict[str, Any]:
    prompt_item: dict[str, Any] = {
        "name": item.get("name"),
        "normalized_name": item.get("normalized_name"),
        "type": item.get("type"),
        "status": item.get("status"),
        "subject": item.get("subject"),
    }
    if item.get("time") is not None:
        prompt_item["time"] = item.get("time")
    attributes = item.get("attributes")
    if attributes:
        prompt_item["attributes"] = deepcopy(attributes)
    return prompt_item


def _bucket_for_type(fact_type: str | None) -> str:
    if fact_type == "medicine":
        return "medications"
    if fact_type == "examination":
        return "examinations"
    if fact_type == "history":
        return "histories"
    if fact_type == "lifestyle":
        return "lifestyle"
    return "other_facts"


def _state_item_from_fact(fact: dict[str, Any], turn_id: int | None) -> dict[str, Any]:
    attributes = fact.get("attribute") if isinstance(fact.get("attribute"), dict) else {}
    attributes = _normalize_attributes(attributes)
    item = {
        "name": fact.get("name"),
        "normalized_name": fact.get("normalized_name") or fact.get("name"),
        "type": fact.get("type"),
        "status": fact.get("status"),
        "subject": fact.get("subject"),
        "standard_code": fact.get("standard_code"),
        "terminology": fact.get("terminology"),
        "time": fact.get("time"),
        "body_part": fact.get("body_part"),
        "attributes": deepcopy(attributes),
        "evidence": [],
        "first_turn_id": turn_id,
        "last_turn_id": turn_id,
    }
    evidence = fact.get("evidence")
    if evidence:
        item["evidence"].append({"turn_id": turn_id, "text": evidence})
    return item


def _identity(item: dict[str, Any]) -> tuple[Any, ...]:
    coded = _coded_identity(item)
    if coded is not None:
        return (
            coded,
            item.get("type"),
            item.get("status"),
            item.get("subject"),
        )
    return (
        item.get("normalized_name"),
        item.get("type"),
        item.get("status"),
        item.get("subject"),
    )


def _append_unique(bucket: list[dict[str, Any]], item: dict[str, Any]) -> dict[str, Any]:
    identity = _identity(item)
    for existing in bucket:
        if _identity(existing) == identity:
            _merge_item(existing, item)
            return existing
    bucket.append(item)
    return item


def _merge_item(existing: dict[str, Any], new: dict[str, Any]) -> None:
    existing["last_turn_id"] = new.get("last_turn_id")
    if not existing.get("standard_code") and new.get("standard_code"):
        existing["standard_code"] = new.get("standard_code")
    if not existing.get("terminology") and new.get("terminology"):
        existing["terminology"] = new.get("terminology")
    if existing.get("normalized_name") == existing.get("name") and new.get("normalized_name"):
        existing["normalized_name"] = new.get("normalized_name")
    if not existing.get("time") and new.get("time"):
        existing["time"] = new.get("time")
    if not existing.get("body_part") and new.get("body_part"):
        existing["body_part"] = new.get("body_part")
    existing.setdefault("attributes", {}).update(_normalize_attributes(new.get("attributes") or {}))
    for evidence in new.get("evidence") or []:
        if evidence not in existing.setdefault("evidence", []):
            existing["evidence"].append(evidence)


def _upsert_problem(state: dict[str, Any], fact: dict[str, Any], turn_id: int | None) -> dict[str, Any]:
    item = _state_item_from_fact(fact, turn_id)
    for problem in state["problems"]:
        if _same_problem(problem, item):
            _merge_item(problem, item)
            return problem
    state["problems"].append(item)
    return item


def _same_problem(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_coded = _coded_identity(a)
    b_coded = _coded_identity(b)
    if a_coded is not None or b_coded is not None:
        return (
            a_coded is not None
            and a_coded == b_coded
            and a.get("type") == b.get("type")
            and a.get("subject") == b.get("subject")
        )
    return (
        a.get("normalized_name") == b.get("normalized_name")
        and a.get("type") == b.get("type")
        and a.get("subject") == b.get("subject")
    )


def _coded_identity(item: dict[str, Any]) -> tuple[str, str] | None:
    terminology = item.get("terminology")
    standard_code = item.get("standard_code")
    if terminology and standard_code:
        return (str(terminology), str(standard_code))
    return None


def _merge_attribute_fact(state: dict[str, Any], fact: dict[str, Any], turn_id: int | None) -> bool:
    attributes = fact.get("attribute") if isinstance(fact.get("attribute"), dict) else {}
    target = str(attributes.get("target") or "").strip()
    target_item = _find_target_problem(state, target)
    if target_item is None:
        return False

    value = attributes.get("value")
    normalized_name = str(fact.get("normalized_name") or fact.get("name") or "")
    key = ATTRIBUTE_ALIASES.get(normalized_name, normalized_name or "attribute")
    if value is not None:
        target_item.setdefault("attributes", {})[key] = value
        if key == "course_duration" and not target_item.get("time"):
            target_item["time"] = value

    for attr_key, attr_value in attributes.items():
        if attr_key in {"target", "value"}:
            continue
        mapped_key = ATTRIBUTE_ALIASES.get(str(attr_key), str(attr_key))
        target_item.setdefault("attributes", {})[mapped_key] = attr_value

    if fact.get("time") and not target_item.get("time"):
        target_item["time"] = fact.get("time")
    if fact.get("body_part") and not target_item.get("body_part"):
        target_item["body_part"] = fact.get("body_part")
    evidence = fact.get("evidence")
    if evidence:
        evidence_item = {"turn_id": turn_id, "text": evidence}
        if evidence_item not in target_item.setdefault("evidence", []):
            target_item["evidence"].append(evidence_item)
    target_item["last_turn_id"] = turn_id
    return True


def _normalize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    return {ATTRIBUTE_ALIASES.get(str(key), str(key)): value for key, value in attributes.items()}


def _find_target_problem(state: dict[str, Any], target: str) -> dict[str, Any] | None:
    if target in UNKNOWN_TARGETS:
        return None
    for problem in reversed(state.get("problems") or []):
        if target in {problem.get("name"), problem.get("normalized_name")}:
            return problem
    for problem in reversed(state.get("problems") or []):
        name = str(problem.get("name") or "")
        normalized = str(problem.get("normalized_name") or "")
        if target and (target in name or target in normalized or name in target or normalized in target):
            return problem
    return None
