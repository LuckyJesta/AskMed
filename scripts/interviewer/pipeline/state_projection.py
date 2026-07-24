from __future__ import annotations

from copy import deepcopy
from typing import Any


STATE_BUCKETS = (
    "problems",
    "negative_findings",
    "medications",
    "examinations",
    "histories",
    "lifestyle",
    "other_facts",
)


def _project_item(item: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": item.get("normalized_name") or item.get("name"),
        "type": item.get("type"),
        "status": item.get("status"),
        "subject": item.get("subject"),
    }
    attributes = item.get("attributes")
    if isinstance(attributes, dict) and attributes:
        result["attributes"] = deepcopy(attributes)
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def project_interviewer_state(state: Any) -> dict[str, Any]:
    """Project the full medical state into the stable view consumed by the interviewer."""
    if not isinstance(state, dict):
        return {}
    projected: dict[str, Any] = {}
    chief_complaint = state.get("chief_complaint")
    if chief_complaint:
        projected["chief_complaint"] = chief_complaint
    for bucket in STATE_BUCKETS:
        items = [_project_item(item) for item in state.get(bucket) or [] if isinstance(item, dict)]
        items = [item for item in items if item.get("name")]
        if items:
            projected[bucket] = items
    return projected
