from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fact_validator import (
    FactValidationError,
    validate_raw_extraction,
    validate_standardized_extraction,
    validate_state_transition,
    validate_training_projection,
)
from .schema import clean_extraction_for_training, normalize_extraction
from .state_manager import merge_facts_into_state, resolve_active_target
from .terminology_normalizer import TerminologyNormalizer


class FactPipelineError(FactValidationError):
    """A dialogue-scoped fact processing failure."""


@dataclass(frozen=True)
class FactPipelineResult:
    runtime_extraction: dict[str, Any]
    training_extraction: dict[str, Any]
    state_after: dict[str, Any]
    normalization_stats: dict[str, Any]
    warnings: list[str]


class FactPipeline:
    def __init__(self, terminology_db: str | Path | None = None) -> None:
        self.normalizer = TerminologyNormalizer(terminology_db)
        if terminology_db is not None and not self.normalizer.enabled:
            raise FileNotFoundError(f"Terminology database not found: {terminology_db}")
        if self.normalizer.enabled:
            self.normalizer.ensure_database_readable()

    @property
    def standardization_enabled(self) -> bool:
        return self.normalizer.enabled

    def process(
        self,
        extraction: dict[str, Any],
        patient_utterance: str,
        state: dict[str, Any],
        turn_id: int | None,
        previous_doctor_question: str | None = None,
        recent_context: list[dict[str, Any]] | None = None,
    ) -> FactPipelineResult:
        del recent_context
        raw = normalize_extraction(extraction)
        raw_errors = validate_raw_extraction(raw, patient_utterance)
        if raw_errors:
            raise FactPipelineError("raw_validation_error", raw_errors)

        active_target = resolve_active_target(state, previous_doctor_question)
        raw, rewrite_warnings = _rewrite_continuation_facts(raw, state, active_target)
        if rewrite_warnings:
            rewritten_errors = validate_raw_extraction(raw, patient_utterance)
            if rewritten_errors:
                raise FactPipelineError("raw_validation_error", rewritten_errors)

        runtime = self.normalizer.normalize_extraction_for_runtime(raw)
        normalization_errors = validate_standardized_extraction(raw, runtime, self.normalizer)
        if normalization_errors:
            raise FactPipelineError("normalization_error", normalization_errors)

        state_before = deepcopy(state)
        state_after = merge_facts_into_state(
            state_before,
            runtime.get("facts") or [],
            turn_id,
            active_target_name=active_target,
        )
        state_errors = validate_state_transition(state_before, state_after)
        if state_errors:
            raise FactPipelineError("state_error", state_errors)

        training = clean_extraction_for_training(runtime)
        projection_errors = validate_training_projection(runtime, training)
        if projection_errors:
            raise FactPipelineError("projection_error", projection_errors)

        coded = sum(
            1
            for fact in runtime.get("facts") or []
            if isinstance(fact, dict) and fact.get("standard_code") and fact.get("terminology")
        )
        return FactPipelineResult(
            runtime_extraction=runtime,
            training_extraction=training,
            state_after=state_after,
            normalization_stats={
                "enabled": self.standardization_enabled,
                "facts": len(runtime.get("facts") or []),
                "coded_facts": coded,
            },
            warnings=rewrite_warnings,
        )


ATTRIBUTE_DISPLAY_NAMES = {
    "time": "发生时间",
    "body_part": "部位",
    "duration": "持续时间",
    "episode_duration": "单次持续时间",
    "frequency": "频率",
    "character": "性质",
    "severity": "程度",
    "trigger": "诱因",
    "aggravating_factor": "加重因素",
    "relieving_factor": "缓解因素",
    "color": "颜色",
    "amount": "数量",
    "effect": "效果",
    "dose": "剂量",
    "result": "结果",
    "route": "给药途径",
    "side_effect": "不良反应",
}


def _rewrite_continuation_facts(
    extraction: dict[str, Any],
    state: dict[str, Any],
    active_target_name: str | None,
) -> tuple[dict[str, Any], list[str]]:
    if not active_target_name:
        return extraction, []
    active_matches = [
        item
        for item in state.get("problems") or []
        if active_target_name
        in {
            str(item.get("name") or "").strip(),
            str(item.get("normalized_name") or "").strip(),
            *(str(alias).strip() for alias in item.get("aliases") or []),
        }
    ]
    if len(active_matches) != 1:
        return extraction, []
    active = active_matches[0]
    rewritten = deepcopy(extraction)
    warnings: list[str] = []
    for idx, fact in enumerate(rewritten.get("facts") or []):
        if not _is_rewritable_continuation(fact, active):
            continue
        attributes = dict(fact.get("attribute") or {})
        attributes["target"] = active_target_name
        concrete_keys = [key for key in attributes if key != "target"]
        display_name = (
            ATTRIBUTE_DISPLAY_NAMES.get(concrete_keys[0], concrete_keys[0])
            if len(concrete_keys) == 1
            else "症状属性"
        )
        fact["name"] = display_name
        fact["normalized_name"] = display_name
        fact["type"] = "attribute"
        fact["attribute"] = attributes
        warnings.append(f"facts[{idx}] rewritten as attribute targeting {active_target_name}")
    return rewritten, warnings


def _is_rewritable_continuation(fact: Any, active: dict[str, Any]) -> bool:
    if not isinstance(fact, dict) or fact.get("type") != "symptom" or fact.get("status") != "present":
        return False
    if fact.get("subject") != active.get("subject"):
        return False
    attributes = fact.get("attribute")
    if not isinstance(attributes, dict) or not attributes:
        return False
    old_names = {
        str(active.get("name") or "").strip(),
        str(active.get("normalized_name") or "").strip(),
        *(str(alias).strip() for alias in active.get("aliases") or []),
    }
    new_names = {
        str(fact.get("name") or "").strip(),
        str(fact.get("normalized_name") or "").strip(),
    }
    for old_name in old_names:
        for new_name in new_names:
            if not old_name or not new_name:
                continue
            if old_name in new_name or new_name in old_name:
                return True
            if _is_pain_descriptor_pair(old_name, new_name):
                return True
    return False


def _is_pain_descriptor_pair(old_name: str, new_name: str) -> bool:
    """Allow short pain descriptors without treating every shared suffix as an alias."""
    return old_name.endswith("痛") and new_name.endswith("痛") and len(new_name) <= 3
