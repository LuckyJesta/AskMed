from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from scripts.extractor.pipeline.fact_pipeline import FactPipeline, FactPipelineError
from scripts.extractor.pipeline.schema import clean_extraction_for_training, resolve_from_root
from scripts.extractor.pipeline.state_manager import (
    compact_patient_state,
    empty_patient_state,
    prompt_patient_state,
)


ATTRIBUTE_ALIASES = {
    "stool_character": "character",
    "dosage": "dose",
    "degree": "severity",
    "onset_time": "time",
    "start_time": "time",
    "onset_timing": "time",
    "occurrence_time": "time",
    "discovery_time": "time",
    "result_value": "result",
    "quantity": "amount",
    "count": "frequency",
    "times": "frequency",
    "number": "frequency",
    "gender": "sex",
    "年龄": "age",
    "伴随症状": "associated_symptom",
    "疼痛": "associated_pain",
    "smell": "odor",
}
HEDGE_MARKERS = ("好像", "可能", "也许", "似乎", "应该", "大概", "估计", "不确定")
SHORT_HEDGED_FACT = re.compile(
    r"^(?:好像|可能|也许|似乎|应该)(?:没有|没|有|有点|有些|是|不是).{0,12}$"
)
GENERIC_HEDGED_FACT = re.compile(
    r"^(?:好像|可能|也许|似乎|应该).{0,24}(?:不适|异常|问题)$"
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            yield row


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def strip_format_characters(value: Any) -> Any:
    if isinstance(value, str):
        return "".join(char for char in value if unicodedata.category(char) != "Cf")
    if isinstance(value, list):
        return [strip_format_characters(item) for item in value]
    if isinstance(value, dict):
        return {
            strip_format_characters(key) if isinstance(key, str) else key: strip_format_characters(item)
            for key, item in value.items()
        }
    return value


def merge_attribute_value(existing: Any, incoming: Any) -> str:
    parts: list[str] = []
    for value in (existing, incoming):
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None:
                continue
            for part in str(item).split("；"):
                text = part.strip()
                if text and text not in parts:
                    parts.append(text)
    return "；".join(parts)


def clean_attributes(fact: dict[str, Any], actions: Counter[str]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for raw_key, value in (fact.get("attribute") or {}).items():
        key = str(raw_key).strip()
        if key == "value" and fact.get("type") == "examination":
            key = "result"
            actions["attribute_alias:value->result"] += 1
        else:
            mapped = ATTRIBUTE_ALIASES.get(key, key)
            if mapped != key:
                actions[f"attribute_alias:{key}->{mapped}"] += 1
            key = mapped
        if value in (None, "", [], {}):
            actions["empty_attribute_removed"] += 1
            continue
        if key in cleaned:
            cleaned[key] = merge_attribute_value(cleaned[key], value)
            actions[f"attribute_collision_merged:{key}"] += 1
        else:
            cleaned[key] = value
    return cleaned


def normalized_clause(text: str) -> str:
    return re.sub(r"[\s，。！？!?、；;：:（）()]+", "", text or "")


def should_drop_hedged_fact(fact: dict[str, Any]) -> bool:
    evidence = normalized_clause(str(fact.get("evidence") or ""))
    return bool(SHORT_HEDGED_FACT.fullmatch(evidence) or GENERIC_HEDGED_FACT.fullmatch(evidence))


def clean_extraction(
    extraction: dict[str, Any],
    actions: Counter[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    projected = clean_extraction_for_training(strip_format_characters(deepcopy(extraction)))
    cleaned_facts: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for fact in projected.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        item = deepcopy(fact)
        item["attribute"] = clean_attributes(item, actions)
        if should_drop_hedged_fact(item):
            dropped.append(item)
            actions["hedged_fact_removed"] += 1
            continue
        cleaned_facts.append(item)
    return {"facts": cleaned_facts}, dropped


def has_hedge(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in HEDGE_MARKERS)
    if isinstance(value, list):
        return any(has_hedge(item) for item in value)
    if isinstance(value, dict):
        return any(has_hedge(item) for item in value.values())
    return False


def checkpoint_row(dialogue_id: str, rows: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    return {
        "dialogue_id": dialogue_id,
        "completed_turn_id": rows[-1]["input"]["turn_id"] if rows else None,
        "patient_state": compact_patient_state(state),
        "finished": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a conservatively cleaned extractor dataset and replay dialogue state."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--terminology-db", type=Path, default=None)
    args = parser.parse_args()

    input_path = resolve_from_root(args.input)
    output_prefix = resolve_from_root(args.output_prefix)
    terminology_db = resolve_from_root(args.terminology_db) if args.terminology_db else None
    pipeline = FactPipeline(terminology_db)

    output_rows: list[dict[str, Any]] = []
    final_states: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    dialogue_rows: list[dict[str, Any]] = []
    current_dialogue_id: str | None = None
    state: dict[str, Any] | None = None

    def finish_dialogue() -> None:
        if current_dialogue_id is None or state is None:
            return
        final_states.append(
            {"dialogue_id": current_dialogue_id, "patient_state": compact_patient_state(state)}
        )
        checkpoints.append(checkpoint_row(current_dialogue_id, dialogue_rows, state))

    for source_row in read_jsonl(input_path):
        row = strip_format_characters(deepcopy(source_row))
        input_row = row.get("input") or {}
        dialogue_id = str(input_row.get("dialogue_id") or "")
        if not dialogue_id:
            raise ValueError("Every row must contain input.dialogue_id")
        if dialogue_id != current_dialogue_id:
            finish_dialogue()
            current_dialogue_id = dialogue_id
            state = empty_patient_state(dialogue_id)
            dialogue_rows = []
        assert state is not None

        input_row["patient_state_before_turn"] = prompt_patient_state(state)
        raw_extraction, dropped = clean_extraction(row.get("parsed_output") or {}, actions)
        try:
            processed = pipeline.process(
                raw_extraction,
                patient_utterance=str(input_row.get("patient_utterance") or ""),
                state=state,
                turn_id=input_row.get("turn_id"),
                previous_doctor_question=input_row.get("previous_doctor_question"),
                recent_context=input_row.get("recent_context") or [],
            )
        except FactPipelineError as exc:
            raise RuntimeError(
                f"{dialogue_id}#{input_row.get('turn_id')} cleaning failed at {exc.stage}: {exc.errors}"
            ) from exc

        state = processed.state_after
        cleaned_row = {
            "input": input_row,
            "raw_response": json.dumps(
                processed.training_extraction,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "parsed_output": processed.runtime_extraction,
            "patient_state_after_turn": compact_patient_state(state),
            "normalization_stats": processed.normalization_stats,
            "model": row.get("model"),
            "api_format": row.get("api_format"),
        }
        output_rows.append(cleaned_row)
        dialogue_rows.append(cleaned_row)

        if dropped or has_hedge(input_row.get("patient_utterance")):
            candidates.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_id": input_row.get("turn_id"),
                    "patient_utterance": input_row.get("patient_utterance"),
                    "dropped_facts": dropped,
                    "remaining_facts": processed.runtime_extraction.get("facts") or [],
                    "reason": "hedged_expression_review",
                }
            )

    finish_dialogue()

    synthesized = Path(str(output_prefix) + "_synthesized.jsonl")
    final_states_path = Path(str(output_prefix) + "_final_states.jsonl")
    checkpoints_path = Path(str(output_prefix) + "_checkpoints.jsonl")
    candidates_path = Path(str(output_prefix) + "_cleaning_candidates.jsonl")
    report_path = Path(str(output_prefix) + "_cleaning_report.json")
    write_jsonl_atomic(synthesized, output_rows)
    write_jsonl_atomic(final_states_path, final_states)
    write_jsonl_atomic(checkpoints_path, checkpoints)
    write_jsonl_atomic(candidates_path, candidates)
    report = {
        "input": str(input_path),
        "output": str(synthesized),
        "rows": len(output_rows),
        "dialogues": len(final_states),
        "facts": sum(len(row["parsed_output"].get("facts") or []) for row in output_rows),
        "actions": dict(sorted(actions.items())),
        "hedged_candidate_rows": len(candidates),
        "outputs": {
            "synthesized": str(synthesized),
            "final_states": str(final_states_path),
            "checkpoints": str(checkpoints_path),
            "candidates": str(candidates_path),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
