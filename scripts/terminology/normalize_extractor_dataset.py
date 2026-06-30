from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extractor_synthesis"))

from extractor_common import DEFAULT_WORK_DIR, read_jsonl, resolve_from_root, write_jsonl
from split_extractor_dataset import main as split_main
from state_manager import compact_patient_state, empty_patient_state, merge_facts_into_state, prompt_patient_state
from terminology_normalizer import TerminologyNormalizer


def dialogue_turn(row: dict[str, Any]) -> tuple[str, int]:
    source = row.get("input") or {}
    return str(source.get("dialogue_id") or ""), int(source.get("turn_id") or 0)


def make_training_projection(runtime_extraction: dict[str, Any], normalizer: TerminologyNormalizer) -> dict[str, Any]:
    return normalizer.project_extraction_for_training(runtime_extraction)


def normalize_rows(
    rows: list[dict[str, Any]],
    normalizer: TerminologyNormalizer,
    collect_candidates: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = sorted(rows, key=dialogue_turn)
    output_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    final_states: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "rows": len(rows),
        "facts": 0,
        "normalized_facts": 0,
        "runtime_coded_facts": 0,
        "by_type": Counter(),
        "normalized_by_type": Counter(),
        "coded_by_terminology": Counter(),
    }

    current_dialogue: str | None = None
    state: dict[str, Any] | None = None
    for row in tqdm(rows, desc="Normalize", unit="row", dynamic_ncols=True):
        dialogue_id, turn_id = dialogue_turn(row)
        if current_dialogue != dialogue_id:
            if current_dialogue is not None and state is not None:
                final_states.append({"dialogue_id": current_dialogue, "patient_state": compact_patient_state(state)})
            current_dialogue = dialogue_id
            state = empty_patient_state(dialogue_id)

        assert state is not None
        source = dict(row.get("input") or {})
        source["patient_state_before_turn"] = prompt_patient_state(state)
        extraction = row.get("parsed_output") or {"facts": []}
        runtime_extraction = normalizer.normalize_extraction_for_runtime(extraction)
        training_extraction = make_training_projection(runtime_extraction, normalizer)
        state_after = merge_facts_into_state(state, runtime_extraction.get("facts") or [], turn_id)

        out_row = dict(row)
        out_row["input"] = source
        out_row["parsed_output"] = training_extraction
        out_row["patient_state_after_turn"] = compact_patient_state(state_after)
        output_rows.append(out_row)
        trajectories.append(
            {
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "patient_state": compact_patient_state(state_after),
            }
        )

        for original, runtime in zip(extraction.get("facts") or [], runtime_extraction.get("facts") or []):
            if not isinstance(original, dict) or not isinstance(runtime, dict):
                continue
            fact_type = str(original.get("type"))
            report["facts"] += 1
            report["by_type"][fact_type] += 1
            if original.get("normalized_name") != runtime.get("normalized_name"):
                report["normalized_facts"] += 1
                report["normalized_by_type"][fact_type] += 1
            if runtime.get("standard_code") and runtime.get("terminology"):
                report["runtime_coded_facts"] += 1
                report["coded_by_terminology"][str(runtime.get("terminology"))] += 1
            elif collect_candidates and fact_type in {"disease", "examination"}:
                fact_candidates = normalizer.candidates(original)
                if fact_candidates:
                    candidates.append(
                        {
                            "dialogue_id": dialogue_id,
                            "turn_id": turn_id,
                            "fact": original,
                            "candidates": fact_candidates,
                        }
                    )

        state = state_after

    if current_dialogue is not None and state is not None:
        final_states.append({"dialogue_id": current_dialogue, "patient_state": compact_patient_state(state)})

    report["by_type"] = dict(report["by_type"])
    report["normalized_by_type"] = dict(report["normalized_by_type"])
    report["coded_by_terminology"] = dict(report["coded_by_terminology"])
    report["terminology"] = normalizer.metadata()
    report["candidate_rows"] = len(candidates)
    return output_rows, trajectories, final_states, report | {"candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create terminology-normalized extractor dataset derivatives.")
    parser.add_argument("--input", type=Path, default=DEFAULT_WORK_DIR / "MedDG_extractor_15k_validated.jsonl")
    parser.add_argument("--terminology-db", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_WORK_DIR / "MedDG_extractor_15k_terminology")
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.9, 0.05, 0.05))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional dry-run row limit.")
    parser.add_argument(
        "--write-candidates",
        action="store_true",
        help="Compute fuzzy terminology candidates for unmatched disease/examination facts. Slower.",
    )
    args = parser.parse_args()

    args.input = resolve_from_root(args.input)
    args.terminology_db = resolve_from_root(args.terminology_db)
    args.output_prefix = resolve_from_root(args.output_prefix)
    normalizer = TerminologyNormalizer(args.terminology_db)
    if not normalizer.enabled:
        raise SystemExit(f"Terminology database not found: {args.terminology_db}")

    print(f"Reading input: {args.input}", flush=True)
    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Loaded rows: {len(rows)}", flush=True)
    print(f"Terminology: {normalizer.metadata()}", flush=True)
    normalized_rows, trajectories, final_states, report_with_candidates = normalize_rows(
        rows,
        normalizer,
        collect_candidates=args.write_candidates,
    )
    candidates = report_with_candidates.pop("candidates")
    validated_path = args.output_prefix.parent / f"{args.output_prefix.name}_validated.jsonl"
    trajectories_path = args.output_prefix.parent / f"{args.output_prefix.name}_state_trajectories.jsonl"
    final_states_path = args.output_prefix.parent / f"{args.output_prefix.name}_final_states.jsonl"
    report_path = args.output_prefix.parent / f"{args.output_prefix.name}_normalization_report.json"
    candidates_path = args.output_prefix.parent / f"{args.output_prefix.name}_candidates.jsonl"

    print(f"Writing validated: {validated_path}", flush=True)
    write_jsonl(validated_path, normalized_rows)
    print(f"Writing state trajectories: {trajectories_path}", flush=True)
    write_jsonl(trajectories_path, trajectories)
    print(f"Writing final states: {final_states_path}", flush=True)
    write_jsonl(final_states_path, final_states)
    print(f"Writing candidates: {candidates_path}", flush=True)
    write_jsonl(candidates_path, candidates)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_with_candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    old_argv = sys.argv
    try:
        train_ratio, valid_ratio, test_ratio = args.split_ratios
        print("Splitting and converting to Alpaca...", flush=True)
        sys.argv = [
            "split_extractor_dataset.py",
            "--input",
            str(validated_path),
            "--output-prefix",
            str(args.output_prefix),
            "--train-ratio",
            str(train_ratio),
            "--valid-ratio",
            str(valid_ratio),
            "--test-ratio",
            str(test_ratio),
            "--seed",
            str(args.split_seed),
        ]
        split_main()
    finally:
        sys.argv = old_argv

    print(
        json.dumps(
            {
                "validated": str(validated_path),
                "state_trajectories": str(trajectories_path),
                "final_states": str(final_states_path),
                "report": str(report_path),
                "candidates": str(candidates_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
