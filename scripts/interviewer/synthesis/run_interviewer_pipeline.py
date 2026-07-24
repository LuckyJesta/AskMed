from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.extractor.pipeline.schema import resolve_from_root


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTRACTOR_DIR = Path("data/synthetic_extractor/MedDG_extractor_prompt_v3_1_30k")
DEFAULT_RUN_NAME = "MedDG_interviewer_v2_2_from_v3_1_30k"


def run(command: list[str], api_key: str | None = None) -> None:
    displayed = list(command)
    if api_key:
        displayed = ["***REDACTED***" if value == api_key else value for value in displayed]
    print(" ".join(displayed), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_report(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interviewer synthesis, validation, and Alpaca conversion.")
    parser.add_argument("--source", type=Path, default=Path("data/MedDG_clean.jsonl"))
    parser.add_argument("--extractor-dir", type=Path, default=DEFAULT_EXTRACTOR_DIR)
    parser.add_argument("--work-dir", type=Path, default=Path("data/synthetic_interviewer"))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--api-format", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-dialogues", type=int)
    parser.add_argument("--max-context-messages", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--repair-failed-dialogues", action="store_true")
    parser.add_argument("--no-ssl-verify", action="store_true")
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-alpaca", action="store_true")
    args = parser.parse_args()

    source = resolve_from_root(args.source)
    extractor_dir = resolve_from_root(args.extractor_dir)
    run_dir = resolve_from_root(args.work_dir) / args.run_name
    prefix = run_dir / args.run_name
    extractor_name = extractor_dir.name
    validated = extractor_dir / f"{extractor_name}_validated.jsonl"
    train = extractor_dir / f"{extractor_name}_train_validated.jsonl"
    valid = extractor_dir / f"{extractor_name}_valid_validated.jsonl"
    test = extractor_dir / f"{extractor_name}_test_validated.jsonl"
    for path in (source, validated, train, valid, test):
        if not path.exists():
            raise SystemExit(f"required input not found: {path}")
    run_dir.mkdir(parents=True, exist_ok=True)

    synthesized = prefix.with_name(prefix.name + "_synthesized.jsonl")
    failed = prefix.with_name(prefix.name + "_failed_dialogues.jsonl")
    checkpoint = prefix.with_name(prefix.name + "_checkpoints.jsonl")
    validated_output = prefix.with_name(prefix.name + "_validated.jsonl")
    report = prefix.with_name(prefix.name + "_validation_report.json")
    synthesis_report = prefix.with_name(prefix.name + "_synthesis_report.json")
    conversion_report = prefix.with_name(prefix.name + "_conversion_report.json")
    pipeline_report = prefix.with_name(prefix.name + "_pipeline_report.json")

    if not args.skip_synthesis:
        if not args.api_key:
            raise SystemExit("--api-key is required unless --skip-synthesis is used")
        command = [
            sys.executable,
            "-m",
            "scripts.interviewer.synthesis.synthesize_interviewer_dialogues",
            "--source", str(source),
            "--extractor-validated", str(validated),
            "--extractor-train", str(train),
            "--extractor-valid", str(valid),
            "--extractor-test", str(test),
            "--output", str(synthesized),
            "--failed", str(failed),
            "--checkpoint", str(checkpoint),
            "--report", str(synthesis_report),
            "--dialogue-work-dir", str(run_dir / "_dialogue_work"),
            "--api-format", args.api_format,
            "--api-key", args.api_key,
            "--base-url", args.base_url,
            "--model", args.model,
            "--max-output-tokens", str(args.max_output_tokens),
            "--workers", str(args.workers),
            "--max-context-messages", str(args.max_context_messages),
            "--max-context-chars", str(args.max_context_chars),
            "--timeout", str(args.timeout),
            "--max-retries", str(args.max_retries),
        ]
        if args.max_dialogues is not None:
            command.extend(["--max-dialogues", str(args.max_dialogues)])
        if args.repair_failed_dialogues:
            command.append("--repair-failed-dialogues")
        if args.no_ssl_verify:
            command.append("--no-ssl-verify")
        print("Step 1/3: interviewer synthesis", flush=True)
        run(command, args.api_key)

    if not args.skip_validation:
        print("Step 2/3: dialogue validation", flush=True)
        run([
            sys.executable,
            "-m",
            "scripts.interviewer.synthesis.validate_interviewer_data",
            "--input", str(synthesized),
            "--output", str(validated_output),
            "--failed", str(failed),
            "--source", str(source),
            "--report", str(report),
        ])

    if not args.skip_alpaca:
        print("Step 3/3: inherited Alpaca conversion", flush=True)
        run([
            sys.executable,
            "-m",
            "scripts.interviewer.synthesis.convert_to_alpaca",
            "--input", str(validated_output),
            "--output-prefix", str(prefix),
        ])

    synthesis_stats = read_report(synthesis_report)
    validation_stats = read_report(report)
    conversion_stats = read_report(conversion_report)
    pipeline_stats = {
        "run_name": args.run_name,
        "selected_parent_dialogues": synthesis_stats.get("selected_parent_dialogues", 0),
        "synthesis_succeeded_parent_dialogues": synthesis_stats.get("completed_parent_dialogues", 0),
        "synthesis_failed_parent_dialogues": synthesis_stats.get("remaining_failed_parent_dialogues", 0),
        "validation_succeeded_parent_dialogues": validation_stats.get("valid_parent_dialogues", 0),
        "validation_failed_parent_dialogues": validation_stats.get("failed_parent_dialogues", 0),
        "sessions": validation_stats.get("sessions", synthesis_stats.get("sessions", 0)),
        "session_splits": validation_stats.get("session_splits", synthesis_stats.get("session_splits", 0)),
        "ambiguous_session_boundaries": validation_stats.get(
            "ambiguous_session_boundaries",
            synthesis_stats.get("ambiguous_session_boundaries", 0),
        ),
        "actions": validation_stats.get("actions", {}),
        "skip_reasons": validation_stats.get("skip_reasons", {}),
        "target_attributes": validation_stats.get("target_attributes", {}),
        "answerability_methods": validation_stats.get("answerability_methods", {}),
        "lexical_matches": validation_stats.get("lexical_matches", 0),
        "short_answer_matches": validation_stats.get("short_answer_matches", 0),
        "target_not_resolved": validation_stats.get("target_not_resolved", 0),
        "rejected_unrelated_same_type_deltas": validation_stats.get(
            "rejected_unrelated_same_type_deltas", 0
        ),
        "teacher_api_attempts": synthesis_stats.get("teacher_api_attempts", 0),
        "teacher_api_retries": synthesis_stats.get("teacher_api_retries", 0),
        "alpaca_rows": conversion_stats.get("all", 0),
    }
    pipeline_report.write_text(json.dumps(pipeline_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(pipeline_stats, ensure_ascii=False, indent=2))

    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
