from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from scripts.extractor.pipeline.schema import (
    DEFAULT_SOURCE,
    DEFAULT_WORK_DIR,
    read_jsonl,
    resolve_from_root,
    select_source_dialogues,
)
from scripts.extractor.synthesis.dialogue_work import load_dialogue_shards
from scripts.extractor.synthesis.failure_ledger import (
    load_failure_ledger,
    select_failure_dialogue_ids,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ASKMED_ROOT = SCRIPT_DIR.parents[2]


class PipelineLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str = "") -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(message + "\n")


def emit(message: str, logger: PipelineLogger) -> None:
    print(message, flush=True)
    logger.log(message)


def redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for idx, value in enumerate(redacted):
        if value == "--api-key" and idx + 1 < len(redacted):
            redacted[idx + 1] = "***REDACTED***"
        elif value.startswith("--api-key="):
            redacted[idx] = "--api-key=***REDACTED***"
    return redacted


def load_checkpoints(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        dialogue_id = row.get("dialogue_id")
        if dialogue_id:
            checkpoints[str(dialogue_id)] = row
    return checkpoints


def count_pending_dialogues(
    source: Path,
    checkpoint: Path,
    dialogue_work_dir: Path,
    max_dialogues: int | None,
    max_user_turns: int | None,
) -> int:
    checkpoints = load_checkpoints(checkpoint)
    shards = load_dialogue_shards(dialogue_work_dir)
    selected, _ = select_source_dialogues(source, max_dialogues, max_user_turns)
    pending = 0
    for dialog in selected:
        dialogue_id = str(dialog.get("dialogue_id"))
        checkpoint_row = checkpoints.get(dialogue_id)
        if (checkpoint_row and checkpoint_row.get("finished")) or dialogue_id in shards:
            continue
        pending += 1
    return pending


def grouped_run_prefix(work_dir: Path, run_name: str, flat_output: bool) -> Path:
    if flat_output:
        return work_dir / run_name
    return work_dir / run_name / run_name


def backup_before_rebuild(paths: list[Path], run_dir: Path) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    backup_dir = run_dir / "_backup_before_rebuild" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def run_with_progress(command: list[str], total: int, env: dict[str, str], logger: PipelineLogger) -> None:
    logger.log(" ".join(redact_command(command)))
    process = subprocess.Popen(
        command,
        cwd=str(ASKMED_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    with tqdm(total=total, desc="Synthesis", unit="dialogue", dynamic_ncols=True) as progress:
        for line in process.stdout:
            stripped = line.strip()
            if stripped.startswith(("dialogue-ok ", "dialogue-failed ", "ok ", "rule-empty ", "failed ")):
                logger.log(stripped)
                progress.update(1)
            elif stripped:
                logger.log(stripped)
                tqdm.write(stripped)
    return_code = process.wait()
    logger.log(f"return_code={return_code}")
    if return_code != 0:
        raise SystemExit(return_code)


def run_plain(command: list[str], env: dict[str, str], logger: PipelineLogger) -> None:
    command_text = " ".join(redact_command(command))
    emit(command_text, logger)
    process = subprocess.Popen(
        command,
        cwd=str(ASKMED_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped:
            emit(stripped, logger)
    return_code = process.wait()
    logger.log(f"return_code={return_code}")
    if return_code != 0:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run extractor synthesis, validation, and Alpaca conversion in order."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--run-name", default="extractor_run")
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write files directly under --work-dir using the legacy flat layout.",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-format", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--terminology-db",
        type=Path,
        default=None,
        help="Optional local terminology SQLite database used during synthesis state updates.",
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable TLS certificate verification for API requests. Use only for trusted API gateways.",
    )
    parser.add_argument(
        "--max-dialogues",
        type=int,
        default=None,
        help="Total source-prefix dialogue limit, including dialogues already completed in checkpoints.",
    )
    parser.add_argument(
        "--max-user-turns",
        type=int,
        default=None,
        help="Select a stable source prefix reaching this many patient turns; the final dialogue stays intact.",
    )
    parser.add_argument("--max-context-messages", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of complete dialogues synthesized concurrently. Dialogue turns remain sequential.",
    )
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-alpaca", action="store_true")
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Retry rows from {run_name}_synthesis_failed.jsonl and append successful rows to synthesized output.",
    )
    parser.add_argument(
        "--repair-failed-dialogues",
        action="store_true",
        help="Retry whole failed dialogues, rebuild synthesized data, then validate and convert.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Pipeline log path. Defaults to {work-dir}/{run-name}_pipeline.log.",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALID", "TEST"),
        default=None,
        help="If set, split validated data by dialogue_id and write train/valid/test Alpaca files.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_dialogues is not None and args.max_dialogues < 0:
        raise SystemExit("--max-dialogues must be non-negative.")
    if args.max_user_turns is not None and args.max_user_turns < 0:
        raise SystemExit("--max-user-turns must be non-negative.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    args.source = resolve_from_root(args.source)
    args.work_dir = resolve_from_root(args.work_dir)
    if args.terminology_db is not None:
        args.terminology_db = resolve_from_root(args.terminology_db)
        if not args.terminology_db.exists():
            raise SystemExit(f"--terminology-db not found: {args.terminology_db}")
    if args.split_ratios is not None:
        if any(ratio < 0 for ratio in args.split_ratios) or sum(args.split_ratios) <= 0:
            raise SystemExit("--split-ratios values must be non-negative and sum to a positive value.")
    if not args.skip_synthesis and not args.api_key:
        raise SystemExit("--api-key is required unless --skip-synthesis is set.")
    if args.log_file is not None:
        args.log_file = resolve_from_root(args.log_file)
    run_prefix = grouped_run_prefix(args.work_dir, args.run_name, args.flat_output)
    run_dir = run_prefix.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    synthesized = run_dir / f"{run_prefix.name}_synthesized.jsonl"
    failed_dialogues = run_dir / f"{run_prefix.name}_failed_dialogues.jsonl"
    repair_succeeded = run_dir / f"{run_prefix.name}_repair_succeeded.jsonl"
    final_states = run_dir / f"{run_prefix.name}_final_states.jsonl"
    checkpoint = run_dir / f"{run_prefix.name}_checkpoints.jsonl"
    validated = run_dir / f"{run_prefix.name}_validated.jsonl"
    report = run_dir / f"{run_prefix.name}_report.json"
    rebuild_report = run_dir / f"{run_prefix.name}_rebuild_report.json"
    alpaca = run_dir / f"{run_prefix.name}_alpaca.jsonl"
    split_report = run_dir / f"{run_prefix.name}_split_report.json"
    dialogue_work_dir = run_dir / "_dialogue_work"
    pipeline_log = args.log_file or (run_dir / f"{run_prefix.name}_pipeline.log")
    logger = PipelineLogger(pipeline_log)
    logger.log(f"run_name={args.run_name}")
    logger.log(f"run_dir={run_dir}")
    logger.log(f"workers={args.workers}")
    logger.log(f"log_file={pipeline_log}")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    if not args.skip_synthesis:
        if args.retry_failed_only:
            raise SystemExit("--retry-failed-only is deprecated. Use --repair-failed-dialogues.")
        if args.repair_failed_dialogues:
            failure_records = load_failure_ledger(failed_dialogues)
            pending = len(
                select_failure_dialogue_ids(
                    failure_records,
                    args.source,
                    max_dialogues=args.max_dialogues,
                    max_user_turns=args.max_user_turns,
                )
            )
        else:
            pending = count_pending_dialogues(
                args.source,
                checkpoint,
                dialogue_work_dir,
                args.max_dialogues,
                args.max_user_turns,
            )
        synth_command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.extractor.synthesis.synthesize_extractor_dialogues",
            "--source",
            str(args.source),
            "--output",
            str(repair_succeeded if args.repair_failed_dialogues else synthesized),
            "--failed",
            str(failed_dialogues),
            "--failed-dialogues",
            str(failed_dialogues),
            "--final-states",
            str(final_states),
            "--checkpoint",
            str(checkpoint),
            "--dialogue-work-dir",
            str(dialogue_work_dir),
            "--model",
            args.model,
            "--api-format",
            args.api_format,
            "--api-key",
            args.api_key or "",
            "--base-url",
            args.base_url,
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--max-context-messages",
            str(args.max_context_messages),
            "--max-context-chars",
            str(args.max_context_chars),
            "--sleep",
            str(args.sleep),
            "--max-retries",
            str(args.max_retries),
            "--timeout",
            str(args.timeout),
            "--workers",
            str(args.workers),
        ]
        if args.repair_failed_dialogues:
            synth_command.extend(
                [
                    "--repair-failed-dialogues",
                    "--repair-succeeded",
                    str(repair_succeeded),
                ]
            )
        if args.no_ssl_verify:
            synth_command.append("--no-ssl-verify")
        if args.terminology_db is not None:
            synth_command.extend(["--terminology-db", str(args.terminology_db)])
        if args.max_dialogues is not None:
            synth_command.extend(["--max-dialogues", str(args.max_dialogues)])
        if args.max_user_turns is not None:
            synth_command.extend(["--max-user-turns", str(args.max_user_turns)])
        step_name = "repair failed dialogues" if args.repair_failed_dialogues else "synthesis"
        emit(f"Step 1/3: {step_name} -> {synthesized}", logger)
        run_with_progress(synth_command, pending, env, logger)

        if args.repair_failed_dialogues:
            backup_dir = backup_before_rebuild(
                [synthesized, validated, alpaca, final_states, checkpoint, report, split_report],
                run_dir,
            )
            if backup_dir is not None:
                emit(f"Backup before rebuild -> {backup_dir}", logger)
        rebuild_command = [
            sys.executable,
            "-m",
            "scripts.extractor.synthesis.rebuild_extractor_dataset",
            "--source",
            str(args.source),
            "--synthesized",
            str(synthesized),
            "--repair-succeeded",
            str(repair_succeeded),
            "--failed-dialogues",
            str(failed_dialogues),
            "--dialogue-work-dir",
            str(dialogue_work_dir),
            "--output",
            str(synthesized),
            "--final-states",
            str(final_states),
            "--checkpoint",
            str(checkpoint),
            "--report",
            str(rebuild_report),
            "--max-context-messages",
            str(args.max_context_messages),
            "--max-context-chars",
            str(args.max_context_chars),
        ]
        if args.max_dialogues is not None:
            rebuild_command.extend(["--max-dialogues", str(args.max_dialogues)])
        if args.max_user_turns is not None:
            rebuild_command.extend(["--max-user-turns", str(args.max_user_turns)])
        if args.terminology_db is not None:
            rebuild_command.extend(["--terminology-db", str(args.terminology_db)])
        emit(f"Step 1b/3: rebuild -> {synthesized}", logger)
        run_plain(rebuild_command, env, logger)
        if dialogue_work_dir.exists():
            shutil.rmtree(dialogue_work_dir)

    if not args.skip_validation:
        validate_command = [
            sys.executable,
            "-m",
            "scripts.extractor.synthesis.validate_extractor_data",
            "--input",
            str(synthesized),
            "--output",
            str(validated),
            "--failed",
            str(failed_dialogues),
            "--failed-dialogues",
            str(failed_dialogues),
            "--source",
            str(args.source),
            "--report",
            str(report),
        ]
        if args.terminology_db is not None:
            validate_command.extend(["--terminology-db", str(args.terminology_db)])
        emit(f"Step 2/3: validation -> {validated}", logger)
        run_plain(validate_command, env, logger)

    if not args.skip_alpaca:
        if args.split_ratios is None:
            convert_command = [
                sys.executable,
                "-m",
                "scripts.extractor.synthesis.convert_to_alpaca",
                "--input",
                str(validated),
                "--output",
                str(alpaca),
            ]
            emit(f"Step 3/3: Alpaca conversion -> {alpaca}", logger)
            run_plain(convert_command, env, logger)
        else:
            train_ratio, valid_ratio, test_ratio = args.split_ratios
            split_command = [
                sys.executable,
                "-m",
                "scripts.extractor.synthesis.split_extractor_dataset",
                "--input",
                str(validated),
                "--output-prefix",
                str(run_prefix),
                "--train-ratio",
                str(train_ratio),
                "--valid-ratio",
                str(valid_ratio),
                "--test-ratio",
                str(test_ratio),
                "--seed",
                str(args.split_seed),
            ]
            emit(f"Step 3/3: split + Alpaca conversion -> {split_report}", logger)
            run_plain(split_command, env, logger)

    summary = {
        "synthesized": str(synthesized),
        "failed_dialogues": str(failed_dialogues),
        "repair_succeeded": str(repair_succeeded) if args.repair_failed_dialogues else None,
        "final_states": str(final_states),
        "checkpoint": str(checkpoint),
        "validated": str(validated),
        "report": str(report),
        "rebuild_report": str(rebuild_report) if not args.skip_synthesis else None,
        "alpaca": str(alpaca) if args.split_ratios is None else None,
        "split_report": str(split_report) if args.split_ratios is not None else None,
        "pipeline_log": str(pipeline_log),
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(summary_text)
    logger.log(summary_text)


if __name__ == "__main__":
    main()
