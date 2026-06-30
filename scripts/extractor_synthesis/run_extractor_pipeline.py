from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from extractor_common import DEFAULT_SOURCE, DEFAULT_WORK_DIR, read_jsonl, resolve_from_root


SCRIPT_DIR = Path(__file__).resolve().parent
ASKMED_ROOT = SCRIPT_DIR.parents[1]


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


def count_pending_user_turns(source: Path, checkpoint: Path, max_dialogues: int | None) -> int:
    checkpoints = load_checkpoints(checkpoint)
    pending = 0
    processed_dialogues = 0
    for dialog in read_jsonl(source):
        if max_dialogues is not None and processed_dialogues >= max_dialogues:
            break
        dialogue_id = str(dialog.get("dialogue_id"))
        messages = dialog.get("messages") or []
        checkpoint_row = checkpoints.get(dialogue_id)
        if checkpoint_row and checkpoint_row.get("finished"):
            continue
        completed_turn_id = None
        if checkpoint_row and isinstance(checkpoint_row.get("completed_turn_id"), int):
            completed_turn_id = checkpoint_row["completed_turn_id"]
        processed_dialogues += 1
        for idx, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            if completed_turn_id is not None and idx <= completed_turn_id:
                continue
            pending += 1
    return pending


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
    with tqdm(total=total, desc="Synthesis", unit="turn", dynamic_ncols=True) as progress:
        for line in process.stdout:
            stripped = line.strip()
            if stripped.startswith(("ok ", "rule-empty ", "failed ")):
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
    parser.add_argument("--max-dialogues", type=int, default=None)
    parser.add_argument("--max-user-turns", type=int, default=None)
    parser.add_argument("--max-context-messages", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-alpaca", action="store_true")
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
    args.work_dir.mkdir(parents=True, exist_ok=True)
    synthesized = args.work_dir / f"{args.run_name}_synthesized.jsonl"
    synth_failed = args.work_dir / f"{args.run_name}_synthesis_failed.jsonl"
    final_states = args.work_dir / f"{args.run_name}_final_states.jsonl"
    checkpoint = args.work_dir / f"{args.run_name}_checkpoints.jsonl"
    validated = args.work_dir / f"{args.run_name}_validated.jsonl"
    validation_failed = args.work_dir / f"{args.run_name}_validation_failed.jsonl"
    report = args.work_dir / f"{args.run_name}_report.json"
    alpaca = args.work_dir / f"{args.run_name}_alpaca.jsonl"
    split_report = args.work_dir / f"{args.run_name}_split_report.json"
    pipeline_log = args.log_file or (args.work_dir / f"{args.run_name}_pipeline.log")
    logger = PipelineLogger(pipeline_log)
    logger.log(f"run_name={args.run_name}")
    logger.log(f"log_file={pipeline_log}")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    if not args.skip_synthesis:
        pending = count_pending_user_turns(args.source, checkpoint, args.max_dialogues)
        if args.max_user_turns is not None:
            pending = min(pending, args.max_user_turns)
        synth_command = [
            sys.executable,
            "-u",
            str(SCRIPT_DIR / "synthesize_extractor_dialogues.py"),
            "--source",
            str(args.source),
            "--output",
            str(synthesized),
            "--failed",
            str(synth_failed),
            "--final-states",
            str(final_states),
            "--checkpoint",
            str(checkpoint),
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
        ]
        if args.no_ssl_verify:
            synth_command.append("--no-ssl-verify")
        if args.terminology_db is not None:
            synth_command.extend(["--terminology-db", str(args.terminology_db)])
        if args.max_dialogues is not None:
            synth_command.extend(["--max-dialogues", str(args.max_dialogues)])
        if args.max_user_turns is not None:
            synth_command.extend(["--max-user-turns", str(args.max_user_turns)])
        emit(f"Step 1/3: synthesis -> {synthesized}", logger)
        run_with_progress(synth_command, pending, env, logger)

    if not args.skip_validation:
        validate_command = [
            sys.executable,
            str(SCRIPT_DIR / "validate_extractor_data.py"),
            "--input",
            str(synthesized),
            "--output",
            str(validated),
            "--failed",
            str(validation_failed),
            "--report",
            str(report),
        ]
        emit(f"Step 2/3: validation -> {validated}", logger)
        run_plain(validate_command, env, logger)

    if not args.skip_alpaca:
        if args.split_ratios is None:
            convert_command = [
                sys.executable,
                str(SCRIPT_DIR / "convert_to_alpaca.py"),
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
                str(SCRIPT_DIR / "split_extractor_dataset.py"),
                "--input",
                str(validated),
                "--output-prefix",
                str(args.work_dir / args.run_name),
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
        "synthesis_failed": str(synth_failed),
        "final_states": str(final_states),
        "checkpoint": str(checkpoint),
        "validated": str(validated),
        "validation_failed": str(validation_failed),
        "report": str(report),
        "alpaca": str(alpaca) if args.split_ratios is None else None,
        "split_report": str(split_report) if args.split_ratios is not None else None,
        "pipeline_log": str(pipeline_log),
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(summary_text)
    logger.log(summary_text)


if __name__ == "__main__":
    main()
