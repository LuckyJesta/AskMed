from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from openai import OpenAI

from scripts.extractor.pipeline import FactPipeline, FactPipelineError
from scripts.extractor.pipeline.schema import (
    TRAINING_SYSTEM_PROMPT,
    extract_json_object,
    format_alpaca_input,
    normalize_extraction,
    resolve_from_root,
)
from scripts.extractor.pipeline.state_manager import empty_patient_state, prompt_patient_state


def build_runtime_row(request: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return {
        "dialogue_id": str(request.get("dialogue_id") or "default"),
        "turn_id": request.get("turn_id"),
        "previous_doctor_question": request.get("previous_doctor_question"),
        "patient_utterance": str(request.get("patient_utterance") or "").strip(),
        "patient_state_before_turn": prompt_patient_state(state),
        "recent_context": request.get("recent_context") or [],
    }


def call_extractor(
    client: OpenAI,
    model: str,
    row: dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    raw_response = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                    {"role": "user", "content": format_alpaca_input(row)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                timeout=timeout,
            )
            raw_response = response.choices[0].message.content or ""
            if not raw_response:
                raise ValueError("model returned empty content")
            return normalize_extraction(extract_json_object(raw_response)), raw_response
        except Exception as exc:  # The API SDK exposes several transport-specific subclasses.
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
    assert last_error is not None
    raise last_error


def run_stream(
    input_stream: TextIO,
    output_stream: TextIO,
    client: OpenAI,
    model: str,
    fact_pipeline: FactPipeline,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> None:
    states: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(input_stream, 1):
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            dialogue_id = str(request.get("dialogue_id") or "default")
            if request.get("reset") or dialogue_id not in states:
                states[dialogue_id] = empty_patient_state(dialogue_id)
            state_before = states[dialogue_id]
            row = build_runtime_row(request, state_before)
            if not row["patient_utterance"]:
                raise ValueError("patient_utterance is required")
            processing_error: FactPipelineError | None = None
            for attempt in range(1, max_retries + 1):
                extraction, raw_response = call_extractor(
                    client,
                    model,
                    row,
                    timeout,
                    1,
                    retry_sleep,
                )
                try:
                    processed = fact_pipeline.process(
                        extraction,
                        patient_utterance=row["patient_utterance"],
                        state=state_before,
                        turn_id=row["turn_id"],
                        previous_doctor_question=row["previous_doctor_question"],
                        recent_context=row["recent_context"],
                    )
                    break
                except FactPipelineError as exc:
                    processing_error = exc
                    if attempt < max_retries:
                        time.sleep(retry_sleep * attempt)
            else:
                assert processing_error is not None
                raise processing_error
            states[dialogue_id] = processed.state_after
            response_row = {
                "ok": True,
                "dialogue_id": dialogue_id,
                "turn_id": row["turn_id"],
                "raw_response": raw_response,
                "facts": processed.runtime_extraction["facts"],
                "patient_state_after_turn": processed.state_after,
                "normalization_stats": processed.normalization_stats,
                "warnings": processed.warnings,
            }
        except FactPipelineError as exc:
            response_row = {
                "ok": False,
                "dialogue_id": str(request.get("dialogue_id") or "default"),
                "turn_id": request.get("turn_id"),
                "failure_type": exc.stage,
                "errors": exc.errors,
                "state_updated": False,
            }
        except Exception as exc:
            response_row = {
                "ok": False,
                "dialogue_id": str(request.get("dialogue_id") or "default"),
                "turn_id": request.get("turn_id"),
                "failure_type": "inference_error",
                "errors": [f"line {line_no}: {type(exc).__name__}: {exc}"],
                "state_updated": False,
            }
        output_stream.write(json.dumps(response_row, ensure_ascii=False, separators=(",", ":")) + "\n")
        output_stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AskMed extractor inference through a LLaMA-Factory API server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--terminology-db", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None, help="JSONL input; defaults to stdin.")
    parser.add_argument("--output", type=Path, default=None, help="JSONL output; defaults to stdout.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=0.5)
    args = parser.parse_args()

    terminology_db = resolve_from_root(args.terminology_db) if args.terminology_db else None
    fact_pipeline = FactPipeline(terminology_db)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    input_stream = args.input.open("r", encoding="utf-8") if args.input else sys.stdin
    output_stream = args.output.open("w", encoding="utf-8", newline="\n") if args.output else sys.stdout
    try:
        run_stream(
            input_stream,
            output_stream,
            client,
            args.model,
            fact_pipeline,
            args.timeout,
            args.max_retries,
            args.retry_sleep,
        )
    finally:
        if args.input:
            input_stream.close()
        if args.output:
            output_stream.close()


if __name__ == "__main__":
    main()
