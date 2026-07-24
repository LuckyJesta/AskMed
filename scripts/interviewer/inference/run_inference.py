from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from openai import OpenAI

from scripts.interviewer.inference.client import call_interviewer
from scripts.interviewer.pipeline.state_projection import project_interviewer_state


def run_stream(
    input_stream: TextIO,
    output_stream: TextIO,
    client: OpenAI,
    model: str,
    timeout: int,
    max_retries: int,
) -> None:
    controls: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(input_stream, 1):
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            dialogue_id = str(request.get("dialogue_id") or "default")
            if request.get("reset") or dialogue_id not in controls:
                controls[dialogue_id] = {"asked_targets": [], "turn_count": 0, "ended": False}
            control = controls[dialogue_id]
            state = project_interviewer_state(request.get("medical_state") or request.get("patient_state") or {})
            patient_block = request.get("patient_block")
            if not isinstance(patient_block, list):
                utterance = str(request.get("patient_utterance") or "").strip()
                patient_block = [utterance] if utterance else []
            if not patient_block:
                raise ValueError("patient_block or patient_utterance is required")
            decision, raw = call_interviewer(
                client,
                model=model,
                patient_block=[str(item) for item in patient_block],
                patient_state=state,
                recent_context=request.get("recent_context") or [],
                asked_targets=control["asked_targets"],
                timeout=timeout,
                max_retries=max_retries,
            )
            if decision["action"] == "ask":
                control["asked_targets"].append(dict(decision["next_question_target"]))
            else:
                control["ended"] = True
            control["turn_count"] += 1
            response = {
                "ok": True,
                "dialogue_id": dialogue_id,
                "decision": decision,
                "dialogue_control": control,
                "raw_response": raw,
            }
        except Exception as exc:
            response = {
                "ok": False,
                "dialogue_id": str(request.get("dialogue_id") or "default"),
                "failure_type": "interviewer_inference_error",
                "errors": [f"line {line_no}: {type(exc).__name__}: {exc}"],
                "control_updated": False,
            }
        output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        output_stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Call an AskMed interviewer OpenAI-compatible API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    input_stream = args.input.open("r", encoding="utf-8") if args.input else sys.stdin
    output_stream = args.output.open("w", encoding="utf-8", newline="\n") if args.output else sys.stdout
    try:
        run_stream(
            input_stream,
            output_stream,
            OpenAI(api_key=args.api_key, base_url=args.base_url),
            args.model,
            args.timeout,
            args.max_retries,
        )
    finally:
        if args.input:
            input_stream.close()
        if args.output:
            output_stream.close()


if __name__ == "__main__":
    main()
