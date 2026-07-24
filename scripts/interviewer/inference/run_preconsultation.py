from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from openai import OpenAI

from scripts.extractor.inference.run_inference import build_runtime_row, call_extractor
from scripts.extractor.pipeline import FactPipeline, FactPipelineError
from scripts.extractor.pipeline.schema import resolve_from_root
from scripts.extractor.pipeline.state_manager import empty_patient_state
from scripts.interviewer.inference.client import call_interviewer
from scripts.interviewer.pipeline.state_projection import project_interviewer_state


def new_session(session_id: str) -> dict[str, Any]:
    return {
        "medical_state": empty_patient_state(session_id),
        "dialogue_control": {"asked_targets": [], "turn_count": 0, "ended": False},
        "history": [],
        "previous_doctor_question": None,
    }


def run_stream(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    extractor_client: OpenAI,
    extractor_model: str,
    interviewer_client: OpenAI,
    interviewer_model: str,
    fact_pipeline: FactPipeline,
    timeout: int,
    max_retries: int,
    max_context_messages: int,
) -> None:
    sessions: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(input_stream, 1):
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            session_id = str(request.get("session_id") or request.get("dialogue_id") or "default")
            if request.get("reset") or session_id not in sessions:
                sessions[session_id] = new_session(session_id)
            session = sessions[session_id]
            if session["dialogue_control"]["ended"]:
                raise ValueError("session has ended; send reset=true to start again")
            patient_utterance = str(request.get("patient_utterance") or "").strip()
            if not patient_utterance:
                raise ValueError("patient_utterance is required")

            turn_id = session["dialogue_control"]["turn_count"]
            context = session["history"][-max_context_messages:]
            extractor_row = build_runtime_row(
                {
                    "dialogue_id": session_id,
                    "turn_id": turn_id,
                    "previous_doctor_question": session["previous_doctor_question"],
                    "patient_utterance": patient_utterance,
                    "recent_context": context,
                },
                session["medical_state"],
            )
            extraction, extractor_raw = call_extractor(
                extractor_client,
                extractor_model,
                extractor_row,
                timeout,
                max_retries,
                0.5,
            )
            processed = fact_pipeline.process(
                extraction,
                patient_utterance=patient_utterance,
                state=session["medical_state"],
                turn_id=turn_id,
                previous_doctor_question=session["previous_doctor_question"],
                recent_context=context,
            )
            session["medical_state"] = processed.state_after
            projected_state = project_interviewer_state(processed.state_after)

            decision, interviewer_raw = call_interviewer(
                interviewer_client,
                model=interviewer_model,
                patient_block=[patient_utterance],
                patient_state=projected_state,
                recent_context=context,
                asked_targets=session["dialogue_control"]["asked_targets"],
                timeout=timeout,
                max_retries=max_retries,
            )
            session["history"].append({"role": "user", "content": patient_utterance})
            if decision["action"] == "ask":
                session["dialogue_control"]["asked_targets"].append(dict(decision["next_question_target"]))
                session["previous_doctor_question"] = decision["utterance"]
                session["history"].append({"role": "assistant", "content": decision["utterance"]})
            else:
                session["dialogue_control"]["ended"] = True
                session["previous_doctor_question"] = None
            session["dialogue_control"]["turn_count"] += 1

            response = {
                "ok": True,
                "session_id": session_id,
                "facts": processed.runtime_extraction["facts"],
                "patient_state": projected_state,
                "dialogue_control": session["dialogue_control"],
                "interviewer": decision,
                "normalization_stats": processed.normalization_stats,
                "raw": {"extractor": extractor_raw, "interviewer": interviewer_raw},
            }
        except FactPipelineError as exc:
            response = {
                "ok": False,
                "session_id": str(request.get("session_id") or "default"),
                "failure_type": exc.stage,
                "errors": exc.errors,
                "state_updated": False,
            }
        except Exception as exc:
            response = {
                "ok": False,
                "session_id": str(request.get("session_id") or "default"),
                "failure_type": "preconsultation_error",
                "errors": [f"line {line_no}: {type(exc).__name__}: {exc}"],
            }
        output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        output_stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AskMed extractor and interviewer as one session pipeline.")
    parser.add_argument("--extractor-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--extractor-api-key", default="EMPTY")
    parser.add_argument("--extractor-model", required=True)
    parser.add_argument("--interviewer-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--interviewer-api-key", default="EMPTY")
    parser.add_argument("--interviewer-model", required=True)
    parser.add_argument("--terminology-db", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-context-messages", type=int, default=4)
    args = parser.parse_args()

    terminology_db = resolve_from_root(args.terminology_db) if args.terminology_db else None
    input_stream = args.input.open("r", encoding="utf-8") if args.input else sys.stdin
    output_stream = args.output.open("w", encoding="utf-8", newline="\n") if args.output else sys.stdout
    try:
        run_stream(
            input_stream,
            output_stream,
            extractor_client=OpenAI(api_key=args.extractor_api_key, base_url=args.extractor_base_url),
            extractor_model=args.extractor_model,
            interviewer_client=OpenAI(api_key=args.interviewer_api_key, base_url=args.interviewer_base_url),
            interviewer_model=args.interviewer_model,
            fact_pipeline=FactPipeline(terminology_db),
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_context_messages=args.max_context_messages,
        )
    finally:
        if args.input:
            input_stream.close()
        if args.output:
            output_stream.close()


if __name__ == "__main__":
    main()
