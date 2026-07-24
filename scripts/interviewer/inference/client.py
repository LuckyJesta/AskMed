from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from scripts.interviewer.pipeline.schema import (
    INTERVIEWER_SYSTEM_PROMPT,
    build_interviewer_input,
    extract_json_object,
    validate_decision,
)


def call_interviewer(
    client: OpenAI,
    *,
    model: str,
    patient_block: list[str],
    patient_state: dict[str, Any],
    recent_context: list[dict[str, str]],
    asked_targets: list[dict[str, Any]],
    timeout: int = 120,
    max_retries: int = 2,
    retry_sleep: float = 0.5,
) -> tuple[dict[str, Any], str]:
    prompt = build_interviewer_input(patient_block, patient_state, recent_context, asked_targets)
    last_error: Exception | None = None
    raw_response = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=256,
                timeout=timeout,
            )
            raw_response = response.choices[0].message.content or ""
            decision = extract_json_object(raw_response)
            errors = validate_decision(decision, state=patient_state, asked_targets=asked_targets)
            if errors:
                raise ValueError("; ".join(errors))
            return decision, raw_response
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
    assert last_error is not None
    raise RuntimeError(f"interviewer failed after {max_retries} attempts: {last_error}") from last_error
