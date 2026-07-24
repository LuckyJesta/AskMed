from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.extractor.pipeline.schema import read_jsonl, select_source_dialogues


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_failure_ledger(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        dialogue_id = row.get("dialogue_id")
        if dialogue_id:
            records[str(dialogue_id)] = row
    return records


def select_failure_dialogue_ids(
    records: dict[str, dict[str, Any]],
    source: Path,
    max_dialogues: int | None = None,
    max_user_turns: int | None = None,
) -> list[str]:
    """Return failed dialogue IDs inside the same stable source prefix used for synthesis."""
    if max_dialogues is None and max_user_turns is None:
        return sorted(records)

    selected, _ = select_source_dialogues(
        source,
        max_dialogues=max_dialogues,
        max_user_turns=max_user_turns,
    )
    return [
        dialogue_id
        for dialogue in selected
        if (dialogue_id := str(dialogue.get("dialogue_id") or "")) in records
    ]


def write_failure_ledger(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        if path.exists():
            path.unlink()
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for dialogue_id in sorted(records):
            f.write(json.dumps(records[dialogue_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def update_failed_dialogue(
    path: Path,
    *,
    dialogue_id: str,
    dialogue: dict[str, Any],
    failure_type: str,
    failed_at: str,
    failed_turn_id: int | None = None,
    input_row: dict[str, Any] | None = None,
    raw_response: str = "",
    error: str = "",
    errors: list[str] | None = None,
    model: str | None = None,
    api_format: str | None = None,
) -> None:
    records = load_failure_ledger(path)
    upsert_failed_dialogue(
        records,
        dialogue_id=dialogue_id,
        dialogue=dialogue,
        failure_type=failure_type,
        failed_at=failed_at,
        failed_turn_id=failed_turn_id,
        input_row=input_row,
        raw_response=raw_response,
        error=error,
        errors=errors,
        model=model,
        api_format=api_format,
    )
    write_failure_ledger(path, records)


def upsert_failed_dialogue(
    records: dict[str, dict[str, Any]],
    *,
    dialogue_id: str,
    dialogue: dict[str, Any],
    failure_type: str,
    failed_at: str,
    failed_turn_id: int | None = None,
    input_row: dict[str, Any] | None = None,
    raw_response: str = "",
    error: str = "",
    errors: list[str] | None = None,
    model: str | None = None,
    api_format: str | None = None,
) -> None:
    previous = records.get(dialogue_id, {})
    attempt_count = int(previous.get("attempt_count") or 0) + 1
    timestamp = now_iso()
    records[dialogue_id] = {
        "dialogue_id": dialogue_id,
        "dialogue": dialogue,
        "status": "failed",
        "failure_type": failure_type,
        "failed_at": failed_at,
        "failed_turn_id": failed_turn_id,
        "input": input_row or {},
        "attempt_count": attempt_count,
        "first_failed_at": previous.get("first_failed_at") or failed_at,
        "last_failed_at": failed_at,
        "last_error": error,
        "last_errors": errors or [],
        "last_raw_response": raw_response,
        "model": model,
        "api_format": api_format,
        "updated_at": timestamp,
    }


def remove_failed_dialogue(path: Path, dialogue_id: str) -> None:
    records = load_failure_ledger(path)
    if dialogue_id in records:
        del records[dialogue_id]
        write_failure_ledger(path, records)
