from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar


T = TypeVar("T")


def execute_dialogue_tasks(
    dialogues: Iterable[dict[str, Any]],
    workers: int,
    task: Callable[[dict[str, Any]], T],
) -> Iterator[tuple[dict[str, Any], T]]:
    dialogue_list = list(dialogues)
    if workers == 1:
        for dialogue in dialogue_list:
            yield dialogue, task(dialogue)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="askmed-synthesis") as executor:
        futures = {executor.submit(task, dialogue): dialogue for dialogue in dialogue_list}
        for future in as_completed(futures):
            yield futures[future], future.result()


def dialogue_shard_path(work_dir: Path, dialogue_id: str) -> Path:
    digest = hashlib.sha256(dialogue_id.encode("utf-8")).hexdigest()[:24]
    return work_dir / f"{digest}.json"


def write_dialogue_shard(
    work_dir: Path,
    dialogue_id: str,
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    user_turns: int,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = dialogue_shard_path(work_dir, dialogue_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "dialogue_id": dialogue_id,
        "rows": rows,
        "patient_state": state,
        "completed_turn_id": rows[-1]["input"]["turn_id"] if rows else None,
        "user_turns": user_turns,
    }
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_dialogue_shards(work_dir: Path) -> dict[str, dict[str, Any]]:
    if not work_dir.exists():
        return {}
    shards: dict[str, dict[str, Any]] = {}
    for path in sorted(work_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid dialogue work shard {path}: {exc}") from exc
        dialogue_id = str(payload.get("dialogue_id") or "")
        rows = payload.get("rows")
        if not dialogue_id or not isinstance(rows, list):
            raise ValueError(f"Invalid dialogue work shard shape: {path}")
        shards[dialogue_id] = payload
    return shards
