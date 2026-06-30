from __future__ import annotations

import argparse
import random
from pathlib import Path

from extractor_common import (
    DEFAULT_SOURCE,
    DEFAULT_WORK_DIR,
    has_weak_labels,
    is_question_targeted,
    is_short_context_answer,
    nearest_previous_assistant,
    recent_context,
    weak_labels_from,
    read_jsonl,
    write_jsonl,
)


def build_turn_rows(source: Path, max_context_messages: int, max_context_chars: int) -> dict[str, list[dict]]:
    buckets = {"labeled": [], "targeted_unlabeled": [], "ordinary_unlabeled": []}
    for dialog in read_jsonl(source):
        dialogue_id = dialog.get("dialogue_id")
        messages = dialog.get("messages") or []
        for idx, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            previous_question = nearest_previous_assistant(messages, idx)
            labels = weak_labels_from(message)
            content = str(message.get("content") or "").strip()
            row = {
                "dialogue_id": dialogue_id,
                "turn_id": idx,
                "previous_doctor_question": previous_question,
                "patient_utterance": content,
                "recent_context": recent_context(
                    messages,
                    idx,
                    max_messages=max_context_messages,
                    max_chars=max_context_chars,
                ),
                "meddg_weak_labels": labels,
                "is_context_dependent": is_short_context_answer(content) or is_question_targeted(previous_question),
            }
            if has_weak_labels(message):
                row["sample_bucket"] = "labeled"
                buckets["labeled"].append(row)
            elif is_question_targeted(previous_question):
                row["sample_bucket"] = "targeted_unlabeled"
                buckets["targeted_unlabeled"].append(row)
            else:
                row["sample_bucket"] = "ordinary_unlabeled"
                buckets["ordinary_unlabeled"].append(row)
    return buckets


def choose_samples(buckets: dict[str, list[dict]], total: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ratios = {
        "labeled": 0.5,
        "targeted_unlabeled": 0.3,
        "ordinary_unlabeled": 0.2,
    }
    selected: list[dict] = []
    leftovers: list[dict] = []
    for bucket, rows in buckets.items():
        rows = rows[:]
        rng.shuffle(rows)
        want = int(total * ratios[bucket])
        take = min(want, len(rows))
        selected.extend(rows[:take])
        leftovers.extend(rows[take:])
    if len(selected) < total:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])
    rng.shuffle(selected)
    return selected[:total]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample MedDG patient turns for extractor data synthesis.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR / "extractor_inputs_sample.jsonl")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-messages", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=1500)
    args = parser.parse_args()

    buckets = build_turn_rows(args.source, args.max_context_messages, args.max_context_chars)
    rows = choose_samples(buckets, args.sample_size, args.seed)
    count = write_jsonl(args.output, rows)
    print(f"Wrote {count} sampled turns to {args.output}")
    print({name: len(items) for name, items in buckets.items()})


if __name__ == "__main__":
    main()
