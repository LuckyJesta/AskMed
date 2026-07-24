from __future__ import annotations

from collections import defaultdict
import re
from typing import Any


PATIENT_CLOSURE_TERMS = (
    "好",
    "好的",
    "好吧",
    "嗯",
    "嗯嗯",
    "谢谢",
    "谢谢医生",
    "知道了",
    "明白了",
    "再见",
    "晚安",
)
DOCTOR_CLOSURE_TERMS = (
    "不客气",
    "不用谢",
    "祝你",
    "有事再联系",
    "有问题再联系",
    "再见",
    "晚安",
    "好的",
)
NEW_TOPIC_CUES = (
    "还有一个问题",
    "还有个问题",
    "另外咨询",
    "另外问",
    "再问一个",
    "顺便再问",
    "顺便问",
)
NEW_COMPLAINT_PATTERNS = (
    re.compile(r"最近|近来|这几天|这段时间|新出现|又出现"),
    re.compile(r"从.{0,12}(?:开始|起)"),
    re.compile(r"(?:已经|持续|断断续续).{0,10}(?:天|周|月|年)"),
)


def _compact_text(value: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:~～]+", "", value).lower()


def _is_closure_block(messages: list[str], terms: tuple[str, ...], max_chars: int) -> bool:
    if not messages:
        return False
    text = _compact_text("".join(messages))
    if not text or len(text) > max_chars:
        return False
    return all(term in text for term in ("谢谢",)) or any(
        text == _compact_text(term) or text.startswith(_compact_text(term))
        for term in terms
    )


def _block_facts(block: dict[str, Any], rows_by_turn: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for turn_id in block.get("patient_turn_ids") or []:
        row = rows_by_turn.get(turn_id) or {}
        facts.extend(
            fact
            for fact in (row.get("parsed_output") or {}).get("facts") or []
            if isinstance(fact, dict)
        )
    return facts


def _fact_names(fact: dict[str, Any]) -> set[str]:
    names = {
        _compact_text(str(fact.get("name") or "")),
        _compact_text(str(fact.get("normalized_name") or "")),
    }
    attribute = fact.get("attribute") if isinstance(fact.get("attribute"), dict) else {}
    names.add(_compact_text(str(attribute.get("target") or "")))
    names.discard("")
    return names


def _names_overlap(left: set[str], right: set[str]) -> bool:
    return any(a == b or a in b or b in a for a in left for b in right)


def _explicit_new_topic(block: dict[str, Any]) -> bool:
    text = _compact_text("".join(block.get("patient_block") or []))
    return any(_compact_text(cue) in text for cue in NEW_TOPIC_CUES)


def _independent_new_complaint(
    previous_blocks: list[dict[str, Any]],
    following: dict[str, Any],
    rows_by_turn: dict[int, dict[str, Any]],
) -> bool:
    following_facts = [
        fact
        for fact in _block_facts(following, rows_by_turn)
        if fact.get("type") in {"symptom", "disease"}
        and fact.get("status") != "absent"
    ]
    if not following_facts:
        return False
    patient_text = "".join(following.get("patient_block") or [])
    if not any(pattern.search(patient_text) for pattern in NEW_COMPLAINT_PATTERNS):
        return False
    previous_facts = [
        fact
        for block in previous_blocks
        for fact in _block_facts(block, rows_by_turn)
    ]
    previous_names = {name for fact in previous_facts for name in _fact_names(fact)}
    following_names = {name for fact in following_facts for name in _fact_names(fact)}
    return bool(following_names) and not _names_overlap(previous_names, following_names)


def group_dialogue_messages(dialogue: dict[str, Any]) -> list[dict[str, Any]]:
    messages = dialogue.get("messages") or []
    blocks: list[dict[str, Any]] = []
    index = 0
    block_id = 0
    while index < len(messages):
        if messages[index].get("role") != "user":
            index += 1
            continue
        patient_messages: list[str] = []
        patient_turn_ids: list[int] = []
        while index < len(messages) and messages[index].get("role") == "user":
            patient_messages.append(str(messages[index].get("content") or "").strip())
            patient_turn_ids.append(index)
            index += 1
        doctor_messages: list[str] = []
        doctor_turn_ids: list[int] = []
        while index < len(messages) and messages[index].get("role") == "assistant":
            doctor_messages.append(str(messages[index].get("content") or "").strip())
            doctor_turn_ids.append(index)
            index += 1
        blocks.append(
            {
                "block_id": block_id,
                "patient_block": patient_messages,
                "patient_turn_ids": patient_turn_ids,
                "original_doctor_block": doctor_messages,
                "doctor_turn_ids": doctor_turn_ids,
            }
        )
        block_id += 1

    for position, block in enumerate(blocks):
        block["next_patient_block"] = blocks[position + 1]["patient_block"] if position + 1 < len(blocks) else []
        block["is_terminal"] = position == len(blocks) - 1
    return blocks


def split_dialogue_sessions(
    dialogue: dict[str, Any],
    rows_by_turn: dict[int, dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Conservatively split a source dialogue after a closed exchange and new medical content."""
    blocks = group_dialogue_messages(dialogue)
    if not blocks:
        return []
    boundaries = [0]
    boundary_reasons: dict[int, str] = {}
    boundary_warnings: dict[int, str] = {}
    for index in range(len(blocks) - 1):
        current = blocks[index]
        following = blocks[index + 1]
        patient_closed = _is_closure_block(
            current.get("patient_block") or [],
            PATIENT_CLOSURE_TERMS,
            24,
        )
        doctor_closed = _is_closure_block(
            current.get("original_doctor_block") or [],
            DOCTOR_CLOSURE_TERMS,
            50,
        )
        if not (patient_closed and doctor_closed):
            continue
        following_facts = _block_facts(following, rows_by_turn)
        if not following_facts:
            continue
        if _explicit_new_topic(following):
            boundaries.append(index + 1)
            boundary_reasons[index + 1] = "explicit_new_topic"
        elif _independent_new_complaint(blocks[: index + 1], following, rows_by_turn):
            boundaries.append(index + 1)
            boundary_reasons[index + 1] = "independent_new_complaint"
        else:
            boundary_warnings[index + 1] = "ambiguous_session_boundary"
    boundaries.append(len(blocks))

    source_dialogue_id = str(dialogue.get("dialogue_id") or "")
    sessions: list[list[dict[str, Any]]] = []
    for session_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        session: list[dict[str, Any]] = []
        for local_block_id, source_block in enumerate(blocks[start:end]):
            block = dict(source_block)
            block["source_block_id"] = source_block["block_id"]
            block["block_id"] = local_block_id
            block["session_index"] = session_index
            block["session_id"] = f"{source_dialogue_id}#session{session_index}"
            source_index = int(source_block["block_id"])
            if source_index in boundary_reasons:
                block["session_split_reason"] = boundary_reasons[source_index]
            if source_index in boundary_warnings:
                block["session_boundary_warning"] = boundary_warnings[source_index]
            session.append(block)
        for position, block in enumerate(session):
            block["next_patient_block"] = session[position + 1]["patient_block"] if position + 1 < len(session) else []
            block["is_terminal"] = position == len(session) - 1
        sessions.append(session)
    return sessions


def build_split_map(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split, rows in split_rows.items():
        for row in rows:
            dialogue_id = str((row.get("input") or {}).get("dialogue_id") or "")
            if not dialogue_id:
                continue
            previous = result.get(dialogue_id)
            if previous is not None and previous != split:
                raise ValueError(f"dialogue {dialogue_id} appears in both {previous} and {split}")
            result[dialogue_id] = split
    return result


def index_extractor_rows(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    indexed: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        input_row = row.get("input") or {}
        dialogue_id = str(input_row.get("dialogue_id") or "")
        turn_id = input_row.get("turn_id")
        if dialogue_id and isinstance(turn_id, int):
            indexed[dialogue_id][turn_id] = row
    return dict(indexed)


def recent_context_before(
    messages: list[dict[str, Any]],
    turn_id: int,
    max_messages: int,
    max_chars: int,
    start_turn_id: int = 0,
) -> list[dict[str, str]]:
    context: list[dict[str, str]] = []
    used = 0
    for message in reversed(messages[start_turn_id:turn_id]):
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        content = content[-remaining:]
        context.append({"role": str(message.get("role") or ""), "content": content})
        used += len(content)
        if len(context) >= max_messages:
            break
    context.reverse()
    return context
