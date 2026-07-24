from __future__ import annotations

import json
import re
from typing import Any


FACT_TYPES = {
    "symptom",
    "disease",
    "medicine",
    "examination",
    "attribute",
    "history",
    "lifestyle",
    "other",
}
QUESTION_TARGET_TYPES = FACT_TYPES - {"attribute"}

INTERVIEWER_SYSTEM_PROMPT = """你是一个严谨的中文预问诊器。根据患者当前表达、当前问诊状态、最近对话上下文和已问目标，决定提出一个必要且未重复的问题，或结束问诊。
规则：
1. action只能是ask或end，只输出JSON。
2. ask时一次只问一个主要问题，不重复询问状态中已有的信息或已问目标。
3. 优先补充主诉的关键属性、相关伴随症状、既往史、用药史和检查史。
4. 不诊断，不推测疾病，不提供药物、检查或治疗建议。
5. end仅用于没有必要继续收集预问诊信息的情况。
6. 询问某个实体的属性时，name填写状态中的真实实体名称，type填写该实体类型，禁止填写attribute；attribute使用英文规范键duration、body_part、frequency、character、severity、time、trigger、aggravating_factor、relieving_factor、effect、amount、dose、result、route或side_effect。
ask输出：{"action":"ask","next_question_target":{"name":"目标名称","type":"symptom","attribute":"duration或null"},"utterance":"一个问题？"}
end输出：{"action":"end","next_question_target":null,"utterance":""}"""

TEACHER_SYSTEM_PROMPT = """你是中文预问诊训练数据标注专家。你的任务仅是解析原医生回复中实际存在的问题，并把每个安全问题改写成一个单目标预问诊问题。
输入不会提供下一段患者回复或下一患者状态。不得猜测患者之后会说什么，不得根据未来答案创造问题。

规则：
1. questions按原医生问题出现顺序排列；原医生没有安全问题时输出空列表。
2. source_text必须逐字复制原医生回复中的连续原文片段。
3. 每个候选只能询问一个主要目标或一个attribute，不能把病程、频率、部位等多个槽位合并到同一句。
4. 不输出诊断、疾病推测、药物建议、检查建议或治疗建议。
5. 具体目标名称必须出现在source_text中；仅当问题使用“多久、哪里、什么性质”等省略表达且当前状态中只有一个对应实体时，才可使用该已有实体名称。
6. 对尚未出现的开放类别使用泛化目标：伴随症状、用药情况、检查情况、既往史、既往疾病、生活方式。不得从未知答案中猜具体药物、检查或症状名称。
7. next_question_target必须包含name、type、attribute三个字段。type只能是symptom、disease、medicine、examination、history、lifestyle、other，禁止使用attribute。属性问题的name和type必须指向当前状态中的真实实体。
8. attribute只能是null或以下英文规范键之一：duration、body_part、frequency、character、severity、time、trigger、aggravating_factor、relieving_factor、effect、amount、dose、result、route、side_effect。病程或持续时间使用duration，部位使用body_part，次数使用frequency，性质使用character，程度使用severity，诱因使用trigger。

必须只输出以下JSON结构：
{"questions":[{"source_text":"原医生连续原文","next_question_target":{"name":"目标名称","type":"symptom","attribute":null},"utterance":"一个中文问题？"}],"skip_reason_code":null,"skip_reason_detail":""}

questions为空时，skip_reason_code只能是NO_SAFE_DOCTOR_QUESTION、UNSAFE_QUESTION或COMPOUND_QUESTION之一；questions非空时skip_reason_code必须为null。只输出JSON，不要Markdown代码块或解释。"""

UNSAFE_PATTERNS = (
    "建议你",
    "建议您",
    "你可以",
    "您可以",
    "可以吃",
    "吃点",
    "需要服用",
    "应该服用",
    "可以服用",
    "做个检查",
    "建议检查",
    "最好检查",
    "考虑是",
    "可能是",
    "诊断为",
    "治疗方案",
)

CANONICAL_ATTRIBUTES = {
    "duration",
    "body_part",
    "frequency",
    "character",
    "severity",
    "time",
    "trigger",
    "aggravating_factor",
    "relieving_factor",
    "effect",
    "amount",
    "dose",
    "result",
    "route",
    "side_effect",
}

ATTRIBUTE_ALIASES = {
    "病程": "duration",
    "时长": "duration",
    "持续时间": "duration",
    "多久": "duration",
    "多长时间": "duration",
    "部位": "body_part",
    "位置": "body_part",
    "身体部位": "body_part",
    "次数": "frequency",
    "频次": "frequency",
    "频率": "frequency",
    "性质": "character",
    "性状": "character",
    "症状性质": "character",
    "程度": "severity",
    "严重程度": "severity",
    "发生时间": "time",
    "时间": "time",
    "诱因": "trigger",
    "诱发因素": "trigger",
    "加重因素": "aggravating_factor",
    "缓解因素": "relieving_factor",
    "效果": "effect",
    "疗效": "effect",
    "量": "amount",
    "数量": "amount",
    "剂量": "dose",
    "检查结果": "result",
    "结果": "result",
    "给药途径": "route",
    "用药途径": "route",
    "副作用": "side_effect",
    "不良反应": "side_effect",
}


def normalize_text(value: Any) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:]", "", str(value or "")).lower()


def canonical_attribute(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in CANONICAL_ATTRIBUTES:
        return lowered
    return ATTRIBUTE_ALIASES.get(normalize_text(raw))


def canonicalize_target(target: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(target)
    if target.get("attribute") is not None:
        canonical = canonical_attribute(target.get("attribute"))
        if canonical is not None:
            normalized["attribute"] = canonical
    return normalized


def canonicalize_teacher_response(response: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(response)
    questions: list[Any] = []
    for candidate in response.get("questions") or []:
        if not isinstance(candidate, dict):
            questions.append(candidate)
            continue
        item = dict(candidate)
        target = item.get("next_question_target")
        if isinstance(target, dict):
            item["next_question_target"] = canonicalize_target(target)
        questions.append(item)
    normalized["questions"] = questions
    return normalized


def extract_json_object(text: str) -> dict[str, Any]:
    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("top-level value must be an object")
    return value


def target_key(target: Any) -> tuple[str, str, str] | None:
    if not isinstance(target, dict):
        return None
    name = normalize_text(target.get("name"))
    fact_type = normalize_text(target.get("type"))
    attribute = canonical_attribute(target.get("attribute")) or normalize_text(target.get("attribute"))
    if not name or not fact_type:
        return None
    return fact_type, name, attribute


def _state_items(state: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in (
        "problems",
        "negative_findings",
        "medications",
        "examinations",
        "histories",
        "lifestyle",
        "other_facts",
    ):
        items.extend(item for item in state.get(key) or [] if isinstance(item, dict))
    return items


def _item_names(item: dict[str, Any]) -> set[str]:
    names = {
        normalize_text(item.get("name")),
        normalize_text(item.get("normalized_name")),
    }
    names.update(normalize_text(alias) for alias in item.get("aliases") or [])
    names.discard("")
    return names


def _canonical_attributes(item: dict[str, Any]) -> dict[str, Any]:
    attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
    normalized: dict[str, Any] = {}
    for key, value in attributes.items():
        canonical = canonical_attribute(key)
        if canonical is not None:
            normalized[canonical] = value
    return normalized


def target_is_known(target: dict[str, Any], state: dict[str, Any]) -> bool:
    wanted_name = normalize_text(target.get("name"))
    wanted_type = target.get("type")
    wanted_attribute = canonical_attribute(target.get("attribute"))
    for item in _state_items(state):
        if wanted_name not in _item_names(item) or item.get("type") != wanted_type:
            continue
        if not wanted_attribute:
            return True
        attributes = _canonical_attributes(item)
        if attributes.get(wanted_attribute) not in (None, "", [], {}):
            return True
    return False


GENERIC_TARGET_TYPES = {
    "伴随症状": "symptom",
    "其他症状": "symptom",
    "既往用药": "medicine",
    "用药情况": "medicine",
    "既往检查": "examination",
    "检查情况": "examination",
    "既往史": "history",
    "既往疾病": "disease",
    "生活方式": "lifestyle",
}
GENERIC_TARGET_NAMES = set(GENERIC_TARGET_TYPES)

SKIP_REASON_CODES = {
    "NO_SAFE_DOCTOR_QUESTION",
    "ANSWER_NOT_OBSERVED",
    "TARGET_ALREADY_KNOWN",
    "TARGET_ALREADY_ASKED",
    "UNSAFE_QUESTION",
    "COMPOUND_QUESTION",
    "SOURCE_NOT_TRACEABLE",
    "TARGET_NOT_RESOLVED",
}
TEACHER_SKIP_REASON_CODES = {
    "NO_SAFE_DOCTOR_QUESTION",
    "UNSAFE_QUESTION",
    "COMPOUND_QUESTION",
}


def _new_state_items(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> list[dict[str, Any]]:
    before_signatures = {
        (tuple(sorted(_item_names(item))), item.get("type"), item.get("status"), item.get("subject"))
        for item in _state_items(before_state)
    }
    return [
        item
        for item in _state_items(after_state)
        if (tuple(sorted(_item_names(item))), item.get("type"), item.get("status"), item.get("subject"))
        not in before_signatures
    ]


def _new_attribute_candidates(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> list[tuple[dict[str, Any], str]]:
    candidates: list[tuple[dict[str, Any], str]] = []
    before_items = _state_items(before_state)
    for after_item in _state_items(after_state):
        matching_before = next(
            (
                item
                for item in before_items
                if item.get("type") == after_item.get("type")
                and item.get("status") == after_item.get("status")
                and item.get("subject") == after_item.get("subject")
                and _item_names(item) & _item_names(after_item)
            ),
            None,
        )
        if matching_before is None:
            continue
        before_attributes = _canonical_attributes(matching_before)
        for key, value in _canonical_attributes(after_item).items():
            if value in (None, "", [], {}) or before_attributes.get(key) not in (None, "", [], {}):
                continue
            candidates.append((after_item, key))
    return candidates


AFFIRMATIVE_SHORT_ANSWERS = {"有", "是", "对", "对的", "正常", "正常的", "嗯", "嗯嗯"}
NEGATIVE_SHORT_ANSWERS = {"没有", "没有吧", "没", "不是", "不", "没有的", "没得", "无"}
ATTRIBUTE_SHORT_ANSWER_PATTERNS = {
    "duration": re.compile(r"^(?:\d+(?:\.\d+)?|[一二三四五六七八九十半两几]+)(?:分钟|小时|天|周|星期|个月|月|年)(?:左右|了)?$"),
    "frequency": re.compile(r"^(?:每天|每周|每月|一天|一周|一个月)?(?:\d+|[一二三四五六七八九十几两]+)次(?:左右)?$"),
    "body_part": re.compile(r"^[上下左右前后内外中]?(?:腹|腹部|胸|胸部|背|背部|腰|头|咽|喉|胃|肚脐|肋|肩|腿|手|脚|肛门|小腹).{0,6}$"),
}

UNCERTAINTY_CUES = ("不知道", "不清楚", "不确定", "可能", "好像")
OMITTED_ENTITY_CUES = (
    "这个症状",
    "这种症状",
    "这个情况",
    "这种情况",
    "多久",
    "多长时间",
    "哪里",
    "哪个部位",
    "什么位置",
    "什么性质",
    "怎么疼",
)
YES_NO_QUESTION_CUES = ("有没有", "有无", "是否", "是不是", "吗", "么")


def _contains_uncertainty(text: str) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(cue) in normalized for cue in UNCERTAINTY_CUES)


def _chinese_chunks(text: Any) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]+", normalize_text(text))


def _has_lexical_relation(left: Any, right: Any) -> bool:
    left_text = normalize_text(left)
    right_text = normalize_text(right)
    if not left_text or not right_text:
        return False
    if left_text in right_text or right_text in left_text:
        return True
    left_chunks = _chinese_chunks(left_text)
    right_chunks = _chinese_chunks(right_text)
    for left_chunk in left_chunks:
        for right_chunk in right_chunks:
            for start in range(len(left_chunk) - 1):
                if left_chunk[start : start + 2] in right_chunk:
                    return True
    return False


def _item_display_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("normalized_name") or "").strip()


def resolve_attribute_target(
    target: dict[str, Any],
    state: dict[str, Any],
    question_text: str = "",
) -> dict[str, Any] | None:
    """Bind an attribute question to one real entity in the current state."""
    attribute = canonical_attribute(target.get("attribute"))
    target_type = target.get("type")
    if attribute is None or target_type not in QUESTION_TARGET_TYPES:
        return None

    wanted_name = normalize_text(target.get("name"))
    compatible = [item for item in _state_items(state) if item.get("type") == target_type]
    exact = [item for item in compatible if wanted_name and wanted_name in _item_names(item)]
    if len(exact) == 1:
        resolved = dict(target)
        resolved["name"] = _item_display_name(exact[0])
        resolved["type"] = exact[0].get("type")
        resolved["attribute"] = attribute
        return resolved

    normalized_question = normalize_text(question_text)
    omitted = any(normalize_text(cue) in normalized_question for cue in OMITTED_ENTITY_CUES)
    if omitted and len(compatible) == 1:
        resolved = dict(target)
        resolved["name"] = _item_display_name(compatible[0])
        resolved["type"] = compatible[0].get("type")
        resolved["attribute"] = attribute
        return resolved
    return None


def _short_answer_method(
    target: dict[str, Any],
    next_patient_block: list[str] | None,
    question_text: str,
) -> str | None:
    if not next_patient_block:
        return None
    raw_answer = "".join(next_patient_block)
    answer = normalize_text(raw_answer)
    if not answer or len(answer) > 16:
        return None
    if _contains_uncertainty(raw_answer):
        return None
    attribute = canonical_attribute(target.get("attribute"))
    if attribute is None:
        normalized_question = normalize_text(question_text)
        is_yes_no = any(normalize_text(cue) in normalized_question for cue in YES_NO_QUESTION_CUES)
        if not is_yes_no or is_compound_question(question_text, target):
            return None
        if answer in AFFIRMATIVE_SHORT_ANSWERS or answer in NEGATIVE_SHORT_ANSWERS:
            return "explicit_short_answer"
        if re.fullmatch(r"(?:有|是|对)?(?:一点|有一点|很)?(?:轻微|轻|微弱|明显|严重)(?:的)?", answer):
            return "explicit_short_answer"
        return None
    slots = question_slot_keys(question_text)
    if slots != {attribute}:
        return None
    pattern = ATTRIBUTE_SHORT_ANSWER_PATTERNS.get(attribute)
    if pattern is not None and pattern.fullmatch(answer):
        return "short_attribute_answer"
    return None


def answerability_match_method(
    target: dict[str, Any],
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    *,
    next_patient_block: list[str] | None = None,
    question_text: str = "",
) -> str | None:
    """Return the first conservative rule proving that the next block answered a target."""
    if target_is_known(target, before_state):
        return None
    if target_is_known(target, after_state):
        return "exact_state_match"

    target_name = str(target.get("name") or "").strip()
    target_type = target.get("type")
    attribute = canonical_attribute(target.get("attribute"))
    if attribute is not None:
        candidates = [
            item
            for item, key in _new_attribute_candidates(before_state, after_state)
            if key == attribute and item.get("type") == target_type
        ]
        named = [item for item in candidates if normalize_text(target_name) in _item_names(item)]
        if named:
            return "canonical_attribute_match"
        compatible_before = [item for item in _state_items(before_state) if item.get("type") == target_type]
        if len(candidates) == 1 and len(compatible_before) == 1:
            return "unique_attribute_delta"
    else:
        new_items = [item for item in _new_state_items(before_state, after_state) if item.get("type") == target_type]
        if target_name in GENERIC_TARGET_NAMES and new_items:
            return "generic_type_delta"

        target_names = {normalize_text(target_name)}
        lexical_items = [
            item
            for item in new_items
            if any(_has_lexical_relation(target_name_value, item_name) for target_name_value in target_names for item_name in _item_names(item))
        ]
        if lexical_items:
            return "lexical_state_delta"

        answer_text = "".join(next_patient_block or [])
        if not _contains_uncertainty(answer_text) and _has_lexical_relation(target_name, answer_text):
            return "lexical_answer_match"

    return _short_answer_method(target, next_patient_block, question_text)


def target_answered_by_state_change(
    target: dict[str, Any],
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> bool:
    return answerability_match_method(target, before_state, after_state) is not None


def answerable_target_candidates(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build canonical question targets from facts newly supplied by the next patient block."""
    before_items = _state_items(before_state)
    after_items = _state_items(after_state)
    before_by_entity: dict[tuple[tuple[str, ...], str, str], list[dict[str, Any]]] = {}
    for item in before_items:
        key = (
            tuple(sorted(_item_names(item))),
            str(item.get("type") or ""),
            str(item.get("subject") or ""),
        )
        before_by_entity.setdefault(key, []).append(item)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in after_items:
        name = str(item.get("name") or "").strip()
        fact_type = str(item.get("type") or "").strip()
        if not name or fact_type not in FACT_TYPES:
            continue
        entity_key = (tuple(sorted(_item_names(item))), fact_type, str(item.get("subject") or ""))
        previous_items = before_by_entity.get(entity_key, [])
        same_status = next(
            (previous for previous in previous_items if previous.get("status") == item.get("status")),
            None,
        )
        if same_status is None:
            candidate = {"name": name, "type": fact_type, "attribute": None}
            key = target_key(candidate)
            if key is not None and key not in seen:
                candidates.append(candidate)
                seen.add(key)
            continue

        before_attributes = _canonical_attributes(same_status)
        after_attributes = _canonical_attributes(item)
        for attribute, value in after_attributes.items():
            if value in (None, "", [], {}):
                continue
            if before_attributes.get(attribute) not in (None, "", [], {}):
                continue
            candidate = {"name": name, "type": fact_type, "attribute": str(attribute)}
            key = target_key(candidate)
            if key is not None and key not in seen:
                candidates.append(candidate)
                seen.add(key)
    return candidates


def target_entity_exists(target: dict[str, Any], state: dict[str, Any]) -> bool:
    wanted_name = normalize_text(target.get("name"))
    wanted_type = target.get("type")
    matches = 0
    for item in _state_items(state):
        if wanted_name in _item_names(item) and item.get("type") == wanted_type:
            matches += 1
    return matches == 1


def question_slot_keys(text: str) -> set[str]:
    normalized = normalize_text(text)
    slots: set[str] = set()
    interval_frequency = any(
        cue in normalized
        for cue in ("多久一次", "多长时间一次", "几天一次", "几日一次", "隔多久一次")
    )
    cue_groups = {
        "duration": ("多久", "多长时间", "持续时间", "持续了"),
        "frequency": ("几次", "多少次", "频率", "经常", "偶尔"),
        "body_part": ("哪里", "哪个部位", "什么位置", "具体位置"),
        "character": ("什么性质", "怎么疼", "什么样的疼", "怎样疼"),
        "severity": ("严重吗", "多严重", "疼得厉害", "程度"),
        "time": ("什么时候", "几点", "白天还是晚上"),
        "trigger": ("什么引起", "什么诱发", "和什么有关"),
        "relieving_factor": ("怎么缓解", "什么能缓解", "缓解因素"),
        "effect": ("效果怎么样", "有没有效果", "是否有效"),
        "amount": ("多少量", "量有多少", "量大吗"),
        "dose": ("多大剂量", "每次吃多少", "一次吃多少", "剂量"),
        "result": ("检查结果", "结果怎么样", "结果如何", "是否正常"),
        "route": ("怎么用药", "怎么服用", "什么途径"),
        "side_effect": ("副作用", "不良反应"),
        "aggravating_factor": ("什么会加重", "加重因素", "怎样会加重"),
    }
    for slot, cues in cue_groups.items():
        if slot == "duration" and interval_frequency:
            continue
        if any(normalize_text(cue) in normalized for cue in cues):
            slots.add(slot)
    if interval_frequency:
        slots.add("frequency")
    return slots


def is_compound_question(utterance: str, target: dict[str, Any]) -> bool:
    slots = question_slot_keys(utterance)
    if len(slots) > 1:
        return True
    interrogative_starts = re.findall(
        r"(?:有没有|是否|多久|多长时间|几次|多少次|什么时候|哪里|哪个部位|什么性质|怎么疼)",
        utterance,
    )
    return len(interrogative_starts) > 1


def target_slot_matches_question(utterance: str, target: dict[str, Any]) -> bool:
    attribute = canonical_attribute(target.get("attribute"))
    slots = question_slot_keys(utterance)
    return not attribute or not slots or slots == {attribute}


def validate_teacher_candidates(response: dict[str, Any], row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(response) != {"questions", "skip_reason_code", "skip_reason_detail"}:
        return ["teacher response fields must be questions, skip_reason_code, skip_reason_detail"]
    questions = response.get("questions")
    if not isinstance(questions, list):
        return ["questions must be a list"]
    code = response.get("skip_reason_code")
    detail = response.get("skip_reason_detail")
    if not isinstance(detail, str):
        errors.append("skip_reason_detail must be a string")
    if questions:
        if code is not None:
            errors.append("skip_reason_code must be null when questions are present")
    elif code not in TEACHER_SKIP_REASON_CODES:
        errors.append("empty questions require a valid teacher skip_reason_code")

    doctor_messages = [str(message or "") for message in row.get("original_doctor_block") or []]
    state = row.get("patient_state") or {}
    for index, candidate in enumerate(questions):
        prefix = f"questions[{index}]"
        if not isinstance(candidate, dict) or set(candidate) != {
            "source_text",
            "next_question_target",
            "utterance",
        }:
            errors.append(f"{prefix} has invalid fields")
            continue
        source_text = str(candidate.get("source_text") or "").strip()
        target = candidate.get("next_question_target")
        utterance = str(candidate.get("utterance") or "").strip()
        if not source_text or not any(source_text in message for message in doctor_messages):
            errors.append(f"{prefix}.source_text is not a continuous span of original_doctor_block")
        if not isinstance(target, dict) or set(target) != {"name", "type", "attribute"}:
            errors.append(f"{prefix}.next_question_target is invalid")
            continue
        name = str(target.get("name") or "").strip()
        target_type = target.get("type")
        if not name or target_type not in QUESTION_TARGET_TYPES:
            errors.append(f"{prefix}.next_question_target has invalid name or type")
        if target.get("attribute") is not None and not isinstance(target.get("attribute"), str):
            errors.append(f"{prefix}.attribute must be string or null")
        generic_type = GENERIC_TARGET_TYPES.get(name)
        if generic_type is not None and target_type != generic_type:
            errors.append(f"{prefix} generic target type is invalid")
        elif generic_type is None:
            name_in_source = normalize_text(name) in normalize_text(source_text)
            resolved_attribute_target = (
                resolve_attribute_target(target, state, source_text + utterance)
                if target.get("attribute") is not None
                else None
            )
            if not name_in_source and target.get("attribute") is None:
                errors.append(f"{prefix} target name is not grounded in source_text or unique current entity")
            elif target.get("attribute") is not None and resolved_attribute_target is None:
                # Semantic ambiguity is handled as TARGET_NOT_RESOLVED during selection.
                pass
        decision = {
            "action": "ask",
            "next_question_target": target,
            "utterance": utterance,
        }
        decision_errors = validate_decision(
            decision,
            state={},
            asked_targets=[],
            allow_unsupported_attribute=True,
            allow_unresolved_attribute=True,
        )
        errors.extend(f"{prefix}: {error}" for error in decision_errors)
        if is_compound_question(utterance, target):
            errors.append(f"{prefix}: utterance asks more than one information slot")
        elif not target_slot_matches_question(utterance, target):
            errors.append(f"{prefix}: target attribute does not match the question slot")
    return errors


def make_unusable_decision(code: str, detail: str = "") -> dict[str, Any]:
    if code not in SKIP_REASON_CODES:
        raise ValueError(f"invalid skip reason code: {code}")
    return {
        "usable": False,
        "action": None,
        "next_question_target": None,
        "utterance": "",
        "skip_reason_code": code,
        "skip_reason_detail": detail,
    }


def select_teacher_question(
    response: dict[str, Any],
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    questions = response.get("questions") or []
    if not questions:
        return make_unusable_decision(
            str(response.get("skip_reason_code")),
            str(response.get("skip_reason_detail") or ""),
        ), None

    state = row.get("patient_state") or {}
    next_state = row.get("next_patient_state") or {}
    asked_keys = {
        key
        for item in row.get("asked_targets_before") or []
        if (key := target_key(item)) is not None
    }
    saw_known = False
    saw_asked = False
    saw_unsupported_attribute = False
    saw_unresolved_target = False
    saw_unrelated_same_type_delta = False
    for candidate in questions:
        target = canonicalize_target(candidate["next_question_target"])
        if target.get("attribute") is not None and canonical_attribute(target.get("attribute")) is None:
            saw_unsupported_attribute = True
            continue
        question_text = str(candidate.get("source_text") or "") + str(candidate.get("utterance") or "")
        if target.get("attribute") is not None:
            resolved_target = resolve_attribute_target(target, state, question_text)
            if resolved_target is None:
                saw_unresolved_target = True
                continue
            target = resolved_target
        key = target_key(target)
        if key is not None and key in asked_keys:
            saw_asked = True
            continue
        if target_is_known(target, state):
            saw_known = True
            continue
        method = answerability_match_method(
            target,
            state,
            next_state,
            next_patient_block=row.get("next_patient_block") or [],
            question_text=question_text,
        )
        if method is None:
            if target.get("attribute") is None and any(
                item.get("type") == target.get("type")
                for item in _new_state_items(state, next_state)
            ):
                saw_unrelated_same_type_delta = True
            continue
        selected = dict(candidate)
        selected["next_question_target"] = dict(target)
        selected["answerability_method"] = method
        return {
            "usable": True,
            "action": "ask",
            "next_question_target": dict(target),
            "utterance": str(candidate["utterance"]).strip(),
        }, selected

    if saw_asked:
        code = "TARGET_ALREADY_ASKED"
    elif saw_known:
        code = "TARGET_ALREADY_KNOWN"
    elif saw_unresolved_target:
        code = "TARGET_NOT_RESOLVED"
    elif saw_unsupported_attribute:
        code = "SOURCE_NOT_TRACEABLE"
    else:
        code = "ANSWER_NOT_OBSERVED"
    if saw_unsupported_attribute and code == "SOURCE_NOT_TRACEABLE":
        detail = "candidate attribute is not a supported canonical attribute"
    elif code == "TARGET_NOT_RESOLVED":
        detail = "attribute target cannot be uniquely resolved to a current-state entity"
    elif saw_unrelated_same_type_delta:
        detail = "unrelated same-type state delta rejected"
    else:
        detail = "no doctor-grounded candidate was newly answered"
    return make_unusable_decision(code, detail), None


def validate_teacher_answerability(decision: dict[str, Any], row: dict[str, Any]) -> list[str]:
    if not decision.get("usable", True) or decision.get("action") != "ask":
        return []
    target = decision.get("next_question_target")
    if not isinstance(target, dict):
        return []
    if not row.get("next_patient_block"):
        return ["ask sample requires a following patient answer"]
    selected = row.get("selected_teacher_candidate") or {}
    method = answerability_match_method(
        target,
        row.get("patient_state") or {},
        row.get("next_patient_state") or {},
        next_patient_block=row.get("next_patient_block") or [],
        question_text=str(selected.get("source_text") or "") + str(decision.get("utterance") or ""),
    )
    if method is None:
        return ["next patient state does not newly answer next_question_target"]
    return []


def validate_decision(
    decision: dict[str, Any],
    *,
    state: dict[str, Any],
    asked_targets: list[dict[str, Any]],
    teacher_mode: bool = False,
    is_terminal: bool = False,
    allow_unsupported_attribute: bool = False,
    allow_unresolved_attribute: bool = False,
) -> list[str]:
    errors: list[str] = []
    usable = decision.get("usable", True) if teacher_mode else True
    if teacher_mode and not isinstance(usable, bool):
        errors.append("usable must be boolean")
        return errors
    if teacher_mode and not usable:
        if decision.get("action") is not None or decision.get("next_question_target") is not None:
            errors.append("unusable decision must have null action and target")
        if decision.get("utterance") not in (None, ""):
            errors.append("unusable decision must have empty utterance")
        if decision.get("skip_reason_code") not in SKIP_REASON_CODES:
            errors.append("unusable decision requires a valid skip_reason_code")
        if not isinstance(decision.get("skip_reason_detail", ""), str):
            errors.append("skip_reason_detail must be a string")
        return errors

    action = decision.get("action")
    target = decision.get("next_question_target")
    utterance = str(decision.get("utterance") or "").strip()
    if action not in {"ask", "end"}:
        errors.append("action must be ask or end")
        return errors

    if action == "end":
        if teacher_mode and not is_terminal:
            errors.append("end is only valid for a terminal patient block")
        if target is not None:
            errors.append("end must have null next_question_target")
        if utterance:
            errors.append("end must have empty utterance")
        return errors

    if not isinstance(target, dict):
        errors.append("ask requires next_question_target object")
        return errors
    if set(target) != {"name", "type", "attribute"}:
        errors.append("next_question_target fields must be name, type, attribute")
    if not str(target.get("name") or "").strip():
        errors.append("target name is required")
    if target.get("type") not in QUESTION_TARGET_TYPES:
        errors.append("target type is invalid")
    if target.get("attribute") is not None and not isinstance(target.get("attribute"), str):
        errors.append("target attribute must be string or null")
    elif (
        target.get("attribute") is not None
        and canonical_attribute(target.get("attribute")) is None
        and not allow_unsupported_attribute
    ):
        errors.append("target attribute is not a supported canonical attribute")
    elif (
        target.get("attribute") is not None
        and not allow_unresolved_attribute
        and resolve_attribute_target(target, state, utterance) is None
    ):
        errors.append("attribute target is not a uniquely resolved current-state entity")
    if not utterance:
        errors.append("ask requires utterance")
    if utterance.count("？") + utterance.count("?") != 1:
        errors.append("utterance must contain exactly one question")
    if any(pattern in utterance for pattern in UNSAFE_PATTERNS):
        errors.append("utterance contains diagnosis, treatment, medicine, or examination advice")

    key = target_key(target)
    asked_keys = {item_key for item in asked_targets if (item_key := target_key(item)) is not None}
    if key is not None and key in asked_keys:
        errors.append("target has already been asked")
    if target_is_known(target, state):
        errors.append("target is already known in patient state")
    return errors


def build_interviewer_input(
    patient_block: list[str],
    patient_state: dict[str, Any],
    recent_context: list[dict[str, str]],
    asked_targets: list[dict[str, Any]],
) -> str:
    parts = [
        "患者当前表达：" + "\n".join(patient_block),
        "当前问诊状态：" + json.dumps(patient_state, ensure_ascii=False, separators=(",", ":")),
        "最近对话上下文：" + json.dumps(recent_context, ensure_ascii=False, separators=(",", ":")),
        "已问目标：" + json.dumps(asked_targets, ensure_ascii=False, separators=(",", ":")),
    ]
    return "\n".join(parts)


def build_teacher_prompt(row: dict[str, Any]) -> str:
    payload = {
        "patient_block": row["patient_block"],
        "patient_state": row["patient_state"],
        "recent_context": row["recent_context"],
        "asked_targets": row["asked_targets_before"],
        "original_doctor_block": row["original_doctor_block"],
    }
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n请按规则输出标注JSON。"
    )
