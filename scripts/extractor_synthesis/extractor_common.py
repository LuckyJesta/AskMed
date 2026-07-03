from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


ASKMED_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE = Path(os.environ.get("ASKMED_SOURCE", ASKMED_ROOT / "data" / "MedDG_clean.jsonl"))
DEFAULT_WORK_DIR = ASKMED_ROOT / "data" / "synthetic_extractor"

MEDDG_FIELDS = ("symptom", "disease", "medicine", "examination", "attribute")
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
STATUSES = {"present", "absent", "uncertain"}
SUBJECTS = {"patient", "family", "other", "unknown"}


def resolve_from_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ASKMED_ROOT / path

CONTEXT_DEPENDENT_ANSWERS = {
    "没有",
    "无",
    "正常",
    "不清楚",
    "不知道",
    "有",
    "是的",
    "不是",
    "三天了",
    "两三天了",
    "黄色的",
    "白色的",
}

QUESTION_TARGET_KEYWORDS = (
    "多久",
    "多长时间",
    "几天",
    "发烧",
    "发热",
    "拉肚子",
    "腹泻",
    "大便",
    "小便",
    "痰",
    "颜色",
    "几次",
    "用药",
    "吃药",
    "检查",
    "过敏",
    "家族史",
    "诱因",
    "生冷",
    "刺激",
    "疼",
    "痛",
    "恶心",
    "呕吐",
    "胸闷",
    "胸痛",
    "气短",
    "咳嗽",
    "咳痰",
)

NON_MEDICAL_CHATTER = {
    "好",
    "好的",
    "好吧",
    "行",
    "可以",
    "嗯",
    "嗯嗯",
    "恩",
    "恩恩",
    "哦",
    "哦哦",
    "噢",
    "知道了",
    "明白了",
    "了解了",
    "谢谢",
    "谢谢医生",
    "感谢",
    "感谢医生",
    "不客气",
    "再见",
    "拜拜",
    "麻烦了",
    "辛苦了",
}

MEDICAL_SIGNAL_KEYWORDS = (
    "药",
    "吃",
    "服",
    "用",
    "检查",
    "复查",
    "化验",
    "验",
    "做",
    "去",
    "明天",
    "今天",
    "昨天",
    "现在",
    "刚才",
    "又",
    "还",
    "发烧",
    "发热",
    "疼",
    "痛",
    "咳",
    "痰",
    "拉",
    "吐",
    "恶心",
    "腹",
    "头晕",
    "胸",
    "血",
    "尿",
    "便",
    "过敏",
)

SYSTEM_PROMPT = """你是一个严谨的中文医学事实抽取器。你的任务是把【患者当前发言】转成可验证的医学事实JSON。

核心边界：
1. 只抽取患者当前发言中明确表达存在或不存在的医学事实。
2. 上一轮医生问题、当前问诊状态、最近对话上下文只用于理解短回答、消解指代、判断否定对象、绑定属性目标，不能作为事实来源或evidence。
3. 禁止诊断、扩写、推测、补充患者未说出的内容。
4. MedDG弱标签可能不完整或有误，只能作为合成阶段的弱参考，不能覆盖患者原文。

事实定义：
1. 患者明确表达有、存在、发生、做过、查过、用过、正在用，status=present。
2. 患者明确表达没有、不存在、未发生、没做过、没查过、没用过，status=absent。
3. 不输出uncertain。患者咨询、猜测、担忧、请求建议、诊断咨询、用药咨询、检查咨询都不作为facts。
4. 一句话同时包含明确事实和咨询时，只抽取明确事实部分。

输出格式：
1. 只输出紧凑JSON，不要Markdown或解释；顶层格式必须是 {"facts":[...]}。
2. 无医学事实时输出 {"facts":[]}。
3. 每个fact必须包含字段：name, normalized_name, type, status, subject, time, body_part, attribute, evidence, standard_code, terminology。
4. type只能是 symptom/disease/medicine/examination/attribute/history/lifestyle/other。
5. status只能是 present/absent。
6. subject只能是 patient/family/other/unknown。
7. attribute必须是JSON对象；没有补充属性时填 {}。time和body_part没有信息时填null。
8. evidence必须是患者当前发言中的连续原文片段，不能拼接不相邻片段。
9. standard_code和terminology必须为null。

抽取策略：
1. name尽量保留患者当前发言中的医学表达；normalized_name使用规范表达，不能确定时与name一致。不要把症状推断成疾病。
2. 症状本体和部位、性质、程度、频率、颜色、诱因、病程等属性同时出现时，优先合并到同一个symptom/disease fact，不要重复输出attribute fact。
3. 只有当前发言是依赖上下文的短回答或纯属性回答时，才单独输出type=attribute，例如“三天了”“五六次”“黄色的”“饭后更疼”。attribute必须包含target和value；无法判断target时输出 {"facts":[]}。
4. 否定短回答要在能确定被否定对象时抽取absent；无法判断对象时输出 {"facts":[]}。
5. 单独出现解剖部位不是fact，不要输出type=body_part。

正反例：
患者“我有胃炎” -> disease present。
患者“我没有反流” -> symptom absent。
患者“我吃了奥美拉唑” -> medicine present。
患者“我做过胃镜” -> examination present。
患者“我没做过肠镜” -> examination absent。
患者“能不能吃奥美拉唑？” -> {"facts":[]}。
患者“还要做肠镜吗？” -> {"facts":[]}。
患者“是不是胃炎？” -> {"facts":[]}。
患者“胃疼三天了，能吃奥美拉唑吗？” -> 只抽取胃疼present和病程三天，不抽取奥美拉唑。
医生问“有发热吗？”，患者答“没有” -> 发热absent，evidence为“没有”。
医生问“疼多久了？”，患者答“三天了”，且状态中已有腹痛 -> 输出病程attribute，target=腹痛，value=三天。"""


TRAINING_SYSTEM_PROMPT = """你是一个严谨的中文医学事实抽取器。只根据【患者当前发言】抽取明确存在或明确不存在的医学事实，并输出JSON。

规则：
1. 只抽取患者当前发言中明确表达有、没有、发生、未发生、做过、没做过、用过、没用过的事实。
2. status只能为present或absent，不输出uncertain。
3. 患者咨询、猜测、担忧、请求建议、诊断咨询、用药咨询、检查咨询不作为facts；一句话同时包含明确事实和咨询时，只抽明确事实。
4. 上一轮医生问题、当前问诊状态和最近对话上下文只能用于理解短回答、判断否定对象和绑定attribute.target，不能作为事实来源或evidence。
5. evidence必须来自患者当前发言中的连续原文片段；无事实输出 {"facts":[]}。
6. 输出顶层格式为 {"facts":[...]}；每个fact包含name, normalized_name, type, status, subject, time, body_part, attribute, evidence, standard_code, terminology。
7. type只能是symptom/disease/medicine/examination/attribute/history/lifestyle/other；subject只能是patient/family/other/unknown。
8. attribute必须是JSON对象；time和body_part无信息时为null；standard_code和terminology固定为null。
9. 短回答如“三天了”“五六次”“没有”只有能确定目标时才抽取；无法判断目标时输出 {"facts":[]}。"""


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def has_weak_labels(message: dict[str, Any]) -> bool:
    return any(message.get(field) for field in MEDDG_FIELDS)


def weak_labels_from(message: dict[str, Any]) -> dict[str, list[str]]:
    return {field: list(message.get(field) or []) for field in MEDDG_FIELDS}


def nearest_previous_assistant(messages: list[dict[str, Any]], index: int) -> str | None:
    for prev in reversed(messages[:index]):
        if prev.get("role") == "assistant":
            content = str(prev.get("content") or "").strip()
            return content or None
    return None


def is_question_targeted(question: str | None) -> bool:
    if not question:
        return False
    return any(keyword in question for keyword in QUESTION_TARGET_KEYWORDS)


def is_short_context_answer(text: str) -> bool:
    stripped = re.sub(r"[。！？!?，,\s]", "", text or "")
    if stripped in CONTEXT_DEPENDENT_ANSWERS:
        return True
    return len(stripped) <= 4 and bool(stripped)


def normalize_chatter_text(text: str) -> str:
    return re.sub(r"[。！？!?，,、~～\.\s]", "", text or "")


def is_pure_non_medical_chatter(text: str) -> bool:
    normalized = normalize_chatter_text(text)
    if not normalized:
        return False
    if normalized in NON_MEDICAL_CHATTER:
        return True
    if any(keyword in normalized for keyword in MEDICAL_SIGNAL_KEYWORDS):
        return False
    remainder = normalized
    for phrase in sorted(NON_MEDICAL_CHATTER, key=len, reverse=True):
        remainder = remainder.replace(phrase, "")
    return remainder == ""


def recent_context(
    messages: list[dict[str, Any]],
    index: int,
    max_messages: int = 8,
    max_chars: int = 1500,
) -> list[dict[str, str]]:
    context: list[dict[str, str]] = []
    total_chars = 0
    for message in reversed(messages[:index]):
        role = str(message.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        item = {"role": role, "content": content}
        next_total = total_chars + len(role) + len(content)
        if context and (len(context) >= max_messages or next_total > max_chars):
            break
        context.append(item)
        total_chars = next_total
        if len(context) >= max_messages or total_chars >= max_chars:
            break
    context.reverse()
    return context


def sample_id(row: dict[str, Any]) -> str:
    return f"{row.get('dialogue_id')}#{row.get('turn_id')}"


def format_context(context: list[dict[str, str]]) -> str:
    if not context:
        return "无"
    role_name = {"user": "患者", "assistant": "医生"}
    return "\n".join(f"{role_name.get(item['role'], item['role'])}：{item['content']}" for item in context)


def format_alpaca_input(row: dict[str, Any]) -> str:
    previous = row.get("previous_doctor_question") or "无"
    utterance = row.get("patient_utterance") or ""
    patient_state = row.get("patient_state_before_turn")
    context = format_context(row.get("recent_context") or [])
    state_text = json.dumps(patient_state, ensure_ascii=False, separators=(",", ":")) if patient_state else "无"
    return (
        f"上一轮医生问题：{previous}\n"
        f"患者当前发言：{utterance}\n"
        f"当前问诊状态：{state_text}\n"
        f"最近对话上下文：\n{context}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.I)
        content = re.sub(r"\s*```$", "", content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON value must be an object")
    return parsed


def normalize_extraction(extraction: dict[str, Any]) -> dict[str, Any]:
    """Normalize common model formatting slips without changing fact semantics."""
    facts = extraction.get("facts")
    if not isinstance(facts, list):
        return extraction
    normalized_facts: list[Any] = []
    for fact in facts:
        if not isinstance(fact, dict):
            normalized_facts.append(fact)
            continue
        item = dict(fact)
        if item.get("attribute") in ("", None):
            item["attribute"] = {}
        if item.get("time") == "":
            item["time"] = None
        if item.get("body_part") == "":
            item["body_part"] = None
        if item.get("standard_code") == "":
            item["standard_code"] = None
        if item.get("terminology") == "":
            item["terminology"] = None
        normalized_facts.append(item)
    return {"facts": normalized_facts}


def validate_extraction(
    extraction: dict[str, Any],
    patient_utterance: str,
    strict_standard_null: bool = True,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    facts = extraction.get("facts")
    if not isinstance(facts, list):
        return False, ["facts must be a list"]

    required = {
        "name",
        "normalized_name",
        "type",
        "status",
        "subject",
        "time",
        "body_part",
        "attribute",
        "evidence",
        "standard_code",
        "terminology",
    }
    for idx, fact in enumerate(facts):
        prefix = f"facts[{idx}]"
        if not isinstance(fact, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = sorted(required - set(fact))
        if missing:
            errors.append(f"{prefix} missing fields: {','.join(missing)}")
        if fact.get("type") not in FACT_TYPES:
            errors.append(f"{prefix}.type invalid: {fact.get('type')!r}")
        if fact.get("status") not in STATUSES:
            errors.append(f"{prefix}.status invalid: {fact.get('status')!r}")
        if fact.get("subject") not in SUBJECTS:
            errors.append(f"{prefix}.subject invalid: {fact.get('subject')!r}")
        if "attribute" in fact and not isinstance(fact.get("attribute"), dict):
            errors.append(f"{prefix}.attribute must be an object")
        evidence = fact.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(f"{prefix}.evidence must be a non-empty string")
        elif evidence not in patient_utterance:
            errors.append(f"{prefix}.evidence not found in patient_utterance: {evidence!r}")
        if strict_standard_null and fact.get("standard_code") is not None:
            errors.append(f"{prefix}.standard_code must be null in v1")
        if strict_standard_null and fact.get("terminology") is not None:
            errors.append(f"{prefix}.terminology must be null in v1")
    return not errors, errors
