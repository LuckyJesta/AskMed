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

SYSTEM_PROMPT = """你是一个严谨的中文医学事实抽取器。你的唯一任务是把【患者当前发言】转成可验证的医学事实JSON。

核心边界：
1. 只能抽取患者当前发言中实际表达的新信息。
2. 上一轮医生问题、当前问诊状态、最近对话上下文只用于理解短回答、消解指代、判断否定对象、绑定属性目标。
3. 禁止把医生问题、当前问诊状态或历史上下文直接当作本轮患者事实。
4. 禁止诊断、扩写、推测、补充患者未说出的内容。
5. MedDG弱标签可能不完整或有误，只能作为合成阶段的弱参考，不能覆盖患者原文。

输出格式：
1. 必须只输出纯JSON，不要Markdown，不要解释；JSON必须使用紧凑格式，不要换行、缩进或多余空格。
2. 顶层格式必须是 {"facts":[...]}。
3. 无医学事实时输出 {"facts":[]}。
4. 每个fact必须包含字段：name, normalized_name, type, status, subject, time, body_part, attribute, evidence, standard_code, terminology。
5. type只能是 symptom/disease/medicine/examination/attribute/history/lifestyle/other。
6. status只能是 present/absent/uncertain。
7. subject只能是 patient/family/other/unknown。
8. attribute必须是JSON对象；没有补充属性时填 {}，不能填空字符串。
9. time和body_part没有信息时填null，不能填空字符串。
10. evidence必须是患者当前发言中的连续原文片段，不能省略中间文本，不能把多个不相邻片段拼接成一个evidence。
11. 第一版不做标准编码，standard_code和terminology必须为null。

抽取策略：
1. 如果患者当前发言同时包含症状本体和该症状的部位、性质、程度、频率、颜色、诱因、病程或单次发作持续时间等属性，应优先合并成同一个symptom/disease fact，不要再额外生成重复的attribute fact。
2. 只有当患者当前发言本身是依赖上下文的短回答或纯属性回答时，才单独输出type=attribute，例如“两三天了”“黄色的”“三四次”“饭后更疼”“没有”。
3. 当type为attribute时，normalized_name必须是属性名称而不是属性取值，例如“病程”“单次发作持续时间”“频率”“颜色”“程度”“诱因”“否定情况”；attribute中必须包含target和value。
4. attribute.target表示该属性补充的目标医学问题，优先从当前问诊状态problems中选择；其次参考上一轮医生问题；无法判断时填"unknown"。
5. attribute.value表示患者当前发言给出的属性取值，不要把取值写进normalized_name。
6. attribute内部字段优先使用这些标准key：duration, episode_duration, frequency, character, severity, trigger, aggravating_factor, relieving_factor, color, amount, stool_character, appetite, sleep, effect。不要随意创造同义字段；确实无法归类时才使用中文字段。
7. 如果一个fact需要多个不连续证据，不要拼接evidence；应拆成多个fact，或选择最能支持该fact的单个连续片段。
8. 区分“病程”和“单次发作持续时间”：“两三天了”“三天了”这类从发病到现在多久，用normalized_name="病程"；“疼一会儿就好”“每次几秒钟”“发作几分钟”这类一次发作持续多久，用normalized_name="单次发作持续时间"。
9. 否定短回答要抽取被否定的医学对象。例如医生问“有没有发烧？”患者答“没有”，应输出发烧 absent，evidence为“没有”；如果无法判断被否定对象，则输出 {"facts":[]}。
10. 当前发言只是问候、感谢、确认、寒暄或告别，例如“好的”“谢谢”“嗯嗯”“知道了”“再见”，且没有包含新的医学事实、用药计划、检查计划或症状变化，则输出 {"facts":[]}。

正例：
患者当前发言“肚脐周围隐隐作痛”：
{"facts":[{"name":"腹痛","normalized_name":"腹痛","type":"symptom","status":"present","subject":"patient","time":null,"body_part":"肚脐周围","attribute":{"character":"隐痛"},"evidence":"肚脐周围隐隐作痛","standard_code":null,"terminology":null}]}

患者当前发言“两三天了”，当前问诊状态中已有腹痛：
{"facts":[{"name":"病程","normalized_name":"病程","type":"attribute","status":"present","subject":"patient","time":null,"body_part":null,"attribute":{"target":"腹痛","value":"两三天"},"evidence":"两三天了","standard_code":null,"terminology":null}]}

患者当前发言“感觉被针扎了一下，几秒钟就好了”，当前问诊状态中已有腹痛：
{"facts":[{"name":"腹痛","normalized_name":"腹痛","type":"symptom","status":"present","subject":"patient","time":null,"body_part":null,"attribute":{"character":"针刺样痛","episode_duration":"几秒钟"},"evidence":"感觉被针扎了一下，几秒钟就好了","standard_code":null,"terminology":null}]}

患者当前发言“没有”，上一轮医生问“有没有发烧？”：
{"facts":[{"name":"发烧","normalized_name":"发热","type":"symptom","status":"absent","subject":"patient","time":null,"body_part":null,"attribute":{},"evidence":"没有","standard_code":null,"terminology":null}]}

反例：
1. 不要因为医生问“有没有腹泻”就输出腹泻，除非患者当前发言确认或否认。
2. 不要把“隐隐作痛”既写进腹痛fact，又单独输出一个“性质”attribute fact。
3. 不要把当前问诊状态中的既往腹痛再次作为本轮事实输出，除非患者当前发言重新表达了它。"""


TRAINING_SYSTEM_PROMPT = """你是一个严谨的中文医学事实抽取器。你的任务是把【患者当前发言】转成可验证的医学事实JSON。

核心边界：
1. 只能抽取患者当前发言中实际表达的新信息。
2. 上一轮医生问题、当前问诊状态、最近对话上下文只用于理解短回答、消解指代、判断否定对象、绑定属性目标。
3. 禁止把医生问题、当前问诊状态或历史上下文直接当作本轮患者事实。
4. 禁止诊断、扩写、推测、补充患者未说出的内容。

输出格式：
1. 必须只输出纯JSON，不要Markdown，不要解释；JSON必须使用紧凑格式，不要换行、缩进或多余空格。
2. 顶层格式必须是 {"facts":[...]}。
3. 无医学事实时输出 {"facts":[]}。
4. 每个fact必须包含字段：name, normalized_name, type, status, subject, time, body_part, attribute, evidence, standard_code, terminology。
5. type只能是 symptom/disease/medicine/examination/attribute/history/lifestyle/other。
6. status只能是 present/absent/uncertain。
7. subject只能是 patient/family/other/unknown。
8. attribute必须是JSON对象；没有补充属性时填 {}。
9. time和body_part没有信息时填null。
10. evidence必须是患者当前发言中的连续原文片段，不能省略中间文本，不能把多个不相邻片段拼接成一个evidence。
11. standard_code和terminology固定为null。

抽取策略：
1. 如果患者当前发言同时包含症状本体和该症状的部位、性质、程度、频率、颜色、诱因、病程或单次发作持续时间等属性，应优先合并成同一个symptom/disease fact，不要再额外生成重复的attribute fact。
2. 只有当患者当前发言本身是依赖上下文的短回答或纯属性回答时，才单独输出type=attribute，例如“两三天了”“黄色的”“三四次”“饭后更疼”。
3. 当type为attribute时，normalized_name必须是属性名称而不是属性取值，例如“病程”“单次发作持续时间”“频率”“颜色”“程度”“诱因”；attribute中必须包含target和value。
4. attribute.target优先从当前问诊状态problems中选择；其次参考上一轮医生问题；无法判断时填"unknown"。
5. attribute内部字段优先使用这些标准key：duration, episode_duration, frequency, character, severity, trigger, aggravating_factor, relieving_factor, color, amount, stool_character, appetite, sleep, effect。不要随意创造同义字段；确实无法归类时才使用中文字段。
6. 如果一个fact需要多个不连续证据，不要拼接evidence；应拆成多个fact，或选择最能支持该fact的单个连续片段。
7. 区分“病程”和“单次发作持续时间”：“两三天了”“三天了”表示从发病到现在多久；“疼一会儿就好”“每次几秒钟”“发作几分钟”表示一次发作持续多久。
8. 否定短回答要抽取被否定的医学对象；如果无法判断被否定对象，则输出 {"facts":[]}。
9. 当前发言只是问候、感谢、确认、寒暄或告别，且没有新的医学事实、用药计划、检查计划或症状变化，则输出 {"facts":[]}。"""


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
