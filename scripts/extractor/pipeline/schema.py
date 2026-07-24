from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


ASKMED_ROOT = Path(__file__).resolve().parents[3]

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
STATUSES = {"present", "absent"}
SUBJECTS = {"patient", "family", "other"}

CORE_FACT_FIELD_ORDER = (
    "name",
    "normalized_name",
    "type",
    "status",
    "subject",
    "evidence",
    "standard_code",
    "terminology",
    "attribute",
)
CORE_FACT_FIELDS = set(CORE_FACT_FIELD_ORDER)
FORBIDDEN_ATTRIBUTE_KEYS = CORE_FACT_FIELDS - {"attribute"}
UNKNOWN_ATTRIBUTE_TARGETS = {"", "unknown", "未知", "不明确", "当前主要症状"}
RECOMMENDED_ATTRIBUTE_KEYS = (
    "time",
    "body_part",
    "duration",
    "episode_duration",
    "frequency",
    "character",
    "severity",
    "trigger",
    "aggravating_factor",
    "relieving_factor",
    "color",
    "amount",
    "appetite",
    "sleep",
    "effect",
    "dose",
    "result",
    "route",
    "side_effect",
    "target",
)
ATTRIBUTE_KEY_ALIASES = {
    "stool_character": "character",
    "dosage": "dose",
    "degree": "severity",
    "onset_time": "time",
    "start_time": "time",
    "onset_timing": "time",
    "occurrence_time": "time",
    "discovery_time": "time",
    "result_value": "result",
    "quantity": "amount",
    "count": "frequency",
    "times": "frequency",
    "number": "frequency",
}


def resolve_from_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ASKMED_ROOT / path


def clean_extraction_for_training(extraction: dict[str, Any]) -> dict[str, Any]:
    """Project facts to the v3 fixed-core, semi-open-attribute training schema."""
    facts = extraction.get("facts")
    if not isinstance(facts, list):
        return extraction

    cleaned_facts: list[Any] = []
    for fact in facts:
        if not isinstance(fact, dict):
            cleaned_facts.append(fact)
            continue
        if set(fact) == CORE_FACT_FIELDS and isinstance(fact.get("attribute"), dict):
            item = deepcopy(fact)
            item["standard_code"] = None
            item["terminology"] = None
            cleaned_facts.append(item)
        else:
            cleaned_facts.append(project_fact_to_core_attribute_schema(fact))
    return {"facts": cleaned_facts}


def project_fact_to_core_attribute_schema(fact: dict[str, Any]) -> dict[str, Any]:
    item = deepcopy(fact)
    if item.get("subject") == "unknown":
        item["subject"] = "other"
    attributes = clean_fact_attributes(item)
    attributes = rewrite_legacy_value_attribute(item, attributes)
    for legacy_key in ("time", "body_part"):
        legacy_value = item.get(legacy_key)
        if legacy_value not in (None, "") and legacy_key not in attributes:
            attributes[legacy_key] = legacy_value
    projected = {field: item.get(field) for field in CORE_FACT_FIELD_ORDER if field != "attribute"}
    projected["standard_code"] = None
    projected["terminology"] = None
    projected["attribute"] = attributes
    return projected


def rewrite_legacy_value_attribute(fact: dict[str, Any], attributes: dict[str, Any]) -> dict[str, Any]:
    if "value" not in attributes:
        return attributes
    value = attributes.pop("value")
    text = " ".join(
        str(part or "")
        for part in (
            fact.get("name"),
            fact.get("normalized_name"),
            fact.get("type"),
            fact.get("evidence"),
            attributes.get("target"),
        )
    )
    if any(keyword in text for keyword in ("病程", "持续", "多久", "时长", "几天", "多长时间")):
        key = "duration"
    elif any(keyword in text for keyword in ("开始", "发病时间", "昨", "今", "前", "后", "时间")):
        key = "time"
    elif any(keyword in text for keyword in ("次数", "频率", "几次", "一天", "每", "间隔", "次/")):
        key = "frequency"
    elif any(keyword in text for keyword in ("颜色", "色")):
        key = "color"
    elif any(keyword in text for keyword in ("性状", "形状", "干", "稀", "水样", "香蕉形")):
        key = "character"
    elif any(keyword in text for keyword in ("部位", "位置", "上腹", "下腹", "右", "左", "肚脐")):
        key = "body_part"
    elif any(keyword in text for keyword in ("程度", "严重", "轻", "重", "剧烈")):
        key = "severity"
    elif any(keyword in text for keyword in ("结果", "谷丙", "谷草", "转氨酶", "指标", "阳性", "阴性")):
        key = "result"
    elif any(keyword in text for keyword in ("体温", "温度", "发热", "发烧", "低烧")):
        key = "temperature"
    elif any(keyword in text for keyword in ("剂量", "用量")):
        key = "dose"
    elif any(keyword in text for keyword in ("缓解", "效果", "改善")):
        key = "effect"
    elif any(keyword in text for keyword in ("食欲", "胃口")):
        key = "appetite"
    elif any(keyword in text for keyword in ("睡眠", "入睡", "醒")):
        key = "sleep"
    elif any(keyword in text for keyword in ("量", "多", "少", "喝水", "饮食", "体重", "身高", "年龄")):
        key = "amount"
    else:
        key = "description"
    if key in attributes and attributes[key] != value:
        if isinstance(attributes[key], list):
            if value not in attributes[key]:
                attributes[key].append(value)
        else:
            attributes[key] = [attributes[key], value]
    else:
        attributes[key] = value
    return attributes


def clean_fact_attributes(fact: dict[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    raw_attribute = fact.get("attribute")
    if isinstance(raw_attribute, dict):
        attributes.update(raw_attribute)
    legacy_attributes = fact.get("attributes")
    if isinstance(legacy_attributes, dict):
        for key, value in legacy_attributes.items():
            attributes.setdefault(key, value)
    if not attributes:
        return {}

    cleaned: dict[str, Any] = {}
    for raw_key, value in attributes.items():
        if value in (None, ""):
            continue
        key = str(raw_key).strip()
        key = ATTRIBUTE_KEY_ALIASES.get(key, key)
        if not key or key in FORBIDDEN_ATTRIBUTE_KEYS:
            continue
        if key in cleaned and cleaned[key] != value:
            if isinstance(cleaned[key], list):
                if value not in cleaned[key]:
                    cleaned[key].append(value)
            else:
                cleaned[key] = [cleaned[key], value]
        else:
            cleaned[key] = value
    return cleaned

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
2. 上一轮医生问题、当前问诊状态、最近对话上下文可以用于理解短回答、消解医学对象和绑定属性目标，但不能单独建立患者事实，也不能作为evidence。
3. 禁止诊断、扩写、推测、补充患者未说出的内容。
4. MedDG弱标签可能不完整或有误，只能作为合成阶段的弱参考，不能覆盖患者原文。
5. 当前问诊状态和最近对话上下文中的既有事实不要重复输出，除非患者当前发言重新明确表达了该事实。
6. 如果当前发言只是在补充已有事实的性质、程度、时间、频率、部位、诱因、缓解或加重因素，应输出type=attribute，并将target写成当前问诊状态中的原名称，不要为“疼痛”“刺痛”等描述新建重复症状。

事实定义：
1. 患者明确表达有、存在、发生、做过、查过、用过、正在用，status=present。
2. 患者明确表达没有、不存在、未发生、没做过、没查过、没用过，status=absent。
3. 患者咨询、猜测、担忧、请求建议、诊断咨询、用药咨询、检查咨询都不作为facts。
4. 一句话同时包含明确事实和咨询时，只抽取明确事实部分。
5. “没有用药”“没有吃药”“没有检查”等没有具体医学对象的泛化否定不作为facts；“没吃奥美拉唑”“没做胃镜”等具有具体对象的否定仍应抽取。

输出格式：
1. 只输出紧凑JSON，不要Markdown或解释；顶层格式必须是 {"facts":[...]}。
2. 无医学事实时输出 {"facts":[]}。
3. 每个fact只能包含字段：name, normalized_name, type, status, subject, evidence, standard_code, terminology, attribute。
4. type只能是 symptom/disease/medicine/examination/attribute/history/lifestyle/other。
5. status只能是 present/absent。
6. subject只能是 patient/family/other。
7. attribute必须是JSON对象；没有补充属性时填 {}。
8. 时间、部位、剂量、结果、频率、程度、性质、诱因、缓解或加重因素等补充信息都写入attribute。
9. evidence必须是患者当前发言中的连续原文片段，不能拼接不相邻片段。
10. standard_code和terminology必须为null。
11. 输出前逐条自检：如果某个fact的evidence不是患者当前发言的连续子串，必须删除该fact，不要改写或借用上下文证据。

attribute半开放规则：
1. 优先使用这些推荐key：time, body_part, duration, episode_duration, frequency, character, severity, trigger, aggravating_factor, relieving_factor, color, amount, appetite, sleep, effect, dose, result, route, side_effect, target。
2. 如果事实属性确实无法用推荐key表达，可以创建清晰、简短、语义明确的新key。
3. 如果含义能被推荐key表达，必须使用推荐key，不要创建推荐key的同义key，例如用severity表示程度，不要用degree；用time表示发生时间，不要用onset_time。
4. 不要把core字段重复放入attribute：name, normalized_name, type, status, subject, evidence, standard_code, terminology。

抽取策略：
1. name直接填写当前问答所表达的简洁医学事实名称，允许对患者表达进行语义等价概括，也允许结合上一轮医生问题解析短回答，不要求是原文片段。normalized_name使用规范表达，不能确定时与name一致。不得诊断、推测或把症状改写成疾病。
2. 症状本体和属性同时出现时，优先合并到同一个symptom/disease fact，不要重复输出attribute fact。
3. 只有当前发言是依赖上下文的短回答或纯属性回答时，才单独输出type=attribute，例如“三天了”“五六次”“黄色的”“饭后更疼”。attribute必须包含target和对应属性；无法判断target时输出 {"facts":[]}。
4. 否定短回答要在能确定被否定对象时抽取absent；evidence必须使用患者当前发言中的否定词原文，如“没有”“没”“无”，不能复制医生问题；无法判断对象时输出 {"facts":[]}。
5. 单独出现解剖部位不是fact，不要输出type=body_part。
6. 已有腹痛后患者只说“针扎样，几秒钟就好了”，应输出腹痛属性attribute，target=腹痛，character=针扎样，episode_duration=几秒钟，不要新建疼痛或刺痛症状。

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
医生问“疼多久了？”，患者答“三天了”，且状态中已有腹痛 -> 输出病程attribute，target=腹痛，duration=三天。"""


TRAINING_SYSTEM_PROMPT = """你是一个严谨的中文医学事实抽取器。根据【上一轮医生问题】和【患者当前发言】抽取患者明确表达存在或不存在的医学事实，并输出JSON。

规则：
1. 只抽取患者当前发言中明确表达有、没有、发生、未发生、做过、没做过、用过、没用过的事实。
2. status只能使用present或absent。
3. 患者咨询、猜测、担忧、请求建议、诊断咨询、用药咨询、检查咨询不作为facts；一句话同时包含明确事实和咨询时，只抽明确事实。
4. 上一轮医生问题、当前问诊状态和最近对话上下文可以用于理解短回答、消解医学对象和绑定attribute.target，但不能单独建立患者事实，也不能作为evidence。name填写由当前问答直接支持的简洁医学事实名称，不要求是原文片段。
5. 当前问诊状态和最近对话上下文中的既有事实不要重复输出，除非患者当前发言重新明确表达了该事实。
5a. 当前发言只补充已有事实的性质、程度、时间、频率、部位、诱因、缓解或加重因素时，输出type=attribute并复用状态中的原名称作为target，不要新建重复症状。
6. evidence必须来自患者当前发言中的连续原文片段；输出前逐条自检，若evidence不是患者当前发言的连续子串，删除该fact；无事实输出 {"facts":[]}。
7. 输出顶层格式为 {"facts":[...]}；每个fact只能包含name, normalized_name, type, status, subject, evidence, standard_code, terminology, attribute。
8. type只能是symptom/disease/medicine/examination/attribute/history/lifestyle/other；subject只能是patient/family/other。
9. attribute必须是JSON对象；时间、部位、剂量、结果、频率、程度、性质等补充信息都写入attribute；standard_code和terminology固定为null。
10. attribute优先使用time, body_part, duration, episode_duration, frequency, character, severity, trigger, aggravating_factor, relieving_factor, color, amount, appetite, sleep, effect, dose, result, route, side_effect, target；确实无法匹配时可以创建清晰、简短、语义明确的新key。
11. 如果含义能被推荐key表达，必须使用推荐key，不要创建推荐key的同义key，例如用severity表示程度，不要用degree；用time表示发生时间，不要用onset_time。
12. 不要把name, normalized_name, type, status, subject, evidence, standard_code, terminology重复放入attribute。
13. 短回答如“三天了”“五六次”“没有”只有能确定目标时才抽取；否定短回答必须输出absent且evidence使用患者否定词原文，不能复制医生问题；无法判断目标时输出 {"facts":[]}。
14. “没有用药”“没有吃药”“没有检查”等没有具体对象的泛化否定不作为facts；具体到某药或某检查的否定仍应抽取。"""

NAME_EXTRACTION_RULE = """name提取规则：name直接填写当前问答明确支持的、简洁且语义完整的医学事实名称，不要求复制原文。例如“肚脐周围隐隐作痛”可填写name=腹痛，“大便时有血”可填写name=便血；医生问“最近腹泻吗”，患者答“有点”可填写name=腹泻。允许语义等价概括和上下文指代消解，但禁止根据模型常识、弱标签、既往状态或更早对话诊断、推测或补充本轮未确认的事实。对于type=attribute，name填写持续时间、频率、性质、程度、检查结果等属性类别，医学对象写入attribute.target，具体值写入对应属性key。evidence始终必须是患者当前发言中的连续原文片段。"""

ATTRIBUTE_TARGET_RULE = """硬性补充规则：如果输出type=attribute，attribute必须同时包含target和至少一个具体属性key，例如duration/frequency/severity/character/color/result等；target必须指向当前问诊状态或当前发言中可以确定的症状、疾病、检查或用药目标。无法确定target时不要输出该attribute fact，直接输出 {"facts":[]} 或仅输出其他可确定事实。"""

ATTRIBUTE_UNIQUE_TARGET_RULE = """attribute唯一目标规则：每个type=attribute的fact只能绑定一个明确target。目标按“患者当前发言中明确出现的对象、上一轮医生问题中的对象、当前问诊状态中最近且唯一相关的对象”依次选择；同一evidence和同一属性key不得重复绑定到不同target。存在多个合理目标且无法唯一确定时，不输出该attribute fact。"""

MEDICINE_USAGE_RULE = """药物事实边界：medicine fact只表示患者明确已经服用、正在使用、曾经使用，或明确没有服用、没有使用某药。仅表示家中有药、持有或购买了药，或者询问某药能否服用，不等于使用药物，不输出medicine fact。"""

CONTINUATION_ATTRIBUTE_RULE = """续述绑定规则：如果当前患者发言只是在补充当前问诊状态中已有事实的性质、程度、持续时间、频率、部位、诱因、缓解或加重因素，必须输出type=attribute，并把attribute.target写成状态中的原事实名称；不要把性质描述另建为疼痛、刺痛等重复症状。"""

GENERIC_NEGATIVE_RULE = """泛化否定规则：“没有用药”“没有吃药”“没有检查”等没有具体药物或检查对象的表达不输出facts；“没吃奥美拉唑”“没做胃镜”等有具体对象的否定仍应抽取。"""

SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n{NAME_EXTRACTION_RULE}"
TRAINING_SYSTEM_PROMPT = f"{TRAINING_SYSTEM_PROMPT}\n{NAME_EXTRACTION_RULE}"
SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n{ATTRIBUTE_TARGET_RULE}"
TRAINING_SYSTEM_PROMPT = f"{TRAINING_SYSTEM_PROMPT}\n{ATTRIBUTE_TARGET_RULE}"
SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n{ATTRIBUTE_UNIQUE_TARGET_RULE}"
TRAINING_SYSTEM_PROMPT = f"{TRAINING_SYSTEM_PROMPT}\n{ATTRIBUTE_UNIQUE_TARGET_RULE}"
SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n{MEDICINE_USAGE_RULE}"
TRAINING_SYSTEM_PROMPT = f"{TRAINING_SYSTEM_PROMPT}\n{MEDICINE_USAGE_RULE}"
SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n{CONTINUATION_ATTRIBUTE_RULE}\n{GENERIC_NEGATIVE_RULE}"
TRAINING_SYSTEM_PROMPT = f"{TRAINING_SYSTEM_PROMPT}\n{CONTINUATION_ATTRIBUTE_RULE}\n{GENERIC_NEGATIVE_RULE}"


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


def dialogue_user_turn_count(dialogue: dict[str, Any]) -> int:
    return sum(1 for message in dialogue.get("messages") or [] if message.get("role") == "user")


def select_source_dialogues(
    source: Path,
    max_dialogues: int | None = None,
    max_user_turns: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Select a stable source prefix while keeping the final dialogue intact."""
    selected: list[dict[str, Any]] = []
    selected_user_turns = 0
    for dialogue in read_jsonl(source):
        if max_dialogues is not None and len(selected) >= max_dialogues:
            break
        if max_user_turns is not None and selected_user_turns >= max_user_turns:
            break
        selected.append(dialogue)
        selected_user_turns += dialogue_user_turn_count(dialogue)
    return selected, selected_user_turns


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
    patient_state = model_visible_patient_state(row.get("patient_state_before_turn"))
    context = format_context(row.get("recent_context") or [])
    state_text = json.dumps(patient_state, ensure_ascii=False, separators=(",", ":")) if patient_state else "无"
    return (
        f"上一轮医生问题：{previous}\n"
        f"患者当前发言：{utterance}\n"
        f"当前问诊状态：{state_text}\n"
        f"最近对话上下文：\n{context}"
    )


def model_visible_patient_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    hidden_keys = {"dialogue_id", "turn_id", "uncertain_findings"}
    return {
        key: value
        for key, value in state.items()
        if key not in hidden_keys and value not in (None, [], {})
    }


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
        elif isinstance(item.get("attribute"), dict):
            item["attribute"] = {
                key: _normalize_attribute_value(value)
                for key, value in item["attribute"].items()
            }
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


def _normalize_attribute_value(value: Any) -> Any:
    """Flatten a scalar list while leaving structurally invalid values visible to validation."""
    if not isinstance(value, list):
        return value
    if any(isinstance(item, (dict, list)) for item in value):
        return value

    parts: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text and text not in parts:
            parts.append(text)
    return "；".join(parts)


def validate_extraction(
    extraction: dict[str, Any],
    patient_utterance: str,
    strict_standard_null: bool = True,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    facts = extraction.get("facts")
    if not isinstance(facts, list):
        return False, ["facts must be a list"]

    attribute_bindings: dict[tuple[str, str], tuple[str, int]] = {}
    for idx, fact in enumerate(facts):
        prefix = f"facts[{idx}]"
        if not isinstance(fact, dict):
            errors.append(f"{prefix} must be an object")
            continue
        field_names = set(fact)
        missing = sorted(CORE_FACT_FIELDS - field_names)
        if missing:
            errors.append(f"{prefix} missing fields: {','.join(missing)}")
        extra = sorted(field_names - CORE_FACT_FIELDS)
        if extra:
            errors.append(f"{prefix} has extra top-level fields: {','.join(extra)}")
        if fact.get("type") not in FACT_TYPES:
            errors.append(f"{prefix}.type invalid: {fact.get('type')!r}")
        if fact.get("status") not in STATUSES:
            errors.append(f"{prefix}.status invalid: {fact.get('status')!r}")
        if fact.get("subject") not in SUBJECTS:
            errors.append(f"{prefix}.subject invalid: {fact.get('subject')!r}")
        if "attribute" in fact and not isinstance(fact.get("attribute"), dict):
            errors.append(f"{prefix}.attribute must be an object")
        elif isinstance(fact.get("attribute"), dict):
            attributes = fact["attribute"]
            forbidden = sorted(set(attributes) & FORBIDDEN_ATTRIBUTE_KEYS)
            if forbidden:
                errors.append(f"{prefix}.attribute repeats core fields: {','.join(forbidden)}")
            if fact.get("type") == "attribute":
                target = attributes.get("target")
                if not isinstance(target, str) or target.strip() in UNKNOWN_ATTRIBUTE_TARGETS:
                    errors.append(f"{prefix}.attribute.target must be a known non-empty string for type=attribute")
                concrete_keys = [
                    key
                    for key, value in attributes.items()
                    if key != "target" and value not in (None, "", [], {})
                ]
                if not concrete_keys:
                    errors.append(f"{prefix}.attribute must include at least one concrete attribute key besides target")
                if isinstance(target, str) and target.strip() not in UNKNOWN_ATTRIBUTE_TARGETS:
                    evidence_key = str(fact.get("evidence") or "").strip()
                    normalized_target = target.strip()
                    for attribute_key in concrete_keys:
                        binding_key = (evidence_key, str(attribute_key))
                        previous = attribute_bindings.get(binding_key)
                        if previous is not None and previous[0] != normalized_target:
                            errors.append(
                                f"{prefix}.attribute.{attribute_key} binds evidence {evidence_key!r} to target "
                                f"{normalized_target!r}, conflicting with facts[{previous[1]}] target {previous[0]!r}"
                            )
                        else:
                            attribute_bindings[binding_key] = (normalized_target, idx)
        name = fact.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix}.name must be a non-empty string")
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
