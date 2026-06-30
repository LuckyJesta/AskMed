from __future__ import annotations

import re
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_TYPES = {
    "disease": "ICD-10-CN",
    "examination": "LOINC",
}

GENERIC_EXAMINATION_KEYS = {
    "检查",
    "检验",
    "化验",
    "检测",
    "复查",
    "体检",
    "查体",
    "化验检查",
    "相关检查",
    "进一步检查",
}


def normalize_term_key(text: str | None) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[，,。；;：:、（）()【】\[\]{}<>《》\"'“”‘’]", "", value)
    return value


@dataclass(frozen=True)
class TerminologyMatch:
    semantic_type: str
    terminology: str | None
    standard_code: str | None
    preferred_name: str
    matched_alias: str


class TerminologyNormalizer:
    """Exact-match terminology normalizer for AskMed facts."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path).resolve() if db_path else None
        self._exact_index: dict[tuple[str, str, str], TerminologyMatch] | None = None
        self._supplemental_examination_index = _build_supplemental_examination_index()

    @property
    def enabled(self) -> bool:
        return self.db_path is not None and self.db_path.exists()

    def metadata(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("select key, value from metadata order by key").fetchall()
        return {"enabled": True, "db_path": str(self.db_path), **{row["key"]: row["value"] for row in rows}}

    def normalize_fact_for_runtime(self, fact: dict[str, Any]) -> dict[str, Any]:
        item = deepcopy(fact)
        match = self.match_fact(item)
        if match is None:
            if not item.get("normalized_name"):
                item["normalized_name"] = item.get("name")
            return item
        item["normalized_name"] = match.preferred_name
        item["standard_code"] = match.standard_code
        item["terminology"] = match.terminology
        return item

    def project_fact_for_training(self, fact: dict[str, Any]) -> dict[str, Any]:
        item = deepcopy(fact)
        item["standard_code"] = None
        item["terminology"] = None
        return item

    def normalize_extraction_for_runtime(self, extraction: dict[str, Any]) -> dict[str, Any]:
        facts = extraction.get("facts")
        if not isinstance(facts, list):
            return deepcopy(extraction)
        return {"facts": [self.normalize_fact_for_runtime(fact) if isinstance(fact, dict) else fact for fact in facts]}

    def project_extraction_for_training(self, extraction: dict[str, Any]) -> dict[str, Any]:
        facts = extraction.get("facts")
        if not isinstance(facts, list):
            return deepcopy(extraction)
        return {"facts": [self.project_fact_for_training(fact) if isinstance(fact, dict) else fact for fact in facts]}

    def match_fact(self, fact: dict[str, Any]) -> TerminologyMatch | None:
        semantic_type = str(fact.get("type") or "")
        expected_terminology = SUPPORTED_TYPES.get(semantic_type)
        if expected_terminology is None:
            return None

        if self.enabled:
            self._ensure_exact_index()

        names = [
            fact.get("name"),
            fact.get("normalized_name"),
        ]
        for raw_name in names:
            key = normalize_term_key(raw_name)
            if not key:
                continue
            if semantic_type == "examination":
                if key in GENERIC_EXAMINATION_KEYS:
                    continue
                supplemental = self._supplemental_examination_index.get(key)
                if supplemental is not None:
                    return supplemental
            if self.enabled:
                match = self._lookup_exact(semantic_type, expected_terminology, key)
                if match is not None:
                    return match
        return None

    def candidates(self, fact: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        semantic_type = str(fact.get("type") or "")
        expected_terminology = SUPPORTED_TYPES.get(semantic_type)
        if expected_terminology is None:
            return []
        query = normalize_term_key(fact.get("name") or fact.get("normalized_name"))
        if not query:
            return []
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                select terminology, semantic_type, code, preferred_name, alias
                from terms
                where semantic_type = ? and terminology = ?
                """,
                (semantic_type, expected_terminology),
            ).fetchall()
        scored: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            score = fuzz.ratio(query, normalize_term_key(row["alias"]))
            key = (row["terminology"], row["code"])
            if score < 75 or key in seen:
                continue
            seen.add(key)
            scored.append(
                {
                    "score": score,
                    "semantic_type": row["semantic_type"],
                    "terminology": row["terminology"],
                    "standard_code": row["code"],
                    "preferred_name": row["preferred_name"],
                    "matched_alias": row["alias"],
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def _lookup_exact(self, semantic_type: str, terminology: str, normalized_key: str) -> TerminologyMatch | None:
        if self._exact_index is not None:
            return self._exact_index.get((semantic_type, terminology, normalized_key))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                select terminology, semantic_type, code, preferred_name, alias
                from terms
                where semantic_type = ? and terminology = ? and normalized_key = ?
                order by preferred_name = alias desc, id asc
                limit 1
                """,
                (semantic_type, terminology, normalized_key),
            ).fetchone()
        if row is None:
            return None
        return TerminologyMatch(
            semantic_type=row["semantic_type"],
            terminology=row["terminology"],
            standard_code=row["code"],
            preferred_name=row["preferred_name"],
            matched_alias=row["alias"],
        )

    def _ensure_exact_index(self) -> None:
        if self._exact_index is not None or not self.enabled:
            return
        index: dict[tuple[str, str, str], TerminologyMatch] = {}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                select terminology, semantic_type, code, preferred_name, alias, normalized_key
                from terms
                where terminology in ('ICD-10-CN', 'LOINC')
                order by preferred_name = alias desc, id asc
                """
            )
            for row in rows:
                key = (row["semantic_type"], row["terminology"], row["normalized_key"])
                if key in index:
                    continue
                index[key] = TerminologyMatch(
                    semantic_type=row["semantic_type"],
                    terminology=row["terminology"],
                    standard_code=row["code"],
                    preferred_name=row["preferred_name"],
                    matched_alias=row["alias"],
                )
        self._exact_index = index


def _build_supplemental_examination_index() -> dict[str, TerminologyMatch]:
    rows = [
        {
            "aliases": ["肠镜", "结肠镜", "结肠镜检查"],
            "preferred_name": "结肠镜检查",
            "standard_code": "18746-8",
            "terminology": "LOINC",
        },
        {
            "aliases": ["胃镜", "胃镜检查", "上消化道内镜", "上消化道内镜检查"],
            "preferred_name": "胃镜检查",
            "standard_code": "28014-9",
            "terminology": "LOINC",
        },
        {
            "aliases": ["血常规", "血常规检查"],
            "preferred_name": "血常规检查",
            "standard_code": "58410-2",
            "terminology": "LOINC",
        },
        {
            "aliases": ["幽门螺杆菌", "幽门螺旋杆菌", "幽门螺杆菌检测", "幽门螺旋杆菌检测"],
            "preferred_name": "幽门螺杆菌检测",
            "standard_code": None,
            "terminology": None,
        },
    ]
    index: dict[str, TerminologyMatch] = {}
    for row in rows:
        for alias in row["aliases"]:
            key = normalize_term_key(alias)
            if not key:
                continue
            index[key] = TerminologyMatch(
                semantic_type="examination",
                terminology=row["terminology"],
                standard_code=row["standard_code"],
                preferred_name=row["preferred_name"],
                matched_alias=alias,
            )
    return index
