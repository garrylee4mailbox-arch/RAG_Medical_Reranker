from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd

from src.utils import clean_text, infer_question_department

FEATURE_COLUMNS = [
    "embedding_score",
    "keyword_overlap",
    "department_match",
    "question_length",
    "context_length",
]


def chinese_char_bigrams(text: str) -> set[str]:
    text = clean_text(text)
    chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text)
    if len(chars) < 2:
        return set(chars)
    return {"".join(chars[i:i+2]) for i in range(len(chars)-1)}


def keyword_overlap(question: str, context: str) -> float:
    q = chinese_char_bigrams(question)
    c = chinese_char_bigrams(context)
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["embedding_score"] = pd.to_numeric(df["embedding_score"], errors="coerce").fillna(0.0)
    out["keyword_overlap"] = [keyword_overlap(q, c) for q, c in zip(df["question"], df["candidate_context"])]
    inferred = [infer_question_department(q) for q in df["question"]]
    out["department_match"] = [1.0 if a != "unknown" and a == b else 0.0 for a, b in zip(inferred, df["department"].astype(str))]
    out["question_length"] = df["question"].astype(str).str.len().clip(0, 500) / 500.0
    out["context_length"] = df["candidate_context"].astype(str).str.len().clip(0, 1500) / 1500.0
    return out[FEATURE_COLUMNS].astype(float)
