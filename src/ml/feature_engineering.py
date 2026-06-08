from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd

from src.utils import clean_text

FEATURE_COLUMNS = [
    "embedding_score",
    "keyword_overlap",
    "department_match",
    "question_length",
    "context_length",
    "length_ratio",
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
    out["department_match"] = [
        1.0 if q_dept == dept else 0.0
        for q_dept, dept in zip(df["question_department"].astype(str), df["department"].astype(str))
    ]
    question_lengths = df["question"].astype(str).str.len().astype(float)
    context_lengths = df["candidate_context"].astype(str).str.len().astype(float)
    out["question_length"] = question_lengths.clip(0, 500) / 500.0
    out["context_length"] = context_lengths.clip(0, 1500) / 1500.0
    ratios = np.divide(
        np.minimum(question_lengths, context_lengths),
        np.maximum(question_lengths, context_lengths),
        out=np.zeros(len(df), dtype=float),
        where=np.maximum(question_lengths, context_lengths) > 0,
    )
    out["length_ratio"] = ratios
    return out[FEATURE_COLUMNS].astype(float)
