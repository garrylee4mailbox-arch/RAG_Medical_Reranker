from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

REQUIRED_CANDIDATE_COLUMNS = [
    "qid", "question", "context_id", "candidate_context", "label",
    "department", "source_type", "embedding_score",
]

REQUIRED_SCORE_COLUMNS = ["qid", "context_id", "score", "rank", "method", "latency_ms"]


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def clean_text(x: object) -> str:
    text = "" if pd.isna(x) else str(x)
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def read_csv_auto(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    last_err: Exception | None = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "gb2312"]:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read {path} with common encodings. Last error: {last_err}")


def normalize_department_from_path_or_name(value: str) -> str:
    v = clean_text(value).lower()
    if any(k in v for k in ["oncology", "肿瘤", "癌", "瘤"]):
        return "oncology"
    if any(k in v for k in ["pediatric", "pediatrics", "儿科", "小儿", "新生儿", "营养保健"]):
        return "pediatrics"
    return "unknown"


def infer_question_department(question: str) -> str:
    q = clean_text(question).lower()
    if any(k in q for k in ["癌", "肿瘤", "化疗", "放疗", "甲状腺", "肺癌", "胃癌", "乳腺"]):
        return "oncology"
    if any(k in q for k in ["小儿", "儿童", "宝宝", "婴儿", "孩子", "新生儿", "发烧", "腹泻"]):
        return "pediatrics"
    return "unknown"


def make_context(title: object, ask: object, answer: object, max_chars: int = 900) -> str:
    parts = []
    title_s, ask_s, ans_s = clean_text(title), clean_text(ask), clean_text(answer)
    if title_s:
        parts.append(f"标题：{title_s}")
    if ask_s:
        parts.append(f"患者问题：{ask_s}")
    if ans_s:
        parts.append(f"医生回答：{ans_s}")
    return clean_text("\n".join(parts))[:max_chars]


def rank_scores(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df.copy()
    out["score"] = out["score"].astype(float)
    out = out.sort_values(["qid", "score", "context_id"], ascending=[True, False, True]).copy()
    out["rank"] = out.groupby("qid").cumcount() + 1
    out["method"] = method
    return out[["qid", "context_id", "score", "rank", "method", "latency_ms"]]


def validate_candidates(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDIDATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"candidates.csv missing columns: {missing}")
    group_sizes = df.groupby("qid").size()
    if group_sizes.empty:
        raise ValueError("candidates.csv has no rows")
    positives = df.groupby("qid")["label"].sum()
    bad = positives[positives < 1]
    if len(bad):
        raise ValueError(f"Some qid groups have no positive label: {bad.index[:5].tolist()}")


def validate_scores(df: pd.DataFrame, method: str | None = None) -> None:
    missing = [c for c in REQUIRED_SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"score file missing columns: {missing}")
    if method is not None and set(df["method"].astype(str)) != {method}:
        raise ValueError(f"Expected method={method}, got {set(df['method'].astype(str))}")


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self.t0) * 1000
