from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils import ensure_parent, validate_candidates, validate_scores

DEFAULT_SCORE_FILES = {
    "embedding_baseline": "results/embedding_baseline_scores.csv",
    "bge": "results/bge_scores.csv",
    "bce": "results/bce_scores.csv",
    "lightweight_ml": "results/ml_scores.csv",
}


def precision_at_k(group: pd.DataFrame, k: int) -> float:
    top = group.sort_values("rank").head(k)
    return 1.0 if top["label"].sum() > 0 else 0.0


def mrr(group: pd.DataFrame) -> float:
    g = group.sort_values("rank")
    rel = g[g["label"] == 1]
    if rel.empty:
        return 0.0
    return 1.0 / float(rel["rank"].min())


def ndcg_at_k(group: pd.DataFrame, k: int) -> float:
    g = group.sort_values("rank").head(k)
    gains = g["label"].astype(float).to_numpy()
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = sorted(group["label"].astype(float).tolist(), reverse=True)[:k]
    if not ideal or sum(ideal) == 0:
        return 0.0
    idcg = float(np.sum(np.asarray(ideal) * discounts[:len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_method(candidates: pd.DataFrame, scores: pd.DataFrame, method: str) -> Dict[str, float | str | int]:
    validate_scores(scores, method)
    merged = scores.merge(candidates[["qid", "context_id", "label"]], on=["qid", "context_id"], how="inner")
    if merged.empty:
        raise ValueError(f"No matching qid/context_id for method {method}")
    qids = sorted(merged["qid"].unique())
    per_q = []
    for qid, group in merged.groupby("qid"):
        per_q.append({
            "qid": qid,
            "p1": precision_at_k(group, 1),
            "p3": precision_at_k(group, 3),
            "p5": precision_at_k(group, 5),
            "mrr": mrr(group),
            "ndcg5": ndcg_at_k(group, 5),
        })
    qdf = pd.DataFrame(per_q)
    return {
        "method": method,
        "num_eval_qids": len(qids),
        "precision_at_1": qdf["p1"].mean(),
        "precision_at_3": qdf["p3"].mean(),
        "precision_at_5": qdf["p5"].mean(),
        "mrr": qdf["mrr"].mean(),
        "ndcg_at_5": qdf["ndcg5"].mean(),
        "avg_latency_ms": float(pd.to_numeric(scores["latency_ms"], errors="coerce").fillna(0.0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--out", default="results/summary_metrics.csv")
    parser.add_argument("--score-files", nargs="*", default=None, help="Optional list like method=path")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    validate_candidates(candidates)

    score_files = DEFAULT_SCORE_FILES.copy()
    if args.score_files:
        for item in args.score_files:
            method, path = item.split("=", 1)
            score_files[method] = path

    rows = []
    for method, path in score_files.items():
        p = Path(path)
        if not p.exists():
            print(f"[WARN] skip missing score file: {method} -> {path}")
            continue
        scores = pd.read_csv(p)
        rows.append(evaluate_method(candidates, scores, method))

    if not rows:
        raise RuntimeError("No score files found. Run at least embedding_baseline first.")
    out = pd.DataFrame(rows)
    metric_cols = [c for c in out.columns if c not in ["method", "num_eval_qids"]]
    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
