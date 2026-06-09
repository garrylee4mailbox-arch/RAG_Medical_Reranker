from __future__ import annotations

import argparse

import pandas as pd

from src.utils import ensure_parent, rank_scores, validate_candidates


METHOD = "embedding_department_bonus"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding baseline with a same-department score bonus.")
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--out", default="results/embedding_department_bonus_scores.csv")
    parser.add_argument("--bonus", type=float, default=0.05)
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    validate_candidates(candidates)
    if "question_department" not in candidates.columns:
        raise ValueError("candidates CSV missing required column for department bonus: question_department")

    out = candidates[["qid", "context_id", "embedding_score", "question_department", "department"]].copy()
    base = pd.to_numeric(out["embedding_score"], errors="coerce").fillna(0.0)
    matches = out["question_department"].astype(str) == out["department"].astype(str)

    scores = out[["qid", "context_id"]].copy()
    scores["score"] = base + matches.astype(float) * float(args.bonus)
    scores["latency_ms"] = 0.0     # 因为embedding已经算好并存储到candidates(_hard).csv里了，所以这里的推理时间近似为0
    scores = rank_scores(scores, METHOD)

    ensure_parent(args.out)
    scores.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out}: {len(scores)} rows")
    print(scores.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
