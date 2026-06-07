from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.utils import ensure_parent, rank_scores, validate_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--out", default="results/embedding_baseline_scores.csv")
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates)
    validate_candidates(cand)
    scores = cand[["qid", "context_id", "embedding_score"]].rename(columns={"embedding_score": "score"})
    scores["latency_ms"] = 0.0
    out = rank_scores(scores, "embedding_baseline")
    ensure_parent(args.out)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out}: {len(out)} rows")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
