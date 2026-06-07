from __future__ import annotations

import argparse
import time

import pandas as pd
from tqdm import tqdm

from src.utils import ensure_parent, rank_scores, validate_candidates

METHOD = "bce"
MODEL_NAME = "maidalun1020/bce-reranker-base_v1"
FALLBACK_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"


def load_model(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, trust_remote_code=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--out", default="results/bce_scores.csv")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample", type=int, default=None, help="Debug only: score first N rows")
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates)
    validate_candidates(cand)
    if args.sample:
        cand = cand.head(args.sample).copy()

    pairs = cand[["question", "candidate_context"]].astype(str).values.tolist()
    try:
        model = load_model(args.model)
    except Exception as exc:
        if args.model == MODEL_NAME:
            print(f"[WARN] BCE failed, fallback to {FALLBACK_MODEL_NAME}. Reason: {exc}")
            model = load_model(FALLBACK_MODEL_NAME)
        else:
            raise

    t0 = time.perf_counter()
    scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
    total_ms = (time.perf_counter() - t0) * 1000
    per_pair_ms = total_ms / max(len(scores), 1)

    raw = cand[["qid", "context_id"]].copy()
    raw["score"] = [float(x) for x in scores]
    raw["latency_ms"] = per_pair_ms
    out = rank_scores(raw, METHOD)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out}: {len(out)} rows; avg latency={per_pair_ms:.2f} ms/pair")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
