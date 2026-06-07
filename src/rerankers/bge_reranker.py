from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import ensure_parent, rank_scores, validate_candidates

METHOD = "bge"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def load_model(use_fp16: bool = True):
    try:
        from FlagEmbedding import FlagReranker
        return "flag", FlagReranker(MODEL_NAME, use_fp16=use_fp16)
    except Exception as flag_exc:
        print(f"[WARN] FlagEmbedding failed, fallback to sentence-transformers CrossEncoder. Reason: {flag_exc}")
        from sentence_transformers import CrossEncoder
        return "cross_encoder", CrossEncoder(MODEL_NAME, trust_remote_code=True)


def score_pairs(model_type: str, model, pairs: list[list[str]], batch_size: int, max_length: int) -> list[float]:
    if model_type == "flag":
        scores = []
        for i in tqdm(range(0, len(pairs), batch_size), desc="BGE scoring"):
            batch = pairs[i:i + batch_size]
            s = model.compute_score(batch, batch_size=batch_size, max_length=max_length, normalize=True)
            if isinstance(s, float):
                s = [s]
            scores.extend([float(x) for x in s])
        return scores
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return [float(x) for x in scores]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--out", default="results/bge_scores.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--sample", type=int, default=None, help="Debug only: score first N rows")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates)
    validate_candidates(cand)
    if args.sample:
        cand = cand.head(args.sample).copy()

    pairs = cand[["question", "candidate_context"]].astype(str).values.tolist()
    model_type, model = load_model(use_fp16=not args.no_fp16)
    t0 = time.perf_counter()
    scores = score_pairs(model_type, model, pairs, args.batch_size, args.max_length)
    total_ms = (time.perf_counter() - t0) * 1000
    per_pair_ms = total_ms / max(len(scores), 1)

    raw = cand[["qid", "context_id"]].copy()
    raw["score"] = scores
    raw["latency_ms"] = per_pair_ms
    out = rank_scores(raw, METHOD)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out}: {len(out)} rows; avg latency={per_pair_ms:.2f} ms/pair")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
