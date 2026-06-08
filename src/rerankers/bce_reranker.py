from __future__ import annotations

import argparse
import time
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import ensure_parent, rank_scores, validate_candidates


METHOD = "bce"
DEFAULT_MODEL_NAME = "maidalun1020/bce-reranker-base_v1"
DEFAULT_CANDIDATES_PATH = "data/processed/candidates.csv"
DEFAULT_OUTPUT_PATH = "results/bce_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BCE reranker on shared candidates.csv.")

    parser.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATES_PATH,
        help="Input shared candidates CSV. Do not regenerate this file.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_PATH,
        help="Output score CSV path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name or local model path.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Debug only: score the first N rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for reranker inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length for query-context pair.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device. Use cuda on GPU server.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 inference on CUDA.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested by --device cuda, but torch.cuda.is_available() is False.")

    return torch.device(device_arg)


def load_bce_model(model_name: str, device: torch.device, fp16: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    if fp16:
        if device.type != "cuda":
            raise ValueError("--fp16 should only be used with --device cuda or auto-resolved CUDA.")
        model.half()

    return tokenizer, model


def iter_batches(items: list[tuple[str, str]], batch_size: int) -> Iterable[list[tuple[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def score_pairs(
    pairs: list[tuple[str, str]],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[float]:
    scores: list[float] = []

    for batch in tqdm(iter_batches(pairs, batch_size), total=(len(pairs) + batch_size - 1) // batch_size, desc="BCE scoring"):
        queries = [q for q, _ in batch]
        contexts = [c for _, c in batch]

        encoded = tokenizer(
            queries,
            contexts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        outputs = model(**encoded)
        logits = outputs.logits

        # For BCE reranker / cross-encoder style models:
        # - one-logit output: larger logit means more relevant
        # - two-logit output: use positive-class logit, larger means more relevant
        if logits.ndim == 2 and logits.shape[1] > 1:
            batch_scores = logits[:, -1]
        else:
            batch_scores = logits.view(-1)

        scores.extend(batch_scores.detach().float().cpu().tolist())

    return [float(score) for score in scores]


def main() -> None:
    args = parse_args()

    candidates = pd.read_csv(args.candidates)
    validate_candidates(candidates)

    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError("--sample must be a positive integer.")
        candidates = candidates.head(args.sample).copy()

    pairs = list(
        zip(
            candidates["question"].astype(str).tolist(),
            candidates["candidate_context"].astype(str).tolist(),
        )
    )

    device = resolve_device(args.device)
    print(f"[INFO] Loading BCE model: {args.model}")
    print(f"[INFO] Device: {device}; fp16={args.fp16}; batch_size={args.batch_size}; max_length={args.max_length}")

    tokenizer, model = load_bce_model(args.model, device=device, fp16=args.fp16)

    start = time.perf_counter()
    scores = score_pairs(
        pairs=pairs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    total_ms = (time.perf_counter() - start) * 1000.0
    latency_ms = total_ms / max(len(scores), 1)

    raw_scores = candidates[["qid", "context_id"]].copy()
    raw_scores["score"] = scores
    raw_scores["latency_ms"] = latency_ms

    # rank_scores sorts within each qid by score descending.
    # Therefore, larger BCE score means more relevant and gets a smaller rank number.
    output = rank_scores(raw_scores, METHOD)

    ensure_parent(args.out)
    output.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote {args.out}: {len(output)} rows")
    print(f"[OK] average latency: {latency_ms:.2f} ms/pair")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
