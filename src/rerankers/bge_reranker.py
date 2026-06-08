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
    # 优先使用 FlagEmbedding 官方封装。这个封装对 BGE reranker 支持更直接，
    # 并且可以通过 use_fp16 在支持的 GPU 上减少显存占用、提升推理速度。
    try:
        from FlagEmbedding import FlagReranker
        return "flag", FlagReranker(MODEL_NAME, use_fp16=use_fp16)
    except Exception as flag_exc:
        # 如果本地没有装 FlagEmbedding，或者环境不兼容，就退回到
        # sentence-transformers 的 CrossEncoder，保证脚本仍然可以跑。
        print(f"[WARN] FlagEmbedding failed, fallback to sentence-transformers CrossEncoder. Reason: {flag_exc}")
        from sentence_transformers import CrossEncoder
        return "cross_encoder", CrossEncoder(MODEL_NAME, trust_remote_code=True)


def score_pairs(model_type: str, model, pairs: list[list[str]], batch_size: int, max_length: int) -> list[float]:
    """给每个 [问题, 候选文本] 二元组打相关性分数。

    pairs 的每一项都是 [question, candidate_context]。
    reranker 不是分别给 question 和 context 做向量，而是把二者作为一对输入模型，
    让模型判断“这个候选文本是否能回答这个问题”。返回值顺序和 pairs 完全一致。
    """
    if model_type == "flag":
        # FlagReranker 的 compute_score 可以直接接收一批文本对。
        # 这里手动切 batch，是为了控制一次送入模型的样本数量，避免显存/内存爆掉。
        scores = []
        for i in tqdm(range(0, len(pairs), batch_size), desc="BGE scoring"):
            batch = pairs[i:i + batch_size]
            # max_length 控制 tokenizer 后的最长 token 数；过长的文本会被截断。
            # normalize=True 会把原始 logits 转成更容易比较的归一化分数。
            s = model.compute_score(batch, batch_size=batch_size, max_length=max_length, normalize=True)
            # 当 batch 里只有 1 条数据时，部分版本会直接返回 float，而不是 list。
            # 统一成 list 后，下面的 extend 逻辑就不用区分单条/多条。
            if isinstance(s, float):
                s = [s]
            # 明确转成 Python float，避免 numpy/torch 标量写 CSV 时出现类型兼容问题。
            scores.extend([float(x) for x in s])
        return scores

    # fallback 的 CrossEncoder 接口更简单：predict 会自己按 batch_size 分批。
    # 这里没有传 max_length，因为 sentence-transformers 的 CrossEncoder 通常在模型
    # 或 tokenizer 配置里处理长度限制，接口也不一定接受同名参数。
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return [float(x) for x in scores]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates_hard.csv")
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

    # 组装 reranker 输入：每一行候选样本变成 [问题, 候选上下文]。
    # 后续 score_pairs() 返回的第 N 个分数，对应这里第 N 个 pair。
    pairs = cand[["question", "candidate_context"]].astype(str).values.tolist()
    model_type, model = load_model(use_fp16=not args.no_fp16)
    t0 = time.perf_counter()
    scores = score_pairs(model_type, model, pairs, args.batch_size, args.max_length)
    total_ms = (time.perf_counter() - t0) * 1000
    per_pair_ms = total_ms / max(len(scores), 1)

    # 只保留排序所需的标识列和模型分数；rank_scores 会按 qid 分组生成排名。
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
