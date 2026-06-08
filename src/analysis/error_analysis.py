from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.utils import ensure_parent, validate_candidates

try:
    from src.utils import validate_scores
except ImportError:
    validate_scores = None


DEFAULT_SCORE_FILES = [
    "embedding_baseline=results_hard/embedding_baseline_scores.csv",
    "embedding_department_bonus=results_hard/embedding_department_bonus_scores.csv",
    "lightweight_ml=results_hard/ml_scores.csv",
    "bce=results_hard/bce_scores.csv",
    "bge=results_hard/bge_scores.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard benchmark error analysis: compare embedding baseline, BCE, and lightweight ML ranking behavior."
    )
    parser.add_argument("--candidates", default="data/processed/candidates_hard.csv")
    parser.add_argument(
        "--score-files",
        nargs="*",
        default=DEFAULT_SCORE_FILES,
        help="method=path pairs. Missing score files are skipped with a warning.",
    )
    parser.add_argument("--out-cases", default="results_hard/error_cases.csv")
    parser.add_argument("--out-summary", default="results_hard/error_summary.csv")
    parser.add_argument("--max-context-chars", type=int, default=500)
    return parser.parse_args()


def parse_score_files(items: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --score-files item {item!r}. Expected method=path.")
        method, path = item.split("=", 1)
        method = method.strip()
        path = path.strip()
        if not method or not path:
            raise ValueError(f"Invalid --score-files item {item!r}. Expected method=path.")
        out[method] = path
    return out


def find_context_column(df: pd.DataFrame) -> str:
    for col in ["candidate_context", "context", "context_text", "passage", "text", "document"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find context text column. Expected one of: candidate_context, context, context_text, passage, text, document."
    )


def make_positive_mask(df: pd.DataFrame) -> pd.Series:
    if "label" in df.columns:
        return pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int) == 1

    if "is_positive" in df.columns:
        return pd.to_numeric(df["is_positive"], errors="coerce").fillna(0).astype(int) == 1

    if "source_type" in df.columns:
        return df["source_type"].astype(str).str.lower().isin(
            {"positive", "gold", "answer", "relevant"}
        )

    raise ValueError(
        "Could not identify positive candidate. Expected label, is_positive, or source_type."
    )


def truncate(value, max_chars: int) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def load_candidates(path: str) -> Tuple[pd.DataFrame, str]:
    candidates = pd.read_csv(path)
    validate_candidates(candidates)

    required = {"qid", "context_id", "question"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidates file missing required columns: {sorted(missing)}")

    context_col = find_context_column(candidates)

    candidates = candidates.copy()
    candidates["qid"] = candidates["qid"].astype(str)
    candidates["context_id"] = candidates["context_id"].astype(str)
    candidates["_is_positive"] = make_positive_mask(candidates)

    pos_counts = candidates.groupby("qid")["_is_positive"].sum()
    bad = pos_counts[pos_counts != 1]
    if not bad.empty:
        raise ValueError(
            "Each qid must have exactly one positive row. "
            f"Invalid examples: {bad.head(10).to_dict()}"
        )

    return candidates, context_col


def load_scores(score_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    loaded: Dict[str, pd.DataFrame] = {}

    for method, path in score_files.items():
        p = Path(path)
        if not p.exists():
            print(f"[WARN] Missing score file for {method}: {path}; skipping.", file=sys.stderr)
            continue

        df = pd.read_csv(p)
        required = {"qid", "context_id", "score", "rank"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        if validate_scores is not None:
            try:
                validate_scores(df)
            except TypeError:
                try:
                    validate_scores(df, method)
                except TypeError:
                    print(
                        f"[WARN] validate_scores signature incompatible for {method}; continuing after local checks.",
                        file=sys.stderr,
                    )

        df = df[["qid", "context_id", "score", "rank"]].copy()
        df["qid"] = df["qid"].astype(str)
        df["context_id"] = df["context_id"].astype(str)
        df["score"] = pd.to_numeric(df["score"], errors="raise")
        df["rank"] = pd.to_numeric(df["rank"], errors="raise")
        loaded[method] = df

    if not loaded:
        raise FileNotFoundError("No score files available; cannot run error analysis.")

    return loaded


def build_method_stats(candidates: pd.DataFrame, scores_by_method: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = candidates[["qid", "context_id", "_is_positive"]].copy()
    rows: List[dict] = []

    for method, scores in scores_by_method.items():
        merged = key.merge(scores, on=["qid", "context_id"], how="left")

        missing = merged["rank"].isna().sum()
        if missing:
            print(f"[WARN] {method}: {missing} candidate rows missing rank after merge.", file=sys.stderr)

        pos = merged[merged["_is_positive"]].copy()
        for _, row in pos.iterrows():
            rows.append(
                {
                    "qid": row["qid"],
                    "method": method,
                    "positive_rank": row["rank"],
                    "positive_score": row["score"],
                }
            )

        ranked = merged.dropna(subset=["rank"]).copy()
        if not ranked.empty:
            top1 = (
                ranked.sort_values(["qid", "rank", "score"], ascending=[True, True, False])
                .groupby("qid", as_index=False)
                .first()
            )
            for _, row in top1.iterrows():
                rows.append(
                    {
                        "qid": row["qid"],
                        "method": method,
                        "top1_context_id": row["context_id"],
                        "top1_score": row["score"],
                    }
                )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("No qid-method stats created.")

    pos_rank = (
        long_df.dropna(subset=["positive_rank"])
        .drop_duplicates(["qid", "method"])
        .pivot(index="qid", columns="method", values="positive_rank")
    )
    pos_rank.columns = [f"{method}_positive_rank" for method in pos_rank.columns]

    pos_score = (
        long_df.dropna(subset=["positive_score"])
        .drop_duplicates(["qid", "method"])
        .pivot(index="qid", columns="method", values="positive_score")
    )
    pos_score.columns = [f"{method}_positive_score" for method in pos_score.columns]

    top1_id = (
        long_df.dropna(subset=["top1_context_id"])
        .drop_duplicates(["qid", "method"])
        .pivot(index="qid", columns="method", values="top1_context_id")
    )
    top1_id.columns = [f"{method}_top1_context_id" for method in top1_id.columns]

    top1_score = (
        long_df.dropna(subset=["top1_score"])
        .drop_duplicates(["qid", "method"])
        .pivot(index="qid", columns="method", values="top1_score")
    )
    top1_score.columns = [f"{method}_top1_score" for method in top1_score.columns]

    return (
        pos_rank.join(pos_score, how="outer")
        .join(top1_id, how="outer")
        .join(top1_score, how="outer")
        .reset_index()
    )


def rank_is(value, rank: int) -> bool:
    return pd.notna(value) and float(value) == float(rank)


def rank_le(value, k: int) -> bool:
    return pd.notna(value) and float(value) <= float(k)


def rank_gt1_or_missing(value) -> bool:
    return pd.isna(value) or float(value) > 1.0


def identify_case_types(row: pd.Series, methods: List[str]) -> List[str]:
    case_types: List[str] = []

    emb_rank = row.get("embedding_baseline_positive_rank")
    dept_rank = row.get("embedding_department_bonus_positive_rank")
    ml_rank = row.get("lightweight_ml_positive_rank")
    bce_rank = row.get("bce_positive_rank")

    if rank_gt1_or_missing(emb_rank) and rank_is(ml_rank, 1):
        case_types.append("embedding_wrong_ml_correct")

    if "bce" in methods and rank_gt1_or_missing(bce_rank) and rank_is(ml_rank, 1):
        case_types.append("bce_wrong_ml_correct")

    if "bce" in methods and rank_is(emb_rank, 1) and rank_gt1_or_missing(bce_rank):
        case_types.append("embedding_correct_bce_wrong")

    if "bce" in methods and rank_is(bce_rank, 1) and rank_gt1_or_missing(emb_rank):
        case_types.append("bce_correct_embedding_wrong")

    # Priority from teammate: BCE top-1 wrong, but top-k still improves / contains the positive.
    if "bce" in methods and rank_gt1_or_missing(bce_rank) and rank_le(bce_rank, 3):
        case_types.append("bce_top1_wrong_but_top3_hit")

    if "bce" in methods and rank_gt1_or_missing(bce_rank) and rank_le(bce_rank, 5):
        case_types.append("bce_top1_wrong_but_top5_hit")

    if (
        "bce" in methods
        and pd.notna(bce_rank)
        and pd.notna(emb_rank)
        and float(bce_rank) < float(emb_rank)
        and rank_gt1_or_missing(bce_rank)
    ):
        case_types.append("bce_improves_embedding_but_not_top1")

    if (
        "embedding_department_bonus" in methods
        and rank_gt1_or_missing(dept_rank)
        and rank_is(ml_rank, 1)
    ):
        case_types.append("department_bonus_wrong_ml_correct")

    rank_cols = [f"{method}_positive_rank" for method in methods if f"{method}_positive_rank" in row.index]
    if rank_cols and all(rank_gt1_or_missing(row[col]) for col in rank_cols):
        case_types.append("all_available_wrong")

    return case_types


def identify_cases(stats: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    rows: List[dict] = []

    for _, row in stats.iterrows():
        for case_type in identify_case_types(row, methods):
            record = row.to_dict()
            record["case_type"] = case_type
            rows.append(record)

    return pd.DataFrame(rows)


def attach_context_text(
    cases: pd.DataFrame,
    candidates: pd.DataFrame,
    context_col: str,
    methods: List[str],
    max_context_chars: int,
) -> pd.DataFrame:
    if cases.empty:
        return cases

    meta_cols = ["qid", "context_id", "question", context_col]
    if "source_type" in candidates.columns:
        meta_cols.append("source_type")

    meta = candidates[meta_cols + ["_is_positive"]].copy()

    positives = meta[meta["_is_positive"]].rename(
        columns={"context_id": "positive_context_id", context_col: "positive_context"}
    )
    cases = cases.merge(
        positives[["qid", "question", "positive_context_id", "positive_context"]],
        on="qid",
        how="left",
    )
    cases["positive_context"] = cases["positive_context"].map(lambda x: truncate(x, max_context_chars))

    for method in methods:
        top_id_col = f"{method}_top1_context_id"
        if top_id_col not in cases.columns:
            continue

        rename = {
            "context_id": top_id_col,
            context_col: f"{method}_top1_context",
        }
        if "source_type" in meta.columns:
            rename["source_type"] = f"{method}_top1_source_type"

        method_meta = meta.rename(columns=rename)
        keep = ["qid", top_id_col, f"{method}_top1_context"]
        if f"{method}_top1_source_type" in method_meta.columns:
            keep.append(f"{method}_top1_source_type")

        cases = cases.merge(method_meta[keep], on=["qid", top_id_col], how="left")
        cases[f"{method}_top1_context"] = cases[f"{method}_top1_context"].map(
            lambda x: truncate(x, max_context_chars)
        )

    front = ["qid", "case_type", "question", "positive_context_id", "positive_context"]
    rank_cols = [c for c in cases.columns if c.endswith("_positive_rank")]
    score_cols = [c for c in cases.columns if c.endswith("_positive_score")]
    top1_id_cols = [c for c in cases.columns if c.endswith("_top1_context_id")]
    top1_score_cols = [c for c in cases.columns if c.endswith("_top1_score")]
    top1_source_cols = [c for c in cases.columns if c.endswith("_top1_source_type")]
    top1_text_cols = [c for c in cases.columns if c.endswith("_top1_context")]

    ordered = front + rank_cols + score_cols + top1_id_cols + top1_score_cols + top1_source_cols + top1_text_cols
    rest = [c for c in cases.columns if c not in ordered]
    return cases[[c for c in ordered + rest if c in cases.columns]]


def build_summary(cases: pd.DataFrame, stats: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    n_qids = stats["qid"].nunique()
    rows: List[dict] = []

    if cases.empty:
        rows.append({"section": "case_type", "item": "no_cases_found", "count": 0, "percentage": 0.0})
    else:
        for case_type, count in cases["case_type"].value_counts().sort_index().items():
            rows.append(
                {
                    "section": "case_type",
                    "item": case_type,
                    "count": int(count),
                    "percentage": float(count / n_qids) if n_qids else 0.0,
                }
            )

    for method in methods:
        col = f"{method}_positive_rank"
        if col not in stats.columns:
            continue

        ranks = pd.to_numeric(stats[col], errors="coerce")
        buckets = {
            "rank_1": ranks == 1,
            "rank_2_to_3": (ranks >= 2) & (ranks <= 3),
            "rank_4_to_5": (ranks >= 4) & (ranks <= 5),
            "rank_6_plus_or_missing": (ranks >= 6) | ranks.isna(),
            "top3_hit": ranks <= 3,
            "top5_hit": ranks <= 5,
        }

        for bucket, mask in buckets.items():
            count = int(mask.sum())
            rows.append(
                {
                    "section": "method_rank",
                    "item": f"{method}_{bucket}",
                    "count": count,
                    "percentage": float(count / n_qids) if n_qids else 0.0,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    candidates, context_col = load_candidates(args.candidates)
    score_files = parse_score_files(args.score_files)
    scores_by_method = load_scores(score_files)
    methods = list(scores_by_method.keys())

    stats = build_method_stats(candidates, scores_by_method)
    cases = identify_cases(stats, methods)
    cases = attach_context_text(cases, candidates, context_col, methods, args.max_context_chars)
    summary = build_summary(cases, stats, methods)

    ensure_parent(args.out_cases)
    ensure_parent(args.out_summary)

    cases.to_csv(args.out_cases, index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    print(f"[OK] Available methods: {', '.join(methods)}")
    print(f"[OK] Wrote {args.out_cases}: {len(cases)} rows")
    print(f"[OK] Wrote {args.out_summary}: {len(summary)} rows")


if __name__ == "__main__":
    main()
