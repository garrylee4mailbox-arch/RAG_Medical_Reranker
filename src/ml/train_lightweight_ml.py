from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.feature_engineering import FEATURE_COLUMNS, build_features
from src.utils import ensure_parent, rank_scores, validate_candidates

METHOD = "lightweight_ml"


def split_by_qid(df: pd.DataFrame, test_size: float, seed: int):
    qids = sorted(df["qid"].unique())
    train_qids, test_qids = train_test_split(qids, test_size=test_size, random_state=seed)
    train = df[df["qid"].isin(train_qids)].copy()
    test = df[df["qid"].isin(test_qids)].copy()
    return train, test


def make_model(kind: str, seed: int):
    if kind == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ])
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=seed)
    raise ValueError(f"Unknown model kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--out", default="results/ml_scores.csv")
    parser.add_argument("--metrics-out", default="results/ml_classification_metrics.csv")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-all", action="store_true", help="Score all qids after training. Default scores test qids only.")
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates)
    validate_candidates(cand)
    train, test = split_by_qid(cand, args.test_size, args.seed)

    X_train = build_features(train)
    y_train = train["label"].astype(int)
    X_test = build_features(test)
    y_test = test["label"].astype(int)

    model = make_model(args.model, args.seed)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    cls_metrics = pd.DataFrame([{
        "model": args.model,
        "num_train_qids": train["qid"].nunique(),
        "num_test_qids": test["qid"].nunique(),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "features": ",".join(FEATURE_COLUMNS),
    }])
    ensure_parent(args.metrics_out)
    cls_metrics.to_csv(args.metrics_out, index=False, encoding="utf-8-sig")

    score_df = cand if args.score_all else test
    X_score = build_features(score_df)
    t0 = time.perf_counter()
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_score)[:, 1]
    else:
        raw = model.decision_function(X_score)
        scores = 1 / (1 + np.exp(-raw))
    total_ms = (time.perf_counter() - t0) * 1000
    per_pair_ms = total_ms / max(len(score_df), 1)

    raw_out = score_df[["qid", "context_id"]].copy()
    raw_out["score"] = scores.astype(float)
    raw_out["latency_ms"] = per_pair_ms
    out = rank_scores(raw_out, METHOD)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote {args.out}: {len(out)} rows; avg latency={per_pair_ms:.4f} ms/pair")
    print(f"[OK] wrote {args.metrics_out}")
    print(cls_metrics.to_string(index=False))
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
