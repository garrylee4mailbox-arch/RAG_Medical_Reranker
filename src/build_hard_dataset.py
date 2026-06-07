from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import (
    clean_text,
    cosine_matrix,
    ensure_parent,
    make_context,
    read_csv_auto,
    validate_candidates,
)


REQUIRED_RAW_COLUMNS = ["department", "title", "ask", "answer"]
OUTPUT_COLUMNS = [
    "qid",
    "question",
    "context_id",
    "candidate_context",
    "label",
    "department",
    "source_type",
    "embedding_score",
    "question_department",
]
ALLOWED_SOURCE_TYPES = {"positive", "embedding_hard_negative", "random_negative"}


def load_medical_csv(path: Path, canonical_department: str) -> pd.DataFrame:
    """Load raw medical QA CSVs with the same cleaning contract as build_dataset.py."""
    df = read_csv_auto(path)
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df[REQUIRED_RAW_COLUMNS].copy()
    for col in df.columns:
        df[col] = df[col].map(clean_text)

    df = df[(df["ask"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(["ask", "answer"]).reset_index(drop=True)
    df["canonical_department"] = canonical_department
    df["raw_id"] = [f"{canonical_department}_{i:06d}" for i in range(len(df))]
    df["context"] = [make_context(t, q, a) for t, q, a in zip(df["title"], df["ask"], df["answer"])]
    df["answer_key"] = df["answer"].map(lambda x: clean_text(x).lower())
    return df


class EmbeddingScorer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.backend = "tfidf"
        self._ollama = None
        self._cache: Dict[str, np.ndarray] = {}

        if model_name.lower() in {"tfidf", "tf-idf"}:
            print("[INFO] using TF-IDF char ngram embeddings")
            return

        try:
            import ollama

            # Probe once so an unavailable Ollama daemon fails before the full run.
            client = ollama.Client(timeout=10.0)
            client.embeddings(model=model_name, prompt="probe")
            self._ollama = client
            self.backend = "ollama"
            print(f"[INFO] using Ollama embeddings: {model_name}")
        except Exception as exc:
            print(f"[WARN] Ollama embedding unavailable, fallback to TF-IDF char ngrams. Reason: {exc}")

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray | None:
        if self.backend != "ollama" or self._ollama is None:
            return None

        embeddings: List[np.ndarray] = []
        for text in texts:
            prompt = str(text)[:900]
            if prompt not in self._cache:
                response = self._ollama.embeddings(model=self.model_name, prompt=prompt)
                self._cache[prompt] = np.asarray(response["embedding"], dtype=float)
                if len(self._cache) % 100 == 0:
                    print(f"[INFO] Ollama embedded {len(self._cache)} unique texts")
            embeddings.append(self._cache[prompt])
        return np.vstack(embeddings)

    def similarity_matrix(self, questions: list[str], contexts: list[str]) -> np.ndarray:
        if self.backend == "ollama":
            q_emb = self.embed_texts(questions)
            c_emb = self.embed_texts(contexts)
            if q_emb is not None and c_emb is not None:
                return cosine_matrix(q_emb, c_emb)

        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        mat = vectorizer.fit_transform(questions + contexts)
        q_mat = mat[: len(questions)]
        c_mat = mat[len(questions) :]
        return (q_mat @ c_mat.T).toarray()


def answer_jaccard(a: str, b: str) -> float:
    a_chars = set(clean_text(a).lower())
    b_chars = set(clean_text(b).lower())
    a_chars.discard(" ")
    b_chars.discard(" ")
    if not a_chars or not b_chars:
        return 0.0
    return len(a_chars & b_chars) / len(a_chars | b_chars)


def load_all_raw(raw_dir: Path) -> pd.DataFrame:
    files = {
        "pediatrics": raw_dir / "pediatric_sample.csv",
        "oncology": raw_dir / "oncology_sample.csv",
    }
    frames = []
    for department, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Put raw CSVs under data/raw/ first.")
        df = load_medical_csv(path, department)
        print(f"[INFO] loaded {department}: {len(df)} rows from {path}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def sample_questions(all_df: pd.DataFrame, num_questions: int, seed: int) -> pd.DataFrame:
    if len(all_df) < num_questions:
        raise ValueError(f"Not enough raw rows: requested {num_questions}, got {len(all_df)}")

    departments = sorted(all_df["canonical_department"].unique())
    base = num_questions // len(departments)
    remainder = num_questions % len(departments)
    sampled = []
    for idx, department in enumerate(departments):
        target = base + (1 if idx < remainder else 0)
        pool = all_df[all_df["canonical_department"] == department]
        if len(pool) < target:
            raise ValueError(f"Not enough {department} rows: requested {target}, got {len(pool)}")
        sampled.append(pool.sample(n=target, random_state=seed + idx))

    return pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def candidate_pool_for_question(all_df: pd.DataFrame, qrow: pd.Series, min_needed: int, qid: str) -> pd.DataFrame:
    positive_context = qrow["context"]
    positive_answer = qrow["answer_key"]

    pool = all_df[all_df["raw_id"] != qrow["raw_id"]].copy()
    pool = pool[pool["context"] != positive_context]
    pool = pool[pool["answer_key"] != positive_answer]
    pool = pool.drop_duplicates("context").reset_index(drop=True)

    filtered = pool[pool["answer"].map(lambda ans: answer_jaccard(qrow["answer"], ans) <= 0.80)].reset_index(drop=True)
    if len(filtered) >= min_needed:
        return filtered

    print(
        f"[WARN] {qid}: high-overlap answer filter left {len(filtered)} negatives; "
        f"using {len(pool)} exact-filtered negatives instead"
    )
    return pool


def validate_hard_candidates(df: pd.DataFrame, num_questions: int, hard_negatives: int, random_negatives: int) -> None:
    validate_candidates(df)
    missing = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Hard candidates missing required columns: {missing}")

    expected_group_size = 1 + hard_negatives + random_negatives
    expected_rows = num_questions * expected_group_size
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")
    if df["qid"].nunique() != num_questions:
        raise ValueError(f"Expected {num_questions} qids, got {df['qid'].nunique()}")

    group_sizes = df.groupby("qid").size()
    bad_sizes = group_sizes[group_sizes != expected_group_size]
    if len(bad_sizes):
        raise ValueError(f"Some qids do not have {expected_group_size} candidates: {bad_sizes.head().to_dict()}")

    positives = df.groupby("qid")["label"].sum()
    bad_positives = positives[positives != 1]
    if len(bad_positives):
        raise ValueError(f"Some qids do not have exactly one positive: {bad_positives.head().to_dict()}")

    expected_source_counts = {
        "positive": 1,
        "embedding_hard_negative": hard_negatives,
        "random_negative": random_negatives,
    }
    source_counts = df.groupby(["qid", "source_type"]).size().unstack(fill_value=0)
    for source_type, expected_count in expected_source_counts.items():
        actual = source_counts.get(source_type, pd.Series(0, index=source_counts.index))
        bad = actual[actual != expected_count]
        if len(bad):
            raise ValueError(f"Some qids do not have {expected_count} {source_type}: {bad.head().to_dict()}")

    invalid_sources = set(df["source_type"].astype(str)) - ALLOWED_SOURCE_TYPES
    if invalid_sources:
        raise ValueError(f"Invalid source_type values: {sorted(invalid_sources)}")


def build_hard_candidates(
    raw_dir: Path,
    out_path: Path,
    num_questions: int,
    hard_negatives: int,
    random_negatives: int,
    seed: int,
    embedding_model: str,
) -> pd.DataFrame:
    if num_questions <= 0:
        raise ValueError("--num-questions must be positive")
    if hard_negatives < 0 or random_negatives < 0:
        raise ValueError("Negative counts are not allowed")

    rng = random.Random(seed)
    all_df = load_all_raw(raw_dir)
    questions_df = sample_questions(all_df, num_questions, seed)

    scorer = EmbeddingScorer(embedding_model)
    question_texts = questions_df["ask"].astype(str).tolist()
    context_texts = all_df["context"].astype(str).tolist()
    sim = scorer.similarity_matrix(question_texts, context_texts)

    all_index_by_raw_id = {raw_id: i for i, raw_id in enumerate(all_df["raw_id"])}
    rows = []
    context_counter = 1

    for qi, qrow in questions_df.iterrows():
        qid = f"qh{qi + 1:04d}"
        question = qrow["ask"]
        q_dept = qrow["canonical_department"]
        selected_contexts = {qrow["context"]}

        def add_candidate(crow: pd.Series, label: int, source_type: str, score: float) -> None:
            nonlocal context_counter
            rows.append(
                {
                    "qid": qid,
                    "question": question,
                    "context_id": f"ch{context_counter:06d}",
                    "candidate_context": crow["context"],
                    "label": int(label),
                    "department": crow["canonical_department"],
                    "source_type": source_type,
                    "embedding_score": float(score),
                    "question_department": q_dept,
                }
            )
            context_counter += 1

        positive_score = sim[qi, all_index_by_raw_id[qrow["raw_id"]]]
        add_candidate(qrow, 1, "positive", positive_score)

        min_needed = hard_negatives + random_negatives
        pool = candidate_pool_for_question(all_df, qrow, min_needed=min_needed, qid=qid)
        if len(pool) < hard_negatives + random_negatives:
            raise ValueError(
                f"Not enough valid negatives for qid {qid}: need {hard_negatives + random_negatives}, got {len(pool)}"
            )

        pool = pool.copy()
        pool["similarity"] = [sim[qi, all_index_by_raw_id[raw_id]] for raw_id in pool["raw_id"]]
        hard_pool = pool.sort_values(["similarity", "raw_id"], ascending=[False, True])
        hard = hard_pool.head(hard_negatives)
        for _, crow in hard.iterrows():
            selected_contexts.add(crow["context"])
            add_candidate(crow, 0, "embedding_hard_negative", crow["similarity"])

        random_pool = pool[~pool["context"].isin(selected_contexts)].copy()
        if len(random_pool) < random_negatives:
            raise ValueError(f"Not enough random negatives for qid {qid}: need {random_negatives}, got {len(random_pool)}")
        random_indices = rng.sample(list(random_pool.index), random_negatives)
        random_rows = random_pool.loc[random_indices]
        for _, crow in random_rows.iterrows():
            add_candidate(crow, 0, "random_negative", crow["similarity"])

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = out.sort_values(["qid", "source_type", "context_id"]).reset_index(drop=True)
    validate_hard_candidates(out, num_questions, hard_negatives, random_negatives)

    ensure_parent(out_path)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def print_summary(out_path: Path, df: pd.DataFrame) -> None:
    print(f"[OK] wrote {out_path}")
    print(f"total rows: {len(df)}")
    print(f"number of qids: {df['qid'].nunique()}")
    print("label distribution:")
    print(df["label"].value_counts().sort_index().to_string())
    print("source_type distribution:")
    print(df["source_type"].value_counts().to_string())
    print("qid group size distribution:")
    print(df.groupby("qid").size().value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval-hard medical QA candidate benchmark.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/candidates_hard.csv")
    parser.add_argument("--num-questions", type=int, default=500)
    parser.add_argument("--hard-negatives", type=int, default=7)
    parser.add_argument("--random-negatives", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    args = parser.parse_args()

    df = build_hard_candidates(
        raw_dir=Path(args.raw_dir),
        out_path=Path(args.out),
        num_questions=args.num_questions,
        hard_negatives=args.hard_negatives,
        random_negatives=args.random_negatives,
        seed=args.seed,
        embedding_model=args.embedding_model,
    )
    print_summary(Path(args.out), df)


if __name__ == "__main__":
    main()
