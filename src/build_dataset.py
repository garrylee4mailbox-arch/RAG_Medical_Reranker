from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import clean_text, ensure_parent, make_context, normalize_department_from_path_or_name, read_csv_auto, validate_candidates


def load_medical_csv(path: Path, canonical_department: str) -> pd.DataFrame:
    df = read_csv_auto(path)
    for col in ["department", "title", "ask", "answer"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")
    df = df[["department", "title", "ask", "answer"]].copy()
    for col in df.columns:
        df[col] = df[col].map(clean_text)
    df = df[(df["ask"] != "") & (df["answer"] != "")].drop_duplicates(["ask", "answer"]).reset_index(drop=True)
    df["canonical_department"] = canonical_department
    df["raw_id"] = [f"{canonical_department}_{i:06d}" for i in range(len(df))]
    df["context"] = [make_context(t, q, a) for t, q, a in zip(df["title"], df["ask"], df["answer"])]
    return df


def compute_similarity_scores(rows: pd.DataFrame) -> List[float]:
    """Use Ollama embeddings first; otherwise fallback to TF-IDF cosine."""
    questions = rows["question"].tolist()
    contexts = rows["candidate_context"].tolist()
    try:
        import ollama

        embedding_cache: Dict[str, np.ndarray] = {}

        def embed_text(text: str) -> np.ndarray:
            prompt = str(text)[:900]
            if prompt not in embedding_cache:
                response = ollama.embeddings(model="nomic-embed-text", prompt=prompt)
                embedding_cache[prompt] = np.asarray(response["embedding"], dtype=float)
                if len(embedding_cache) % 50 == 0:
                    print(f"[INFO] Ollama embedded {len(embedding_cache)} unique texts")
            return embedding_cache[prompt]

        q_emb = np.vstack([embed_text(text) for text in questions])
        c_emb = np.vstack([embed_text(text) for text in contexts])
        numerator = np.sum(q_emb * c_emb, axis=1)
        denominator = np.linalg.norm(q_emb, axis=1) * np.linalg.norm(c_emb, axis=1)
        scores = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)
        return scores.astype(float).tolist()
    except Exception as exc:
        print(f"[WARN] Ollama embedding failed, fallback to TF-IDF. Reason: {exc}")
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        all_text = questions + contexts
        mat = vectorizer.fit_transform(all_text)
        q_mat = mat[:len(questions)]
        c_mat = mat[len(questions):]
        scores = q_mat.multiply(c_mat).sum(axis=1).A1
        return scores.astype(float).tolist()


def build_candidates(raw_dir: Path, out_path: Path, num_questions: int, hard_negatives: int, random_negatives: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    files = {
        "pediatrics": raw_dir / "pediatric_sample.csv",
        "oncology": raw_dir / "oncology_sample.csv",
    }
    dfs: Dict[str, pd.DataFrame] = {}
    for dept, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Put raw CSVs under data/raw/ first.")
        dfs[dept] = load_medical_csv(path, dept)
        print(f"[INFO] loaded {dept}: {len(dfs[dept])} rows from {path}")

    all_df = pd.concat(dfs.values(), ignore_index=True)
    if len(all_df) < num_questions:
        raise ValueError(f"Not enough raw rows: requested {num_questions}, got {len(all_df)}")

    # Balanced sampling across departments as much as possible.
    half = num_questions // 2
    sampled = []
    for dept, target in [("pediatrics", half), ("oncology", num_questions - half)]:
        pool = dfs[dept]
        sampled.append(pool.sample(n=min(target, len(pool)), random_state=seed))
    questions_df = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    rows = []
    context_counter = 1
    for qi, qrow in questions_df.iterrows():
        qid = f"q{qi+1:04d}"
        q_dept = qrow["canonical_department"]
        question = qrow["ask"]

        def add_candidate(crow: pd.Series, label: int, source_type: str):
            nonlocal context_counter
            rows.append({
                "qid": qid,
                "question": question,
                "context_id": f"c{context_counter:06d}",
                "candidate_context": crow["context"],
                "label": int(label),
                "department": crow["canonical_department"],
                "source_type": source_type,
                "embedding_score": 0.0,
                "question_department": q_dept,
            })
            context_counter += 1

        add_candidate(qrow, 1, "positive")

        same_pool = dfs[q_dept][dfs[q_dept]["raw_id"] != qrow["raw_id"]]
        hard = same_pool.sample(n=min(hard_negatives, len(same_pool)), random_state=seed + qi)
        for _, crow in hard.iterrows():
            add_candidate(crow, 0, "hard_negative")

        other_dept = "oncology" if q_dept == "pediatrics" else "pediatrics"
        other_pool = dfs[other_dept]
        rand = other_pool.sample(n=min(random_negatives, len(other_pool)), random_state=seed + 10000 + qi)
        for _, crow in rand.iterrows():
            add_candidate(crow, 0, "random_negative")

    out = pd.DataFrame(rows)
    out["embedding_score"] = compute_similarity_scores(out)
    out = out.sort_values(["qid", "source_type", "context_id"]).reset_index(drop=True)
    validate_candidates(out)
    ensure_parent(out_path)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/candidates.csv")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--hard-negatives", type=int, default=4)
    parser.add_argument("--random-negatives", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_candidates(Path(args.raw_dir), Path(args.out), args.num_questions, args.hard_negatives, args.random_negatives, args.seed)
    print(f"[OK] wrote {args.out}: {len(df)} rows, {df['qid'].nunique()} questions")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
