# A Comparative Study of Lightweight ML Reranker and Open-Source Rerankers for Medical QA Retrieval

## 1. Introduction

This project studies the reranking stage in a Chinese medical QA retrieval pipeline. Instead of building a complete RAG system, we isolate the context selection problem: given one medical question and multiple candidate medical contexts, which method can rank the truly relevant context higher?

## 2. Task Formulation

Each data instance is a question-context pair. For each question, we construct one positive context from the original answer, several hard negatives from the same broad department, and several random negatives from another department. Each reranker produces a relevance score for every pair. The candidates are then sorted within each question group.

## 3. Compared Methods

- Embedding baseline: ranks candidates by the initial semantic similarity score.
- BGE reranker: uses `BAAI/bge-reranker-v2-m3` as a multilingual cross-encoder reranker.
- BCE reranker: uses `maidalun1020/bce-reranker-base_v1` as a bilingual Chinese-English reranker.
- Lightweight ML reranker: uses engineered features such as embedding score, keyword overlap, department match, question length, and context length.

## 4. Dataset Construction

The dataset is built from Chinese medical QA records in pediatrics and oncology. Each question has 10 candidate contexts: 1 positive context, 4 hard negatives, and 5 random negatives. The final dataset is saved as `data/processed/candidates.csv`.

## 5. Evaluation Metrics

We use Precision@1, Precision@3, Precision@5, MRR, NDCG@5, and average latency. These metrics focus on whether the relevant medical context is placed near the top of the ranked list.

## 6. Results

See `results/summary_metrics.csv`.

## 7. Discussion

The comparison shows whether general-purpose multilingual rerankers and a lightweight ML model can improve over the initial embedding baseline. The lightweight ML model is easier to run and interpret, while BGE and BCE may provide stronger semantic matching but require more computation.

## 8. Limitations and Future Work

This project only evaluates reranking quality, not final answer generation. It uses automatically constructed labels from the original QA pairs, so the negatives may contain partially relevant medical information. Future work can add human-labeled relevance, more departments, larger datasets, and compatibility tests across different LLM generators.
