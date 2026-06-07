# CPS3830 Medical QA Reranker Comparison

## Project Goal

This project compares four reranking methods for Chinese medical QA retrieval. Given a medical question and multiple candidate medical contexts, each method assigns a relevance score and ranks the candidates within the same question group.

Methods:

1. `embedding_baseline`: original embedding similarity baseline.
2. `bge`: `BAAI/bge-reranker-v2-m3` open-source reranker.
3. `bce`: `maidalun1020/bce-reranker-base_v1` open-source reranker.
4. `lightweight_ml`: our lightweight ML reranker using engineered features.

This is not a full RAG generation system. The task is isolated as context selection / retrieval reranking.

## Repository Structure

```text
CPS3830-Medical-Reranker/
├── data/
│   ├── raw/
│   │   ├── pediatric_sample.csv
│   │   └── oncology_sample.csv
│   └── processed/
│       └── candidates.csv
├── src/
│   ├── build_dataset.py
│   ├── embedding_baseline.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── ml/
│   │   ├── feature_engineering.py
│   │   └── train_lightweight_ml.py
│   └── rerankers/
│       ├── bge_reranker.py
│       └── bce_reranker.py
├── results/
├── report/
└── slides/
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Step 1: Build Candidate Dataset

Input:

- `data/raw/pediatric_sample.csv`
- `data/raw/oncology_sample.csv`

Output:

- `data/processed/candidates.csv`

Run:

```bash
python -m src.build_dataset --num-questions 100 --hard-negatives 4 --random-negatives 5 --seed 42
```

Expected output: about 1000 rows, because each question has 10 candidate contexts.

## Step 2: Run Embedding Baseline

```bash
python -m src.embedding_baseline
```

Output:

- `results/embedding_baseline_scores.csv`

## Step 3: Run BGE Reranker

For teammate A:

```bash
python -m src.rerankers.bge_reranker --sample 20
python -m src.rerankers.bge_reranker
```

Output:

- `results/bge_scores.csv`

## Step 4: Run BCE Reranker

For teammate B:

```bash
python -m src.rerankers.bce_reranker --sample 20
python -m src.rerankers.bce_reranker
```

Output:

- `results/bce_scores.csv`

If BCE fails, the script automatically tries `jinaai/jina-reranker-v2-base-multilingual`.

## Step 5: Train Lightweight ML Reranker

```bash
python -m src.ml.train_lightweight_ml --model logreg --score-all
```

Outputs:

- `results/ml_scores.csv`
- `results/ml_classification_metrics.csv`

## Step 6: Evaluate All Methods

```bash
python -m src.evaluate
```

Output:

- `results/summary_metrics.csv`

Metrics:

- Precision@1
- Precision@3
- Precision@5
- MRR
- NDCG@5
- Average latency

## Output Format Contract

Every score file must contain exactly these columns:

```text
qid,context_id,score,rank,method,latency_ms
```

Allowed method names:

```text
embedding_baseline
bge
bce
lightweight_ml
```

Ranking must be computed within each `qid` group. Higher score means more relevant.

## Team Division

- Garry: repository setup, dataset construction, embedding baseline, lightweight ML, final evaluation, integration.
- Teammate A: BGE reranker, `results/bge_scores.csv`, method explanation.
- Teammate B: BCE reranker, `results/bce_scores.csv`, figures/report/slides support.
