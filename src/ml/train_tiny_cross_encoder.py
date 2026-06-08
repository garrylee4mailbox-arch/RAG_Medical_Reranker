from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.utils import clean_text, ensure_parent, rank_scores, validate_candidates

METHOD = "tiny_cross_encoder"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_by_qid(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    qids = sorted(df["qid"].unique())
    train_qids, test_qids = train_test_split(qids, test_size=test_size, random_state=seed)
    train = df[df["qid"].isin(train_qids)].copy()
    test = df[df["qid"].isin(test_qids)].copy()
    return train, test


def text_chars(value: object) -> List[str]:
    return list(clean_text(value))


def build_vocab(train_df: pd.DataFrame) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in train_df[["question", "candidate_context"]].itertuples(index=False):
        counts.update(text_chars(row.question))
        counts.update(text_chars(row.candidate_context))

    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for char, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab


@dataclass(frozen=True)
class EncodedPair:
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]


def truncate_pair(question_chars: List[str], context_chars: List[str], max_content_length: int) -> Tuple[List[str], List[str]]:
    q_chars = list(question_chars)
    c_chars = list(context_chars)
    while len(q_chars) + len(c_chars) > max_content_length:
        if len(c_chars) >= len(q_chars) and c_chars:
            c_chars.pop()
        elif q_chars:
            q_chars.pop()
        else:
            break
    return q_chars, c_chars


def encode_pair(question: object, context: object, vocab: Dict[str, int], max_length: int) -> EncodedPair:
    if max_length < 3:
        raise ValueError("--max-length must be at least 3")

    q_chars, c_chars = truncate_pair(text_chars(question), text_chars(context), max_length - 3)

    input_ids = [CLS_ID]
    token_type_ids = [0]

    input_ids.extend(vocab.get(ch, UNK_ID) for ch in q_chars)
    token_type_ids.extend([0] * len(q_chars))
    input_ids.append(SEP_ID)
    token_type_ids.append(0)

    input_ids.extend(vocab.get(ch, UNK_ID) for ch in c_chars)
    token_type_ids.extend([1] * len(c_chars))
    input_ids.append(SEP_ID)
    token_type_ids.append(1)

    attention_mask = [1] * len(input_ids)
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids.extend([PAD_ID] * pad_len)
        token_type_ids.extend([0] * pad_len)
        attention_mask.extend([0] * pad_len)

    return EncodedPair(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Dict[str, int], max_length: int, include_labels: bool = True) -> None:
        self.labels = df["label"].astype(np.float32).to_numpy() if include_labels else None
        encoded = [
            encode_pair(row.question, row.candidate_context, vocab, max_length)
            for row in df[["question", "candidate_context"]].itertuples(index=False)
        ]
        self.input_ids = torch.tensor([item.input_ids for item in encoded], dtype=torch.long)
        self.token_type_ids = torch.tensor([item.token_type_ids for item in encoded], dtype=torch.long)
        self.attention_mask = torch.tensor([item.attention_mask for item in encoded], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        item = {
            "input_ids": self.input_ids[idx],
            "token_type_ids": self.token_type_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class TinyCrossEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.segment_embedding = nn.Embedding(2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(token_type_ids)
        )
        hidden = self.dropout(hidden)
        key_padding_mask = attention_mask == 0
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        cls_hidden = encoded[:, 0, :]
        return self.classifier(cls_hidden).squeeze(-1)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
    return total_loss / max(total_rows, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().cpu().numpy().tolist()
        labels = batch["labels"].long().cpu().numpy().tolist()
        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
        all_preds.extend(preds)
        all_labels.extend(labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "test_loss": total_loss / max(total_rows, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def score_rows(
    model: nn.Module,
    df: pd.DataFrame,
    vocab: Dict[str, int],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    dataset = PairDataset(df, vocab, max_length, include_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    scores: List[float] = []
    total_forward_ms = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
        probs = torch.sigmoid(logits)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_forward_ms += (time.perf_counter() - t0) * 1000
        scores.extend(probs.cpu().numpy().astype(float).tolist())

    latency_ms = total_forward_ms / max(len(df), 1)
    return np.asarray(scores, dtype=float), latency_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates_hard.csv")
    parser.add_argument("--out", default="results_hard/tiny_cross_encoder_scores.csv")
    parser.add_argument("--metrics-out", default="results_hard/tiny_cross_encoder_training_metrics.csv")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=9.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-all", action="store_true", help="Score all qids after training. Default scores test qids only.")
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu.")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[INFO] device={device}")

    candidates = pd.read_csv(args.candidates)
    validate_candidates(candidates)
    train_df, test_df = split_by_qid(candidates, args.test_size, args.seed)
    train_qids = set(train_df["qid"].unique())
    test_qids = set(test_df["qid"].unique())
    overlap = train_qids.intersection(test_qids)
    if overlap:
        raise RuntimeError(f"Qid leakage detected between train and test: {sorted(overlap)[:5]}")

    vocab = build_vocab(train_df)
    train_dataset = PairDataset(train_df, vocab, args.max_length, include_labels=True)
    test_dataset = PairDataset(test_df, vocab, args.max_length, include_labels=True)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    model = TinyCrossEncoder(
        vocab_size=len(vocab),
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metric_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "num_train_qids": len(train_qids),
            "num_test_qids": len(test_qids),
            "max_length": args.max_length,
            "vocab_size": len(vocab),
            "seed": args.seed,
        }
        metric_rows.append(row)
        print(
            f"[EPOCH {epoch:02d}] train_loss={train_loss:.4f} "
            f"test_loss={metrics['test_loss']:.4f} f1={metrics['f1']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
        )

    metrics_out = pd.DataFrame(metric_rows)
    ensure_parent(args.metrics_out)
    metrics_out.to_csv(args.metrics_out, index=False, encoding="utf-8-sig")

    score_df = candidates.copy() if args.score_all else test_df.copy()
    scores, latency_ms = score_rows(model, score_df, vocab, args.max_length, args.batch_size, device)
    raw_out = score_df[["qid", "context_id"]].copy()
    raw_out["score"] = scores
    raw_out["latency_ms"] = latency_ms
    out = rank_scores(raw_out, METHOD)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote {args.out}: {len(out)} rows; avg latency={latency_ms:.4f} ms/pair")
    print(f"[OK] wrote {args.metrics_out}")
    print(metrics_out.tail(1).to_string(index=False))
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
