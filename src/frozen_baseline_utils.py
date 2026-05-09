"""Reusable utilities for frozen-backbone baseline notebooks."""

from __future__ import annotations

import ast
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    """Load a split CSV and normalize key text fields used in prompting."""
    path = data_dir / f"{split}.csv"
    df = pd.read_csv(path)
    for col in ["question", "hint", "lecture"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def apply_sanity_subset(df: pd.DataFrame, sanity_check: bool, n: int, seed: int) -> pd.DataFrame:
    """Return a deterministic subset for quick smoke tests."""
    if not sanity_check:
        return df.copy()
    n = max(1, min(int(n), len(df)))
    sampled = df.sample(n=n, random_state=seed).copy()
    return sampled.reset_index(drop=True)


def parse_choices(choices_raw: Any) -> list[str]:
    """Parse choices from JSON-like strings into a string list."""
    if isinstance(choices_raw, list):
        return [str(x) for x in choices_raw]
    if isinstance(choices_raw, str):
        try:
            parsed = ast.literal_eval(choices_raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return []
    return []


def build_prompt(row: pd.Series, include_hint: bool = True, include_lecture: bool = True) -> str:
    """Build a structured prompt with numeric choice indices."""
    choices = parse_choices(row.get("choices", "[]"))
    choice_lines = [f"{i}: {choice}" for i, choice in enumerate(choices)]
    sections = [
        "You are solving a science multiple-choice question from an image and text.",
        "Return only the integer index of the best answer (for example: 0).",
        "",
        f"Question: {row.get('question', '')}",
    ]
    hint = str(row.get("hint", "")).strip()
    lecture = str(row.get("lecture", "")).strip()
    if include_hint and hint:
        sections.extend(["", f"Hint: {hint}"])
    if include_lecture and lecture:
        sections.extend(["", f"Lecture: {lecture}"])
    sections.extend(["", "Choices:", *choice_lines, "", "Answer index:"])
    return "\n".join(sections)


def resolve_image_path(data_dir: Path, image_path: str) -> Path:
    """Resolve image path from CSV metadata."""
    return data_dir / str(image_path)


class FrozenQADataset(Dataset):
    """Simple dataset wrapper that serves rows for multimodal processing."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        item = {
            "id": row["id"],
            "image_path": row["image_path"],
            "num_choices": int(row["num_choices"]),
            "question": row.get("question", ""),
            "hint": row.get("hint", ""),
            "lecture": row.get("lecture", ""),
            "choices": row.get("choices", "[]"),
        }
        if "answer" in self.df.columns:
            item["answer"] = int(row["answer"])
        return item


@dataclass
class Batch:
    """Batch container for typed access in train/eval loops."""

    ids: list[Any]
    model_inputs: dict[str, torch.Tensor]
    num_choices: torch.Tensor
    choice_mask: torch.Tensor
    labels: torch.Tensor | None


def make_collate_fn(
    processor: Any,
    data_dir: Path,
    include_hint: bool,
    include_lecture: bool,
    max_choices: int,
):
    """Create a collate function that tokenizes text and images consistently."""

    def collate_fn(items: list[dict[str, Any]]) -> Batch:
        prompts: list[str] = []
        images: list[Image.Image] = []
        ids: list[Any] = []
        num_choices_list: list[int] = []
        labels: list[int] = []

        for item in items:
            row = pd.Series(item)
            prompt = build_prompt(row, include_hint=include_hint, include_lecture=include_lecture)
            img_path = resolve_image_path(data_dir, item["image_path"])
            image = Image.open(img_path).convert("RGB")

            prompts.append(prompt)
            images.append(image)
            ids.append(item["id"])
            num_choices_list.append(int(item["num_choices"]))
            if "answer" in item:
                labels.append(int(item["answer"]))

        if hasattr(processor, "apply_chat_template"):
            rendered_prompts: list[str] = []
            for prompt in prompts:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                rendered_prompts.append(
                    processor.apply_chat_template(messages, add_generation_prompt=False)
                )
            model_inputs = processor(
                text=rendered_prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        else:
            model_inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")

        num_choices = torch.tensor(num_choices_list, dtype=torch.long)
        choice_mask = (
            torch.arange(max_choices, dtype=torch.long).unsqueeze(0) < num_choices.unsqueeze(1)
        )
        label_tensor = torch.tensor(labels, dtype=torch.long) if labels else None

        return Batch(
            ids=ids,
            model_inputs=model_inputs,
            num_choices=num_choices,
            choice_mask=choice_mask,
            labels=label_tensor,
        )

    return collate_fn


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def extract_representation(
    backbone: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    device: str,
    pooling: str = "last_token",
) -> torch.Tensor:
    """Extract fixed-size representations from frozen hidden states."""
    inputs = {k: v.to(device) for k, v in model_inputs.items()}
    with torch.no_grad():
        outputs = backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]

    if pooling == "mean":
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return hidden[:, -1, :]
    last_token_idx = attention_mask.sum(dim=1).clamp_min(1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_token_idx, :]


class LinearHead(nn.Module):
    """Simple linear classification head."""

    def __init__(self, in_dim: int, max_choices: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, max_choices)

    def forward(self, x: torch.Tensor, choice_mask: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        return logits.masked_fill(~choice_mask, -1e9)


class MLPHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(self, in_dim: int, max_choices: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_choices),
        )

    def forward(self, x: torch.Tensor, choice_mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.masked_fill(~choice_mask, -1e9)


class ChoiceWiseScoringHead(nn.Module):
    """Choice-wise scorer using a shared MLP with learned choice index embeddings."""

    def __init__(
        self,
        in_dim: int,
        max_choices: int,
        choice_emb_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.choice_embedding = nn.Embedding(max_choices, choice_emb_dim)
        self.scorer = nn.Sequential(
            nn.Linear(in_dim + choice_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.max_choices = max_choices

    def forward(self, x: torch.Tensor, choice_mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        choice_ids = torch.arange(self.max_choices, device=x.device).unsqueeze(0).expand(batch_size, -1)
        choice_emb = self.choice_embedding(choice_ids)
        x_expanded = x.unsqueeze(1).expand(-1, self.max_choices, -1)
        fused = torch.cat([x_expanded, choice_emb], dim=-1)
        logits = self.scorer(fused).squeeze(-1)
        return logits.masked_fill(~choice_mask, -1e9)


def build_head(
    head_name: str,
    in_dim: int,
    max_choices: int,
    mlp_hidden_dim: int = 512,
    choice_emb_dim: int = 32,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory for supported head types."""
    key = head_name.lower().strip()
    if key == "linear":
        return LinearHead(in_dim=in_dim, max_choices=max_choices)
    if key == "mlp":
        return MLPHead(
            in_dim=in_dim,
            max_choices=max_choices,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )
    if key in {"choicewise", "choice_wise"}:
        return ChoiceWiseScoringHead(
            in_dim=in_dim,
            max_choices=max_choices,
            choice_emb_dim=choice_emb_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported head_name '{head_name}'.")


def make_dataloader(
    df: pd.DataFrame,
    processor: Any,
    data_dir: Path,
    include_hint: bool,
    include_lecture: bool,
    max_choices: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build dataloader with deterministic collate behavior."""
    dataset = FrozenQADataset(df)
    collate_fn = make_collate_fn(
        processor=processor,
        data_dir=data_dir,
        include_hint=include_hint,
        include_lecture=include_lecture,
        max_choices=max_choices,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def infer_representation_dim(
    backbone: nn.Module,
    loader: DataLoader,
    device: str,
    pooling: str,
) -> int:
    """Infer representation dimensionality from one batch."""
    batch = next(iter(loader))
    reps = extract_representation(
        backbone=backbone,
        model_inputs=batch.model_inputs,
        device=device,
        pooling=pooling,
    )
    return int(reps.shape[-1])


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pooling: str,
    max_grad_norm: float = 1.0,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Train one epoch for the trainable head only."""
    head.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    batch_iter = tqdm(loader, desc=progress_desc or "Train batches", leave=False)
    for batch in batch_iter:
        if batch.labels is None:
            raise ValueError("Training loader must contain labels.")

        labels = batch.labels.to(device)
        choice_mask = batch.choice_mask.to(device)

        reps = extract_representation(
            backbone=backbone,
            model_inputs=batch.model_inputs,
            device=device,
            pooling=pooling,
        )
        # Keep head input dtype aligned with head weights (e.g., bf16 backbone -> fp32 head).
        head_dtype = next(head.parameters()).dtype
        reps = reps.to(dtype=head_dtype)
        logits = head(reps, choice_mask=choice_mask)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_examples += int(labels.size(0))
        total_loss += float(loss.item()) * labels.size(0)

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
    }


def evaluate_head(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: str,
    pooling: str,
    progress_desc: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a head and return aggregate metrics plus per-example predictions."""
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    records: list[dict[str, Any]] = []

    eval_iter = tqdm(loader, desc=progress_desc, leave=False) if progress_desc else loader
    with torch.no_grad():
        for batch in eval_iter:
            choice_mask = batch.choice_mask.to(device)
            reps = extract_representation(
                backbone=backbone,
                model_inputs=batch.model_inputs,
                device=device,
                pooling=pooling,
            )
            # Keep head input dtype aligned with head weights (e.g., bf16 backbone -> fp32 head).
            head_dtype = next(head.parameters()).dtype
            reps = reps.to(dtype=head_dtype)
            logits = head(reps, choice_mask=choice_mask)
            preds = logits.argmax(dim=1).cpu()
            num_choices = batch.num_choices.cpu()

            if batch.labels is not None:
                labels = batch.labels.to(device)
                loss = F.cross_entropy(logits, labels)
                total_loss += float(loss.item()) * labels.size(0)
                total_correct += int((preds.to(device) == labels).sum().item())
                total_examples += int(labels.size(0))

            for i, item_id in enumerate(batch.ids):
                rec = {
                    "id": item_id,
                    "pred_answer": int(preds[i].item()),
                    "num_choices": int(num_choices[i].item()),
                }
                if batch.labels is not None:
                    gold = int(batch.labels[i].item())
                    rec["gold_answer"] = gold
                    rec["is_correct"] = int(rec["pred_answer"] == gold)
                records.append(rec)

    metrics: dict[str, float] = {
        "n_examples": float(len(records)),
    }
    if total_examples > 0:
        metrics["loss"] = total_loss / total_examples
        metrics["accuracy"] = total_correct / total_examples
        metrics["percent_correct"] = 100.0 * metrics["accuracy"]
    return metrics, pd.DataFrame(records)


def train_and_select_head(
    backbone: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    pooling: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    max_grad_norm: float = 1.0,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, float], pd.DataFrame]:
    """Train a head and keep the best checkpoint by validation accuracy."""
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    best_state: dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    best_metrics: dict[str, float] = {}
    best_val_preds = pd.DataFrame()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            backbone=backbone,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pooling=pooling,
            max_grad_norm=max_grad_norm,
            progress_desc=f"Epoch {epoch}/{epochs}",
        )
        val_metrics, val_preds = evaluate_head(
            backbone=backbone,
            head=head,
            loader=val_loader,
            device=device,
            pooling=pooling,
            progress_desc=f"Validation {epoch}/{epochs}",
        )
        epoch_row = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics.get("loss", np.nan)),
            "val_accuracy": float(val_metrics.get("accuracy", np.nan)),
            "val_percent_correct": float(val_metrics.get("percent_correct", np.nan)),
        }
        history.append(epoch_row)

        val_acc = float(val_metrics.get("accuracy", -1.0))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            best_metrics = val_metrics
            best_val_preds = val_preds.copy()

    if best_state is None:
        raise RuntimeError("No best checkpoint captured during training.")
    head.load_state_dict(best_state)
    return head, history, best_metrics, best_val_preds


def make_submission(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Build Kaggle-ready submission dataframe from prediction outputs."""
    sub = pred_df[["id", "pred_answer"]].rename(columns={"pred_answer": "answer"}).copy()
    sub["answer"] = sub["answer"].astype(int)
    return sub


def validate_submission(df: pd.DataFrame) -> None:
    """Validate Kaggle submission contract: two columns with integer answers."""
    expected_cols = ["id", "answer"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Submission columns must be {expected_cols}, got {list(df.columns)}")
    if df["id"].isna().any():
        raise ValueError("Submission contains missing id values.")
    if df["id"].duplicated().any():
        raise ValueError("Submission contains duplicate ids.")
    if df["answer"].isna().any():
        raise ValueError("Submission contains missing answers.")
    if not pd.api.types.is_integer_dtype(df["answer"]):
        raise ValueError("Submission answer column must be integer dtype.")


def maybe_save_dataframe(df: pd.DataFrame, path: Path, save: bool) -> Path | None:
    """Save dataframe only if save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def maybe_save_json(payload: dict[str, Any], path: Path, save: bool) -> Path | None:
    """Save JSON payload only if save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def maybe_save_figure(fig: Any, path: Path, save: bool) -> Path | None:
    """Save matplotlib figure only if save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
