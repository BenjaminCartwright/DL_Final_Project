"""Reusable utilities for LoRA baseline notebooks."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.frozen_baseline_utils import (
    ChoiceWiseScoringHead,
    LinearHead,
    MLPHead,
    apply_sanity_subset,
    build_head,
    build_prompt,
    infer_representation_dim,
    load_split,
    make_dataloader,
    make_submission,
    maybe_save_dataframe,
    maybe_save_figure,
    maybe_save_json,
    parse_choices,
    resolve_image_path,
    set_seed,
    validate_submission,
)


def build_lora_model(
    backbone: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
) -> nn.Module:
    """Wrap a backbone with LoRA adapters."""
    task_type_key = str(task_type).upper().strip()
    try:
        peft_task_type = TaskType[task_type_key]
    except KeyError as exc:
        allowed = ", ".join([item.name for item in TaskType])
        raise ValueError(f"Unsupported task_type '{task_type}'. Allowed: {allowed}") from exc

    config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(target_modules),
        bias=str(bias),
        task_type=peft_task_type,
    )
    return get_peft_model(backbone, config)


def get_trainable_param_counts(model: nn.Module, head: nn.Module | None = None) -> dict[str, int]:
    """Return trainable and total parameter counts for model and optional head."""
    model_total = sum(p.numel() for p in model.parameters())
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = {
        "model_total": int(model_total),
        "model_trainable": int(model_trainable),
    }
    if head is not None:
        head_total = sum(p.numel() for p in head.parameters())
        head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        out["head_total"] = int(head_total)
        out["head_trainable"] = int(head_trainable)
        out["combined_trainable"] = int(model_trainable + head_trainable)
    return out


def extract_representation(
    model: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    device: str,
    pooling: str = "last_token",
) -> torch.Tensor:
    """Extract fixed-size representations from LoRA-adapted hidden states."""
    inputs = {k: v.to(device) for k, v in model_inputs.items()}
    outputs = model(
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


def infer_representation_dim_lora(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    pooling: str,
) -> int:
    """Infer representation dimensionality from one batch."""
    batch = next(iter(loader))
    with torch.no_grad():
        reps = extract_representation(
            model=model,
            model_inputs=batch.model_inputs,
            device=device,
            pooling=pooling,
        )
    return int(reps.shape[-1])


def train_one_epoch_lora(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pooling: str,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Train one epoch for LoRA adapters plus trainable head."""
    model.train()
    head.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    grad_accum = max(1, int(gradient_accumulation_steps))
    optimizer.zero_grad(set_to_none=True)

    batch_iter = tqdm(loader, desc=progress_desc or "Train batches", leave=False)
    n_batches = len(loader)
    for step_idx, batch in enumerate(batch_iter, start=1):
        if batch.labels is None:
            raise ValueError("Training loader must contain labels.")

        labels = batch.labels.to(device)
        choice_mask = batch.choice_mask.to(device)

        reps = extract_representation(
            model=model,
            model_inputs=batch.model_inputs,
            device=device,
            pooling=pooling,
        )
        head_dtype = next(head.parameters()).dtype
        reps = reps.to(dtype=head_dtype)
        logits = head(reps, choice_mask=choice_mask)
        loss = F.cross_entropy(logits, labels)
        scaled_loss = loss / grad_accum
        scaled_loss.backward()

        should_step = (step_idx % grad_accum == 0) or (step_idx == n_batches)
        if should_step:
            if max_grad_norm > 0:
                trainable_params = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_examples += int(labels.size(0))
        total_loss += float(loss.item()) * labels.size(0)

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
    }


def evaluate_lora_head(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: str,
    pooling: str,
    progress_desc: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a LoRA+head model and return metrics plus predictions."""
    model.eval()
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
                model=model,
                model_inputs=batch.model_inputs,
                device=device,
                pooling=pooling,
            )
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


def train_and_select_lora_head(
    model: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    pooling: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> tuple[nn.Module, nn.Module, list[dict[str, float]], dict[str, float], pd.DataFrame]:
    """Train LoRA+head for fixed epochs and keep best checkpoint by val accuracy."""
    params = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    best_model_state: dict[str, torch.Tensor] | None = None
    best_head_state: dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    best_metrics: dict[str, float] = {}
    best_val_preds = pd.DataFrame()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_lora(
            model=model,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pooling=pooling,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            progress_desc=f"Epoch {epoch}/{epochs}",
        )
        val_metrics, val_preds = evaluate_lora_head(
            model=model,
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
            best_model_state = copy.deepcopy(model.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
            best_metrics = val_metrics
            best_val_preds = val_preds.copy()

    if best_model_state is None or best_head_state is None:
        raise RuntimeError("No best checkpoint captured during training.")

    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    return model, head, history, best_metrics, best_val_preds


__all__ = [
    "ChoiceWiseScoringHead",
    "LinearHead",
    "MLPHead",
    "apply_sanity_subset",
    "build_head",
    "build_lora_model",
    "build_prompt",
    "evaluate_lora_head",
    "get_trainable_param_counts",
    "infer_representation_dim",
    "infer_representation_dim_lora",
    "load_split",
    "make_dataloader",
    "make_submission",
    "maybe_save_dataframe",
    "maybe_save_figure",
    "maybe_save_json",
    "parse_choices",
    "resolve_image_path",
    "set_seed",
    "train_and_select_lora_head",
    "train_one_epoch_lora",
    "validate_submission",
]
