"""Utilities for LoRA best-candidates full-run notebooks (named ablation spaces)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from src.lora_tuning_block_utils import run_single_lora_ablation


def build_configs_from_named_space(named_space: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    """Expand a named ablation dict into a flat list of run configs.

    Same contract as ``zero_shot_best_candidates_full_run.ipynb``:
    each key is an ablation name; value is either a single params ``dict`` or a
    ``list[dict]`` of variants. Each output dict includes ``ablation_name`` and
    ``ablation_id`` (``{prefix}_{name}`` or ``{prefix}_{name}_{variant_idx:02d}``).
    """
    configs: list[dict[str, Any]] = []
    for name, spec in named_space.items():
        variants = spec if isinstance(spec, list) else [spec]
        for variant_idx, params in enumerate(variants, start=1):
            cfg = dict(params)
            cfg["ablation_name"] = name
            if len(variants) > 1:
                cfg["ablation_id"] = f"{prefix}_{name}_{variant_idx:02d}"
            else:
                cfg["ablation_id"] = f"{prefix}_{name}"
            configs.append(cfg)
    return configs


def run_lora_candidate_on_val(
    config: dict[str, Any],
    fixed_context: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Train one LoRA candidate and evaluate on validation.

    Returns ``(summary_row, val_predictions_df, train_history_df)`` where
    ``summary_row`` merges config keys with validation metrics and timing;
    prediction rows include ``ablation_id`` for concatenation across candidates.
    """
    summary, history, val_pred_df = run_single_lora_ablation(cfg=config, fixed_context=fixed_context)

    val_pred_df = val_pred_df.copy()
    val_pred_df["ablation_id"] = config["ablation_id"]

    history_df = pd.DataFrame(history)
    if len(history_df):
        history_df = history_df.copy()
        history_df["ablation_id"] = config["ablation_id"]

    if fixed_context.get("clear_cuda_between_runs", True) and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, val_pred_df, history_df


__all__ = [
    "build_configs_from_named_space",
    "run_lora_candidate_on_val",
]
