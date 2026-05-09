"""Block-based LoRA tuning helpers for notebook workflows."""

from __future__ import annotations

import time
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import transformers
from peft import PeftModel
from transformers import AutoProcessor

from src.gen_lora_study_utils import (
    build_failure_examples,
    build_lora_model,
    evaluate_option_scoring,
    evaluate_sft_generation,
    get_param_counts,
    make_submission,
    maybe_save_dataframe,
    maybe_save_json,
    maybe_save_model,
    train_option_scoring_objective,
    train_sft_objective,
    validate_submission,
)


BLOCK_SWEEP_KEYS = [
    "training_objective",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "use_dora",
    "target_modules",
    "lr",
    "epochs",
    "weight_decay",
    "batch_size",
    "gradient_accumulation_steps",
    "max_grad_norm",
]


def _cast_config_types(cfg: dict[str, Any]) -> dict[str, Any]:
    """Cast numeric fields to stable runtime types."""
    out = dict(cfg)
    int_keys = [
        "lora_r",
        "lora_alpha",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_new_tokens",
        "num_beams",
        "early_stopping",
        "epochs_per_val_check",
        "early_stopping_val_examples",
        "image_size",
        "epoch_resampling_seed",
        "epoch_train_size",
    ]
    float_keys = [
        "lora_dropout",
        "lr",
        "weight_decay",
        "max_grad_norm",
        "warmup_ratio",
        "epoch_train_fraction",
    ]
    for key in int_keys:
        if key in out and out[key] is not None:
            out[key] = int(out[key])
    for key in float_keys:
        if key in out and out[key] is not None:
            out[key] = float(out[key])
    return out


def build_block_configs(
    fixed_values: dict[str, Any],
    block_space: dict[str, list[Any]],
    ablation_prefix: str,
    sweep_keys: list[str] | None = None,
    alpha_from_r_multiplier: int | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic ablation configs from fixed values + sweep space."""
    keys = sweep_keys if sweep_keys is not None else BLOCK_SWEEP_KEYS
    space: dict[str, list[Any]] = {}
    for key in keys:
        if key in block_space:
            space[key] = list(block_space[key])
        else:
            if key not in fixed_values:
                raise ValueError(f"Missing fixed value for key: {key}")
            space[key] = [fixed_values[key]]

    combos = list(product(*[space[k] for k in keys]))
    configs: list[dict[str, Any]] = []
    for idx, vals in enumerate(combos, start=1):
        cfg = dict(zip(keys, vals))
        if alpha_from_r_multiplier is not None and "lora_r" in cfg and "lora_alpha" not in block_space:
            cfg["lora_alpha"] = int(alpha_from_r_multiplier) * int(cfg["lora_r"])
        cfg = _cast_config_types(cfg)
        cfg["ablation_id"] = f"{ablation_prefix}_{idx:03d}"
        configs.append(cfg)
    return configs


def select_ablation_configs(
    all_configs: list[dict[str, Any]],
    selected_ablation_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter configs by selected IDs while preserving ordering."""
    if not selected_ablation_ids:
        return list(all_configs)
    selected = set(selected_ablation_ids)
    filtered = [cfg for cfg in all_configs if cfg["ablation_id"] in selected]
    missing = sorted(selected - {cfg["ablation_id"] for cfg in all_configs})
    if missing:
        raise ValueError(f"Unknown ablation ids: {missing}")
    return filtered


def _resolve_model_class():
    if hasattr(transformers, "AutoModelForVision2Seq"):
        return transformers.AutoModelForVision2Seq
    if hasattr(transformers, "AutoModelForImageTextToText"):
        return transformers.AutoModelForImageTextToText
    raise ImportError("Upgrade transformers: SmolVLM model class not found.")


def _load_model_with_lora(
    model_id: str,
    device: str,
    torch_dtype: str,
    config: dict[str, Any],
):
    """Load base model and attach LoRA adapters for one ablation run."""
    model_class = _resolve_model_class()
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    selected_dtype = dtype_map.get(str(torch_dtype), "auto")
    base_model = model_class.from_pretrained(model_id, torch_dtype=selected_dtype).to(device)
    model = build_lora_model(
        backbone=base_model,
        lora_r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        bias=str(config["lora_bias"]),
        task_type=str(config["lora_task_type"]),
        use_dora=bool(config.get("use_dora", False)),
        use_rslora=bool(config.get("use_rslora", False)),
    ).to(device)
    return model


def count_trainable_params_for_config(
    cfg: dict[str, Any],
    fixed_context: dict[str, Any],
) -> dict[str, int | float]:
    """Load the LoRA-wrapped model for one merged config and return parameter counts.

    Performs a full ``from_pretrained`` load; intended for pre-run previews in
    notebooks. Deletes the model afterward and clears CUDA cache when available.
    """
    run_cfg = _cast_config_types({**fixed_context, **cfg})
    model = _load_model_with_lora(
        model_id=str(run_cfg["MODEL_ID"]),
        device=str(run_cfg["DEVICE"]),
        torch_dtype=str(run_cfg["torch_dtype"]),
        config=run_cfg,
    )
    counts = get_param_counts(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return counts


def _run_one_ablation(
    cfg: dict[str, Any],
    fixed_context: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, float]], pd.DataFrame]:
    """Train and evaluate one LoRA ablation config."""
    run_cfg = _cast_config_types({**fixed_context, **cfg})
    processor = AutoProcessor.from_pretrained(str(run_cfg["MODEL_ID"]))
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = _load_model_with_lora(
        model_id=str(run_cfg["MODEL_ID"]),
        device=str(run_cfg["DEVICE"]),
        torch_dtype=str(run_cfg["torch_dtype"]),
        config=run_cfg,
    )
    params = get_param_counts(model)
    t0 = time.time()
    objective = str(run_cfg["training_objective"])
    if objective == "sft_next_token":
        _em = run_cfg.get("early_stopping_metric", "accuracy")
        es_metric = str("accuracy" if _em is None else _em).strip().lower()
        history = train_sft_objective(
            model=model,
            processor=processor,
            train_df=fixed_context["train_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
            val_df=fixed_context["val_df"],
            early_stopping=run_cfg.get("early_stopping"),
            epochs_per_val_check=run_cfg.get("epochs_per_val_check"),
            early_stopping_val_examples=run_cfg.get("early_stopping_val_examples"),
        )
        val_metrics, val_pred_df = evaluate_sft_generation(
            model=model,
            processor=processor,
            df=fixed_context["val_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
            compute_eval_loss=(es_metric == "loss"),
            compute_generation=(es_metric == "accuracy"),
        )
        val_metrics.setdefault("accuracy", float("nan"))
        val_metrics.setdefault("percent_correct", float("nan"))
        val_metrics.setdefault("eval_loss", float("nan"))
    elif objective == "option_scoring":
        history = train_option_scoring_objective(
            model=model,
            processor=processor,
            train_df=fixed_context["train_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
            val_df=fixed_context["val_df"],
            early_stopping=run_cfg.get("early_stopping"),
            epochs_per_val_check=run_cfg.get("epochs_per_val_check"),
            early_stopping_val_examples=run_cfg.get("early_stopping_val_examples"),
        )
        val_metrics, val_pred_df = evaluate_option_scoring(
            model=model,
            processor=processor,
            df=fixed_context["val_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
        )
    else:
        raise ValueError("training_objective must be 'sft_next_token' or 'option_scoring'.")
    elapsed_s = float(time.time() - t0)
    adapter_dir: Path | None = None
    if bool(fixed_context.get("save_trained_adapters", False)):
        adapter_root = Path(fixed_context.get("ADAPTER_DIR", Path("outputs") / "adapters"))
        adapter_dir = adapter_root / str(cfg["ablation_id"])
        maybe_save_model(model, adapter_dir, save=True)
    summary = {
        **cfg,
        **val_metrics,
        "ablation_id": cfg["ablation_id"],
        "trainable_params": int(params["trainable"]),
        "total_params": int(params["total"]),
        "trainable_pct": float(params["trainable_pct"]),
        "train_seconds": elapsed_s,
    }
    if adapter_dir is not None:
        summary["adapter_dir"] = str(adapter_dir)
    return summary, history, val_pred_df


def run_single_lora_ablation(
    cfg: dict[str, Any],
    fixed_context: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, float]], pd.DataFrame]:
    """Train and validate one LoRA candidate; public wrapper for notebooks and full-run utilities."""
    return _run_one_ablation(cfg=cfg, fixed_context=fixed_context)


def top_configs(results_df: pd.DataFrame, k: int = 5) -> list[dict[str, Any]]:
    """Return top-k rows sorted by accuracy and parse quality where available."""
    if len(results_df) == 0:
        return []
    if "parse_failure_rate" in results_df.columns:
        sort_cols = ["accuracy", "parse_failure_rate"]
        ascending = [False, True]
    else:
        sort_cols = ["accuracy"]
        ascending = [False]
    return results_df.sort_values(sort_cols, ascending=ascending).head(int(k)).to_dict(orient="records")


def run_lora_tuning_block(
    block_name: str,
    all_configs: list[dict[str, Any]],
    selected_ablation_ids: list[str] | None,
    fixed_context: dict[str, Any],
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Run train+eval for each config in a tuning block."""
    configs_to_run = select_ablation_configs(all_configs, selected_ablation_ids)
    all_results: list[dict[str, Any]] = []
    all_histories: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []
    print(f"\n=== {block_name} ===")
    print("total block configs:", len(all_configs))
    print("configs to run now:", len(configs_to_run))
    for cfg in configs_to_run:
        print(f"\nRunning {cfg['ablation_id']} ...")
        summary, history, val_pred_df = run_single_lora_ablation(cfg=cfg, fixed_context=fixed_context)
        all_results.append(summary)
        hist_df = pd.DataFrame(history)
        if len(hist_df):
            hist_df["ablation_id"] = cfg["ablation_id"]
            all_histories.append(hist_df.to_dict(orient="records"))
        val_pred_df = val_pred_df.copy()
        val_pred_df["ablation_id"] = cfg["ablation_id"]
        all_preds.append(val_pred_df)
        if fixed_context.get("clear_cuda_between_runs", True) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    histories_df = pd.DataFrame([r for rows in all_histories for r in rows]) if all_histories else pd.DataFrame()
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if len(results_df):
        if "parse_failure_rate" in results_df.columns:
            results_df = results_df.sort_values(["accuracy", "parse_failure_rate"], ascending=[False, True]).reset_index(drop=True)
        else:
            results_df = results_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    top_cfgs = top_configs(results_df, k=top_k)
    return results_df, histories_df, preds_df, top_cfgs


def build_display_tables(
    all_configs: list[dict[str, Any]],
    results_df: pd.DataFrame,
    sweep_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build config table and concise metrics table for display."""
    cfg_cols = ["ablation_id"] + [k for k in sweep_keys if any(k in c for c in all_configs)]
    cfg_table = pd.DataFrame(all_configs)[cfg_cols] if len(all_configs) else pd.DataFrame()

    metric_candidates = [
        "ablation_id",
        "percent_correct",
        "accuracy",
        "parse_failure_rate",
        "train_seconds",
        "trainable_params",
    ] + [k for k in sweep_keys if k in results_df.columns]
    metric_cols = [c for c in metric_candidates if c in results_df.columns]
    metrics_table = results_df[metric_cols].copy() if len(results_df) else pd.DataFrame(columns=metric_cols)
    return cfg_table, metrics_table


def save_block_artifacts(
    out_dir: Path,
    block_prefix: str,
    save: bool,
    results_df: pd.DataFrame,
    histories_df: pd.DataFrame,
    val_predictions_df: pd.DataFrame,
    top_cfgs: list[dict[str, Any]],
) -> None:
    """Persist per-block artifacts when save=True."""
    if not save:
        return
    maybe_save_dataframe(results_df, out_dir / f"{block_prefix}_results.csv", save=save)
    maybe_save_dataframe(histories_df, out_dir / f"{block_prefix}_history.csv", save=save)
    maybe_save_dataframe(val_predictions_df, out_dir / f"{block_prefix}_predictions.csv", save=save)
    maybe_save_dataframe(
        build_failure_examples(val_predictions_df, max_examples=200),
        out_dir / f"{block_prefix}_failure_examples.csv",
        save=save,
    )
    maybe_save_json({"top_configs": top_cfgs}, out_dir / f"{block_prefix}_top_configs.json", save=save)


def run_test_submission(
    best_config: dict[str, Any],
    fixed_context: dict[str, Any],
    save: bool,
    out_dir: Path,
    filename_prefix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run best config on test split and return predictions + submission.

    If filename_prefix is non-empty (e.g. ``'generative_'``), saves as
    ``{prefix}test_predictions.csv`` and ``{prefix}submission.csv``; otherwise
    uses ``test_predictions.csv`` and ``submission.csv``.
    """
    run_cfg = _cast_config_types({**fixed_context, **best_config})
    processor = AutoProcessor.from_pretrained(str(run_cfg["MODEL_ID"]))
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = _load_model_with_lora(
        model_id=str(run_cfg["MODEL_ID"]),
        device=str(run_cfg["DEVICE"]),
        torch_dtype=str(run_cfg["torch_dtype"]),
        config=run_cfg,
    )
    objective = str(run_cfg["training_objective"])
    if objective == "sft_next_token":
        _ = train_sft_objective(
            model=model,
            processor=processor,
            train_df=fixed_context["train_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
        )
        _, test_pred_df = evaluate_sft_generation(
            model=model,
            processor=processor,
            df=fixed_context["test_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
            compute_eval_loss=False,
        )
    else:
        _ = train_option_scoring_objective(
            model=model,
            processor=processor,
            train_df=fixed_context["train_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
        )
        _, test_pred_df = evaluate_option_scoring(
            model=model,
            processor=processor,
            df=fixed_context["test_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
        )
    submission_df = make_submission(test_pred_df)
    validate_submission(submission_df)
    if save:
        pred_name = f"{filename_prefix}test_predictions.csv" if filename_prefix else "test_predictions.csv"
        sub_name = f"{filename_prefix}submission.csv" if filename_prefix else "submission.csv"
        adapter_dir = (out_dir / f"{filename_prefix}adapter") if filename_prefix else (out_dir / "adapter")
        maybe_save_dataframe(test_pred_df, out_dir / pred_name, save=save)
        maybe_save_dataframe(submission_df, out_dir / sub_name, save=save)
        maybe_save_model(model, adapter_dir, save=save)
    return test_pred_df, submission_df


def run_test_submission_from_saved_adapter(
    config: dict[str, Any],
    fixed_context: dict[str, Any],
    adapter_dir: Path,
    save: bool,
    out_dir: Path,
    filename_prefix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run test submission from a saved adapter without retraining."""
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    run_cfg = _cast_config_types({**fixed_context, **config})
    processor = AutoProcessor.from_pretrained(str(run_cfg["MODEL_ID"]))
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_class = _resolve_model_class()
    dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}
    selected_dtype = dtype_map.get(str(run_cfg["torch_dtype"]), "auto")
    base_model = model_class.from_pretrained(str(run_cfg["MODEL_ID"]), torch_dtype=selected_dtype).to(str(run_cfg["DEVICE"]))
    model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(str(run_cfg["DEVICE"]))

    objective = str(run_cfg["training_objective"])
    if objective == "sft_next_token":
        _, test_pred_df = evaluate_sft_generation(
            model=model,
            processor=processor,
            df=fixed_context["test_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
            compute_eval_loss=False,
        )
    elif objective == "option_scoring":
        _, test_pred_df = evaluate_option_scoring(
            model=model,
            processor=processor,
            df=fixed_context["test_df"],
            images_root=Path(fixed_context["IMAGE_DIR"]),
            device=str(run_cfg["DEVICE"]),
            config=run_cfg,
        )
    else:
        raise ValueError("training_objective must be 'sft_next_token' or 'option_scoring'.")

    submission_df = make_submission(test_pred_df)
    validate_submission(submission_df)
    if save:
        pred_name = f"{filename_prefix}test_predictions.csv" if filename_prefix else "test_predictions.csv"
        sub_name = f"{filename_prefix}submission.csv" if filename_prefix else "submission.csv"
        maybe_save_dataframe(test_pred_df, out_dir / pred_name, save=save)
        maybe_save_dataframe(submission_df, out_dir / sub_name, save=save)
    return test_pred_df, submission_df


__all__ = [
    "BLOCK_SWEEP_KEYS",
    "build_block_configs",
    "build_display_tables",
    "count_trainable_params_for_config",
    "run_lora_tuning_block",
    "run_single_lora_ablation",
    "run_test_submission",
    "run_test_submission_from_saved_adapter",
    "save_block_artifacts",
    "select_ablation_configs",
    "top_configs",
]
