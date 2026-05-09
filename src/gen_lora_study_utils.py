"""Utilities for generative LoRA study notebooks."""

from __future__ import annotations

import ast
import inspect
import json
import math
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch import nn
from tqdm.auto import tqdm
from src.input_ablation_utils import (
    apply_image_transform,
    build_metadata_prefix,
    build_prompt_ablation_overrides,
    resolve_split_image_path,
)


def maybe_mount_colab_drive(mount_path: str = "/content/drive") -> bool:
    """Attempt to mount Google Drive when running inside Colab."""
    try:
        from google.colab import drive  # type: ignore

        drive.mount(mount_path, force_remount=False)
        return True
    except Exception:
        return False


def detect_device() -> str:
    """Return the preferred torch device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    """Load one split CSV and normalize expected text columns."""
    path = data_dir / f"{split}.csv"
    df = pd.read_csv(path)
    df["split"] = split
    for col in ["question", "hint", "lecture"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def apply_sanity_subset(df: pd.DataFrame, sanity_check: bool, n: int, seed: int) -> pd.DataFrame:
    """Return deterministic subset for quick debug loops."""
    if not sanity_check:
        return df.copy().reset_index(drop=True)
    n = max(1, min(int(n), len(df)))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def cap_validation_rows(df: pd.DataFrame, max_validation_samples: int | None) -> pd.DataFrame:
    """Cap validation rows while preserving order."""
    if max_validation_samples is None:
        return df.copy().reset_index(drop=True)
    n = max(1, min(int(max_validation_samples), len(df)))
    return df.iloc[:n].copy().reset_index(drop=True)


def parse_choices(choices_raw: Any) -> list[str]:
    """Parse list-like choice data from CSV into list[str]."""
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


def _letter_for_index(idx: int) -> str:
    return chr(ord("A") + idx)


def build_choice_lines(choices: list[str], choice_format: str) -> list[str]:
    """Render answer choices in selected format."""
    lines: list[str] = []
    for i, choice in enumerate(choices):
        letter = _letter_for_index(i)
        if choice_format == "letter_paren":
            lines.append(f"({letter}) {choice}")
        elif choice_format == "option_letter":
            lines.append(f"Option {letter}: {choice}")
        else:
            lines.append(f"{letter}. {choice}")
    return lines


def _build_context_lines(row: pd.Series, context_mode: str) -> list[str]:
    hint = str(row.get("hint", "")).strip()
    lecture = str(row.get("lecture", "")).strip()
    lines: list[str] = []
    if context_mode in {"hint_only", "hint_lecture"} and hint:
        lines.extend(["Hint:", hint, ""])
    if context_mode in {"lecture_only", "hint_lecture"} and lecture:
        lines.extend(["Lecture:", lecture, ""])
    return lines


def build_prompt(
    row: pd.Series,
    prompt_structure: str,
    context_mode: str,
    choice_format: str,
    output_format: str,
    prompt_overrides: dict[str, Any] | None = None,
) -> str:
    """Build prompt text for generative training and inference."""
    prompt_overrides = prompt_overrides or {}
    question = str(row.get("question", "")).strip()
    choices = parse_choices(row.get("choices", "[]"))
    choice_lines = build_choice_lines(choices, choice_format)
    context_lines = _build_context_lines(row, context_mode)
    metadata_prefix = build_metadata_prefix(row=row, config=prompt_overrides)
    phrasing = build_prompt_ablation_overrides(prompt_overrides)
    question_label = str(phrasing.get("question_label", "Question:"))
    choices_label = str(phrasing.get("choices_label", "Choices:"))
    instruction_prefix = str(phrasing.get("instruction_prefix", ""))

    if output_format == "answer_prefix":
        answer_prefix_text = str(phrasing.get("answer_prefix_text", "Answer:"))
        output_instruction = f"Return your answer as: {answer_prefix_text} <LETTER>."
    elif output_format == "reason_then_final":
        output_instruction = "Reason briefly, then end with: Final Answer: <LETTER>."
    else:
        output_instruction = "Return only one letter from the answer choices."
    if instruction_prefix:
        output_instruction = f"{instruction_prefix}\n{output_instruction}"

    question_line = f"{question_label} {question}"

    if prompt_structure == "explicit_instruction":
        sections = [
            "You are solving a visual multiple-choice question.",
            output_instruction,
            "",
            metadata_prefix,
            question_line,
            "",
            *context_lines,
            choices_label,
            *choice_lines,
        ]
    elif prompt_structure == "question_first":
        sections = [
            metadata_prefix,
            question_line,
            "",
            *context_lines,
            choices_label,
            *choice_lines,
            "",
            output_instruction,
        ]
    else:
        sections = [metadata_prefix, question, "", *context_lines, choices_label, *choice_lines, "", output_instruction]
    return "\n".join([x for x in sections if x is not None])


def build_target_text(answer_idx: int, output_format: str, prompt_overrides: dict[str, Any] | None = None) -> str:
    """Build target answer string used by SFT objective."""
    phrasing = build_prompt_ablation_overrides(prompt_overrides or {})
    letter = _letter_for_index(int(answer_idx))
    if output_format == "answer_prefix":
        return f"{str(phrasing.get('answer_prefix_text', 'Answer:'))} {letter}"
    return letter


def resolve_image_path(images_root: Path, row: pd.Series, split_name: str | None = None) -> Path:
    """Resolve image paths from numeric references under images/images/<split>/."""
    return resolve_split_image_path(images_root=images_root, row=row, split_name=split_name)


def _load_image_with_ablation(
    images_root: Path,
    row: pd.Series,
    config: dict[str, Any],
    is_train: bool,
) -> Image.Image:
    """Load a row image and apply optional resolution/augmentation transforms."""
    image_path = resolve_image_path(images_root=images_root, row=row, split_name=row.get("split", None))
    image = Image.open(image_path).convert("RGB")
    return apply_image_transform(image=image, config=config, is_train=is_train)


def build_lora_model(
    backbone: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    use_dora: bool = False,
    use_rslora: bool = False,
) -> nn.Module:
    """Attach LoRA adapters to a pretrained backbone."""
    try:
        peft_task_type = TaskType[str(task_type).upper().strip()]
    except KeyError as exc:
        allowed = ", ".join([t.name for t in TaskType])
        raise ValueError(f"Unsupported task_type '{task_type}'. Allowed: {allowed}") from exc

    base_kwargs: dict[str, Any] = {
        "r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "target_modules": list(target_modules),
        "bias": str(bias),
        "task_type": peft_task_type,
    }
    signature = inspect.signature(LoraConfig.__init__)
    supported = set(signature.parameters.keys())
    if bool(use_dora):
        if "use_dora" not in supported:
            raise ValueError("Installed peft version does not support DoRA (use_dora).")
        base_kwargs["use_dora"] = True
    if bool(use_rslora):
        if "use_rslora" not in supported:
            raise ValueError("Installed peft version does not support RsLoRA (use_rslora).")
        base_kwargs["use_rslora"] = True
    config = LoraConfig(**base_kwargs)
    return get_peft_model(backbone, config)


def get_param_counts(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_pct": float(100.0 * trainable / max(1, total)),
    }


def _iter_row_batches(df: pd.DataFrame, batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def _get_tokenizer_padding_side(processor: Any) -> str:
    """Return tokenizer padding side, defaulting to right if unavailable."""
    tok = getattr(processor, "tokenizer", None)
    side = getattr(tok, "padding_side", "right") if tok is not None else "right"
    return "left" if str(side).lower() == "left" else "right"


def _mask_prompt_tokens_for_sft(
    *,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_lens: torch.Tensor | list[int],
    processor: Any,
) -> torch.Tensor:
    """Mask prompt and pad tokens for causal-LM SFT."""
    out = labels.clone()

    if isinstance(prompt_lens, list):
        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=out.device)
    else:
        prompt_lens_t = prompt_lens.to(device=out.device, dtype=torch.long)

    batch_size, seq_len = out.shape
    padding_side = _get_tokenizer_padding_side(processor)

    if attention_mask is not None:
        attn = attention_mask.to(device=out.device)
        full_lens = attn.sum(dim=1).to(dtype=torch.long)
    else:
        attn = None
        full_lens = torch.full(
            (batch_size,),
            fill_value=seq_len,
            dtype=torch.long,
            device=out.device,
        )

    for i in range(batch_size):
        if padding_side == "left":
            prompt_start = int(seq_len - full_lens[i].item())
        else:
            prompt_start = 0

        prompt_end = prompt_start + int(prompt_lens_t[i].item())
        prompt_end = max(0, min(prompt_end, seq_len))
        out[i, :prompt_end] = -100

    if attn is not None:
        out[attn == 0] = -100

    return out


def _render_chat_prompt(processor: Any, prompt: str, add_generation_prompt: bool) -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
    return prompt


def parse_letter_prediction(raw_output: str, num_choices: int, parse_rule: str = "strict_first_letter") -> tuple[int, str]:
    """Parse model output into a 0-based index."""
    text = str(raw_output).strip()
    letters = [_letter_for_index(i) for i in range(max(1, int(num_choices)))]
    rule = str(parse_rule or "strict_first_letter").strip().lower()

    if rule == "answer_prefix":
        match = re.search(r"answer\s*:\s*([A-Z])", text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in letters:
                return letters.index(letter), "ok"

    if rule == "strict_first_letter":
        match = re.match(r"^\s*([A-Z])(?:\b|[\.)\]:-])", text.upper())
        if match:
            letter = match.group(1).upper()
            if letter in letters:
                return letters.index(letter), "ok"
        return 0, "parse_failure"

    match = re.search(r"\b([A-Z])\b", text.upper())
    if match and match.group(1) in letters:
        return letters.index(match.group(1)), "ok"
    return 0, "parse_failure"


def _early_stopping_validation_step(
    *,
    val_df: pd.DataFrame | None,
    epoch: int,
    epochs_per_val_check: int | None,
    patience: int | None,
    best_metric_value: float,
    no_improve_streak: int,
    history_row: dict[str, float],
    evaluate_fn: Callable[[], tuple[dict[str, Any], pd.DataFrame]],
    early_stopping_metric: str = "accuracy",
) -> tuple[float, int, bool]:
    """Periodic validation check for early stopping.

    When ``epoch`` is a multiple of ``epochs_per_val_check``, runs ``evaluate_fn``
    and records ``val_accuracy`` and ``val_loss`` on ``history_row`` when present.
    Stops after ``patience`` consecutive checks without strictly improving the
    monitored metric: higher is better for ``accuracy``, lower for ``loss``.

    Returns ``(best_metric_value, no_improve_streak, break_training)``.
    """
    if epochs_per_val_check is None or patience is None or val_df is None or len(val_df) == 0:
        return best_metric_value, no_improve_streak, False
    if epoch % int(epochs_per_val_check) != 0:
        return best_metric_value, no_improve_streak, False
    metrics, _ = evaluate_fn()
    acc = float(metrics.get("accuracy", 0.0))
    history_row["val_accuracy"] = acc
    eloss = metrics.get("eval_loss")
    if eloss is not None and not (isinstance(eloss, float) and math.isnan(eloss)):
        history_row["val_loss"] = float(eloss)

    m = (early_stopping_metric or "accuracy").strip().lower()
    if m not in ("accuracy", "loss"):
        raise ValueError("early_stopping_metric must be 'accuracy' or 'loss'")
    monitored_value = float(eloss) if (m == "loss" and eloss is not None and not (isinstance(eloss, float) and math.isnan(eloss))) else float(acc)
    tqdm.write(f"[Early stopping check] metric={m}, value={monitored_value:.6f}, epoch={int(epoch)}")

    if m == "loss":
        if eloss is None or (isinstance(eloss, float) and math.isnan(eloss)):
            return best_metric_value, no_improve_streak, False
        lv = float(eloss)
        if lv < best_metric_value - 1e-12:
            return lv, 0, False
    else:
        if acc > best_metric_value + 1e-12:
            return acc, 0, False

    new_streak = no_improve_streak + 1
    if new_streak >= int(patience):
        history_row["stopped_early"] = 1.0
        return best_metric_value, new_streak, True
    return best_metric_value, new_streak, False


def select_epoch_train_df(
    train_df: pd.DataFrame,
    epoch_idx: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Return training rows for one epoch, optionally resampled or shuffled.

    When ``enable_epoch_resampling`` is false or missing, returns ``train_df`` unchanged.
    Otherwise selects ``epoch_train_size`` rows or ``epoch_train_fraction`` of the full
    set (exactly one must be set when resampling is enabled), with optional stratification
    and deterministic RNG ``epoch_resampling_seed + epoch_idx``.
    """
    if not bool(config.get("enable_epoch_resampling", False)):
        return train_df

    mode = str(config.get("epoch_resampling_mode", "without_replacement")).strip().lower()
    if mode not in ("without_replacement", "with_replacement"):
        raise ValueError(
            "epoch_resampling_mode must be 'without_replacement' or 'with_replacement', "
            f"got {config.get('epoch_resampling_mode')!r}"
        )
    replace = mode == "with_replacement"

    n_full = len(train_df)
    if n_full == 0:
        return train_df

    frac = config.get("epoch_train_fraction")
    size_kw = config.get("epoch_train_size")
    if frac is not None and size_kw is not None:
        raise ValueError("Set at most one of epoch_train_fraction and epoch_train_size when epoch resampling is enabled.")
    if frac is None and size_kw is None:
        raise ValueError("When enable_epoch_resampling is True, set epoch_train_fraction or epoch_train_size.")

    if frac is not None:
        f = float(frac)
        if not 0 < f <= 1:
            raise ValueError(f"epoch_train_fraction must be in (0, 1], got {f}")
        n = max(1, int(round(f * n_full)))
    else:
        n = int(size_kw)  # type: ignore[arg-type]
        if n < 1:
            raise ValueError(f"epoch_train_size must be >= 1, got {n}")

    if not replace and n > n_full:
        raise ValueError(
            f"epoch_train_size={n} exceeds len(train_df)={n_full} with without_replacement sampling."
        )

    seed = int(config.get("epoch_resampling_seed", 0))
    rng = np.random.RandomState(seed + int(epoch_idx))

    strat_col = config.get("epoch_resampling_stratify_col")
    if strat_col is not None and str(strat_col).strip():
        col = str(strat_col)
        if col not in train_df.columns:
            raise ValueError(f"epoch_resampling_stratify_col={col!r} not found in train_df columns.")

        if replace:
            props = train_df[col].value_counts(normalize=True)
            weights = train_df[col].map(props).astype(float).to_numpy()
            weights = weights / weights.sum()
            idx = rng.choice(train_df.index.to_numpy(), size=n, replace=True, p=weights)
            return train_df.loc[idx].reset_index(drop=True)

        sizes = train_df.groupby(col, sort=False).size()
        k = int(n)
        exact = sizes * (k / float(n_full))
        alloc = exact.apply(lambda x: int(math.floor(x)))
        remainder = (exact - alloc).sort_values(ascending=False)
        deficit = int(k - int(alloc.sum()))
        for lab in remainder.index:
            if deficit <= 0:
                break
            cap = int(sizes.loc[lab])
            if int(alloc.loc[lab]) < cap:
                alloc.loc[lab] = int(alloc.loc[lab]) + 1
                deficit -= 1
        if deficit > 0:
            spare = sizes - alloc
            spare = spare[spare > 0].sort_values(ascending=False)
            for lab in spare.index:
                while deficit > 0 and int(alloc.loc[lab]) < int(sizes.loc[lab]):
                    alloc.loc[lab] = int(alloc.loc[lab]) + 1
                    deficit -= 1
                if deficit <= 0:
                    break

        parts: list[pd.DataFrame] = []
        for lab, want in alloc.items():
            if int(want) <= 0:
                continue
            g = train_df[train_df[col] == lab]
            take = min(int(want), len(g))
            if take > 0:
                parts.append(
                    g.sample(
                        n=take,
                        replace=False,
                        random_state=int(rng.randint(0, 2**31 - 1)),
                    )
                )
        out = pd.concat(parts, axis=0) if parts else pd.DataFrame(columns=train_df.columns)
        shortfall = k - len(out)
        if shortfall > 0:
            used = set(out.index)
            pool = train_df.loc[~train_df.index.isin(used)]
            if len(pool) < shortfall:
                raise ValueError("Stratified epoch sample could not reach target size; check strata sizes.")
            extra = pool.sample(
                n=shortfall,
                replace=False,
                random_state=int(rng.randint(0, 2**31 - 1)),
            )
            out = pd.concat([out, extra], axis=0)
        elif len(out) > k:
            out = out.sample(
                n=k,
                replace=False,
                random_state=int(rng.randint(0, 2**31 - 1)),
            )
        return out.sample(frac=1.0, random_state=int(rng.randint(0, 2**31 - 1))).reset_index(drop=True)

    if n >= n_full and not replace:
        return train_df.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)

    return train_df.sample(n=n, replace=replace, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)


def train_sft_objective(
    model: nn.Module,
    processor: Any,
    train_df: pd.DataFrame,
    images_root: Path,
    device: str,
    config: dict[str, Any],
    val_df: pd.DataFrame | None = None,
    early_stopping: int | None = None,
    epochs_per_val_check: int | None = None,
    early_stopping_val_examples: int | None = None,
) -> list[dict[str, float]]:
    """Train LoRA adapters with next-token supervised fine-tuning."""
    model.train()
    lr = float(config["lr"])
    epochs = int(config["epochs"])
    weight_decay = float(config["weight_decay"])
    grad_accum = max(1, int(config["gradient_accumulation_steps"]))
    max_grad_norm = float(config["max_grad_norm"])
    batch_size = int(config["batch_size"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    patience_set = early_stopping is not None
    interval_set = epochs_per_val_check is not None
    if patience_set != interval_set:
        raise ValueError(
            "early_stopping (patience) and epochs_per_val_check must both be set to enable early stopping, or both None."
        )
    val_interval: int | None = None
    patience: int | None = None
    if patience_set and interval_set:
        patience = int(early_stopping)  # type: ignore[arg-type]
        val_interval = int(epochs_per_val_check)  # type: ignore[arg-type]
        if patience < 1 or val_interval < 1:
            raise ValueError("early_stopping and epochs_per_val_check must be integers >= 1 when set.")

    val_es_df: pd.DataFrame | None = None
    if val_df is not None and patience_set and interval_set:
        val_es_df = (
            cap_validation_rows(val_df, early_stopping_val_examples)
            if early_stopping_val_examples is not None
            else val_df
        )

    _em = config.get("early_stopping_metric", "accuracy")
    es_metric = str("accuracy" if _em is None else _em).strip().lower()
    if es_metric not in ("accuracy", "loss"):
        raise ValueError("early_stopping_metric must be 'accuracy' or 'loss'")
    best_metric_value = float("inf") if es_metric == "loss" else -1.0
    no_improve_streak = 0

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        epoch_train_df = select_epoch_train_df(train_df, epoch, config)
        if bool(config.get("enable_epoch_resampling", False)):
            tqdm.write(
                f"[epoch {epoch}/{epochs}] epoch_train subset: {len(epoch_train_df)}/{len(train_df)} rows "
                f"(resampling_seed+epoch={int(config.get('epoch_resampling_seed', 0)) + epoch})"
            )
            sc = config.get("epoch_resampling_stratify_col")
            if sc and str(sc).strip() and str(sc) in epoch_train_df.columns:
                tqdm.write(f"  stratify {sc!r} counts: {epoch_train_df[str(sc)].value_counts().head(12).to_dict()}")

        running_loss = 0.0
        total_examples = 0
        optimizer.zero_grad(set_to_none=True)
        num_batches = (len(epoch_train_df) + batch_size - 1) // batch_size
        iter_batches = tqdm(
            _iter_row_batches(epoch_train_df, batch_size),
            total=num_batches,
            desc=f"SFT epoch {epoch}/{epochs}",
        )
        for step_idx, batch_df in enumerate(iter_batches, start=1):
            prompts: list[str] = []
            targets: list[str] = []
            images: list[Image.Image] = []
            for _, row in batch_df.iterrows():
                prompt = build_prompt(
                    row=row,
                    prompt_structure=config["prompt_structure"],
                    context_mode=config["context_mode"],
                    choice_format=config["choice_format"],
                    output_format=config["output_format"],
                    prompt_overrides=config,
                )
                target = build_target_text(int(row["answer"]), output_format=config["output_format"], prompt_overrides=config)
                images.append(_load_image_with_ablation(images_root=images_root, row=row, config=config, is_train=True))
                prompts.append(_render_chat_prompt(processor, prompt, add_generation_prompt=True))
                targets.append(target)

            full_text = [f"{p}{t}" for p, t in zip(prompts, targets)]
            prompt_inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            full_inputs = processor(text=full_text, images=images, padding=True, return_tensors="pt")
            prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)

            inputs = {k: v.to(device) for k, v in full_inputs.items()}
            labels = _mask_prompt_tokens_for_sft(
                labels=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                prompt_lens=prompt_lens,
                processor=processor,
            )

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step_idx % grad_accum == 0) or (step_idx == num_batches):
                if max_grad_norm > 0:
                    params = [p for p in model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(outputs.loss.item()) * len(batch_df)
            total_examples += len(batch_df)

        history_row = {
            "epoch": float(epoch),
            "train_loss": float(running_loss / max(1, total_examples)),
        }
        history.append(history_row)
        best_metric_value, no_improve_streak, do_break = _early_stopping_validation_step(
            val_df=val_es_df,
            epoch=epoch,
            epochs_per_val_check=val_interval,
            patience=patience,
            best_metric_value=best_metric_value,
            no_improve_streak=no_improve_streak,
            history_row=history_row,
            evaluate_fn=lambda: evaluate_sft_generation(
                model=model,
                processor=processor,
                df=val_es_df,  # type: ignore[arg-type]
                images_root=images_root,
                device=device,
                config=config,
                compute_eval_loss=(es_metric == "loss"),
                compute_generation=(es_metric == "accuracy"),
            ),
            early_stopping_metric=es_metric,
        )
        model.train()
        if do_break:
            break
    return history


def _candidate_text(letter: str, output_format: str, prompt_overrides: dict[str, Any] | None = None) -> str:
    phrasing = build_prompt_ablation_overrides(prompt_overrides or {})
    if output_format == "answer_prefix":
        return f"{str(phrasing.get('answer_prefix_text', 'Answer:'))} {letter}"
    return letter


def _candidate_logprob_scores(
    model: nn.Module,
    processor: Any,
    prompt: str,
    image: Image.Image,
    num_choices: int,
    output_format: str,
    device: str,
    prompt_overrides: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Return differentiable candidate scores for one example."""
    n = int(num_choices)
    if n < 1:
        raise ValueError("num_choices must be >= 1")

    rendered_prompt = _render_chat_prompt(
        processor,
        prompt,
        add_generation_prompt=True,
    )

    prompt_inputs = processor(
        text=rendered_prompt,
        images=[image],
        return_tensors="pt",
    )
    prompt_len = int(prompt_inputs["attention_mask"].sum(dim=1).item())

    candidate_texts = [
        _candidate_text(
            _letter_for_index(idx),
            output_format=output_format,
            prompt_overrides=prompt_overrides,
        )
        for idx in range(n)
    ]
    full_texts = [f"{rendered_prompt}{cand}" for cand in candidate_texts]

    full_inputs = processor(
        text=full_texts,
        images=[image] * n,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in full_inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    batch_size, seq_len = input_ids.shape
    padding_side = _get_tokenizer_padding_side(processor)

    if attention_mask is not None:
        full_lens = attention_mask.sum(dim=1).to(dtype=torch.long)
    else:
        full_lens = torch.full(
            (batch_size,),
            fill_value=seq_len,
            dtype=torch.long,
            device=input_ids.device,
        )

    scores: list[torch.Tensor] = []
    for i in range(n):
        full_len_i = int(full_lens[i].item())
        if padding_side == "left":
            first_real_token = seq_len - full_len_i
        else:
            first_real_token = 0

        cand_start_input_pos = first_real_token + prompt_len
        cand_end_input_pos_exclusive = first_real_token + full_len_i

        score_start = max(cand_start_input_pos - 1, 0)
        score_end = max(cand_end_input_pos_exclusive - 1, score_start)
        scores.append(token_log_probs[i, score_start:score_end].sum())

    return torch.stack(scores, dim=0).unsqueeze(0)


def train_option_scoring_objective(
    model: nn.Module,
    processor: Any,
    train_df: pd.DataFrame,
    images_root: Path,
    device: str,
    config: dict[str, Any],
    val_df: pd.DataFrame | None = None,
    early_stopping: int | None = None,
    epochs_per_val_check: int | None = None,
    early_stopping_val_examples: int | None = None,
) -> list[dict[str, float]]:
    """Train LoRA adapters by maximizing score of the gold answer option."""
    model.train()
    lr = float(config["lr"])
    epochs = int(config["epochs"])
    weight_decay = float(config["weight_decay"])
    max_grad_norm = float(config["max_grad_norm"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    patience_set = early_stopping is not None
    interval_set = epochs_per_val_check is not None
    if patience_set != interval_set:
        raise ValueError(
            "early_stopping (patience) and epochs_per_val_check must both be set to enable early stopping, or both None."
        )
    val_interval: int | None = None
    patience: int | None = None
    if patience_set and interval_set:
        patience = int(early_stopping)  # type: ignore[arg-type]
        val_interval = int(epochs_per_val_check)  # type: ignore[arg-type]
        if patience < 1 or val_interval < 1:
            raise ValueError("early_stopping and epochs_per_val_check must be integers >= 1 when set.")

    val_es_df: pd.DataFrame | None = None
    if val_df is not None and patience_set and interval_set:
        val_es_df = (
            cap_validation_rows(val_df, early_stopping_val_examples)
            if early_stopping_val_examples is not None
            else val_df
        )

    _em = config.get("early_stopping_metric", "accuracy")
    es_metric = str("accuracy" if _em is None else _em).strip().lower()
    if es_metric not in ("accuracy", "loss"):
        raise ValueError("early_stopping_metric must be 'accuracy' or 'loss'")
    best_metric_value = float("inf") if es_metric == "loss" else -1.0
    no_improve_streak = 0

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        epoch_train_df = select_epoch_train_df(train_df, epoch, config)
        if bool(config.get("enable_epoch_resampling", False)):
            tqdm.write(
                f"[epoch {epoch}/{epochs}] epoch_train subset: {len(epoch_train_df)}/{len(train_df)} rows "
                f"(resampling_seed+epoch={int(config.get('epoch_resampling_seed', 0)) + epoch})"
            )
            sc = config.get("epoch_resampling_stratify_col")
            if sc and str(sc).strip() and str(sc) in epoch_train_df.columns:
                tqdm.write(f"  stratify {sc!r} counts: {epoch_train_df[str(sc)].value_counts().head(12).to_dict()}")

        running_loss = 0.0
        total_examples = 0
        row_iter = tqdm(epoch_train_df.iterrows(), total=len(epoch_train_df), desc=f"Score epoch {epoch}/{epochs}")
        for _, row in row_iter:
            prompt = build_prompt(
                row=row,
                prompt_structure=config["prompt_structure"],
                context_mode=config["context_mode"],
                choice_format=config["choice_format"],
                output_format=config["output_format"],
                prompt_overrides=config,
            )
            image = _load_image_with_ablation(images_root=images_root, row=row, config=config, is_train=True)
            scores = _candidate_logprob_scores(
                model=model,
                processor=processor,
                prompt=prompt,
                image=image,
                num_choices=int(row["num_choices"]),
                output_format=config["output_format"],
                device=device,
                prompt_overrides=config,
            )
            label = torch.tensor([int(row["answer"])], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(scores, label)
            loss.backward()
            if max_grad_norm > 0:
                params = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            total_examples += 1

        history_row = {"epoch": float(epoch), "train_loss": float(running_loss / max(1, total_examples))}
        history.append(history_row)
        best_metric_value, no_improve_streak, do_break = _early_stopping_validation_step(
            val_df=val_es_df,
            epoch=epoch,
            epochs_per_val_check=val_interval,
            patience=patience,
            best_metric_value=best_metric_value,
            no_improve_streak=no_improve_streak,
            history_row=history_row,
            evaluate_fn=lambda: evaluate_option_scoring(
                model=model,
                processor=processor,
                df=val_es_df,  # type: ignore[arg-type]
                images_root=images_root,
                device=device,
                config=config,
            ),
            early_stopping_metric=es_metric,
        )
        model.train()
        if do_break:
            break
    return history


def _mean_teacher_forced_sft_loss(
    model: nn.Module,
    processor: Any,
    df: pd.DataFrame,
    images_root: Path,
    device: str,
    config: dict[str, Any],
) -> float:
    """Mean next-token CE on assistant tokens (matches SFT training objective)."""
    model.eval()
    batch_size = int(config["batch_size"])
    running = 0.0
    total_examples = 0
    with torch.no_grad():
        batch_iter = tqdm(
            _iter_row_batches(df, batch_size=batch_size),
            total=(len(df) + batch_size - 1) // batch_size,
            desc="Val teacher-forced loss",
        )
        for batch_df in batch_iter:
            prompts: list[str] = []
            targets: list[str] = []
            images: list[Image.Image] = []
            for _, row in batch_df.iterrows():
                prompt = build_prompt(
                    row=row,
                    prompt_structure=config["prompt_structure"],
                    context_mode=config["context_mode"],
                    choice_format=config["choice_format"],
                    output_format=config["output_format"],
                    prompt_overrides=config,
                )
                target = build_target_text(int(row["answer"]), output_format=config["output_format"], prompt_overrides=config)
                images.append(_load_image_with_ablation(images_root=images_root, row=row, config=config, is_train=False))
                prompts.append(_render_chat_prompt(processor, prompt, add_generation_prompt=True))
                targets.append(target)

            full_text = [f"{p}{t}" for p, t in zip(prompts, targets)]
            prompt_inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            full_inputs = processor(text=full_text, images=images, padding=True, return_tensors="pt")
            prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)

            inputs = {key: val.to(device) for key, val in full_inputs.items()}
            labels = _mask_prompt_tokens_for_sft(
                labels=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                prompt_lens=prompt_lens,
                processor=processor,
            )

            outputs = model(**inputs, labels=labels)
            running += float(outputs.loss.item()) * len(batch_df)
            total_examples += len(batch_df)

    return float(running / max(1, total_examples))


def evaluate_sft_generation(
    model: nn.Module,
    processor: Any,
    df: pd.DataFrame,
    images_root: Path,
    device: str,
    config: dict[str, Any],
    compute_eval_loss: bool = True,
    compute_generation: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate SFT model using generation and letter parsing."""
    model.eval()
    records: list[dict[str, Any]] = []
    batch_size = int(config["batch_size"])
    eval_loss_mean = float("nan")
    can_compute_loss = bool(compute_eval_loss) and ("answer" in df.columns)
    if len(df) > 0 and can_compute_loss:
        eval_loss_mean = _mean_teacher_forced_sft_loss(
            model=model,
            processor=processor,
            df=df,
            images_root=images_root,
            device=device,
            config=config,
        )
    if compute_generation:
        with torch.no_grad():
            batch_iter = tqdm(
                _iter_row_batches(df, batch_size=batch_size),
                total=(len(df) + batch_size - 1) // batch_size,
                desc="Val generation",
            )
            for batch_df in batch_iter:
                rows = [row for _, row in batch_df.iterrows()]
                prompts: list[str] = []
                images: list[Image.Image] = []
                for row in rows:
                    prompt = build_prompt(
                        row=row,
                        prompt_structure=config["prompt_structure"],
                        context_mode=config["context_mode"],
                        choice_format=config["choice_format"],
                        output_format=config["output_format"],
                        prompt_overrides=config,
                    )
                    prompts.append(_render_chat_prompt(processor, prompt, add_generation_prompt=True))
                    images.append(_load_image_with_ablation(images_root=images_root, row=row, config=config, is_train=False))

                inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
                input_seq_len = int(inputs["input_ids"].shape[-1])
                inputs = {k: v.to(device) for k, v in inputs.items()}
                generate_kwargs = {
                    "do_sample": False,
                    "max_new_tokens": int(config["max_new_tokens"]),
                }
                if str(config["decoding_strategy"]).lower() == "beam":
                    generate_kwargs["num_beams"] = int(config.get("num_beams", 3))
                generated = model.generate(**inputs, **generate_kwargs)

                for i, row in enumerate(rows):
                    continuation = generated[i : i + 1, input_seq_len:]
                    raw_output = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
                    pred_idx, parse_status = parse_letter_prediction(
                        raw_output=raw_output,
                        num_choices=int(row["num_choices"]),
                        parse_rule=str(config["parse_rule"]),
                    )
                    rec = {
                        "id": row["id"],
                        "pred_answer": int(pred_idx),
                        "raw_output": raw_output,
                        "parse_status": parse_status,
                        "num_choices": int(row["num_choices"]),
                    }
                    if "answer" in row.index:
                        rec["gold_answer"] = int(row["answer"])
                        rec["is_correct"] = int(rec["pred_answer"] == rec["gold_answer"])
                    records.append(rec)

    pred_df = pd.DataFrame(records)
    metrics = summarize_predictions(pred_df) if compute_generation else {"n_examples": int(len(df))}
    metrics.setdefault("accuracy", float("nan"))
    metrics.setdefault("percent_correct", float("nan"))
    if not math.isnan(eval_loss_mean):
        metrics["eval_loss"] = eval_loss_mean
    else:
        metrics.setdefault("eval_loss", float("nan"))
    return metrics, pred_df


def evaluate_option_scoring(
    model: nn.Module,
    processor: Any,
    df: pd.DataFrame,
    images_root: Path,
    device: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate option-scoring objective by candidate log-prob ranking."""
    model.eval()
    records: list[dict[str, Any]] = []
    total_ce = 0.0
    n_loss_rows = 0
    with torch.no_grad():
        row_iter = tqdm(df.iterrows(), total=len(df), desc="Val option scoring")
        for _, row in row_iter:
            prompt = build_prompt(
                row=row,
                prompt_structure=config["prompt_structure"],
                context_mode=config["context_mode"],
                choice_format=config["choice_format"],
                output_format=config["output_format"],
                prompt_overrides=config,
            )
            image = _load_image_with_ablation(images_root=images_root, row=row, config=config, is_train=False)
            scores = _candidate_logprob_scores(
                model=model,
                processor=processor,
                prompt=prompt,
                image=image,
                num_choices=int(row["num_choices"]),
                output_format=config["output_format"],
                device=device,
                prompt_overrides=config,
            )
            pred = int(torch.argmax(scores, dim=1).item())
            if "answer" in row.index and pd.notna(row["answer"]):
                label = torch.tensor([int(row["answer"])], dtype=torch.long, device=device)
                total_ce += float(torch.nn.functional.cross_entropy(scores, label).item())
                n_loss_rows += 1
            rec = {
                "id": row["id"],
                "pred_answer": pred,
                "num_choices": int(row["num_choices"]),
            }
            if "answer" in row.index:
                rec["gold_answer"] = int(row["answer"])
                rec["is_correct"] = int(pred == rec["gold_answer"])
            records.append(rec)

    pred_df = pd.DataFrame(records)
    metrics = summarize_predictions(pred_df)
    if n_loss_rows > 0:
        metrics["eval_loss"] = total_ce / n_loss_rows
    return metrics, pred_df


def summarize_predictions(pred_df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact metrics dictionary for val/test tables."""
    metrics: dict[str, Any] = {
        "n_examples": int(len(pred_df)),
    }
    if "is_correct" in pred_df.columns and len(pred_df) > 0:
        acc = float(pred_df["is_correct"].mean())
        metrics["accuracy"] = acc
        metrics["percent_correct"] = 100.0 * acc
    if "parse_status" in pred_df.columns and len(pred_df) > 0:
        metrics["parse_failure_rate"] = float((pred_df["parse_status"] != "ok").mean())
    return metrics


def make_submission(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Convert prediction frame to Kaggle submission schema."""
    submission = pred_df[["id", "pred_answer"]].rename(columns={"pred_answer": "answer"}).copy()
    submission["answer"] = submission["answer"].astype(int)
    return submission


def validate_submission(df: pd.DataFrame) -> None:
    """Basic schema checks for submission format."""
    expected_cols = ["id", "answer"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Submission columns must be {expected_cols}, got {list(df.columns)}")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("Submission id column contains nulls or duplicates.")
    if df["answer"].isna().any():
        raise ValueError("Submission answer column contains nulls.")


def build_failure_examples(pred_df: pd.DataFrame, max_examples: int = 200) -> pd.DataFrame:
    """Return a compact set of incorrect predictions for error analysis."""
    if "is_correct" not in pred_df.columns:
        return pd.DataFrame(columns=pred_df.columns)
    return pred_df[pred_df["is_correct"] == 0].head(max_examples).reset_index(drop=True)


def maybe_save_dataframe(df: pd.DataFrame, path: Path, save: bool) -> Path | None:
    """Write DataFrame to CSV when save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def maybe_save_json(payload: dict[str, Any], path: Path, save: bool) -> Path | None:
    """Write JSON when save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def maybe_save_model(model: nn.Module, out_dir: Path, save: bool) -> Path | None:
    """Save adapter weights to disk when save=True."""
    if not save:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return out_dir


__all__ = [
    "apply_sanity_subset",
    "build_failure_examples",
    "build_lora_model",
    "build_prompt",
    "build_target_text",
    "cap_validation_rows",
    "detect_device",
    "evaluate_option_scoring",
    "evaluate_sft_generation",
    "get_param_counts",
    "load_split",
    "make_submission",
    "maybe_mount_colab_drive",
    "maybe_save_dataframe",
    "maybe_save_json",
    "maybe_save_model",
    "parse_choices",
    "parse_letter_prediction",
    "resolve_image_path",
    "set_seed",
    "select_epoch_train_df",
    "summarize_predictions",
    "train_option_scoring_objective",
    "train_sft_objective",
    "validate_submission",
]
