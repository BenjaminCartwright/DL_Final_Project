"""Reusable utilities for the zero-shot baseline notebook."""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    """Set Python and NumPy seeds for deterministic notebook behavior."""
    random.seed(seed)
    np.random.seed(seed)


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    """Load a split CSV and normalize key text fields used in prompting."""
    path = data_dir / f"{split}.csv"
    df = pd.read_csv(path)
    for col in ["question", "hint", "lecture"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def apply_sanity_subset(df: pd.DataFrame, sanity_check: bool, n: int, seed: int) -> pd.DataFrame:
    """Return a deterministic subset for fast pipeline checks."""
    if not sanity_check:
        return df.copy()
    n = max(1, min(int(n), len(df)))
    sampled = df.sample(n=n, random_state=seed).copy()
    return sampled.reset_index(drop=True)


def parse_choices(choices_raw: Any) -> list[str]:
    """Parse choices from JSON-like text into a list of strings."""
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
    """Create a structured prompt that asks for a numeric answer index only."""
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
    """Resolve relative image paths from CSV rows against data root."""
    return data_dir / str(image_path)


def parse_answer_index(text: str, num_choices: int, fallback: int = 0) -> tuple[int, str]:
    """Parse generated text into an answer index and return parse status."""
    if num_choices <= 0:
        return fallback, "invalid_num_choices"
    match = re.search(r"-?\d+", str(text))
    if not match:
        safe = min(max(fallback, 0), num_choices - 1)
        return safe, "no_integer_found"
    value = int(match.group(0))
    if value < 0 or value >= num_choices:
        safe = min(max(fallback, 0), num_choices - 1)
        return safe, "out_of_range_integer"
    return value, "ok"


def run_zero_shot_inference(
    df: pd.DataFrame,
    data_dir: Path,
    processor: Any,
    model: Any,
    device: str,
    include_hint: bool = True,
    include_lecture: bool = True,
    max_new_tokens: int = 8,
    fallback_index: int = 0,
    progress_desc: str = "Zero-shot inference",
) -> pd.DataFrame:
    """Run generation row-by-row and collect raw outputs plus parsed indices."""
    from PIL import Image
    import torch

    records: list[dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=progress_desc):
        prompt = build_prompt(row, include_hint=include_hint, include_lecture=include_lecture)
        img_path = resolve_image_path(data_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            # Some multimodal processors (e.g., Idefics-style) require text to
            # explicitly include an image placeholder that matches image inputs.
            if hasattr(processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                prompt_with_image = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
                inputs = processor(text=prompt_with_image, images=[image], return_tensors="pt")
            else:
                inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
            new_tokens = generated[:, inputs["input_ids"].shape[-1]:]
            raw_pred = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        pred_idx, parse_status = parse_answer_index(
            raw_pred, int(row["num_choices"]), fallback=fallback_index
        )
        rec = {
            "id": row["id"],
            "pred_answer": pred_idx,
            "raw_output": raw_pred,
            "parse_status": parse_status,
            "num_choices": int(row["num_choices"]),
        }
        if "answer" in row.index:
            rec["gold_answer"] = int(row["answer"])
            rec["is_correct"] = int(pred_idx == int(row["answer"]))
        records.append(rec)

    return pd.DataFrame(records)


def evaluate_predictions(pred_df: pd.DataFrame) -> dict[str, Any]:
    """Compute top-level metrics and simple diagnostics."""
    metrics: dict[str, Any] = {}
    if "is_correct" in pred_df.columns and len(pred_df) > 0:
        metrics["accuracy"] = float(pred_df["is_correct"].mean())
    metrics["n_examples"] = int(len(pred_df))
    metrics["parse_ok_rate"] = float((pred_df["parse_status"] == "ok").mean()) if len(pred_df) else 0.0
    metrics["fallback_rate"] = float((pred_df["parse_status"] != "ok").mean()) if len(pred_df) else 0.0
    if len(pred_df):
        by_choices = pred_df.groupby("num_choices")["parse_status"].apply(
            lambda s: float((s == "ok").mean())
        )
        metrics["parse_ok_rate_by_num_choices"] = {int(k): float(v) for k, v in by_choices.items()}
    return metrics


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
    """Save a dataframe only if save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def maybe_save_json(payload: dict[str, Any], path: Path, save: bool) -> Path | None:
    """Save JSON only if save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def maybe_save_figure(fig: Any, path: Path, save: bool) -> Path | None:
    """Save a matplotlib figure only if save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
