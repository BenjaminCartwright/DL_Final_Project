"""Shared input/data ablation helpers for LoRA notebook experiments."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageEnhance


def resolve_split_image_path(images_root: Path, row: pd.Series, split_name: str | None = None) -> Path:
    """Resolve image paths for numbered or explicit references under split folders."""
    split = str(split_name or row.get("split", "val")).strip().lower()
    if split not in {"train", "val", "test"}:
        split = "val"
    split_dir = images_root / split

    image_path_raw = str(row.get("image_path", "")).strip()
    if image_path_raw:
        # Handles CSV values like "images/train/train_00000.png".
        parts = Path(image_path_raw).parts
        if len(parts) >= 2 and parts[0] == "images":
            candidate = images_root / Path(*parts[1:])
            if candidate.exists():
                return candidate
        # Handles split-relative values.
        candidate = split_dir / image_path_raw
        if candidate.exists():
            return candidate
        # Handles numeric-only IDs.
        if image_path_raw.isdigit():
            numeric = split_dir / f"{split}_{int(image_path_raw):05d}.png"
            if numeric.exists():
                return numeric
        # Handles bare stem values.
        stem = split_dir / f"{image_path_raw}.png"
        if stem.exists():
            return stem

    row_id = str(row.get("id", "")).strip()
    digits = re.findall(r"\d+", row_id)
    if digits:
        from_id = split_dir / f"{split}_{int(digits[-1]):05d}.png"
        if from_id.exists():
            return from_id

    raise FileNotFoundError(f"Could not resolve image for row id={row.get('id', 'unknown')}.")


def build_image_transform_pipeline(config: dict[str, Any], is_train: bool) -> dict[str, Any]:
    """Normalize transform options from config into a compact pipeline dictionary."""
    image_size = config.get("image_size")
    brightness_jitter = float(config.get("brightness_jitter", 0.0) or 0.0)
    slight_rotation_deg = float(config.get("slight_rotation_deg", 0.0) or 0.0)
    augmentation_enabled = bool(config.get("enable_image_augmentation", False)) and is_train
    return {
        "image_size": int(image_size) if image_size is not None else None,
        "brightness_jitter": max(0.0, brightness_jitter) if augmentation_enabled else 0.0,
        "slight_rotation_deg": max(0.0, slight_rotation_deg) if augmentation_enabled else 0.0,
    }


def apply_image_transform(image: Image.Image, config: dict[str, Any], is_train: bool) -> Image.Image:
    """Apply optional resolution and mild image augmentations."""
    pipeline = build_image_transform_pipeline(config=config, is_train=is_train)
    out = image.convert("RGB")

    image_size = pipeline["image_size"]
    if image_size is not None and image_size > 0:
        out = out.resize((image_size, image_size), Image.BICUBIC)

    if pipeline["slight_rotation_deg"] > 0:
        angle = float(pipeline["slight_rotation_deg"])
        sampled_angle = random.uniform(-angle, angle)
        out = out.rotate(angle=sampled_angle, resample=Image.BICUBIC)

    if pipeline["brightness_jitter"] > 0:
        jitter = float(pipeline["brightness_jitter"])
        factor = 1.0 + min(jitter, 0.95) * 0.5
        out = ImageEnhance.Brightness(out).enhance(factor)

    return out


def build_metadata_prefix(row: pd.Series, config: dict[str, Any]) -> str:
    """Render optional metadata lines prepended to prompts."""
    if not bool(config.get("include_metadata_in_prompt", False)):
        return ""
    fields = config.get("metadata_fields", ["subject", "grade", "topic"])
    if not isinstance(fields, list):
        return ""
    lines: list[str] = []
    for field in fields:
        key = str(field)
        value = str(row.get(key, "")).strip()
        if value:
            lines.append(f"{key.capitalize()}: {value}")
    if not lines:
        return ""
    header = str(config.get("metadata_header", "Metadata")).strip() or "Metadata"
    return "\n".join([f"{header}:", *lines, ""])


def build_prompt_ablation_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Collect prompt-phrasing override knobs from run config."""
    return {
        "question_first": bool(config.get("question_first_override", False)),
        "question_label": str(config.get("question_label", "Question:")),
        "choices_label": str(config.get("choices_label", "Choices:")),
        "answer_prefix_text": str(config.get("answer_prefix_text", "Answer:")),
        "instruction_prefix": str(config.get("instruction_prefix", "")).strip(),
    }

