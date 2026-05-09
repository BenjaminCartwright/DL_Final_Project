"""Reusable utilities for phase-1 zero-shot ablation notebooks."""

from __future__ import annotations

import ast
import json
import random
import re
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from src.input_ablation_utils import (
    apply_image_transform,
    build_metadata_prefix,
    build_prompt_ablation_overrides,
    resolve_split_image_path,
)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    """Load one split CSV and normalize core text columns."""
    path = data_dir / f"{split}.csv"
    df = pd.read_csv(path)
    df["split"] = split
    for col in ["question", "hint", "lecture"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def apply_sanity_subset(df: pd.DataFrame, sanity_check: bool, n: int, seed: int) -> pd.DataFrame:
    """Return a deterministic subset for fast debug iterations."""
    if not sanity_check:
        return df.copy().reset_index(drop=True)
    n = max(1, min(int(n), len(df)))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def cap_validation_rows(df: pd.DataFrame, max_val_examples: int | None) -> pd.DataFrame:
    """Cap the validation set size while preserving original order."""
    if max_val_examples is None:
        return df.copy().reset_index(drop=True)
    n = max(1, min(int(max_val_examples), len(df)))
    return df.iloc[:n].copy().reset_index(drop=True)


def parse_choices(choices_raw: Any) -> list[str]:
    """Parse list-like choice text into a clean list of strings."""
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


def _normalize_split_name(split_name: str | None) -> str:
    if split_name is None:
        return "val"
    split_clean = str(split_name).strip().lower()
    return split_clean if split_clean in {"train", "val", "test"} else "val"


def resolve_image_path(images_root: Path, row: pd.Series, split_name: str | None = None) -> Path:
    """Resolve image path from CSV metadata or numeric identifier fallback."""
    # Delegate to shared helper that supports numbered references under
    # data/images/images/{split} plus explicit image_path values.
    split = _normalize_split_name(split_name or row.get("split", None))
    return resolve_split_image_path(images_root=images_root, row=row, split_name=split)


def build_choice_lines(choices: list[str], choice_format: str) -> list[str]:
    """Render answer choices according to ablation formatting."""
    lines: list[str] = []
    for i, choice in enumerate(choices):
        letter = _letter_for_index(i)
        if choice_format == "letter_dot":
            lines.append(f"{letter}. {choice}")
        elif choice_format == "letter_paren":
            lines.append(f"({letter}) {choice}")
        elif choice_format == "option_letter":
            lines.append(f"Option {letter}: {choice}")
        else:
            lines.append(f"{letter}. {choice}")
    return lines


def _build_context_block(row: pd.Series, context_mode: str) -> list[str]:
    hint = str(row.get("hint", "")).strip()
    lecture = str(row.get("lecture", "")).strip()
    sections: list[str] = []
    if context_mode in {"hint_only", "hint_lecture"} and hint:
        sections.extend(["Hint:", hint, ""])
    if context_mode in {"lecture_only", "hint_lecture"} and lecture:
        sections.extend(["Lecture:", lecture, ""])
    return sections


def build_generative_prompt(
    row: pd.Series,
    prompt_structure: str,
    context_mode: str,
    choice_format: str,
    output_format: str,
    prompt_overrides: dict[str, Any] | None = None,
) -> str:
    """Build generative prompt text according to ablation settings."""
    prompt_overrides = prompt_overrides or {}
    phrasing = build_prompt_ablation_overrides(prompt_overrides)
    question = str(row.get("question", "")).strip()
    choices = parse_choices(row.get("choices", "[]"))
    choice_lines = build_choice_lines(choices, choice_format)
    context_lines = _build_context_block(row, context_mode)
    metadata_prefix = build_metadata_prefix(row=row, config=prompt_overrides)
    question_label = str(phrasing.get("question_label", "Question:"))
    choices_label = str(phrasing.get("choices_label", "Choices:"))

    if output_format == "letter_only":
        output_instruction = "Return only one letter from the answer choices."
    elif output_format == "answer_prefix":
        answer_prefix_text = str(phrasing.get("answer_prefix_text", "Answer:"))
        output_instruction = f"Return your answer as: {answer_prefix_text} <LETTER>."
    elif output_format == "no_explanation":
        output_instruction = "Do not include explanations; return only the final letter."
    elif output_format == "reason_then_final":
        output_instruction = "You may reason briefly, but end with a final line: Final Answer: <LETTER>."
    else:
        output_instruction = "Return one final answer letter."

    instruction_prefix = str(phrasing.get("instruction_prefix", "")).strip()
    if instruction_prefix:
        output_instruction = f"{instruction_prefix}\n{output_instruction}"

    question_line = f"{question_label} {question}"

    if prompt_structure == "minimal":
        sections = [metadata_prefix, question, "", choices_label, *choice_lines, "", output_instruction]
    elif prompt_structure == "explicit_instruction":
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
    elif prompt_structure == "context_first":
        sections = [
            "Use the provided context and image to choose the best answer.",
            "",
            metadata_prefix,
            *context_lines,
            question_line,
            "",
            choices_label,
            *choice_lines,
            "",
            output_instruction,
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
        sections = [metadata_prefix, question_line, "", choices_label, *choice_lines, "", output_instruction]
    return "\n".join([x for x in sections if x is not None])


def parse_letter_prediction(raw_output: str, num_choices: int, parse_rule: str = "strict_first_letter") -> tuple[int, str]:
    """Parse model text output into a 0-based answer index."""
    text = str(raw_output).strip()
    if num_choices <= 0:
        return 0, "invalid_num_choices"

    letters = [_letter_for_index(i) for i in range(num_choices)]
    if parse_rule == "strict_first_letter":
        match = re.search(r"\b([A-Z])\b", text.upper())
        if match and match.group(1) in letters:
            return letters.index(match.group(1)), "ok"
    elif parse_rule == "answer_prefix":
        match = re.search(r"answer\s*:\s*([A-Z])", text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in letters:
                return letters.index(letter), "ok"
    else:
        for letter in letters:
            if letter in text.upper():
                return letters.index(letter), "ok"

    return 0, "parse_failure"


def _decode_generated_text(processor: Any, generated_ids: torch.Tensor, prompt_len: int) -> str:
    continuation = generated_ids[:, prompt_len:]
    return processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()


def _iter_row_batches(df: pd.DataFrame, batch_size: int):
    """Yield dataframe slices of at most batch_size rows."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def _prepare_generative_batch(
    batch_df: pd.DataFrame,
    images_root: Path,
    processor: Any,
    config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], list[pd.Series], list[int]]:
    """Build batched model inputs and per-item prompt lengths."""
    rows: list[pd.Series] = [row for _, row in batch_df.iterrows()]
    prompts: list[str] = []
    images: list[Image.Image] = []
    for row in rows:
        prompts.append(
            build_generative_prompt(
                row=row,
                prompt_structure=config["prompt_structure"],
                context_mode=config["context_mode"],
                choice_format=config["choice_format"],
                output_format=config["output_format"],
                prompt_overrides=config,
            )
        )
        image_path = resolve_image_path(images_root, row, split_name=row.get("split", None))
        raw_image = Image.open(image_path).convert("RGB")
        images.append(
            apply_image_transform(
                image=raw_image,
                config=config,
                is_train=bool(config.get("enable_image_augmentation", False)),
            )
        )

    if hasattr(processor, "apply_chat_template"):
        rendered_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            rendered_prompts.append(processor.apply_chat_template(messages, add_generation_prompt=True))
        inputs = processor(text=rendered_prompts, images=images, padding=True, return_tensors="pt")
    else:
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        prompt_lens = attention_mask.sum(dim=1).tolist()
    else:
        prompt_lens = [inputs["input_ids"].shape[-1]] * len(rows)
    prompt_lens = [int(x) for x in prompt_lens]
    return inputs, rows, prompt_lens


def run_generative_ablation(
    df: pd.DataFrame,
    images_root: Path,
    processor: Any,
    model: Any,
    device: str,
    config: dict[str, Any],
    batch_size: int = 1,
) -> pd.DataFrame:
    """Run one generative ablation config over a dataframe.

    batch_size controls throughput vs memory usage. Keep it low if you hit OOM.
    """
    records: list[dict[str, Any]] = []

    total_batches = (len(df) + batch_size - 1) // batch_size
    for batch_df in tqdm(_iter_row_batches(df, batch_size=batch_size), total=total_batches, desc=f"Gen {config['ablation_id']}"):
        inputs, rows, prompt_lens = _prepare_generative_batch(
            batch_df=batch_df,
            images_root=images_root,
            processor=processor,
            config=config,
        )
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if config["decoding_strategy"] == "beam":
                generated = model.generate(
                    **inputs,
                    num_beams=3,
                    do_sample=False,
                    max_new_tokens=int(config["max_new_tokens"]),
                )
            else:
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(config["max_new_tokens"]),
                )

        for i, row in enumerate(rows):
            continuation = generated[i : i + 1, prompt_lens[i] :]
            raw_output = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
            pred_idx, parse_status = parse_letter_prediction(
                raw_output=raw_output,
                num_choices=int(row["num_choices"]),
                parse_rule=config["parse_rule"],
            )
            rec = {
                "ablation_id": config["ablation_id"],
                "id": row["id"],
                "pred_answer": int(pred_idx),
                "raw_output": raw_output,
                "parse_status": parse_status,
                "output_length_chars": len(raw_output),
                "num_choices": int(row["num_choices"]),
            }
            if "answer" in row.index:
                rec["gold_answer"] = int(row["answer"])
                rec["is_correct"] = int(int(pred_idx) == int(row["answer"]))
            records.append(rec)

    return pd.DataFrame(records)


def summarize_generative_predictions(pred_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Compute summary metrics for one generative ablation config."""
    metrics: dict[str, Any] = {**config}
    metrics["n_examples"] = int(len(pred_df))
    metrics["accuracy"] = float(pred_df["is_correct"].mean()) if "is_correct" in pred_df.columns else 0.0
    metrics["percent_correct"] = 100.0 * metrics["accuracy"]
    metrics["parse_failure_rate"] = float((pred_df["parse_status"] != "ok").mean()) if len(pred_df) else 0.0
    metrics["invalid_output_rate"] = float((pred_df["pred_answer"] >= pred_df["num_choices"]).mean()) if len(pred_df) else 0.0
    metrics["mean_output_length_chars"] = float(pred_df["output_length_chars"].mean()) if len(pred_df) else 0.0
    metrics["std_output_length_chars"] = float(pred_df["output_length_chars"].std(ddof=0)) if len(pred_df) else 0.0
    if len(pred_df):
        dist = pred_df["pred_answer"].value_counts(normalize=True).sort_index()
        metrics["prediction_distribution"] = {int(k): float(v) for k, v in dist.items()}
    else:
        metrics["prediction_distribution"] = {}
    return metrics


def build_generative_phase1_configs() -> list[dict[str, Any]]:
    """Build phase-1 generative ablations (36 configs per PDF)."""
    configs: list[dict[str, Any]] = []
    prompt_structures = ["minimal", "explicit_instruction", "question_first"]
    context_modes = ["none", "hint_lecture"]
    output_formats = ["letter_only", "answer_prefix"]
    max_new_tokens_values = [1, 3, 5]
    idx = 1

    for prompt_structure in prompt_structures:
        for context_mode in context_modes:
            for output_format in output_formats:
                for max_new_tokens in max_new_tokens_values:
                    config = {
                        "ablation_id": f"gen_{idx:03d}",
                        "prompt_structure": prompt_structure,
                        "context_mode": context_mode,
                        "choice_format": "letter_dot",
                        "output_format": output_format,
                        "max_new_tokens": int(max_new_tokens),
                        "decoding_strategy": "greedy",
                        "parse_rule": "strict_first_letter",
                    }
                    configs.append(config)
                    idx += 1
    return configs


def _render_nongenerative_candidate(
    choice_text: str,
    letter: str,
    scoring_target: str,
    answer_prefix_text: str = "Answer:",
) -> str:
    if scoring_target == "letter":
        return letter
    if scoring_target == "answer_prefix":
        return f"{answer_prefix_text} {letter}"
    if scoring_target == "full_choice":
        return choice_text
    if scoring_target == "letter_plus_choice":
        return f"{letter}. {choice_text}"
    return letter


def build_nongenerative_prompt(
    row: pd.Series,
    prompt_structure: str,
    context_mode: str,
    choice_format: str,
    choice_order: str,
    prompt_overrides: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Build a scoring prompt and ordered candidate metadata."""
    prompt_overrides = prompt_overrides or {}
    phrasing = build_prompt_ablation_overrides(prompt_overrides)
    question = str(row.get("question", "")).strip()
    choices = parse_choices(row.get("choices", "[]"))
    indexed_choices = list(enumerate(choices))
    if choice_order == "reverse":
        indexed_choices = list(reversed(indexed_choices))

    ordered_choices: list[dict[str, Any]] = []
    for pos, (orig_idx, choice) in enumerate(indexed_choices):
        ordered_choices.append(
            {
                "position": pos,
                "orig_idx": int(orig_idx),
                "letter": _letter_for_index(pos),
                "choice_text": str(choice),
            }
        )

    choice_lines = build_choice_lines([x["choice_text"] for x in ordered_choices], choice_format)
    context_lines = _build_context_block(row, context_mode)
    metadata_prefix = build_metadata_prefix(row=row, config=prompt_overrides)
    question_label = str(phrasing.get("question_label", "Question:"))
    choices_label = str(phrasing.get("choices_label", "Choices:"))
    answer_prefix_text = str(phrasing.get("answer_prefix_text", "Answer:"))
    question_line = f"{question_label} {question}"
    instruction_prefix = str(phrasing.get("instruction_prefix", "")).strip()

    if prompt_structure == "minimal":
        sections = [metadata_prefix, question, "", choices_label, *choice_lines, "", answer_prefix_text]
    elif prompt_structure == "explicit_instruction":
        sections = [
            "Select the best answer choice for the question and image.",
            "You are selecting from the provided options.",
            instruction_prefix,
            "",
            metadata_prefix,
            question_line,
            "",
            *context_lines,
            choices_label,
            *choice_lines,
            "",
            answer_prefix_text,
        ]
    else:
        sections = [
            metadata_prefix,
            question_line,
            "",
            *context_lines,
            choices_label,
            *choice_lines,
            "",
            answer_prefix_text,
        ]
    return "\n".join([x for x in sections if x is not None]), ordered_choices


def _score_candidate_sequence(
    processor: Any,
    model: Any,
    device: str,
    prompt: str,
    image: Image.Image,
    candidate_text: str,
    length_normalize: bool,
) -> float:
    """Score candidate text with teacher-forced log-probability."""
    with torch.no_grad():
        if hasattr(processor, "apply_chat_template"):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            rendered_prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
            full_text = f"{rendered_prompt}{candidate_text}"
            prompt_inputs = processor(text=rendered_prompt, images=[image], return_tensors="pt")
            full_inputs = processor(text=full_text, images=[image], return_tensors="pt")
        else:
            prompt_inputs = processor(text=prompt, images=image, return_tensors="pt")
            full_inputs = processor(text=f"{prompt}{candidate_text}", images=image, return_tensors="pt")

        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        input_ids = full_inputs["input_ids"].to(device)
        attention_mask = full_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        # Candidate token positions correspond to tokens generated after the prompt boundary.
        candidate_start = max(prompt_len - 1, 0)
        log_probs = torch.log_softmax(logits[:, candidate_start:, :], dim=-1)
        target_slice = target_ids[:, candidate_start:]

        token_log_probs = log_probs.gather(dim=-1, index=target_slice.unsqueeze(-1)).squeeze(-1)
        score = float(token_log_probs.sum().item())
        if length_normalize and token_log_probs.numel() > 0:
            score /= float(token_log_probs.numel())
        return score


def _score_candidate_sequences_batch(
    processor: Any,
    model: Any,
    device: str,
    prompt: str,
    image: Image.Image,
    candidate_texts: list[str],
    length_normalize: bool,
    batch_size: int,
) -> list[float]:
    """Score candidate texts in mini-batches for one prompt+image pair."""
    scores: list[float] = []
    for start in range(0, len(candidate_texts), batch_size):
        batch_candidates = candidate_texts[start : start + batch_size]
        with torch.no_grad():
            if hasattr(processor, "apply_chat_template"):
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                rendered_prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
                full_texts = [f"{rendered_prompt}{cand}" for cand in batch_candidates]
                prompt_inputs = processor(
                    text=[rendered_prompt] * len(batch_candidates),
                    images=[image] * len(batch_candidates),
                    padding=True,
                    return_tensors="pt",
                )
                full_inputs = processor(
                    text=full_texts,
                    images=[image] * len(batch_candidates),
                    padding=True,
                    return_tensors="pt",
                )
            else:
                prompt_inputs = processor(
                    text=[prompt] * len(batch_candidates),
                    images=[image] * len(batch_candidates),
                    padding=True,
                    return_tensors="pt",
                )
                full_inputs = processor(
                    text=[f"{prompt}{cand}" for cand in batch_candidates],
                    images=[image] * len(batch_candidates),
                    padding=True,
                    return_tensors="pt",
                )

            prompt_attn = prompt_inputs.get("attention_mask")
            if prompt_attn is not None:
                prompt_lens = prompt_attn.sum(dim=1).tolist()
            else:
                prompt_lens = [prompt_inputs["input_ids"].shape[-1]] * len(batch_candidates)
            prompt_lens = [int(x) for x in prompt_lens]

            input_ids = full_inputs["input_ids"].to(device)
            attention_mask = full_inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)

            if attention_mask is not None:
                target_mask = attention_mask[:, 1:]
            else:
                target_mask = torch.ones_like(target_ids, dtype=torch.long)

            token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            for row_idx in range(len(batch_candidates)):
                candidate_start = max(prompt_lens[row_idx] - 1, 0)
                valid_mask = target_mask[row_idx, candidate_start:].bool()
                cand_token_log_probs = token_log_probs[row_idx, candidate_start:][valid_mask]
                score = float(cand_token_log_probs.sum().item()) if cand_token_log_probs.numel() else 0.0
                if length_normalize and cand_token_log_probs.numel() > 0:
                    score /= float(cand_token_log_probs.numel())
                scores.append(score)

    return scores


def run_nongenerative_ablation(
    df: pd.DataFrame,
    images_root: Path,
    processor: Any,
    model: Any,
    device: str,
    config: dict[str, Any],
    batch_size: int = 1,
) -> pd.DataFrame:
    """Run one non-generative scoring ablation config over a dataframe.

    batch_size controls throughput vs memory usage. Keep it low if you hit OOM.
    """
    records: list[dict[str, Any]] = []

    total_batches = (len(df) + batch_size - 1) // batch_size
    for batch_df in tqdm(_iter_row_batches(df, batch_size=batch_size), total=total_batches, desc=f"NonGen {config['ablation_id']}"):
        for _, row in batch_df.iterrows():
            prompt, ordered_choices = build_nongenerative_prompt(
                row=row,
                prompt_structure=config["prompt_structure"],
                context_mode=config["context_mode"],
                choice_format=config["choice_format"],
                choice_order=config["choice_order"],
                prompt_overrides=config,
            )
            image_path = resolve_image_path(images_root, row, split_name=row.get("split", None))
            raw_image = Image.open(image_path).convert("RGB")
            image = apply_image_transform(
                image=raw_image,
                config=config,
                is_train=bool(config.get("enable_image_augmentation", False)),
            )

            candidate_texts = [
                _render_nongenerative_candidate(
                    choice_text=item["choice_text"],
                    letter=item["letter"],
                    scoring_target=config["scoring_target"],
                    answer_prefix_text=str(config.get("answer_prefix_text", "Answer:")),
                )
                for item in ordered_choices
            ]
            candidate_scores = _score_candidate_sequences_batch(
                processor=processor,
                model=model,
                device=device,
                prompt=prompt,
                image=image,
                candidate_texts=candidate_texts,
                length_normalize=bool(config["length_normalize"]),
                batch_size=batch_size,
            )
            scored = [
                {**item, "candidate_text": candidate_texts[idx], "score": candidate_scores[idx]}
                for idx, item in enumerate(ordered_choices)
            ]

            scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
            top = scored_sorted[0]
            second = scored_sorted[1] if len(scored_sorted) > 1 else scored_sorted[0]
            pred_orig_idx = int(top["orig_idx"])

            rec = {
                "ablation_id": config["ablation_id"],
                "id": row["id"],
                "pred_answer": pred_orig_idx,
                "best_score": float(top["score"]),
                "second_best_score": float(second["score"]),
                "confidence_margin": float(top["score"] - second["score"]),
                "num_choices": int(row["num_choices"]),
            }
            if "answer" in row.index:
                rec["gold_answer"] = int(row["answer"])
                rec["is_correct"] = int(pred_orig_idx == int(row["answer"]))
            records.append(rec)

    return pd.DataFrame(records)


def summarize_nongenerative_predictions(pred_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Compute summary metrics for one non-generative ablation config."""
    metrics: dict[str, Any] = {**config}
    metrics["n_examples"] = int(len(pred_df))
    metrics["accuracy"] = float(pred_df["is_correct"].mean()) if "is_correct" in pred_df.columns else 0.0
    metrics["percent_correct"] = 100.0 * metrics["accuracy"]
    metrics["mean_confidence_margin"] = float(pred_df["confidence_margin"].mean()) if len(pred_df) else 0.0
    metrics["std_confidence_margin"] = float(pred_df["confidence_margin"].std(ddof=0)) if len(pred_df) else 0.0
    if len(pred_df):
        dist = pred_df["pred_answer"].value_counts(normalize=True).sort_index()
        metrics["prediction_distribution"] = {int(k): float(v) for k, v in dist.items()}
    else:
        metrics["prediction_distribution"] = {}
    return metrics


def build_nongenerative_phase1_configs() -> list[dict[str, Any]]:
    """Build phase-1 non-generative ablations (48 configs per PDF)."""
    configs: list[dict[str, Any]] = []
    prompt_structures = ["minimal", "explicit_instruction", "question_first"]
    context_modes = ["none", "hint_lecture"]
    scoring_targets = ["letter", "answer_prefix", "full_choice", "letter_plus_choice"]
    length_norm_options = [False, True]
    idx = 1

    for prompt_structure in prompt_structures:
        for context_mode in context_modes:
            for scoring_target in scoring_targets:
                for length_normalize in length_norm_options:
                    configs.append(
                        {
                            "ablation_id": f"nongen_{idx:03d}",
                            "prompt_structure": prompt_structure,
                            "context_mode": context_mode,
                            "choice_format": "letter_dot",
                            "choice_order": "original",
                            "scoring_target": scoring_target,
                            "length_normalize": bool(length_normalize),
                        }
                    )
                    idx += 1
    return configs


def select_ablation_configs(
    all_configs: list[dict[str, Any]],
    selected_ablation_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter all ablations by selected IDs, preserving original order."""
    if not selected_ablation_ids:
        return list(all_configs)
    selected = set(selected_ablation_ids)
    filtered = [cfg for cfg in all_configs if cfg["ablation_id"] in selected]
    missing = sorted(selected - {cfg["ablation_id"] for cfg in all_configs})
    if missing:
        raise ValueError(f"Unknown ablation ids: {missing}")
    return filtered


def build_block_configs(
    ordered_keys: list[str],
    fixed_values: dict[str, Any],
    block_space: dict[str, list[Any]],
    ablation_prefix: str,
) -> list[dict[str, Any]]:
    """Build cartesian configs for a tuning block with stable key order."""
    space: dict[str, list[Any]] = {}
    for key in ordered_keys:
        if key in block_space:
            space[key] = list(block_space[key])
        else:
            if key not in fixed_values:
                raise ValueError(f"Missing fixed value for key: {key}")
            space[key] = [fixed_values[key]]

    combos = list(product(*[space[k] for k in ordered_keys]))
    configs: list[dict[str, Any]] = []
    for idx, values in enumerate(combos, start=1):
        cfg = dict(zip(ordered_keys, values))
        cfg["ablation_id"] = f"{ablation_prefix}_{idx:03d}"
        if "max_new_tokens" in cfg:
            cfg["max_new_tokens"] = int(cfg["max_new_tokens"])
        if "length_normalize" in cfg:
            cfg["length_normalize"] = bool(cfg["length_normalize"])
        configs.append(cfg)
    return configs


def make_submission(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Convert prediction dataframe into Kaggle submission format."""
    submission = pred_df[["id", "pred_answer"]].rename(columns={"pred_answer": "answer"}).copy()
    submission["answer"] = submission["answer"].astype(int)
    return submission


def validate_submission(df: pd.DataFrame) -> None:
    """Validate submission schema and basic constraints."""
    expected_cols = ["id", "answer"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Submission columns must be {expected_cols}, got {list(df.columns)}")
    if df["id"].isna().any():
        raise ValueError("Submission contains missing ids.")
    if df["id"].duplicated().any():
        raise ValueError("Submission contains duplicate ids.")
    if df["answer"].isna().any():
        raise ValueError("Submission contains missing answers.")
    if not pd.api.types.is_integer_dtype(df["answer"]):
        raise ValueError("Submission answer column must be integer dtype.")


def top_configs(results_df: pd.DataFrame, k: int = 5) -> list[dict[str, Any]]:
    """Return top-k configurations by validation accuracy then parse quality."""
    if len(results_df) == 0:
        return []
    sort_cols = ["accuracy"]
    if "parse_failure_rate" in results_df.columns:
        sort_cols = ["accuracy", "parse_failure_rate"]
        asc = [False, True]
    else:
        asc = [False]
    top_df = results_df.sort_values(sort_cols, ascending=asc).head(int(k)).copy()
    return top_df.to_dict(orient="records")


def build_failure_examples(pred_df: pd.DataFrame, max_examples: int = 100) -> pd.DataFrame:
    """Extract common failure examples for analysis."""
    if "is_correct" not in pred_df.columns:
        return pd.DataFrame(columns=pred_df.columns)
    failed = pred_df[pred_df["is_correct"] == 0].copy()
    return failed.head(max_examples).reset_index(drop=True)


def maybe_save_dataframe(df: pd.DataFrame, path: Path, save: bool) -> Path | None:
    """Save dataframe when save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def maybe_save_json(payload: dict[str, Any], path: Path, save: bool) -> Path | None:
    """Save JSON when save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def maybe_save_figure(fig: Any, path: Path, save: bool) -> Path | None:
    """Save matplotlib figure when save=True."""
    if not save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
