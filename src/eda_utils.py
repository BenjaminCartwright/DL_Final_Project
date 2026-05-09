
"""Reusable utilities for EDA notebooks."""

import ast
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd


def dataset_card(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "split": [name],
        "rows": [len(df)],
        "cols": [df.shape[1]],
        "memory_mb": [df.memory_usage(deep=True).sum() / 1024**2],
        "id_unique": [df["id"].nunique() if "id" in df.columns else np.nan],
    })


def clean_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def parse_choices(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return np.nan


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["question_clean"] = clean_text_series(out["question"])
    out["hint_clean"] = clean_text_series(out["hint"]) if "hint" in out.columns else ""
    out["lecture_clean"] = clean_text_series(out["lecture"]) if "lecture" in out.columns else ""
    out["question_chars"] = out["question_clean"].str.len()
    out["question_words"] = out["question_clean"].str.split().str.len()
    out["question_sentences"] = out["question_clean"].str.count(r"[.!?]+") + 1
    out["question_avg_word_len"] = out["question_clean"].str.findall(r"[A-Za-z]+").map(
        lambda toks: np.mean([len(t) for t in toks]) if toks else 0
    )
    out["question_punct_density"] = out["question_clean"].str.count(r"[^\w\s]") / out["question_chars"].replace(0, np.nan)
    out["question_lex_diversity"] = out["question_clean"].map(
        lambda txt: (lambda toks: len(set(toks))/len(toks) if toks else 0.0)(re.findall(r"[A-Za-z]+", txt.lower()))
    )
    out["hint_chars"] = out["hint_clean"].str.len() if "hint_clean" in out else 0
    out["lecture_chars"] = out["lecture_clean"].str.len() if "lecture_clean" in out else 0
    return out


def add_context_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hint_present = clean_text_series(out["hint"]).str.len() > 0 if "hint" in out.columns else pd.Series(False, index=out.index)
    lecture_present = clean_text_series(out["lecture"]).str.len() > 0 if "lecture" in out.columns else pd.Series(False, index=out.index)
    out["hint_present"] = hint_present
    out["lecture_present"] = lecture_present
    out["context_group"] = np.select(
        [~hint_present & ~lecture_present, hint_present & ~lecture_present, ~hint_present & lecture_present, hint_present & lecture_present],
        ["none", "hint_only", "lecture_only", "hint_and_lecture"],
        default="unknown",
    )
    return out


def text_stats(series: pd.Series, prefix: str) -> pd.DataFrame:
    s = clean_text_series(series)
    chars = s.str.len()
    words = s.str.split().str.len()
    return pd.DataFrame({
        f"{prefix}_char_mean": [chars.mean()],
        f"{prefix}_char_median": [chars.median()],
        f"{prefix}_char_p90": [chars.quantile(0.9)],
        f"{prefix}_char_max": [chars.max()],
        f"{prefix}_word_mean": [words.mean()],
        f"{prefix}_word_p90": [words.quantile(0.9)],
        f"{prefix}_word_max": [words.max()],
    })


def save_csv(df: pd.DataFrame, out_dir: Path, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def tokenize(text: str, stopwords: set[str]):
    toks = re.findall(r"[a-z]+", str(text).lower())
    return [t for t in toks if t not in stopwords and len(t) > 2]


def top_tokens(series: pd.Series, stopwords: set[str], k: int = 40):
    all_tokens = []
    for txt in series:
        all_tokens.extend(tokenize(txt, stopwords))
    return pd.DataFrame(Counter(all_tokens).most_common(k), columns=["token", "count"])


def label_entropy(s: pd.Series) -> float:
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())
