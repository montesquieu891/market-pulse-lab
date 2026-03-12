from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_BOILERPLATE_PATTERNS: tuple[str, ...] = (
    r"(?i)reporting by [^.]+\.",
    r"(?i)editing by [^.]+\.",
    r"(?i)additional reporting by [^.]+\.",
    r"(?i)for more (?:details|information),? click here[^.]*\.",
    r"(?i)disclaimer:[^.]*\.",
    r"(?i)all rights reserved\.?",
)

_TICKER_DOLLAR_RE = re.compile(r"\$([A-Za-z]{1,10})")
_TICKER_EXCHANGE_RE = re.compile(r"\((?:NYSE|NASDAQ|AMEX)\s*:\s*([A-Za-z]{1,10})\)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def _resolve_paths(config: dict[str, Any]) -> Path:
    """Resolve cleaned-news path from configuration.

    Args:
        config: Parsed pipeline configuration dictionary.

    Returns:
        Output parquet path for cleaned news.
    """
    paths_cfg = config.get("paths", {})
    return Path(paths_cfg.get("interim_news", "datasets/interim/news_cleaned.parquet"))


def _remove_boilerplate(text: str) -> str:
    """Strip repeated wire-service boilerplate snippets.

    Args:
        text: Raw news text.

    Returns:
        Text with common boilerplate removed.
    """
    out = text
    for pattern in _BOILERPLATE_PATTERNS:
        out = re.sub(pattern, " ", out)
    return out


def _normalize_ticker_mentions(text: str) -> str:
    """Normalize ticker mentions such as $AAPL and (NASDAQ: AAPL).

    Args:
        text: Input text.

    Returns:
        Text with normalized ticker mentions.
    """
    out = _TICKER_DOLLAR_RE.sub(lambda m: m.group(1).upper(), text)
    out = _TICKER_EXCHANGE_RE.sub(lambda m: f" {m.group(1).upper()} ", out)
    return out


def _clean_text(text: str) -> str:
    """Apply deterministic text-cleaning rules.

    Args:
        text: Input text.

    Returns:
        Cleaned text.
    """
    out = text.replace("\u00a0", " ")
    out = _remove_boilerplate(out)
    out = _normalize_ticker_mentions(out)
    out = out.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out


def _build_base_text(news_df: pd.DataFrame) -> pd.Series:
    """Build source text using article with title fallback.

    Args:
        news_df: News dataframe.

    Returns:
        Text series aligned to input rows.
    """
    article = news_df["article"].fillna("").astype("string").str.strip()
    title = news_df["title"].fillna("").astype("string").str.strip()
    return article.where(article != "", title)


def clean_news(config: dict[str, Any]) -> None:
    """Normalize and enrich news text before sentiment inference.

    This step does not drop records. It adds deterministic cleaned text and
    metadata features needed by downstream NLP stages.

    Args:
        config: Parsed pipeline configuration dictionary.
    """
    output_path = _resolve_paths(config)
    if not output_path.exists():
        raise FileNotFoundError(f"Missing interim news parquet: {output_path}")

    news_df = pd.read_parquet(output_path)

    required = {"ticker", "date", "title", "article", "text_source"}
    missing = sorted(required - set(news_df.columns))
    if missing:
        raise ValueError(f"interim news parquet missing required columns: {missing}")

    original_rows = len(news_df)

    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce", utc=True)
    news_df["ticker"] = news_df["ticker"].astype("string").str.upper()

    base_text = _build_base_text(news_df)
    cleaned_text = base_text.fillna("").astype("string").map(_clean_text)

    news_df["cleaned_text"] = cleaned_text.astype("string")
    news_df["char_count"] = news_df["cleaned_text"].str.len().fillna(0).astype("int32")
    news_df["word_count"] = news_df["cleaned_text"].str.split().str.len().fillna(0).astype("int32")
    safe_word_count = news_df["word_count"].replace(0, pd.NA)
    news_df["avg_word_length"] = (
        news_df["char_count"] / safe_word_count
    ).fillna(0.0).astype("float32")

    if len(news_df) != original_rows:
        raise AssertionError("clean_news must not change row count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    news_df.to_parquet(output_path, index=False)

    article_ratio = float((news_df["text_source"] == "article").mean()) if len(news_df) else 0.0
    title_ratio = float((news_df["text_source"] == "title").mean()) if len(news_df) else 0.0

    log.info(
        "clean_news completed | rows=%s | output=%s | article_ratio=%.3f | title_ratio=%.3f",
        len(news_df),
        output_path,
        article_ratio,
        title_ratio,
    )
