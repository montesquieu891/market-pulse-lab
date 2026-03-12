from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

_EXCHANGE_SUFFIXES = {
    "A",
    "AX",
    "B",
    "C",
    "K",
    "L",
    "N",
    "O",
    "PA",
    "SA",
    "TO",
}


def _normalize_ticker(raw_value: Any) -> str:
    """Normalize ticker text and remove common exchange suffixes."""
    if pd.isna(raw_value):
        return ""

    ticker = str(raw_value).strip().upper()
    ticker = re.sub(r"\s+", "", ticker)
    if not ticker:
        return ""

    if "." in ticker:
        base, suffix = ticker.rsplit(".", 1)
        if suffix in _EXCHANGE_SUFFIXES:
            ticker = base

    ticker = re.sub(r"[^A-Z0-9.\-]", "", ticker)
    return ticker


def _prepare_chunk(chunk: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Clean and normalize one news chunk."""
    lower_cols = {col: col.strip().lower() for col in chunk.columns}
    chunk = chunk.rename(columns=lower_cols)

    if "stock" in chunk.columns and "ticker" not in chunk.columns:
        chunk = chunk.rename(columns={"stock": "ticker"})
    if "headline" in chunk.columns and "title" not in chunk.columns:
        chunk = chunk.rename(columns={"headline": "title"})

    if "date" not in chunk.columns or "ticker" not in chunk.columns:
        return pd.DataFrame()

    if "title" not in chunk.columns:
        chunk["title"] = pd.NA
    if "article" not in chunk.columns:
        chunk["article"] = pd.NA
    if "url" not in chunk.columns:
        chunk["url"] = pd.NA
    if "publisher" not in chunk.columns:
        chunk["publisher"] = pd.NA

    out = chunk[["ticker", "date", "title", "article", "url", "publisher"]].copy()
    out["ticker"] = out["ticker"].map(_normalize_ticker)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out = out.dropna(subset=["date"])
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    out = out[out["ticker"] != ""]

    has_article = out["article"].astype("string").str.strip().fillna("") != ""
    out["text_source"] = has_article.map({True: "article", False: "title"}).astype("string")

    out["ticker"] = out["ticker"].astype("string")
    out["title"] = out["title"].astype("string")
    out["article"] = out["article"].astype("string")
    out["url"] = out["url"].astype("string")
    out["publisher"] = out["publisher"].astype("string")

    return out[["ticker", "date", "title", "article", "url", "publisher", "text_source"]]


def load_news(config: dict[str, Any]) -> None:
    """Load and normalize raw news data for downstream NLP steps.

    Args:
        config: Parsed pipeline configuration dictionary.
    """
    paths_cfg = config.get("paths", {})
    raw_news = Path(paths_cfg.get("raw_news", "datasets/raw_analyst_ratings.csv"))
    out_path = Path(paths_cfg.get("interim_news", "datasets/interim/news_cleaned.parquet"))

    if not raw_news.exists():
        raise FileNotFoundError(f"Raw news file not found: {raw_news}")

    start = pd.to_datetime(config["dates"]["start"], utc=True)
    end = pd.to_datetime(config["dates"]["end"], utc=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    chunks = pd.read_csv(raw_news, chunksize=250_000, low_memory=False)
    for idx, chunk in enumerate(chunks, start=1):
        cleaned = _prepare_chunk(chunk, start=start, end=end)
        if cleaned.empty:
            continue

        table = pa.Table.from_pandas(cleaned, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)

        total_rows += len(cleaned)
        if idx % 5 == 0:
            log.info("News ingestion progress: %s chunks processed", idx)

    if writer is None:
        raise ValueError("News ingestion produced zero rows after cleaning.")

    writer.close()
    log.info("load_news completed | rows=%s | output=%s", total_rows, out_path)
