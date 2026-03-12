from __future__ import annotations

import logging
from csv import Error as CsvError
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def _get_date_bounds(config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Extract and normalize configured date boundaries in UTC."""
    start = pd.to_datetime(config["dates"]["start"], utc=True)
    end = pd.to_datetime(config["dates"]["end"], utc=True)
    return start, end


def _resolve_input_files(raw_dir: Path, tickers: Sequence[str] | None) -> list[Path]:
    """Resolve ticker file paths from the configured directory and optional subset."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw prices directory does not exist: {raw_dir}")

    if tickers:
        resolved: list[Path] = []
        missing: list[str] = []
        for ticker in tickers:
            ticker_file = raw_dir / f"{ticker.lower()}.us.txt"
            if ticker_file.exists():
                resolved.append(ticker_file)
            else:
                missing.append(ticker)
        if missing:
            log.warning("Some requested tickers were not found: %s", sorted(missing))
        return sorted(resolved)

    return sorted(raw_dir.glob("*.us.txt"))


def _load_single_price_file(file_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load and clean one ticker price file."""
    read_attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ",", "engine": "c"},
        {"sep": "\t", "engine": "c"},
    ]

    df: pd.DataFrame | None = None
    last_exc: Exception | None = None
    for kwargs in read_attempts:
        try:
            df = pd.read_csv(file_path, **kwargs)
            break
        except (pd.errors.ParserError, CsvError, UnicodeDecodeError, ValueError, OSError) as exc:
            last_exc = exc

    if df is None:
        log.warning("Skipping file due to read error: %s | %s", file_path.name, last_exc)
        return pd.DataFrame()

    column_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=column_map)

    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        log.warning("Skipping file with missing required columns: %s", file_path.name)
        return pd.DataFrame()

    ticker = file_path.name.replace(".us.txt", "").upper()

    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df[df["close"] > 0]
    if df.empty:
        return df

    df["ticker"] = ticker
    df["volume_zero_flag"] = (df["volume"] == 0).astype("int8")

    # Explicit dtypes keep parquet output compact and stable.
    df["open"] = df["open"].astype("float32")
    df["high"] = df["high"].astype("float32")
    df["low"] = df["low"].astype("float32")
    df["close"] = df["close"].astype("float32")
    df["volume"] = df["volume"].astype("float64")

    return df[["ticker", "date", "open", "high", "low", "close", "volume", "volume_zero_flag"]]


def load_prices(config: dict[str, Any], tickers: Sequence[str] | None = None) -> None:
    """Load raw price files and persist the interim prices dataset.

    Args:
        config: Parsed pipeline configuration dictionary.
        tickers: Optional subset of tickers for development runs.
    """
    paths_cfg = config.get("paths", {})
    raw_dir = Path(paths_cfg.get("raw_prices_dir", "datasets/Data/Stocks/"))
    out_path = Path(paths_cfg.get("interim_prices", "datasets/interim/prices_combined.parquet"))

    start, end = _get_date_bounds(config)
    input_files = _resolve_input_files(raw_dir, tickers)
    if not input_files:
        raise FileNotFoundError(f"No input price files found in {raw_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    processed_files = 0

    for idx, file_path in enumerate(input_files, start=1):
        cleaned = _load_single_price_file(file_path, start=start, end=end)
        if cleaned.empty:
            continue

        table = pa.Table.from_pandas(cleaned, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)

        processed_files += 1
        total_rows += len(cleaned)
        if idx % 500 == 0:
            log.info("Price ingestion progress: %s/%s files", idx, len(input_files))

    if writer is None:
        raise ValueError("Price ingestion produced zero rows after cleaning.")

    writer.close()
    log.info(
        "load_prices completed | files=%s | rows=%s | output=%s",
        processed_files,
        total_rows,
        out_path,
    )
