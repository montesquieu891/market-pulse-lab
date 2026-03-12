from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def _load_fred_series(series_id: str) -> pd.DataFrame:
    """Download one FRED series via the public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    series_df = pd.read_csv(url)

    # Endpoint schema can vary between DATE and observation_date.
    date_col = None
    for candidate in ("DATE", "date", "observation_date", "observation date"):
        if candidate in series_df.columns:
            date_col = candidate
            break

    value_col = series_id if series_id in series_df.columns else None
    if value_col is None:
        non_date_cols = [c for c in series_df.columns if c != date_col]
        value_col = non_date_cols[0] if non_date_cols else None

    if date_col is None or value_col is None:
        raise ValueError(f"Unexpected FRED schema for series {series_id}")

    series_df[date_col] = pd.to_datetime(series_df[date_col], errors="coerce", utc=True)
    series_df[value_col] = pd.to_numeric(series_df[value_col], errors="coerce")
    series_df = series_df.rename(columns={date_col: "date", value_col: series_id})
    return series_df[["date", series_id]].dropna(subset=["date"])


def fetch_macro(config: dict[str, Any]) -> None:
    """Fetch and store macroeconomic indicators used for enrichment.

    Skips the network request if the output file already exists (subsequent
    offline runs). On network errors, falls back to the existing file when
    available rather than crashing the pipeline.

    Args:
        config: Parsed pipeline configuration dictionary.
    """
    series_cfg = config.get("macro", {}).get("fred_series", {})
    if not series_cfg:
        raise ValueError("No FRED series configured under macro.fred_series")

    out_path = Path(
        config.get("paths", {}).get("macro_indicators", "datasets/external/macro_indicators.parquet")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip network fetch on subsequent runs (CLAUDE.md: internet only for first run).
    if out_path.exists():
        log.info("fetch_macro skipped — cached file already exists | output=%s", out_path)
        return

    merged: pd.DataFrame | None = None
    try:
        for _, series_id in series_cfg.items():
            one = _load_fred_series(series_id)
            merged = one if merged is None else merged.merge(one, on="date", how="outer")
    except Exception as exc:  # network errors, timeouts, HTTP errors
        if out_path.exists():
            log.warning(
                "fetch_macro network error (%s); using cached file: %s", exc, out_path
            )
            return
        raise RuntimeError(
            f"fetch_macro failed and no cached file exists at {out_path}: {exc}"
        ) from exc

    if merged is None:
        raise ValueError("No macro data was downloaded.")

    start = pd.to_datetime(config["dates"]["start"], utc=True)
    end = pd.to_datetime(config["dates"]["end"], utc=True)
    merged = merged[(merged["date"] >= start) & (merged["date"] <= end)]
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_parquet(out_path, index=False)

    log.info("fetch_macro completed | rows=%s | output=%s", len(merged), out_path)
