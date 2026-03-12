from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RAW_PRICE_COLUMNS = {"open", "high", "low", "close", "volume", "volume_zero_flag"}
NEWS_FILL_ZERO_COLUMNS = {
    "finbert_sentiment_mean",
    "finbert_pct_positive",
    "finbert_pct_negative",
    "finbert_pct_neutral",
    "news_count",
    "char_count_mean",
    "word_count_mean",
    "avg_word_length_mean",
}
NEWS_WEIGHTED_COLUMNS = [
    "finbert_sentiment_mean",
    "finbert_pct_positive",
    "finbert_pct_negative",
    "finbert_pct_neutral",
    "char_count_mean",
    "word_count_mean",
    "avg_word_length_mean",
]


def _resolve_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Resolve integration input paths.

    Args:
        config: Parsed pipeline configuration dictionary.

    Returns:
        Tuple of price features path, news features path, and macro path.
    """
    paths_cfg = config.get("paths", {})
    return (
        Path(paths_cfg.get("price_features", "datasets/features/price_features.parquet")),
        Path(paths_cfg.get("news_features", "datasets/features/news_features.parquet")),
        Path(paths_cfg.get("macro_indicators", "datasets/external/macro_indicators.parquet")),
    )


def _load_price_features(price_path: Path) -> pd.DataFrame:
    """Load and validate price features parquet.

    Args:
        price_path: Price features parquet path.

    Returns:
        Cleaned price-features dataframe.
    """
    if not price_path.exists():
        raise FileNotFoundError(f"Missing price features parquet: {price_path}")

    price_df = pd.read_parquet(price_path)
    required = {"ticker", "date", "target_realized_vol_5d", "target_start_date", "realized_vol_5d"}
    missing = sorted(required - set(price_df.columns))
    if missing:
        raise ValueError(f"Price features parquet missing required columns: {missing}")

    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce", utc=True)
    price_df["target_start_date"] = pd.to_datetime(price_df["target_start_date"], errors="coerce", utc=True)
    price_df = price_df.dropna(subset=["ticker", "date"]).copy()
    price_df["ticker"] = price_df["ticker"].astype("string").str.upper()
    price_df = price_df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return price_df.reset_index(drop=True)


def _load_news_features(news_path: Path) -> pd.DataFrame:
    """Load and validate news features parquet.

    Args:
        news_path: News features parquet path.

    Returns:
        Cleaned news-features dataframe.
    """
    if not news_path.exists():
        raise FileNotFoundError(f"Missing news features parquet: {news_path}")

    news_df = pd.read_parquet(news_path)
    required = {"ticker", "date", *NEWS_FILL_ZERO_COLUMNS}
    missing = sorted(required - set(news_df.columns))
    if missing:
        raise ValueError(f"News features parquet missing required columns: {missing}")

    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce", utc=True)
    news_df = news_df.dropna(subset=["ticker", "date"]).copy()
    news_df["ticker"] = news_df["ticker"].astype("string").str.upper()
    for col in NEWS_FILL_ZERO_COLUMNS:
        news_df[col] = pd.to_numeric(news_df[col], errors="coerce")

    return news_df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _weighted_daily_news_aggregation(news_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize aggregated news features to ticker x calendar-day granularity.

    Args:
        news_df: News features dataframe.

    Returns:
        News dataframe aggregated at ticker x calendar day level.
    """
    df = news_df.copy()
    df["date"] = df["date"].dt.normalize()
    df["news_count_weight"] = pd.to_numeric(df["news_count"], errors="coerce").fillna(0.0)

    for col in NEWS_WEIGHTED_COLUMNS:
        df[f"__weighted_{col}"] = pd.to_numeric(df[col], errors="coerce") * df["news_count_weight"]

    grouped = df.groupby(["ticker", "date"], observed=True)
    aggregated = grouped.agg(
        news_count=("news_count_weight", "sum"),
        source_rows=("ticker", "size"),
        **{f"__weighted_{col}": (f"__weighted_{col}", "sum") for col in NEWS_WEIGHTED_COLUMNS},
    ).reset_index()

    weight = aggregated["news_count"].replace(0, np.nan)
    for col in NEWS_WEIGHTED_COLUMNS:
        aggregated[col] = aggregated[f"__weighted_{col}"] / weight

    aggregated = aggregated.drop(columns=[f"__weighted_{col}" for col in NEWS_WEIGHTED_COLUMNS])
    return aggregated.sort_values(["ticker", "date"]).reset_index(drop=True)


def _align_news_to_trading_days(news_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align news rows to the next available trading day per ticker.

    Args:
        news_df: Ticker x calendar-day aggregated news features.
        price_df: Price feature dataframe with trading days.

    Returns:
        Tuple with trading-day aligned news dataframe and alignment statistics.
    """
    price_calendar = (
        price_df[["ticker", "date"]]
        .drop_duplicates()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    news_for_alignment = (
        news_df.rename(columns={"date": "news_date"})
        .sort_values(["news_date", "ticker"])
        .reset_index(drop=True)
    )
    trading_dates = (
        price_calendar.rename(columns={"date": "trading_date"})
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )

    aligned = pd.merge_asof(
        news_for_alignment,
        trading_dates,
        left_on="news_date",
        right_on="trading_date",
        by="ticker",
        direction="forward",
        allow_exact_matches=True,
    )
    aligned = aligned.dropna(subset=["trading_date"]).copy()

    stats = {
        "news_rows_before_alignment": int(len(news_df)),
        "news_rows_after_alignment": int(len(aligned)),
        "news_rows_unmapped": int(len(news_df) - len(aligned)),
    }

    if aligned.empty:
        empty = pd.DataFrame(columns=["ticker", "date", *sorted(NEWS_FILL_ZERO_COLUMNS)])
        return empty, stats

    for col in NEWS_WEIGHTED_COLUMNS:
        aligned[f"__weighted_{col}"] = pd.to_numeric(aligned[col], errors="coerce") * aligned["news_count"].fillna(0.0)

    grouped = aligned.groupby(["ticker", "trading_date"], observed=True)
    aligned_daily = grouped.agg(
        news_count=("news_count", "sum"),
        source_rows=("source_rows", "sum"),
        **{f"__weighted_{col}": (f"__weighted_{col}", "sum") for col in NEWS_WEIGHTED_COLUMNS},
    ).reset_index()

    weight = aligned_daily["news_count"].replace(0, np.nan)
    for col in NEWS_WEIGHTED_COLUMNS:
        aligned_daily[col] = aligned_daily[f"__weighted_{col}"] / weight

    aligned_daily = aligned_daily.rename(columns={"trading_date": "date"})
    aligned_daily = aligned_daily.drop(columns=[f"__weighted_{col}" for col in NEWS_WEIGHTED_COLUMNS])
    return aligned_daily.sort_values(["ticker", "date"]).reset_index(drop=True), stats


def _ensure_macro_presence(price_df: pd.DataFrame, macro_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Ensure macro columns exist in the price feature frame.

    Args:
        price_df: Price features dataframe.
        macro_path: Macro indicators parquet path.
        config: Parsed pipeline configuration dictionary.

    Returns:
        Price dataframe containing lagged macro columns.
    """
    if any(col.startswith("macro_") for col in price_df.columns):
        return price_df

    if not macro_path.exists():
        raise FileNotFoundError(f"Missing macro parquet required for integration: {macro_path}")

    macro_df = pd.read_parquet(macro_path)
    macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce", utc=True)
    macro_df = macro_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    series_ids = list(config.get("macro", {}).get("fred_series", {}).values())
    missing = sorted(set(series_ids) - set(macro_df.columns))
    if missing:
        raise ValueError(f"Macro parquet missing configured FRED columns: {missing}")

    out = macro_df[["date", *series_ids]].copy()
    for col in series_ids:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[series_ids] = out[series_ids].ffill()
    out["macro_effective_date"] = out["date"] + pd.DateOffset(months=1)
    rename_map = {col: f"macro_{col}" for col in series_ids}
    out = out.rename(columns=rename_map).drop(columns=["date"]).sort_values("macro_effective_date")

    merged = pd.merge_asof(
        price_df.sort_values("date").reset_index(drop=True),
        out,
        left_on="date",
        right_on="macro_effective_date",
        direction="backward",
    )
    return merged.drop(columns=["macro_effective_date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def _assign_dataset_split(feature_df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Assign temporal split label per row.

    Args:
        feature_df: Integrated feature matrix.
        config: Parsed pipeline configuration dictionary.

    Returns:
        Series with train, validation, or test labels.
    """
    train_end = pd.to_datetime(config["dates"]["train_end"], utc=True)
    validation_end = pd.to_datetime(config["dates"]["validation_end"], utc=True)

    split = np.where(
        feature_df["date"] <= train_end,
        "train",
        np.where(feature_df["date"] <= validation_end, "validation", "test"),
    )
    return pd.Series(split, index=feature_df.index, dtype="string")


def _drop_high_missing_columns(feature_df: pd.DataFrame, max_missing_pct: float) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns exceeding the configured missing threshold.

    Args:
        feature_df: Integrated feature matrix.
        max_missing_pct: Maximum missing fraction allowed.

    Returns:
        Tuple of filtered dataframe and dropped columns.
    """
    protected = {"ticker", "date", "dataset_split", "target_realized_vol_5d", "target_start_date"}
    missing_pct = feature_df.isna().mean()
    drop_cols = [col for col, pct in missing_pct.items() if pct > max_missing_pct and col not in protected]
    if not drop_cols:
        return feature_df, []
    return feature_df.drop(columns=drop_cols), sorted(drop_cols)


def _prepare_model_ready_matrix(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply final row/feature hygiene for a model-ready matrix.

    Args:
        feature_df: Integrated feature matrix.

    Returns:
        Tuple with cleaned feature matrix and hygiene counters.
    """
    out = feature_df.copy()

    realized_vol_cols = [col for col in out.columns if col.startswith("realized_vol_")]
    for col in realized_vol_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    dropped_non_positive = 0
    if "realized_vol_5d" in out.columns:
        non_positive_mask = out["realized_vol_5d"].notna() & (out["realized_vol_5d"] <= 0)
        dropped_non_positive = int(non_positive_mask.sum())
        if dropped_non_positive > 0:
            out = out.loc[~non_positive_mask].copy()

    before_target_drop = len(out)
    out = out[out["target_realized_vol_5d"].notna()].copy()
    dropped_null_target = int(before_target_drop - len(out))
    return out.reset_index(drop=True), {
        "dropped_non_positive_realized_vol_5d": dropped_non_positive,
        "dropped_null_target_rows": dropped_null_target,
    }


def join_all_sources(config: dict[str, Any]) -> pd.DataFrame:
    """Join price, news, and macro features into one matrix.

    Args:
        config: Parsed pipeline configuration dictionary.

    Returns:
        Integrated feature matrix dataframe.
    """
    price_path, news_path, macro_path = _resolve_paths(config)
    price_df = _load_price_features(price_path)
    price_df = _ensure_macro_presence(price_df, macro_path, config)
    news_df = _load_news_features(news_path)

    news_daily = _weighted_daily_news_aggregation(news_df)
    aligned_news, alignment_stats = _align_news_to_trading_days(news_daily, price_df)

    joined = price_df.merge(aligned_news, on=["ticker", "date"], how="left", suffixes=("", "_news"))
    no_news_mask = joined["news_count"].isna() if "news_count" in joined.columns else pd.Series(True, index=joined.index)
    no_news_rate = float(no_news_mask.mean()) if len(joined) else 0.0

    for col in NEWS_FILL_ZERO_COLUMNS:
        if col in joined.columns:
            joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0.0)

    joined = joined.drop(columns=[col for col in RAW_PRICE_COLUMNS if col in joined.columns])
    joined["dataset_split"] = _assign_dataset_split(joined, config)
    joined, hygiene_stats = _prepare_model_ready_matrix(joined)

    max_missing_pct = float(config.get("features", {}).get("max_missing_pct", 0.5))
    joined, dropped_columns = _drop_high_missing_columns(joined, max_missing_pct=max_missing_pct)
    joined = joined.sort_values(["ticker", "date"]).reset_index(drop=True)

    feature_columns = [
        col
        for col in joined.columns
        if col not in {"ticker", "date", "dataset_split", "target_realized_vol_5d", "target_start_date"}
    ]
    joined.attrs["integration_stats"] = {
        **alignment_stats,
        **hygiene_stats,
        "rows": int(len(joined)),
        "columns": int(len(joined.columns)),
        "feature_count": int(len(feature_columns)),
        "rows_without_news_before_fill": int(no_news_mask.sum()),
        "rows_without_news_pct_before_fill": no_news_rate,
        "dropped_high_missing_columns": dropped_columns,
    }

    log.info(
        "join_all_sources completed | rows=%s | cols=%s | no_news_pct=%.4f | dropped_cols=%s",
        len(joined),
        len(joined.columns),
        no_news_rate,
        dropped_columns,
    )
    return joined
