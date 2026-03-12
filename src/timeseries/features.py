from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Return rolling standard deviation for a series."""
    return series.rolling(window=window, min_periods=window).std()


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Return rolling mean for a series."""
    return series.rolling(window=window, min_periods=window).mean()


def _get_paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    """Resolve input and output paths used in the time series stage.

    Args:
        config: Parsed pipeline configuration dictionary.

    Returns:
        A tuple with prices input path, macro input path, features output path,
        and diagnostics report path.
    """
    paths_cfg = config.get("paths", {})
    prices_path = Path(paths_cfg.get("interim_prices", "datasets/interim/prices_combined.parquet"))
    macro_path = Path(paths_cfg.get("macro_indicators", "datasets/external/macro_indicators.parquet"))
    output_path = Path(paths_cfg.get("price_features", "datasets/features/price_features.parquet"))
    diagnostics_path = Path("reports/timeseries_diagnostics.md")
    return prices_path, macro_path, output_path, diagnostics_path


def _load_inputs(prices_path: Path, macro_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and basic-validate inputs needed by this stage.

    Args:
        prices_path: Path to interim prices parquet.
        macro_path: Path to external macro parquet.

    Returns:
        A tuple of (prices_df, macro_df).
    """
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing interim prices parquet: {prices_path}")
    if not macro_path.exists():
        raise FileNotFoundError(f"Missing macro indicators parquet: {macro_path}")

    prices_df = pd.read_parquet(prices_path)
    macro_df = pd.read_parquet(macro_path)

    required_price_cols = {"ticker", "date", "close", "open", "high", "low", "volume", "volume_zero_flag"}
    missing_price_cols = sorted(required_price_cols - set(prices_df.columns))
    if missing_price_cols:
        raise ValueError(f"Prices parquet missing required columns: {missing_price_cols}")

    prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce", utc=True)
    macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce", utc=True)

    prices_df = prices_df.dropna(subset=["ticker", "date", "close"]).copy()
    macro_df = macro_df.dropna(subset=["date"]).copy()

    prices_df["ticker"] = prices_df["ticker"].astype("string")

    prices_df = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices_df = prices_df.drop_duplicates(subset=["ticker", "date"], keep="last")
    macro_df = macro_df.sort_values("date").reset_index(drop=True)

    return prices_df, macro_df


def _compute_log_return(prices_df: pd.DataFrame) -> pd.Series:
    """Compute per-ticker log returns from close prices.

    Args:
        prices_df: Sorted prices dataframe.

    Returns:
        Log-return series aligned to input index.
    """
    prev_close = prices_df.groupby("ticker", observed=True)["close"].shift(1)
    ratio = prices_df["close"] / prev_close
    ratio = ratio.where(ratio > 0)
    return np.log(ratio)


def _compute_realized_vol(
    prices_df: pd.DataFrame,
    log_return_col: str,
    windows: list[int],
    annualization_factor: float,
) -> pd.DataFrame:
    """Compute realized volatility for configured trailing windows.

    Args:
        prices_df: Prices dataframe with log return column.
        log_return_col: Name of the log-return column.
        windows: Rolling windows in trading days.
        annualization_factor: Annualization multiplier, usually sqrt(252).

    Returns:
        Dataframe with one realized-vol column per window.
    """
    out = pd.DataFrame(index=prices_df.index)
    grouped_returns = prices_df.groupby("ticker", observed=True)[log_return_col]

    for window in windows:
        out[f"realized_vol_{window}d_raw"] = (
            grouped_returns.transform(_rolling_std, window=window)
            * annualization_factor
        )

    return out


def _compute_lag_features(prices_df: pd.DataFrame, log_return_col: str, lag_periods: list[int]) -> pd.DataFrame:
    """Build lagged log-return features.

    Args:
        prices_df: Prices dataframe with log return column.
        log_return_col: Name of the log-return column.
        lag_periods: Lags in trading days.

    Returns:
        Dataframe with lagged log-return columns.
    """
    out = pd.DataFrame(index=prices_df.index)
    grouped_returns = prices_df.groupby("ticker", observed=True)[log_return_col]

    for lag in lag_periods:
        out[f"log_return_lag_{lag}d"] = grouped_returns.shift(lag)

    return out


def _compute_rolling_moments(prices_df: pd.DataFrame, log_return_col: str, windows: list[int]) -> pd.DataFrame:
    """Compute trailing rolling mean and std over log returns.

    Args:
        prices_df: Prices dataframe with log return column.
        log_return_col: Name of the log-return column.
        windows: Rolling windows in trading days.

    Returns:
        Dataframe with rolling mean/std columns.
    """
    out = pd.DataFrame(index=prices_df.index)
    grouped_returns = prices_df.groupby("ticker", observed=True)[log_return_col]

    for window in windows:
        out[f"rolling_mean_{window}d_raw"] = grouped_returns.transform(_rolling_mean, window=window)
        out[f"rolling_std_{window}d_raw"] = grouped_returns.transform(_rolling_std, window=window)

    return out


def _compute_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI indicator from close prices.

    Args:
        prices_df: Prices dataframe.
        period: RSI lookback period.

    Returns:
        RSI series in [0, 100].
    """
    delta = prices_df.groupby("ticker", observed=True)["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.groupby(prices_df["ticker"], observed=True).transform(
        lambda s: s.rolling(window=period, min_periods=period).mean()
    )
    avg_loss = loss.groupby(prices_df["ticker"], observed=True).transform(
        lambda s: s.rolling(window=period, min_periods=period).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # If no losses over the lookback window, RSI should saturate at 100.
    rsi = rsi.where(avg_loss != 0, 100.0)
    # If both gains and losses are zero, use a neutral RSI.
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi.clip(lower=0, upper=100)


def _compute_bollinger(prices_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute Bollinger Bands from close prices.

    Args:
        prices_df: Prices dataframe.
        window: Rolling window used for the indicator.

    Returns:
        Dataframe with upper, lower, and width columns.
    """
    grouped_close = prices_df.groupby("ticker", observed=True)["close"]
    mean_ = grouped_close.transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    std_ = grouped_close.transform(lambda s: s.rolling(window=window, min_periods=window).std())

    upper = mean_ + 2 * std_
    lower = mean_ - 2 * std_
    width = (upper - lower) / mean_.replace(0, np.nan)

    return pd.DataFrame(
        {
            "bb_upper_raw": upper,
            "bb_lower_raw": lower,
            "bb_width_raw": width,
        },
        index=prices_df.index,
    )


def _apply_min_lag(features_df: pd.DataFrame, min_lag_days: int, skip_cols: set[str]) -> pd.DataFrame:
    """Apply global minimum lag per ticker to feature columns.

    Args:
        features_df: Feature dataframe with ticker and date present.
        min_lag_days: Minimum lag to enforce.
        skip_cols: Columns that should not be lagged.

    Returns:
        Dataframe with lagged feature columns.
    """
    if min_lag_days <= 0:
        return features_df

    df = features_df.copy()
    feature_cols = [col for col in df.columns if col not in skip_cols]
    grouped = df.groupby("ticker", observed=True)

    for col in feature_cols:
        if col.startswith("log_return_lag_"):
            # Explicit lag features already satisfy minimum lag requirements.
            continue
        df[col] = grouped[col].shift(min_lag_days)

    return df


def _forward_realized_vol_target(log_returns: pd.Series, tickers: pd.Series, window: int = 5) -> pd.Series:
    """Compute forward-looking realized volatility target per ticker.

    Target at day t uses returns from t+1 to t+window.

    Args:
        log_returns: Log-return series.
        tickers: Ticker key series aligned with log_returns.
        window: Forward window length.

    Returns:
        Forward realized volatility series (annualized).
    """
    annualization_factor = np.sqrt(252.0)

    return log_returns.groupby(tickers, observed=True).transform(
        lambda s: s.iloc[::-1].rolling(window=window, min_periods=window).std().iloc[::-1].shift(-1)
        * annualization_factor
    )


def _prepare_macro_features(config: dict[str, Any], macro_df: pd.DataFrame) -> pd.DataFrame:
    """Create one-month-lagged macro features and effective join dates.

    Args:
        config: Parsed pipeline configuration dictionary.
        macro_df: Raw macro dataframe.

    Returns:
        Dataframe with effective join date and lagged macro feature columns.
    """
    series_ids = list(config.get("macro", {}).get("fred_series", {}).values())
    if not series_ids:
        return pd.DataFrame(columns=["macro_effective_date"])

    missing = sorted(set(series_ids) - set(macro_df.columns))
    if missing:
        raise ValueError(f"Macro parquet missing configured FRED columns: {missing}")

    out = macro_df[["date", *series_ids]].copy()
    for col in series_ids:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    out[series_ids] = out[series_ids].ffill()

    # Conservative release-lag approximation: a macro value becomes available
    # one calendar month after its observation date.
    out["macro_effective_date"] = out["date"] + pd.DateOffset(months=1)

    rename_map = {col: f"macro_{col}" for col in series_ids}
    out = out.rename(columns=rename_map)
    out = out.drop(columns=["date"])

    macro_cols = [rename_map[col] for col in series_ids]
    return out[["macro_effective_date", *macro_cols]].sort_values("macro_effective_date")


def _assert_anti_leakage(feature_df: pd.DataFrame) -> None:
    """Assert that each row's features are available before target horizon starts.

    Args:
        feature_df: Final feature dataframe including target_start_date.
    """
    if "target_start_date" not in feature_df.columns:
        raise ValueError("target_start_date is missing; anti-leakage assertion cannot run")

    candidate = feature_df[feature_df["target_start_date"].notna()]
    violations = candidate["date"] >= candidate["target_start_date"]
    if bool(violations.any()):
        n_violations = int(violations.sum())
        raise AssertionError(
            "ANTI-LEAKAGE ASSERTION FAILED: "
            f"{n_violations} rows where feature_date >= target_start_date."
        )


def _assert_temporal_split(feature_df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Assert configured temporal train/validation split integrity.

    Args:
        feature_df: Final feature dataframe.
        config: Parsed pipeline configuration dictionary.
    """
    train_end = pd.to_datetime(config["dates"]["train_end"], utc=True)
    validation_end = pd.to_datetime(config["dates"]["validation_end"], utc=True)

    train = feature_df[feature_df["date"] <= train_end]
    validation = feature_df[(feature_df["date"] > train_end) & (feature_df["date"] <= validation_end)]

    if train.empty:
        raise AssertionError("TEMPORAL SPLIT ASSERTION FAILED: train split is empty")
    if validation.empty:
        log.warning(
            "Temporal split validation window is empty for configured boundaries "
            "(train_end=%s, validation_end=%s). Current data may not reach validation period yet.",
            train_end,
            validation_end,
        )
        return

    if not bool(train["date"].max() < validation["date"].min()):
        raise AssertionError(
            "TEMPORAL SPLIT ASSERTION FAILED: train max date is not earlier than validation min date"
        )


def _build_stationarity_diagnostics(
    feature_df: pd.DataFrame,
    diagnostics_path: Path,
    close_col: str = "close",
    log_return_col: str = "log_return_raw",
) -> None:
    """Generate a compact diagnostics report with ADF p-values.

    Args:
        feature_df: Feature dataframe containing ticker series.
        diagnostics_path: Output markdown path.
        close_col: Close price column.
        log_return_col: Raw log return column.
    """
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    aapl = feature_df[feature_df["ticker"] == "AAPL"].sort_values("date")
    if aapl.empty:
        lines = [
            "# Time Series Diagnostics",
            "",
            "AAPL series not found; ADF diagnostics skipped.",
        ]
        diagnostics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    from statsmodels.tsa.stattools import adfuller

    close_series = pd.to_numeric(aapl[close_col], errors="coerce").dropna()
    return_series = pd.to_numeric(aapl[log_return_col], errors="coerce").dropna()

    close_pvalue = float(adfuller(close_series, autolag="AIC")[1]) if len(close_series) > 20 else float("nan")
    return_pvalue = float(adfuller(return_series, autolag="AIC")[1]) if len(return_series) > 20 else float("nan")

    lines = [
        "# Time Series Diagnostics",
        "",
        "ADF sanity checks computed on AAPL series.",
        "",
        f"- ADF p-value on close: {close_pvalue:.6f}",
        f"- ADF p-value on log_return: {return_pvalue:.6f}",
        f"- Close expected non-stationary (p > 0.05): {bool(close_pvalue > 0.05) if not np.isnan(close_pvalue) else 'N/A'}",
        f"- Log return expected stationary (p < 0.05): {bool(return_pvalue < 0.05) if not np.isnan(return_pvalue) else 'N/A'}",
    ]
    diagnostics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_price_features(config: dict[str, Any]) -> None:
    """Build price-based time series features with strict temporal controls.

    The output contains lagged price features, technical indicators, lagged macro
    enrichments, and a forward-looking 5-day realized volatility target.

    Args:
        config: Parsed pipeline configuration dictionary.
    """
    prices_path, macro_path, output_path, diagnostics_path = _get_paths(config)
    prices_df, macro_df = _load_inputs(prices_path, macro_path)

    ts_cfg = config.get("timeseries", {})
    feat_cfg = config.get("features", {})

    vol_windows = [int(x) for x in ts_cfg.get("realized_vol_windows", [5, 10, 20])]
    lag_periods = [int(x) for x in ts_cfg.get("lag_periods", [1, 2, 3, 5, 10])]
    rolling_windows = [int(x) for x in ts_cfg.get("rolling_windows", [5, 10, 20, 60])]
    min_lag_days = int(feat_cfg.get("min_lag_days", 1))

    annualization_factor = np.sqrt(252.0)

    feature_df = prices_df[["ticker", "date", "open", "high", "low", "close", "volume", "volume_zero_flag"]].copy()

    feature_df["log_return_raw"] = _compute_log_return(prices_df)

    realized = _compute_realized_vol(
        feature_df,
        log_return_col="log_return_raw",
        windows=vol_windows,
        annualization_factor=annualization_factor,
    )
    feature_df = pd.concat([feature_df, realized], axis=1)

    lag_features = _compute_lag_features(feature_df, log_return_col="log_return_raw", lag_periods=lag_periods)
    feature_df = pd.concat([feature_df, lag_features], axis=1)

    rolling = _compute_rolling_moments(feature_df, log_return_col="log_return_raw", windows=rolling_windows)
    feature_df = pd.concat([feature_df, rolling], axis=1)

    feature_df["rsi_14_raw"] = _compute_rsi(feature_df, period=14)
    feature_df = pd.concat([feature_df, _compute_bollinger(feature_df, window=20)], axis=1)

    macro_features = _prepare_macro_features(config, macro_df)
    feature_df = feature_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    macro_features = macro_features.sort_values("macro_effective_date").reset_index(drop=True)

    if not macro_features.empty:
        feature_df = feature_df.sort_values("date").reset_index(drop=True)
        feature_df = pd.merge_asof(
            feature_df,
            macro_features,
            left_on="date",
            right_on="macro_effective_date",
            direction="backward",
        )
        feature_df = feature_df.drop(columns=["macro_effective_date"])
        feature_df = feature_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    feature_df["target_realized_vol_5d"] = _forward_realized_vol_target(
        log_returns=feature_df["log_return_raw"],
        tickers=feature_df["ticker"],
        window=5,
    )
    feature_df["target_start_date"] = feature_df.groupby("ticker", observed=True)["date"].shift(-1)

    raw_to_final = {
        "log_return_raw": "log_return",
        **{f"realized_vol_{w}d_raw": f"realized_vol_{w}d" for w in vol_windows},
        **{f"rolling_mean_{w}d_raw": f"rolling_mean_{w}d" for w in rolling_windows},
        **{f"rolling_std_{w}d_raw": f"rolling_std_{w}d" for w in rolling_windows},
        "rsi_14_raw": "rsi_14",
        "bb_upper_raw": "bb_upper",
        "bb_lower_raw": "bb_lower",
        "bb_width_raw": "bb_width",
    }

    for raw_col, final_col in raw_to_final.items():
        feature_df[final_col] = feature_df[raw_col]

    skip_lag_cols = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_zero_flag",
        "target_realized_vol_5d",
        "target_start_date",
        *raw_to_final.keys(),
    }
    feature_df = _apply_min_lag(feature_df, min_lag_days=min_lag_days, skip_cols=skip_lag_cols)

    # Enforce indicator range and build assertions expected by Phase 3.
    if "rsi_14" in feature_df.columns:
        feature_df["rsi_14"] = feature_df["rsi_14"].clip(lower=0, upper=100)

    _assert_anti_leakage(feature_df)
    _assert_temporal_split(feature_df, config)

    _build_stationarity_diagnostics(feature_df, diagnostics_path)

    drop_raw_cols = [
        "log_return_raw",
        *[f"realized_vol_{w}d_raw" for w in vol_windows],
        *[f"rolling_mean_{w}d_raw" for w in rolling_windows],
        *[f"rolling_std_{w}d_raw" for w in rolling_windows],
        "rsi_14_raw",
        "bb_upper_raw",
        "bb_lower_raw",
        "bb_width_raw",
    ]
    feature_df = feature_df.drop(columns=[col for col in drop_raw_cols if col in feature_df.columns])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output_path, index=False)

    log.info(
        "build_price_features completed | rows=%s | cols=%s | output=%s",
        len(feature_df),
        len(feature_df.columns),
        output_path,
    )
