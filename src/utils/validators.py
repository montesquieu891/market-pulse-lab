from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pandas.api.types import is_datetime64tz_dtype

log = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    """Raised when one or more validation checks fail."""


def _require_file(path: Path) -> None:
    """Ensure an expected file exists."""
    if not path.exists():
        raise ValidationError(f"Required file does not exist: {path}")


def _ensure_required_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> list[str]:
    """Return missing-column validation errors for a dataframe."""
    missing = sorted(required - set(df.columns))
    if not missing:
        return []
    return [f"{dataset_name} missing required columns: {missing}"]


def _is_utc_datetime(series: pd.Series) -> bool:
    """Check whether a pandas series is timezone-aware UTC datetime."""
    if not is_datetime64tz_dtype(series.dtype):
        return False
    tz = getattr(series.dtype, "tz", None)
    return str(tz) == "UTC"


def _validate_date_range(
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    label: str,
    start_grace_days: int = 7,
    end_grace_days: int = 0,
) -> list[str]:
    """Validate that date values cover the configured time window."""
    if series.empty:
        return [f"{label} date series is empty"]

    min_date = series.min()
    max_date = series.max()
    errors: list[str] = []
    start_grace = start + pd.Timedelta(days=start_grace_days)
    end_grace = end - pd.Timedelta(days=end_grace_days)
    if min_date > start_grace:
        errors.append(f"{label} min date {min_date} > configured start grace {start_grace}")
    if max_date < end_grace:
        errors.append(f"{label} max date {max_date} < configured end grace {end_grace}")
    return errors


def _resolve_validation_level(
    config: dict[str, Any],
    level_override: Literal["basic", "strict"] | None,
) -> Literal["basic", "strict"]:
    """Resolve effective validation level from CLI override or config."""
    if level_override is not None:
        return level_override

    configured_level = str(config.get("validation", {}).get("default_level", "strict")).lower()
    if configured_level not in {"basic", "strict"}:
        raise ValidationError(
            f"Invalid validation.default_level '{configured_level}'. Expected 'basic' or 'strict'."
        )
    return configured_level  # type: ignore[return-value]


def _run_ge_validation(
    dataframe: pd.DataFrame,
    suite_name: str,
    validation_name: str,
    expectations: list[Any],
) -> dict[str, Any]:
    """Execute a Great Expectations suite against an in-memory dataframe."""
    import great_expectations as gx
    from great_expectations.core.expectation_suite import ExpectationSuite
    from great_expectations.core.validation_definition import ValidationDefinition

    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(f"{validation_name}_src")
    asset = datasource.add_dataframe_asset(name=f"{validation_name}_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")

    suite = ExpectationSuite(name=suite_name)
    for expectation in expectations:
        suite.add_expectation(expectation)
    context.suites.add(suite)

    validation = ValidationDefinition(name=validation_name, data=batch_definition, suite=suite)
    context.validation_definitions.add(validation)
    result = validation.run(batch_parameters={"dataframe": dataframe})
    return result.to_json_dict()


def _summarize_ge_result(result: dict[str, Any], label: str) -> dict[str, Any]:
    """Summarize a GE validation result into a compact structure."""
    statistics = result.get("statistics", {})
    return {
        "label": label,
        "success": bool(result.get("success", False)),
        "evaluated_expectations": int(statistics.get("evaluated_expectations", 0)),
        "successful_expectations": int(statistics.get("successful_expectations", 0)),
        "unsuccessful_expectations": int(statistics.get("unsuccessful_expectations", 0)),
    }


def _validate_feature_matrix(
    config: dict[str, Any],
    feature_matrix_path: Path,
    pipeline_model_path: Path,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the integrated feature matrix and serialized pipeline."""
    _require_file(feature_matrix_path)

    feature_df = pd.read_parquet(feature_matrix_path)
    required = {"ticker", "date", "target_realized_vol_5d", "target_start_date", "realized_vol_5d"}
    errors = _ensure_required_columns(feature_df, required, "feature_matrix")
    if errors:
        return errors, {}, [], []

    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce", utc=True)
    feature_df["target_start_date"] = pd.to_datetime(feature_df["target_start_date"], errors="coerce", utc=True)
    feature_df = feature_df.dropna(subset=["ticker", "date"]).copy()
    feature_df["ticker"] = feature_df["ticker"].astype("string")
    feature_df = feature_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if not _is_utc_datetime(feature_df["date"]):
        errors.append(f"feature_matrix date dtype is not UTC datetime: {feature_df['date'].dtype}")
    if not _is_utc_datetime(feature_df["target_start_date"]):
        errors.append(f"feature_matrix target_start_date dtype is not UTC datetime: {feature_df['target_start_date'].dtype}")

    start = pd.to_datetime(config["dates"]["start"], utc=True)
    end = pd.to_datetime(config["dates"]["end"], utc=True)
    errors.extend(
        _validate_date_range(
            feature_df["date"],
            start,
            end,
            "feature_matrix",
            start_grace_days=7,
            end_grace_days=10,
        )
    )

    duplicate_keys = int(feature_df.duplicated(subset=["ticker", "date"]).sum())
    if duplicate_keys > 0:
        errors.append(f"feature_matrix has {duplicate_keys} duplicated ticker-date rows")

    train_end = pd.to_datetime(config["dates"]["train_end"], utc=True)
    train_df = feature_df[feature_df["date"] <= train_end].copy()
    ge_results: list[dict[str, Any]] = []

    if train_df.empty:
        errors.append("feature_matrix train slice is empty")
    else:
        from great_expectations.expectations import ExpectColumnValuesToNotBeNull

        target_result = _run_ge_validation(
            dataframe=train_df,
            suite_name="feature_matrix_train_suite",
            validation_name="feature_matrix_train_validation",
            expectations=[ExpectColumnValuesToNotBeNull(column="target_realized_vol_5d")],
        )
        ge_results.append(_summarize_ge_result(target_result, "train_target_not_null"))
        if not bool(target_result.get("success", False)):
            errors.append("Great Expectations failed: target_realized_vol_5d contains nulls in train slice")

    realized_vol_df = feature_df[feature_df["realized_vol_5d"].notna()].copy()
    if realized_vol_df.empty:
        errors.append("feature_matrix realized_vol_5d has no non-null rows to validate")
    else:
        from great_expectations.expectations import ExpectColumnValuesToBeBetween

        vol_result = _run_ge_validation(
            dataframe=realized_vol_df,
            suite_name="feature_matrix_positive_vol_suite",
            validation_name="feature_matrix_positive_vol_validation",
            expectations=[
                ExpectColumnValuesToBeBetween(
                    column="realized_vol_5d",
                    min_value=0,
                    strict_min=True,
                )
            ],
        )
        ge_results.append(_summarize_ge_result(vol_result, "realized_vol_5d_positive"))
        if not bool(vol_result.get("success", False)):
            errors.append("Great Expectations failed: realized_vol_5d contains non-positive values")

    target_window_df = feature_df[feature_df["target_start_date"].notna()].copy()
    if target_window_df.empty:
        errors.append("feature_matrix target_start_date has no non-null rows to validate")
    else:
        from great_expectations.expectations import ExpectColumnPairValuesAToBeGreaterThanB

        future_result = _run_ge_validation(
            dataframe=target_window_df,
            suite_name="feature_matrix_temporal_suite",
            validation_name="feature_matrix_temporal_validation",
            expectations=[
                ExpectColumnPairValuesAToBeGreaterThanB(
                    column_A="target_start_date",
                    column_B="date",
                    or_equal=False,
                    ignore_row_if="either_value_is_missing",
                )
            ],
        )
        ge_results.append(_summarize_ge_result(future_result, "target_window_after_feature_date"))
        if not bool(future_result.get("success", False)):
            errors.append("Great Expectations failed: target_start_date is not strictly after feature date")

    monotonic_failures = int(
        (~feature_df.groupby("ticker", observed=True)["date"].apply(lambda s: bool(s.is_monotonic_increasing))).sum()
    )
    manual_checks = [
        {
            "label": "date_monotonic_increasing_per_ticker",
            "success": monotonic_failures == 0,
            "details": f"tickers_with_non_monotonic_dates={monotonic_failures}",
        }
    ]
    if monotonic_failures > 0:
        errors.append(f"feature_matrix has {monotonic_failures} tickers with non-monotonic date ordering")

    feature_columns = [
        col
        for col in feature_df.columns
        if col not in {"ticker", "date", "dataset_split", "target_realized_vol_5d", "target_start_date"}
    ]
    news_count = feature_df.get("news_count", pd.Series(index=feature_df.index, dtype=float)).fillna(0.0)
    summary = {
        "rows": int(len(feature_df)),
        "columns": int(len(feature_df.columns)),
        "feature_count": int(len(feature_columns)),
        "train_rows": int(len(train_df)),
        "rows_without_news_pct": round(float((news_count == 0).mean()), 6),
        "pipeline_model_path": pipeline_model_path.as_posix(),
    }
    return errors, summary, ge_results, manual_checks


def _write_validation_report(
    report_path: Path,
    raw_summary: dict[str, Any],
    feature_summary: dict[str, Any],
    ge_results: list[dict[str, Any]],
    manual_checks: list[dict[str, Any]],
    overall_success: bool,
) -> None:
    """Write a compact HTML validation report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def render_rows(items: list[dict[str, Any]], key_order: list[str]) -> str:
        if not items:
            return "<p>No checks executed.</p>"
        header = "".join(f"<th>{html.escape(key)}</th>" for key in key_order)
        body_rows: list[str] = []
        for item in items:
            cells = "".join(f"<td>{html.escape(str(item.get(key, '')))}</td>" for key in key_order)
            body_rows.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    raw_items = [{"metric": key, "value": value} for key, value in raw_summary.items()]
    feature_items = [{"metric": key, "value": value} for key, value in feature_summary.items()]

    html_text = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Market Pulse Lab Validation Report</title>
  <style>
    body {{ font-family: Segoe UI, sans-serif; margin: 32px; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    .status {{ font-weight: 700; color: {'#166534' if overall_success else '#991b1b'}; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Market Pulse Lab Validation Report</h1>
  <p class=\"status\">Overall status: {'PASS' if overall_success else 'FAIL'}</p>
  <h2>Ingestion Summary</h2>
  {render_rows(raw_items, ['metric', 'value'])}
  <h2>Feature Matrix Summary</h2>
  {render_rows(feature_items, ['metric', 'value'])}
  <h2>Great Expectations Checks</h2>
  {render_rows(ge_results, ['label', 'success', 'evaluated_expectations', 'successful_expectations', 'unsuccessful_expectations'])}
  <h2>Manual Checks</h2>
  {render_rows(manual_checks, ['label', 'success', 'details'])}
</body>
</html>
""".strip()
    report_path.write_text(html_text + "\n", encoding="utf-8")


def run_all_validations(
    config: dict[str, Any],
    level: Literal["basic", "strict"] | None = None,
) -> None:
    """Run all configured data validation checks.

    Args:
        config: Parsed pipeline configuration dictionary.
        level: Optional validation strictness override.
    """
    effective_level = _resolve_validation_level(config, level)
    strict_enabled = effective_level == "strict"

    validation_cfg = config.get("validation", {})
    paths_cfg = config.get("paths", {})

    prices_path = Path(paths_cfg.get("interim_prices", "datasets/interim/prices_combined.parquet"))
    news_path = Path(paths_cfg.get("interim_news", "datasets/interim/news_cleaned.parquet"))
    macro_path = Path(paths_cfg.get("macro_indicators", "datasets/external/macro_indicators.parquet"))
    feature_matrix_path = Path(paths_cfg.get("feature_matrix", "datasets/features/feature_matrix.parquet"))
    pipeline_model_path = Path(paths_cfg.get("pipeline_model", "models/preprocessing_pipeline.joblib"))
    validation_report_path = Path("reports/data_validation_report.html")

    _require_file(prices_path)
    _require_file(news_path)
    _require_file(macro_path)

    prices_df = pd.read_parquet(prices_path)
    news_df = pd.read_parquet(news_path)
    macro_df = pd.read_parquet(macro_path)

    start = pd.to_datetime(config["dates"]["start"], utc=True)
    end = pd.to_datetime(config["dates"]["end"], utc=True)

    min_rows_prices = int(validation_cfg.get("min_rows_prices", 0))
    min_rows_news = int(validation_cfg.get("min_rows_news", 0))
    max_missing_close = float(validation_cfg.get("max_missing_close", 1.0))
    min_tickers = int(validation_cfg.get("min_tickers", 0))

    errors: list[str] = []

    errors.extend(
        _ensure_required_columns(
            prices_df,
            {"ticker", "date", "open", "high", "low", "close", "volume", "volume_zero_flag"},
            "prices",
        )
    )
    errors.extend(
        _ensure_required_columns(
            news_df,
            {"ticker", "date", "title", "article", "url", "publisher", "text_source"},
            "news",
        )
    )

    macro_required_cols = {"date", *set(config.get("macro", {}).get("fred_series", {}).values())}
    errors.extend(_ensure_required_columns(macro_df, macro_required_cols, "macro"))

    if errors:
        raise ValidationError("Validation checks failed: " + " | ".join(errors))

    if not _is_utc_datetime(prices_df["date"]):
        errors.append(f"prices date dtype is not UTC datetime: {prices_df['date'].dtype}")
    if not _is_utc_datetime(news_df["date"]):
        errors.append(f"news date dtype is not UTC datetime: {news_df['date'].dtype}")
    if not _is_utc_datetime(macro_df["date"]):
        errors.append(f"macro date dtype is not UTC datetime: {macro_df['date'].dtype}")

    errors.extend(_validate_date_range(prices_df["date"], start, end, "prices"))
    errors.extend(_validate_date_range(news_df["date"], start, end, "news"))
    errors.extend(_validate_date_range(macro_df["date"], start, end, "macro"))

    if len(prices_df) < min_rows_prices:
        errors.append(f"prices rows {len(prices_df)} < min_rows_prices {min_rows_prices}")
    if len(news_df) < min_rows_news:
        errors.append(f"news rows {len(news_df)} < min_rows_news {min_rows_news}")

    missing_close = prices_df["close"].isna().mean() if "close" in prices_df.columns else 1.0
    if missing_close > max_missing_close:
        errors.append(f"close missing rate {missing_close:.4f} > max_missing_close {max_missing_close:.4f}")

    n_tickers = prices_df["ticker"].nunique() if "ticker" in prices_df.columns else 0
    if n_tickers < min_tickers:
        errors.append(f"ticker count {n_tickers} < min_tickers {min_tickers}")

    raw_summary = {
        "level": effective_level,
        "prices_rows": int(len(prices_df)),
        "news_rows": int(len(news_df)),
        "macro_rows": int(len(macro_df)),
        "ticker_count": int(n_tickers),
        "close_missing_rate": round(float(missing_close), 6),
    }

    if strict_enabled:
        prices_ticker_ok = prices_df["ticker"].astype(str).str.fullmatch(r"[A-Z0-9._-]+", na=False).all()
        if not bool(prices_ticker_ok):
            errors.append("prices ticker contains invalid/non-uppercase symbols")

        news_ticker_ok = news_df["ticker"].astype(str).str.fullmatch(r"[A-Z0-9.-]+", na=False).all()
        if not bool(news_ticker_ok):
            errors.append("news ticker contains invalid/non-uppercase symbols")

        non_positive_close = int((prices_df["close"] <= 0).sum())
        if non_positive_close > 0:
            errors.append(f"prices has {non_positive_close} rows with close <= 0")

        duplicate_price_keys = int(prices_df.duplicated(subset=["ticker", "date"]).sum())
        if duplicate_price_keys > 0:
            errors.append(f"prices has {duplicate_price_keys} duplicated ticker-date rows")

        valid_flags = prices_df["volume_zero_flag"].isin([0, 1]).all()
        if not bool(valid_flags):
            errors.append("volume_zero_flag contains values outside {0, 1}")

        text_source_ok = news_df["text_source"].dropna().astype(str).isin(["article", "title"]).all()
        if not bool(text_source_ok):
            errors.append("news text_source contains unexpected values")

        both_text_missing = (
            news_df["title"].fillna("").astype(str).str.strip().eq("")
            & news_df["article"].fillna("").astype(str).str.strip().eq("")
        ).sum()
        if int(both_text_missing) > 0:
            errors.append(f"news has {int(both_text_missing)} rows with empty title and article")

        duplicate_macro_dates = int(macro_df.duplicated(subset=["date"]).sum())
        if duplicate_macro_dates > 0:
            errors.append(f"macro has {duplicate_macro_dates} duplicated dates")

        if not bool(macro_df["date"].is_monotonic_increasing):
            errors.append("macro date column is not sorted ascending")

    feature_summary: dict[str, Any] = {}
    ge_results: list[dict[str, Any]] = []
    manual_checks: list[dict[str, Any]] = []
    if feature_matrix_path.exists():
        feature_errors, feature_summary, ge_results, manual_checks = _validate_feature_matrix(
            config=config,
            feature_matrix_path=feature_matrix_path,
            pipeline_model_path=pipeline_model_path,
        )
        errors.extend(feature_errors)
    elif strict_enabled:
        errors.append(f"Missing required integrated artifact in strict mode: {feature_matrix_path}")

    if strict_enabled and not pipeline_model_path.exists():
        errors.append(f"Missing required serialized pipeline in strict mode: {pipeline_model_path}")

    _write_validation_report(
        report_path=validation_report_path,
        raw_summary=raw_summary,
        feature_summary=feature_summary,
        ge_results=ge_results,
        manual_checks=manual_checks,
        overall_success=not errors,
    )

    if errors:
        raise ValidationError("Validation checks failed: " + " | ".join(errors))

    log.info(
        (
            "Validations passed | level=%s | prices_rows=%s | news_rows=%s | macro_rows=%s "
            "| ticker_count=%s | close_missing=%.4f"
        ),
        effective_level,
        len(prices_df),
        len(news_df),
        len(macro_df),
        n_tickers,
        missing_close,
    )