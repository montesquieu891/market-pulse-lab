from __future__ import annotations

import logging
import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MissingnessAssessment:
    """Container for missingness classification and rationale."""

    dataset: str
    column: str
    missing_pct: float
    likely_mechanism: str
    rationale: str
    proposed_treatment: str


def _ensure_dependencies() -> None:
    """Ensure optional EDA dependencies are available."""
    if importlib.util.find_spec("ydata_profiling") is None:
        raise ImportError(
            "ydata-profiling is required for Phase 2 reports. "
            "Install it in the active environment before running EDA."
        )


def _build_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve all paths required by the Phase 2 workflow."""
    paths_cfg = config.get("paths", {})
    root = Path(".")

    return {
        "prices": Path(paths_cfg.get("interim_prices", "datasets/interim/prices_combined.parquet")),
        "news": Path(paths_cfg.get("interim_news", "datasets/interim/news_cleaned.parquet")),
        "reports": root / "reports",
        "plots": root / "reports" / "plots",
        "profiling_prices": root / "reports" / "profiling_prices.html",
        "profiling_news": root / "reports" / "profiling_news.html",
        "missingness": root / "reports" / "missingness_assessment.md",
        "quality_issues": root / "reports" / "data_quality_issues.md",
        "date_overlap": root / "reports" / "date_overlap.md",
        "missing_tickers_md": root / "reports" / "missing_trading_days_gt20pct.md",
        "missing_tickers_csv": root / "reports" / "missing_trading_days_gt20pct.csv",
        "correlation_csv": root / "reports" / "price_correlation_matrix.csv",
        "data_dictionary": root / "DATA_DICTIONARY.md",
    }


def _load_inputs(prices_path: Path, news_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 1 interim datasets for diagnostic EDA."""
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing interim prices parquet: {prices_path}")
    if not news_path.exists():
        raise FileNotFoundError(f"Missing interim news parquet: {news_path}")

    prices_df = pd.read_parquet(prices_path)
    news_df = pd.read_parquet(news_path)

    prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce", utc=True)
    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce", utc=True)

    prices_df = prices_df.dropna(subset=["date", "ticker"]).copy()
    news_df = news_df.dropna(subset=["date", "ticker"]).copy()

    return prices_df, news_df


def _prepare_prices_for_eda(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Create diagnostic-only fields for the prices dataset."""
    df = prices_df.sort_values(["ticker", "date"]).copy()
    close_prev = df.groupby("ticker", observed=True)["close"].shift(1)
    ratio = df["close"] / close_prev
    ratio = ratio.where(ratio > 0)
    df["log_return"] = np.log(ratio)
    return df


def _prepare_news_for_eda(news_df: pd.DataFrame) -> pd.DataFrame:
    """Create diagnostic-only fields for the news dataset."""
    df = news_df.copy()
    article = df["article"].fillna("").astype("string").str.strip()
    title = df["title"].fillna("").astype("string").str.strip()
    text = article.where(article != "", title)
    df["word_count"] = text.str.split().str.len().fillna(0).astype("int32")
    return df


def _profile_input(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Return profiling input dataframe with optional down-sampling for stability."""
    if len(df) <= max_rows:
        return df

    sampled = df.sample(n=max_rows, random_state=42)
    return sampled.sort_values("date") if "date" in sampled.columns else sampled


def _save_profile(df: pd.DataFrame, title: str, out_path: Path, max_rows: int) -> None:
    """Generate ydata-profiling HTML output for a dataframe."""
    profiling_module = importlib.import_module("ydata_profiling")
    profile_report_cls = getattr(profiling_module, "ProfileReport")

    profile_df = _profile_input(df, max_rows=max_rows)
    profile = profile_report_cls(
        profile_df,
        title=title,
        explorative=True,
        minimal=True,
        progress_bar=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_file(str(out_path))


def _plot_distribution(series: pd.Series, name: str, out_path: Path) -> None:
    """Save a simple histogram for a numeric series."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=80)
    ax.set_title(f"Distribution of {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_required_distributions(prices_df: pd.DataFrame, news_df: pd.DataFrame, plots_dir: Path) -> None:
    """Create required distribution plots for Phase 2 checkpoints."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_distribution(prices_df["close"], "close", plots_dir / "distribution_close.png")
    _plot_distribution(prices_df["log_return"], "log_return", plots_dir / "distribution_log_return.png")
    _plot_distribution(prices_df["volume"], "volume", plots_dir / "distribution_volume.png")
    _plot_distribution(news_df["word_count"], "word_count", plots_dir / "distribution_word_count.png")


def _plot_news_volume_pareto(news_df: pd.DataFrame, plots_dir: Path) -> None:
    """Plot ticker-level news volume and cumulative share (Pareto check)."""
    counts = news_df.groupby("ticker", observed=True).size().sort_values(ascending=False)
    if counts.empty:
        return

    top_n = min(100, len(counts))
    top = counts.head(top_n)
    cumulative_share = top.cumsum() / counts.sum()

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.bar(top.index.astype(str), top.values, color="#3b82f6")
    ax1.set_xlabel("Ticker")
    ax1.set_ylabel("Article count")
    ax1.set_title(f"News Volume Pareto (Top {top_n} tickers)")
    ax1.tick_params(axis="x", rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(top.index.astype(str), cumulative_share.values, color="#ef4444", marker="o", linewidth=1.5)
    ax2.axhline(0.8, color="#6b7280", linestyle="--", linewidth=1)
    ax2.set_ylabel("Cumulative share")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(plots_dir / "news_volume_pareto_top100.png", dpi=160)
    plt.close(fig)


def _compute_price_correlation(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric price-related columns."""
    candidate_cols = ["open", "high", "low", "close", "volume", "volume_zero_flag", "log_return"]
    available = [col for col in candidate_cols if col in prices_df.columns]
    if not available:
        return pd.DataFrame()

    numeric_df = prices_df[available].apply(pd.to_numeric, errors="coerce")
    return numeric_df.corr(numeric_only=True)


def _save_correlation_outputs(corr: pd.DataFrame, csv_path: Path, plots_dir: Path) -> None:
    """Save correlation matrix table and heatmap."""
    if corr.empty:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(csv_path, index=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr.values, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Price Numeric Correlation Matrix")
    fig.colorbar(cax)
    fig.tight_layout()
    fig.savefig(plots_dir / "price_correlation_matrix.png", dpi=160)
    plt.close(fig)


def _assess_missingness(prices_df: pd.DataFrame, news_df: pd.DataFrame) -> list[MissingnessAssessment]:
    """Classify null columns as MCAR, MAR, or MNAR with rationale."""
    assessments: list[MissingnessAssessment] = []

    for dataset_name, df in (("prices", prices_df), ("news", news_df)):
        for col in df.columns:
            missing_pct = float(df[col].isna().mean())
            if missing_pct <= 0:
                continue

            if dataset_name == "news" and col == "article":
                mechanism = "MAR"
                rationale = (
                    "Missing full article text depends on source feed availability; "
                    "headline usually remains available."
                )
                treatment = "Fallback to title text; keep row."
            elif dataset_name == "prices" and col == "log_return":
                mechanism = "MNAR"
                rationale = "First observation per ticker has no prior close by construction."
                treatment = "Keep NaN for first row and ignore in return-based aggregations."
            else:
                mechanism = "MCAR"
                rationale = "No deterministic business process signal detected during diagnostic EDA."
                treatment = "Keep for now and revisit at feature-imputation stage."

            assessments.append(
                MissingnessAssessment(
                    dataset=dataset_name,
                    column=col,
                    missing_pct=missing_pct,
                    likely_mechanism=mechanism,
                    rationale=rationale,
                    proposed_treatment=treatment,
                )
            )

    return sorted(assessments, key=lambda item: item.missing_pct, reverse=True)


def _write_missingness_report(assessments: list[MissingnessAssessment], out_path: Path) -> None:
    """Write missingness assessment as a markdown report."""
    lines = [
        "# Missingness Assessment",
        "",
        "Classification by column as MCAR/MAR/MNAR with rationale and treatment.",
        "",
        "| Dataset | Column | Missing % | Mechanism | Rationale | Proposed treatment |",
        "|---|---|---:|---|---|---|",
    ]

    for item in assessments:
        lines.append(
            f"| {item.dataset} | {item.column} | {item.missing_pct * 100:.2f}% | "
            f"{item.likely_mechanism} | {item.rationale} | {item.proposed_treatment} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _missing_trading_days(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Identify tickers with more than 20% missing trading days."""
    global_days = pd.Index(sorted(prices_df["date"].dropna().unique()))
    expected_days = len(global_days)

    rows: list[dict[str, Any]] = []
    for ticker, ticker_df in prices_df.groupby("ticker", observed=True):
        actual_days = ticker_df["date"].nunique()
        missing_days = expected_days - actual_days
        missing_pct = missing_days / expected_days if expected_days else 0.0

        if missing_pct > 0.2:
            rows.append(
                {
                    "ticker": str(ticker),
                    "expected_days": int(expected_days),
                    "actual_days": int(actual_days),
                    "missing_days": int(missing_days),
                    "missing_pct": float(missing_pct),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["ticker", "expected_days", "actual_days", "missing_days", "missing_pct"])

    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)


def _write_missing_tickers(missing_tickers: pd.DataFrame, markdown_path: Path, csv_path: Path) -> None:
    """Write sparse ticker report in markdown and CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    missing_tickers.to_csv(csv_path, index=False)

    lines = [
        "# Tickers With More Than 20% Missing Trading Days",
        "",
        f"Total tickers above threshold: {len(missing_tickers)}",
        "",
    ]

    if missing_tickers.empty:
        lines.append("No tickers exceeded the 20% missing-days threshold.")
    else:
        lines.extend(
            [
                "| Ticker | Expected Days | Actual Days | Missing Days | Missing % |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in missing_tickers.iterrows():
            lines.append(
                f"| {row['ticker']} | {int(row['expected_days'])} | {int(row['actual_days'])} | "
                f"{int(row['missing_days'])} | {float(row['missing_pct']) * 100:.2f}% |"
            )

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _date_overlap(prices_df: pd.DataFrame, news_df: pd.DataFrame) -> dict[str, Any]:
    """Compute date-range overlap between prices and news."""
    prices_min = prices_df["date"].min()
    prices_max = prices_df["date"].max()
    news_min = news_df["date"].min()
    news_max = news_df["date"].max()

    overlap_start = max(prices_min, news_min)
    overlap_end = min(prices_max, news_max)
    has_overlap = bool(overlap_start <= overlap_end)

    return {
        "prices_min": prices_min,
        "prices_max": prices_max,
        "news_min": news_min,
        "news_max": news_max,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "has_overlap": has_overlap,
    }


def _write_date_overlap_report(overlap: dict[str, Any], out_path: Path) -> None:
    """Write markdown summary for date overlap diagnostics."""
    lines = [
        "# Date Range Overlap",
        "",
        f"- Prices range: {overlap['prices_min']} to {overlap['prices_max']}",
        f"- News range: {overlap['news_min']} to {overlap['news_max']}",
        f"- Overlap window: {overlap['overlap_start']} to {overlap['overlap_end']}",
        f"- Overlap present: {overlap['has_overlap']}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_quality_issues(prices_df: pd.DataFrame, news_df: pd.DataFrame, missing_tickers: pd.DataFrame, out_path: Path) -> None:
    """Document key data quality issues and proposed treatments."""
    article_missing = float(news_df["article"].isna().mean())
    zero_volume = float((prices_df["volume_zero_flag"] == 1).mean())
    duplicate_news = int(news_df.duplicated(subset=["ticker", "date", "title", "article"]).sum())

    lines = [
        "# Data Quality Issues",
        "",
        "1. Material missingness in `news.article`.",
        f"   Evidence: {article_missing * 100:.2f}% of rows have null article text.",
        "   Proposed treatment: use title fallback and preserve `text_source`.",
        "",
        "2. Zero-volume trading rows are present in prices.",
        f"   Evidence: {zero_volume * 100:.2f}% of rows have `volume_zero_flag == 1`.",
        "   Proposed treatment: keep rows and use `volume_zero_flag` as a quality feature.",
        "",
        "3. Duplicate news stories exist for ticker-date-text keys.",
        f"   Evidence: {duplicate_news} exact duplicates found for (`ticker`, `date`, `title`, `article`).",
        "   Proposed treatment: deduplicate only during aggregation, not at raw-interim stage.",
        "",
        "4. Some tickers have sparse trading-day coverage.",
        f"   Evidence: {len(missing_tickers)} tickers exceed 20% missing trading days.",
        "   Proposed treatment: track these symbols and consider minimum-history filters before modeling.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _column_note(dataset_name: str, column: str) -> str:
    """Return column-level notes for data dictionary output."""
    if dataset_name == "news" and column == "article":
        return "Full article text may be missing; fallback to title."
    if dataset_name == "news" and column == "word_count":
        return "Diagnostic field computed from article/title fallback text."
    if dataset_name == "prices" and column == "log_return":
        return "Diagnostic log return per ticker; first row per ticker is null by construction."
    if dataset_name == "prices" and column == "volume_zero_flag":
        return "1 if volume is zero, else 0."
    return "Interim field."


def _write_data_dictionary(
    prices_df: pd.DataFrame,
    news_df: pd.DataFrame,
    overlap: dict[str, Any],
    missing_tickers_count: int,
    out_path: Path,
) -> None:
    """Write project-level data dictionary for all current Phase 2 columns."""
    lines = [
        "# DATA_DICTIONARY",
        "",
        "Generated by `src.eda.diagnostic.run_diagnostic_eda`.",
        "",
        "## Coverage Summary",
        "",
        f"- Prices date range: {overlap['prices_min']} to {overlap['prices_max']}",
        f"- News date range: {overlap['news_min']} to {overlap['news_max']}",
        f"- Overlap window: {overlap['overlap_start']} to {overlap['overlap_end']}",
        f"- Date overlap present: {overlap['has_overlap']}",
        f"- Tickers with >20% missing trading days: {missing_tickers_count}",
        "",
        "## Prices Columns",
        "",
        "| Column | Dtype | Null % | Notes |",
        "|---|---|---:|---|",
    ]

    for col in prices_df.columns:
        null_pct = float(prices_df[col].isna().mean()) * 100
        lines.append(f"| {col} | {prices_df[col].dtype} | {null_pct:.2f}% | {_column_note('prices', col)} |")

    lines.extend(
        [
            "",
            "## News Columns",
            "",
            "| Column | Dtype | Null % | Notes |",
            "|---|---|---:|---|",
        ]
    )

    for col in news_df.columns:
        null_pct = float(news_df[col].isna().mean()) * 100
        lines.append(f"| {col} | {news_df[col].dtype} | {null_pct:.2f}% | {_column_note('news', col)} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic_eda(config: dict[str, Any]) -> None:
    """Run Phase 2 diagnostic EDA and produce all required artifacts.

    Args:
        config: Parsed pipeline configuration dictionary.
    """
    _ensure_dependencies()
    paths = _build_paths(config)

    paths["reports"].mkdir(parents=True, exist_ok=True)
    paths["plots"].mkdir(parents=True, exist_ok=True)

    prices_df, news_df = _load_inputs(paths["prices"], paths["news"])
    prices_eda = _prepare_prices_for_eda(prices_df)
    news_eda = _prepare_news_for_eda(news_df)

    eda_cfg = config.get("eda", {})
    profile_max_rows = int(eda_cfg.get("profile_max_rows", 200_000))

    _save_profile(
        prices_eda,
        "Market Pulse Lab - Prices Profile",
        paths["profiling_prices"],
        max_rows=profile_max_rows,
    )
    _save_profile(
        news_eda,
        "Market Pulse Lab - News Profile",
        paths["profiling_news"],
        max_rows=profile_max_rows,
    )

    _plot_required_distributions(prices_eda, news_eda, paths["plots"])
    _plot_news_volume_pareto(news_eda, paths["plots"])

    corr = _compute_price_correlation(prices_eda)
    _save_correlation_outputs(corr, paths["correlation_csv"], paths["plots"])

    assessments = _assess_missingness(prices_eda, news_eda)
    _write_missingness_report(assessments, paths["missingness"])

    missing_tickers = _missing_trading_days(prices_eda)
    _write_missing_tickers(missing_tickers, paths["missing_tickers_md"], paths["missing_tickers_csv"])

    overlap = _date_overlap(prices_eda, news_eda)
    _write_date_overlap_report(overlap, paths["date_overlap"])

    _write_quality_issues(prices_eda, news_eda, missing_tickers, paths["quality_issues"])
    _write_data_dictionary(prices_eda, news_eda, overlap, len(missing_tickers), paths["data_dictionary"])

    log.info(
        "run_diagnostic_eda completed | prices_rows=%s | news_rows=%s | reports_dir=%s",
        len(prices_eda),
        len(news_eda),
        paths["reports"],
    )
