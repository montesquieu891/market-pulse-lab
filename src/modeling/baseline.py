from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.dates import temporal_train_val_test_split

IDENTIFIER_COLUMNS = {
    "ticker",
    "date",
    "dataset_split",
    "target_start_date",
    "target_realized_vol_5d",
}
SENTIMENT_COLUMN_TOKENS = (
    "finbert_",
    "news_count",
    "char_count",
    "word_count",
    "avg_word_length",
)


@dataclass
class BaselineArtifacts:
    """Container for baseline outputs.

    Attributes:
        results: Summary dataframe with model metrics and metadata.
        feature_importance: Long dataframe of per-model feature importance values.
    """

    results: pd.DataFrame
    feature_importance: pd.DataFrame


def _resolve_feature_matrix_path(config: dict[str, Any]) -> Path:
    """Resolve feature matrix path from configuration.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Feature matrix parquet path.
    """
    return Path(config.get("paths", {}).get("feature_matrix", "datasets/features/feature_matrix.parquet"))


def _load_feature_matrix(config: dict[str, Any]) -> pd.DataFrame:
    """Load feature matrix from parquet.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Feature matrix dataframe.

    Raises:
        FileNotFoundError: If the matrix file is missing.
        ValueError: If the required target column is missing.
    """
    path = _resolve_feature_matrix_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix does not exist: {path}")

    frame = pd.read_parquet(path)
    if "target_realized_vol_5d" not in frame.columns:
        raise ValueError("Feature matrix must contain 'target_realized_vol_5d'")
    return frame


def _extract_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Build feature-set definitions used by baseline models.

    Args:
        frame: Feature matrix dataframe.

    Returns:
        Mapping with two feature sets: price_only and price_plus_sentiment.
    """
    numeric_columns = [
        col
        for col in frame.columns
        if col not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(frame[col])
    ]

    sentiment_columns = [
        col for col in numeric_columns if any(token in col for token in SENTIMENT_COLUMN_TOKENS)
    ]
    price_only_columns = [col for col in numeric_columns if col not in sentiment_columns]

    if not price_only_columns:
        raise ValueError("No numeric price features were found for Baseline 1")
    if not sentiment_columns:
        raise ValueError("No sentiment features were found for Baseline 2/3")

    return {
        "price_only": sorted(price_only_columns),
        "price_plus_sentiment": sorted(price_only_columns + sentiment_columns),
    }


def _build_model_pipeline(model: Any) -> Pipeline:
    """Build preprocessing + model pipeline.

    Args:
        model: sklearn-compatible regressor.

    Returns:
        Fitted-ready pipeline where scaler is learned on train only.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _extract_importance(trained_pipeline: Pipeline, feature_columns: list[str], model_name: str) -> pd.DataFrame:
    """Extract feature importance values from trained regressors.

    Args:
        trained_pipeline: Trained preprocessing/model pipeline.
        feature_columns: Ordered list of feature names used during fit.
        model_name: Human-readable model label.

    Returns:
        Dataframe with feature importance per model.
    """
    model = trained_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        raw_importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw_importance = np.abs(np.ravel(np.asarray(model.coef_, dtype=float)))
    else:
        raw_importance = np.zeros(len(feature_columns), dtype=float)

    size = min(len(feature_columns), len(raw_importance))
    return pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_columns[:size],
            "importance": raw_importance[:size],
        }
    ).sort_values("importance", ascending=False)


def _downsample_train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    max_train_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Downsample train split to control runtime in notebook workflows.

    Args:
        x_train: Training feature matrix.
        y_train: Training target vector.
        max_train_rows: Optional cap for the number of train rows.
        random_state: Seed used for reproducible row sampling.

    Returns:
        Possibly downsampled training features and target.
    """
    if max_train_rows is None or len(x_train) <= max_train_rows:
        return x_train, y_train

    sampled_index = x_train.sample(n=max_train_rows, random_state=random_state).index
    return x_train.loc[sampled_index].copy(), y_train.loc[sampled_index].copy()


def run_baseline_suite(
    config: dict[str, Any],
    max_train_rows: int | None = 250_000,
    random_state: int = 42,
) -> BaselineArtifacts:
    """Train and evaluate Phase 6 baseline regressors.

    Args:
        config: Parsed project configuration dictionary.
        max_train_rows: Optional cap for train rows to keep runtimes reasonable.
        random_state: Random seed for reproducibility.

    Returns:
        BaselineArtifacts containing result metrics and feature importances.
    """
    feature_df = _load_feature_matrix(config)
    feature_df = feature_df[feature_df["target_realized_vol_5d"].notna()].copy()

    splits = temporal_train_val_test_split(feature_df, config=config, date_column="date")
    feature_sets = _extract_feature_sets(feature_df)

    models: list[tuple[str, str, Any]] = [
        ("LinearRegression_price_only", "price_only", LinearRegression()),
        (
            "LinearRegression_price_plus_sentiment",
            "price_plus_sentiment",
            LinearRegression(),
        ),
    ]

    try:
        from xgboost import XGBRegressor

        models.append(
            (
                "XGBoostRegressor_price_plus_sentiment",
                "price_plus_sentiment",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
    except ModuleNotFoundError:
        models.append(("XGBoostRegressor_price_plus_sentiment", "price_plus_sentiment", None))

    rows: list[dict[str, Any]] = []
    importance_tables: list[pd.DataFrame] = []

    y_train_full = pd.to_numeric(splits["train"]["target_realized_vol_5d"], errors="coerce")
    y_val = pd.to_numeric(splits["validation"]["target_realized_vol_5d"], errors="coerce")
    y_test = pd.to_numeric(splits["test"]["target_realized_vol_5d"], errors="coerce")

    for model_name, feature_set_name, estimator in models:
        feature_columns = feature_sets[feature_set_name]
        x_train_full = splits["train"][feature_columns].apply(pd.to_numeric, errors="coerce")
        x_val = splits["validation"][feature_columns].apply(pd.to_numeric, errors="coerce")
        x_test = splits["test"][feature_columns].apply(pd.to_numeric, errors="coerce")

        x_train, y_train = _downsample_train(
            x_train=x_train_full,
            y_train=y_train_full,
            max_train_rows=max_train_rows,
            random_state=random_state,
        )

        if estimator is None:
            rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "n_features": len(feature_columns),
                    "train_rows_used": int(len(x_train)),
                    "validation_mae": np.nan,
                    "test_mae": np.nan,
                    "status": "skipped_missing_dependency",
                }
            )
            continue

        pipeline = _build_model_pipeline(estimator)
        pipeline.fit(x_train, y_train)

        y_val_pred = pipeline.predict(x_val)
        y_test_pred = pipeline.predict(x_test)

        rows.append(
            {
                "model": model_name,
                "feature_set": feature_set_name,
                "n_features": len(feature_columns),
                "train_rows_used": int(len(x_train)),
                "validation_mae": float(mean_absolute_error(y_val, y_val_pred)),
                "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
                "status": "ok",
            }
        )
        importance_tables.append(_extract_importance(pipeline, feature_columns, model_name))

    results = pd.DataFrame(rows).sort_values(["status", "validation_mae"], ascending=[True, True]).reset_index(drop=True)
    importances = (
        pd.concat(importance_tables, ignore_index=True)
        if importance_tables
        else pd.DataFrame(columns=["model", "feature", "importance"])
    )

    return BaselineArtifacts(results=results, feature_importance=importances)


def plot_top_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> pd.DataFrame:
    """Plot and save top-N feature importances for the strongest available model.

    Args:
        feature_importance: Long dataframe returned by run_baseline_suite.
        output_path: Destination image path.
        top_n: Number of top features to display.

    Returns:
        Dataframe containing plotted rows.
    """
    if feature_importance.empty:
        raise ValueError("feature_importance dataframe is empty; run baselines before plotting")

    preferred_order = [
        "XGBoostRegressor_price_plus_sentiment",
        "LinearRegression_price_plus_sentiment",
        "LinearRegression_price_only",
    ]
    available_models = feature_importance["model"].drop_duplicates().tolist()
    selected_model = next((name for name in preferred_order if name in available_models), available_models[0])

    top_df = (
        feature_importance.loc[feature_importance["model"] == selected_model]
        .sort_values("importance", ascending=False)
        .head(top_n)
        .iloc[::-1]
        .copy()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_df["feature"], top_df["importance"], color="#1f77b4")
    ax.set_title(f"Top {top_n} Features - {selected_model}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return top_df


def summarize_results(results: pd.DataFrame) -> dict[str, float]:
    """Create a compact metric summary used in README and notebook outputs.

    Args:
        results: Baseline results table.

    Returns:
        Dictionary with key MAE comparisons.
    """
    ok_results = results.loc[results["status"] == "ok"].copy()
    if ok_results.empty:
        return {}

    lookup = ok_results.set_index("model")
    summary: dict[str, float] = {}
    for col in ["validation_mae", "test_mae"]:
        for model_name in lookup.index:
            summary[f"{model_name}_{col}"] = float(lookup.loc[model_name, col])

    if {
        "LinearRegression_price_only",
        "LinearRegression_price_plus_sentiment",
    }.issubset(set(lookup.index)):
        base = float(lookup.loc["LinearRegression_price_only", "validation_mae"])
        with_sent = float(lookup.loc["LinearRegression_price_plus_sentiment", "validation_mae"])
        if base > 0:
            summary["linear_validation_improvement_pct"] = float((base - with_sent) / base * 100.0)

    return summary
