from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def _resolve_output_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve persisted output paths for integration artifacts.

    Args:
        config: Parsed pipeline configuration dictionary.

    Returns:
        Tuple of feature-matrix path and serialized pipeline path.
    """
    paths_cfg = config.get("paths", {})
    return (
        Path(paths_cfg.get("feature_matrix", "datasets/features/feature_matrix.parquet")),
        Path(paths_cfg.get("pipeline_model", "models/preprocessing_pipeline.joblib")),
    )


def _build_and_serialize_preprocessing_pipeline(feature_df: pd.DataFrame, config: dict[str, Any], output_path: Path) -> None:
    """Fit and persist a train-only sklearn preprocessing pipeline.

    Args:
        feature_df: Final integrated feature matrix.
        config: Parsed pipeline configuration dictionary.
        output_path: Joblib output path.
    """
    from joblib import dump
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    excluded = {"ticker", "date", "dataset_split", "target_realized_vol_5d", "target_start_date"}
    candidate_columns = [col for col in feature_df.columns if col not in excluded]
    numeric_columns = [col for col in candidate_columns if pd.api.types.is_numeric_dtype(feature_df[col])]

    if not numeric_columns:
        raise ValueError("No numeric feature columns available to fit preprocessing pipeline")

    if "dataset_split" in feature_df.columns:
        train_mask = feature_df["dataset_split"].astype("string").eq("train")
    else:
        train_end = pd.to_datetime(config["dates"]["train_end"], utc=True)
        train_mask = feature_df["date"] <= train_end

    train_frame = feature_df.loc[train_mask, numeric_columns].copy()
    if train_frame.empty:
        raise ValueError("Training slice is empty; cannot fit preprocessing pipeline")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                            numeric_columns,
                        )
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            )
        ]
    )
    pipeline.fit(train_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, output_path)
    log.info(
        "Serialized preprocessing pipeline | train_rows=%s | feature_count=%s | output=%s",
        len(train_frame),
        len(numeric_columns),
        output_path,
    )


def save_feature_matrix(feature_matrix: Any, config: dict[str, Any]) -> None:
    """Persist the final feature matrix and preprocessing pipeline.

    Args:
        feature_matrix: Final integrated matrix returned by the joiner.
        config: Parsed pipeline configuration dictionary.
    """
    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError("save_feature_matrix expects a pandas DataFrame returned by join_all_sources")

    feature_matrix_path, pipeline_path = _resolve_output_paths(config)

    feature_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_parquet(feature_matrix_path, index=False)
    _build_and_serialize_preprocessing_pipeline(feature_matrix, config, pipeline_path)

    stats = feature_matrix.attrs.get("integration_stats", {})
    log.info(
        "Saved feature matrix | output=%s | rows=%s | cols=%s | feature_count=%s",
        feature_matrix_path,
        len(feature_matrix),
        len(feature_matrix.columns),
        stats.get("feature_count", "unknown"),
    )
