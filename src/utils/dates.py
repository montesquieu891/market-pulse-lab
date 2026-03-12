from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _coerce_utc_datetime(series: pd.Series, column_name: str) -> pd.Series:
    """Coerce a series to UTC datetime and validate conversion success.

    Args:
        series: Raw datetime-like pandas series.
        column_name: Column label used in error messages.

    Returns:
        UTC-aware datetime series.

    Raises:
        ValueError: If one or more values cannot be parsed.
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        missing = int(parsed.isna().sum())
        raise ValueError(f"Column '{column_name}' contains {missing} non-parsable datetime values")
    return parsed


def resolve_split_boundaries(config: dict[str, Any]) -> dict[str, pd.Timestamp]:
    """Resolve train, validation, and test temporal boundaries from config.

    Args:
        config: Parsed project configuration dictionary.

    Returns:
        Dictionary containing UTC timestamps for train_end, validation_end, and test_end.

    Raises:
        ValueError: If split boundaries are missing or not strictly ordered.
    """
    dates_cfg = config.get("dates", {})
    required_keys = ("train_end", "validation_end", "test_end")
    missing_keys = [key for key in required_keys if key not in dates_cfg]
    if missing_keys:
        raise ValueError(f"Missing temporal split boundaries in config: {missing_keys}")

    train_end = pd.to_datetime(dates_cfg["train_end"], utc=True)
    validation_end = pd.to_datetime(dates_cfg["validation_end"], utc=True)
    test_end = pd.to_datetime(dates_cfg["test_end"], utc=True)

    if not (train_end < validation_end < test_end):
        raise ValueError(
            "Temporal boundaries must satisfy train_end < validation_end < test_end "
            f"(received: {train_end}, {validation_end}, {test_end})"
        )

    return {
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": test_end,
    }


def assign_temporal_split(
    frame: pd.DataFrame,
    config: dict[str, Any],
    date_column: str = "date",
) -> pd.Series:
    """Assign train/validation/test labels using configured temporal cutoffs.

    Args:
        frame: Input dataframe containing a date column.
        config: Parsed project configuration dictionary.
        date_column: Name of the datetime column used for split assignment.

    Returns:
        String series with labels: train, validation, test.
    """
    if date_column not in frame.columns:
        raise ValueError(f"Date column '{date_column}' does not exist in the provided dataframe")

    boundaries = resolve_split_boundaries(config)
    date_values = _coerce_utc_datetime(frame[date_column], date_column)

    labels = np.where(
        date_values <= boundaries["train_end"],
        "train",
        np.where(date_values <= boundaries["validation_end"], "validation", "test"),
    )
    return pd.Series(labels, index=frame.index, dtype="string")


def temporal_train_val_test_split(
    frame: pd.DataFrame,
    config: dict[str, Any],
    date_column: str = "date",
    copy: bool = True,
) -> dict[str, pd.DataFrame]:
    """Split a dataframe into temporal train/validation/test partitions.

    Args:
        frame: Input dataframe containing at least one datetime column.
        config: Parsed project configuration dictionary.
        date_column: Name of the datetime column used to split rows.
        copy: Whether to return copies of the split slices.

    Returns:
        Dictionary with train, validation, and test dataframes.

    Raises:
        ValueError: If any split is empty or temporal integrity is broken.
    """
    working = frame.copy() if copy else frame
    working[date_column] = _coerce_utc_datetime(working[date_column], date_column)
    working = working.sort_values(date_column).reset_index(drop=True)
    working["dataset_split"] = assign_temporal_split(working, config, date_column=date_column)

    split_frames = {
        "train": working.loc[working["dataset_split"] == "train"].copy(),
        "validation": working.loc[working["dataset_split"] == "validation"].copy(),
        "test": working.loc[working["dataset_split"] == "test"].copy(),
    }
    for split_name, split_frame in split_frames.items():
        if split_frame.empty:
            raise ValueError(f"Temporal split '{split_name}' is empty. Verify split boundaries in pipeline_config.yaml")

    if split_frames["train"][date_column].max() >= split_frames["validation"][date_column].min():
        raise ValueError("Temporal leakage detected: train and validation date windows overlap")
    if split_frames["validation"][date_column].max() >= split_frames["test"][date_column].min():
        raise ValueError("Temporal leakage detected: validation and test date windows overlap")

    return split_frames
