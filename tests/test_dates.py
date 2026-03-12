from __future__ import annotations

import pandas as pd
import pytest

from src.utils.dates import assign_temporal_split, resolve_split_boundaries, temporal_train_val_test_split


def _base_config() -> dict[str, dict[str, str]]:
    return {
        "dates": {
            "train_end": "2020-12-31",
            "validation_end": "2021-12-31",
            "test_end": "2022-12-31",
        }
    }


def test_resolve_split_boundaries_returns_utc_timestamps() -> None:
    boundaries = resolve_split_boundaries(_base_config())

    assert boundaries["train_end"] == pd.Timestamp("2020-12-31", tz="UTC")
    assert boundaries["validation_end"] == pd.Timestamp("2021-12-31", tz="UTC")
    assert boundaries["test_end"] == pd.Timestamp("2022-12-31", tz="UTC")


def test_resolve_split_boundaries_rejects_invalid_order() -> None:
    bad_config = {
        "dates": {
            "train_end": "2021-12-31",
            "validation_end": "2020-12-31",
            "test_end": "2022-12-31",
        }
    }

    with pytest.raises(ValueError, match="train_end < validation_end < test_end"):
        resolve_split_boundaries(bad_config)


def test_assign_temporal_split_labels_rows_correctly() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2021-06-01", "2022-02-01"],
                utc=True,
            )
        }
    )

    labels = assign_temporal_split(frame, _base_config(), date_column="date")

    assert labels.tolist() == ["train", "validation", "test"]


def test_temporal_train_val_test_split_enforces_non_overlap() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-12-30", "2021-06-01", "2022-03-01"],
                utc=True,
            ),
            "value": [1, 2, 3, 4],
        }
    )

    splits = temporal_train_val_test_split(frame, _base_config(), date_column="date")

    assert len(splits["train"]) == 2
    assert len(splits["validation"]) == 1
    assert len(splits["test"]) == 1
    assert splits["train"]["date"].max() < splits["validation"]["date"].min()
    assert splits["validation"]["date"].max() < splits["test"]["date"].min()
