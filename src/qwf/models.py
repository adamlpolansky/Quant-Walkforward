from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _validate_columns(df: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, pd.Index]:
    _validate_columns(df, feature_cols, label="feature")
    if label_col is not None:
        _validate_columns(df, [label_col], label="label")

    valid_mask = df[feature_cols].notna().all(axis=1)
    if label_col is not None:
        valid_mask &= df[label_col].notna()

    valid_index = df.index[valid_mask]
    x = df.loc[valid_index, feature_cols].to_numpy(dtype=float)
    y = None
    if label_col is not None:
        y = df.loc[valid_index, label_col].to_numpy(dtype=float)

    return x, y, valid_index


def fit_ridge_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    alpha: float = 1.0,
) -> Ridge:
    x, y, valid_index = get_feature_matrix(train_df, feature_cols, label_col=label_col)
    if y is None or len(valid_index) == 0:
        raise ValueError("No valid training rows available after NaN filtering")

    model = Ridge(alpha=alpha)
    model.fit(x, y)
    return model


def predict_scores(
    model: Ridge,
    df: pd.DataFrame,
    feature_cols: list[str],
    score_col: str = "score",
) -> pd.DataFrame:
    x, _, valid_index = get_feature_matrix(df, feature_cols, label_col=None)

    out = df.copy()
    out[score_col] = np.nan

    if len(valid_index) == 0:
        return out

    out.loc[valid_index, score_col] = model.predict(x)
    return out
