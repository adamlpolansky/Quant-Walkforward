from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

SUPPORTED_MODEL_NAMES: tuple[str, ...] = ("linear", "ridge", "lasso", "elasticnet")


def _validate_columns(df: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def _normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    if normalized not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model_name='{model_name}'. Supported models: {list(SUPPORTED_MODEL_NAMES)}"
        )
    return normalized


def _coerce_float(value: Any, *, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Model parameter '{field_name}' must be numeric. Received: {value!r}") from exc

    if not np.isfinite(out):
        raise ValueError(f"Model parameter '{field_name}' must be finite. Received: {value!r}")
    return out


def resolve_model_params(
    model_name: str,
    model_params: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    normalized_name = _normalize_model_name(model_name)
    raw_params = dict(model_params or {})

    if normalized_name == "linear":
        if raw_params:
            raise ValueError(f"Model '{normalized_name}' does not accept parameters: {sorted(raw_params)}")
        return {}

    if normalized_name == "ridge":
        allowed = {"alpha"}
        extra = sorted(set(raw_params) - allowed)
        if extra:
            raise ValueError(f"Model '{normalized_name}' received unsupported parameters: {extra}")

        alpha = _coerce_float(raw_params.get("alpha", 1.0), field_name="alpha")
        if alpha < 0.0:
            raise ValueError("Ridge alpha must be >= 0.0")
        return {"alpha": alpha}

    if normalized_name == "lasso":
        allowed = {"alpha"}
        extra = sorted(set(raw_params) - allowed)
        if extra:
            raise ValueError(f"Model '{normalized_name}' received unsupported parameters: {extra}")

        alpha = _coerce_float(raw_params.get("alpha", 1.0), field_name="alpha")
        if alpha <= 0.0:
            raise ValueError("Lasso alpha must be > 0.0")
        return {"alpha": alpha}

    allowed = {"alpha", "l1_ratio"}
    extra = sorted(set(raw_params) - allowed)
    if extra:
        raise ValueError(f"Model '{normalized_name}' received unsupported parameters: {extra}")

    alpha = _coerce_float(raw_params.get("alpha", 1.0), field_name="alpha")
    l1_ratio = _coerce_float(raw_params.get("l1_ratio", 0.5), field_name="l1_ratio")
    if alpha <= 0.0:
        raise ValueError("ElasticNet alpha must be > 0.0")
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("ElasticNet l1_ratio must be between 0.0 and 1.0")
    return {"alpha": alpha, "l1_ratio": l1_ratio}


def build_model(
    model_name: str,
    model_params: Mapping[str, Any] | None = None,
) -> RegressorMixin:
    normalized_name = _normalize_model_name(model_name)
    params = resolve_model_params(normalized_name, model_params)

    if normalized_name == "linear":
        return LinearRegression()
    if normalized_name == "ridge":
        return Ridge(alpha=params["alpha"])
    if normalized_name == "lasso":
        return Lasso(alpha=params["alpha"], max_iter=10_000)
    return ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=10_000)


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


def fit_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    *,
    model_name: str = "ridge",
    model_params: Mapping[str, Any] | None = None,
) -> RegressorMixin:
    x, y, valid_index = get_feature_matrix(train_df, feature_cols, label_col=label_col)
    if y is None or len(valid_index) == 0:
        raise ValueError("No valid training rows available after NaN filtering")

    model = build_model(model_name, model_params)
    model.fit(x, y)
    return model


def fit_ridge_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    alpha: float = 1.0,
) -> Ridge:
    model = fit_model(
        train_df,
        feature_cols,
        label_col,
        model_name="ridge",
        model_params={"alpha": alpha},
    )
    return model


def predict_scores(
    model: RegressorMixin,
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
