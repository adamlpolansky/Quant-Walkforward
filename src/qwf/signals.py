from __future__ import annotations

from typing import Callable

import pandas as pd

SUPPORTED_SCORE_NORMALIZATIONS: tuple[str, ...] = ("none", "zscore", "rank_to_minus1_plus1")
SUPPORTED_SCORE_SMOOTHING: tuple[str, ...] = ("none", "ema_3", "ema_5")


def _normalize_method_name(method: str, *, supported: tuple[str, ...], label: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in supported:
        raise ValueError(f"Unsupported {label}='{method}'. Supported values: {list(supported)}")
    return normalized


def smooth_scores_by_ticker(
    predictions: pd.DataFrame,
    *,
    score_col: str = "score",
    smoothing: str = "none",
    out_col: str = "score_smoothed",
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    method = _normalize_method_name(
        smoothing,
        supported=SUPPORTED_SCORE_SMOOTHING,
        label="score_smoothing",
    )
    required_cols = [ticker_col, date_col, score_col]
    missing = [col for col in required_cols if col not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions are missing required smoothing columns: {missing}")

    out = predictions.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.sort_values([ticker_col, date_col], kind="mergesort").reset_index(drop=True)

    if method == "none":
        out[out_col] = out[score_col]
    else:
        span = int(method.split("_", 1)[1])

        def _ema(series: pd.Series) -> pd.Series:
            smoothed = series.ewm(span=span, adjust=False, min_periods=1).mean()
            smoothed[series.isna()] = float("nan")
            return smoothed

        out[out_col] = out.groupby(ticker_col, sort=False)[score_col].transform(_ema)

    return out.sort_values([date_col, ticker_col], kind="mergesort").reset_index(drop=True)


def normalize_scores_cross_sectionally(
    predictions: pd.DataFrame,
    *,
    score_col: str = "score",
    normalization: str = "none",
    out_col: str = "score_normalized",
    date_col: str = "date",
) -> pd.DataFrame:
    method = _normalize_method_name(
        normalization,
        supported=SUPPORTED_SCORE_NORMALIZATIONS,
        label="score_normalization",
    )
    required_cols = [date_col, score_col]
    missing = [col for col in required_cols if col not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions are missing required normalization columns: {missing}")

    out = predictions.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")

    if method == "none":
        out[out_col] = out[score_col]
        sort_cols = [col for col in [date_col, "ticker"] if col in out.columns]
        return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if method == "zscore":
        def _zscore(series: pd.Series) -> pd.Series:
            valid = pd.to_numeric(series, errors="coerce")
            std = float(valid.std(ddof=0))
            if not pd.notna(std) or std <= 0.0:
                out = pd.Series(float("nan"), index=series.index, dtype=float)
                out.loc[valid.notna()] = 0.0
                return out
            return (valid - float(valid.mean())) / std

        transform_fn: Callable[[pd.Series], pd.Series] = _zscore
    else:
        def _rank_to_minus1_plus1(series: pd.Series) -> pd.Series:
            valid = pd.to_numeric(series, errors="coerce")
            n = int(valid.notna().sum())
            if n <= 1:
                out = pd.Series(float("nan"), index=series.index, dtype=float)
                out.loc[valid.notna()] = 0.0
                return out
            ranks = valid.rank(method="average")
            return 2.0 * (ranks - 1.0) / float(n - 1) - 1.0

        transform_fn = _rank_to_minus1_plus1

    out[out_col] = out.groupby(date_col, sort=False)[score_col].transform(transform_fn)
    sort_cols = [col for col in [date_col, "ticker"] if col in out.columns]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def prepare_cross_sectional_scores(
    predictions: pd.DataFrame,
    *,
    raw_score_col: str = "score",
    out_col: str = "score_signal",
    score_smoothing: str = "none",
    score_normalization: str = "none",
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    smoothed_col = "__score_smoothed"
    out = smooth_scores_by_ticker(
        predictions,
        score_col=raw_score_col,
        smoothing=score_smoothing,
        out_col=smoothed_col,
        ticker_col=ticker_col,
        date_col=date_col,
    )
    out = normalize_scores_cross_sectionally(
        out,
        score_col=smoothed_col,
        normalization=score_normalization,
        out_col=out_col,
        date_col=date_col,
    )
    return out.drop(columns=[smoothed_col]).sort_values([date_col, ticker_col], kind="mergesort").reset_index(drop=True)
