from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from qwf import metrics
from qwf.models import fit_model, predict_scores, resolve_model_params
from qwf.portfolio import build_daily_top_bottom_portfolio


def _normalize_plan(plan: pd.DataFrame) -> pd.DataFrame:
    required = ["fold_id", "train_start", "train_end", "test_start", "test_end"]
    missing = [col for col in required if col not in plan.columns]
    if missing:
        raise ValueError(f"Split plan is missing required columns: {missing}")

    out = plan.copy()
    for col in required[1:]:
        out[col] = pd.to_datetime(out[col])

    return out.sort_values("fold_id", kind="mergesort").reset_index(drop=True)


def _resolve_train_label_cutoff(
    global_dates: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    label_horizon: int,
) -> pd.Timestamp:
    train_end_pos = int(global_dates.get_indexer([pd.Timestamp(train_end)])[0])
    if train_end_pos < 0:
        raise ValueError(f"train_end={train_end} is not present in the shared panel calendar")

    cutoff_pos = train_end_pos - label_horizon
    if cutoff_pos < 0:
        raise ValueError(
            f"Not enough in-sample history to fit with label_horizon={label_horizon} at train_end={train_end}"
        )

    return pd.Timestamp(global_dates[cutoff_pos])


def run_cross_sectional_walkforward_experiment(
    panel: pd.DataFrame,
    plan: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str,
    label_horizon: int = 1,
    model_name: str = "ridge",
    model_params: Mapping[str, Any] | None = None,
    alpha: float = 1.0,
    k: int = 3,
    cost_bps_per_turnover: float = 0.0,
    score_col: str = "score",
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    if "date" not in panel.columns or "ticker" not in panel.columns:
        raise ValueError("Panel must contain 'date' and 'ticker' columns")
    if label_col not in panel.columns:
        raise ValueError(f"Panel is missing label column '{label_col}'")

    panel_sorted = panel.copy()
    panel_sorted["date"] = pd.to_datetime(panel_sorted["date"])
    panel_sorted = panel_sorted.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)

    global_dates = pd.DatetimeIndex(panel_sorted["date"].drop_duplicates().sort_values())
    plan_df = _normalize_plan(plan)
    fold_predictions: list[pd.DataFrame] = []
    normalized_model_name = str(model_name).strip().lower()
    effective_model_params = dict(model_params or {})
    if not effective_model_params and normalized_model_name in {"ridge", "lasso", "elasticnet"}:
        effective_model_params["alpha"] = alpha
    resolved_model_params = resolve_model_params(normalized_model_name, effective_model_params)

    for row in plan_df.itertuples(index=False):
        train_label_cutoff = _resolve_train_label_cutoff(global_dates, row.train_end, label_horizon)

        train_mask = (panel_sorted["date"] >= row.train_start) & (panel_sorted["date"] <= train_label_cutoff)
        test_mask = (panel_sorted["date"] >= row.test_start) & (panel_sorted["date"] <= row.test_end)

        train_df = panel_sorted.loc[train_mask].copy()
        test_df = panel_sorted.loc[test_mask].copy()
        if test_df.empty:
            continue

        model = fit_model(
            train_df,
            feature_cols,
            label_col=label_col,
            model_name=normalized_model_name,
            model_params=resolved_model_params,
        )
        scored_test = predict_scores(model, test_df, feature_cols, score_col=score_col)
        scored_test["fold_id"] = row.fold_id
        scored_test["train_start"] = row.train_start
        scored_test["train_end"] = row.train_end
        scored_test["test_start"] = row.test_start
        scored_test["test_end"] = row.test_end
        fold_predictions.append(scored_test)

    if not fold_predictions:
        raise RuntimeError("No test predictions were generated from the supplied panel and split plan")

    predictions = pd.concat(fold_predictions, ignore_index=True, sort=False)
    predictions = predictions.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)

    portfolio_detail, portfolio_daily = build_daily_top_bottom_portfolio(
        predictions,
        score_col=score_col,
        forward_ret_col=label_col,
        k=k,
        cost_bps_per_turnover=cost_bps_per_turnover,
    )
    ic_daily = metrics.daily_cross_sectional_ic(
        predictions,
        score_col=score_col,
        forward_ret_col=label_col,
    )
    spread_daily = metrics.daily_long_short_spread(portfolio_detail)

    summary: dict[str, Any] = {
        "n_folds": int(plan_df["fold_id"].nunique()),
        "n_prediction_rows": int(len(predictions)),
        "n_scored_rows": int(predictions[score_col].notna().sum()),
        "n_portfolio_days": int(len(portfolio_daily)),
        "n_tickers": int(predictions["ticker"].nunique()),
        "model_name": normalized_model_name,
        "model_params": dict(resolved_model_params),
        "alpha": float(resolved_model_params.get("alpha", alpha)),
        "k": int(k),
        "cost_bps_per_turnover": float(cost_bps_per_turnover),
        "label_horizon": int(label_horizon),
        "label_col": label_col,
        "score_col": score_col,
        "turnover_convention": "one_way_half_abs_change",
        "feature_cols": list(feature_cols),
        **metrics.summarize_ic(ic_daily),
        **metrics.summarize_spread(spread_daily),
        **metrics.portfolio_perf_summary(portfolio_daily),
    }

    return {
        "predictions": predictions,
        "portfolio_detail": portfolio_detail,
        "portfolio_daily": portfolio_daily,
        "ic_daily": ic_daily,
        "spread_daily": spread_daily,
        "summary": summary,
    }
