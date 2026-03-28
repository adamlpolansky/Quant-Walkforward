from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_required_columns(df: pd.DataFrame, required_cols: list[str], *, source_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def build_daily_top_bottom_portfolio(
    pred_df: pd.DataFrame,
    score_col: str,
    forward_ret_col: str,
    k: int = 3,
    cost_bps_per_turnover: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a daily dollar-neutral long-short portfolio from cross-sectional scores.

    Turnover convention:
      one-way turnover = 0.5 * sum(abs(weight_t - weight_{t-1})) across tickers
    Cost convention:
      cost = turnover * (cost_bps_per_turnover / 10000)
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    required_cols = ["date", "ticker", score_col, forward_ret_col]
    _validate_required_columns(pred_df, required_cols, source_name="Prediction data")

    detail = pred_df.copy()
    detail["date"] = pd.to_datetime(detail["date"])
    detail["forward_ret"] = pd.to_numeric(detail[forward_ret_col], errors="coerce")
    detail = detail.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    detail["weight"] = 0.0
    detail["is_constant_score_day"] = False
    detail["n_valid_scores"] = 0

    for _, idx in detail.groupby("date", sort=True).groups.items():
        day_idx = pd.Index(idx)
        day = detail.loc[day_idx].copy()
        candidates = day[day[score_col].notna() & day["forward_ret"].notna()].copy()
        is_constant_score_day = len(candidates) > 0 and int(candidates[score_col].nunique(dropna=True)) <= 1
        detail.loc[day_idx, "is_constant_score_day"] = bool(is_constant_score_day)
        detail.loc[day_idx, "n_valid_scores"] = int(len(candidates))

        if is_constant_score_day:
            continue

        select_k = min(k, len(candidates) // 2)

        if select_k == 0:
            continue

        long_side = candidates.sort_values([score_col, "ticker"], ascending=[False, True], kind="mergesort").head(select_k)
        short_side = candidates.sort_values([score_col, "ticker"], ascending=[True, True], kind="mergesort").head(select_k)

        detail.loc[long_side.index, "weight"] = 1.0 / select_k
        detail.loc[short_side.index, "weight"] = -1.0 / select_k

    detail = detail.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    detail["prev_weight"] = detail.groupby("ticker", sort=False)["weight"].shift(1).fillna(0.0)
    detail["abs_weight_change"] = (detail["weight"] - detail["prev_weight"]).abs()
    detail["contribution"] = np.where(
        detail["weight"] != 0.0,
        detail["weight"] * detail["forward_ret"],
        0.0,
    )

    detail = detail.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)

    daily = (
        detail.groupby("date", sort=True)
        .agg(
            portfolio_ret_gross=("contribution", "sum"),
            turnover=("abs_weight_change", lambda s: 0.5 * float(s.sum())),
            gross_exposure=("weight", lambda s: float(s.abs().sum())),
            net_exposure=("weight", "sum"),
            n_longs=("weight", lambda s: int((s > 0.0).sum())),
            n_shorts=("weight", lambda s: int((s < 0.0).sum())),
            is_constant_score_day=("is_constant_score_day", "max"),
            n_valid_scores=("n_valid_scores", "max"),
        )
        .reset_index()
    )
    daily["cost"] = daily["turnover"] * (cost_bps_per_turnover / 10_000.0)
    daily["portfolio_ret_net"] = daily["portfolio_ret_gross"] - daily["cost"]
    daily["equity"] = (1.0 + daily["portfolio_ret_net"]).cumprod()
    daily["is_constant_score_day"] = daily["is_constant_score_day"].astype(bool)

    return detail, daily


def summarize_ticker_selection(
    portfolio_detail: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
    score_col: str = "score",
    forward_ret_col: str = "forward_ret",
    weight_col: str = "weight",
    contribution_col: str = "contribution",
) -> pd.DataFrame:
    required_cols = [ticker_col, date_col, score_col, forward_ret_col, weight_col, contribution_col]
    _validate_required_columns(portfolio_detail, required_cols, source_name="Portfolio detail")

    detail = portfolio_detail.copy()
    detail[date_col] = pd.to_datetime(detail[date_col])
    detail[score_col] = pd.to_numeric(detail[score_col], errors="coerce")
    detail[forward_ret_col] = pd.to_numeric(detail[forward_ret_col], errors="coerce")
    detail[weight_col] = pd.to_numeric(detail[weight_col], errors="coerce").fillna(0.0)
    detail[contribution_col] = pd.to_numeric(detail[contribution_col], errors="coerce").fillna(0.0)

    total_dates = int(detail[date_col].nunique())
    if total_dates <= 0:
        raise ValueError("Portfolio detail must contain at least one date")

    rows: list[dict[str, float | int | str]] = []
    for ticker, group in detail.groupby(ticker_col, sort=True):
        long_mask = group[weight_col] > 0.0
        short_mask = group[weight_col] < 0.0
        selected_mask = group[weight_col] != 0.0

        rows.append(
            {
                "ticker": str(ticker),
                "n_long_days": int(long_mask.sum()),
                "n_short_days": int(short_mask.sum()),
                "avg_score": float(group[score_col].mean()),
                "avg_forward_return_when_long": float(group.loc[long_mask, forward_ret_col].mean()),
                "avg_forward_return_when_short": float(group.loc[short_mask, forward_ret_col].mean()),
                "mean_contribution": float(group[contribution_col].mean()),
                "total_contribution": float(group[contribution_col].sum()),
                "avg_weight_when_selected": float(group.loc[selected_mask, weight_col].abs().mean()),
                "selection_rate_long": float(long_mask.sum() / total_dates),
                "selection_rate_short": float(short_mask.sum() / total_dates),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    return summary.sort_values(["total_contribution", "ticker"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
