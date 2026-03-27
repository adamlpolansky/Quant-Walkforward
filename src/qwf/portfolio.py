from __future__ import annotations

import numpy as np
import pandas as pd


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
    missing = [col for col in required_cols if col not in pred_df.columns]
    if missing:
        raise ValueError(f"Prediction data is missing required columns: {missing}")

    detail = pred_df.copy()
    detail["date"] = pd.to_datetime(detail["date"])
    detail["forward_ret"] = pd.to_numeric(detail[forward_ret_col], errors="coerce")
    detail = detail.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    detail["weight"] = 0.0

    for _, idx in detail.groupby("date", sort=True).groups.items():
        day_idx = pd.Index(idx)
        day = detail.loc[day_idx].copy()
        candidates = day[day[score_col].notna() & day["forward_ret"].notna()].copy()
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
        )
        .reset_index()
    )
    daily["cost"] = daily["turnover"] * (cost_bps_per_turnover / 10_000.0)
    daily["portfolio_ret_net"] = daily["portfolio_ret_gross"] - daily["cost"]
    daily["equity"] = (1.0 + daily["portfolio_ret_net"]).cumprod()

    return detail, daily
