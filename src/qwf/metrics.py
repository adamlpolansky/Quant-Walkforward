# src/qwf/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def compute_pnl_series(
    detail: pd.DataFrame,
    *,
    ret_col: str = "ret",
    pos_col: str = "pos",
    pos_lag_col: str = "pos_lag",
    pnl_col: str = "pnl",
    fill_missing_ret_with_zero: bool = True,
) -> pd.Series:
    """
    Strategy period return:
      pnl(t) = pos(t-1) * ret(t)
    Preference:
      - use pnl_col if present
      - else compute from pos_lag (or lag(pos)) and ret
    """
    if pnl_col in detail.columns:
        pnl = _to_float(detail[pnl_col])
        return pnl.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if ret_col not in detail.columns:
        raise ValueError(f"Missing '{ret_col}' and '{pnl_col}' -> cannot compute pnl.")

    ret = _to_float(detail[ret_col])
    if fill_missing_ret_with_zero:
        ret = ret.fillna(0.0)

    if pos_lag_col in detail.columns:
        pos_lag = _to_float(detail[pos_lag_col]).fillna(0.0)
    elif pos_col in detail.columns:
        pos_lag = _to_float(detail[pos_col]).shift(1).fillna(0.0)
    else:
        raise ValueError(f"Missing '{pos_col}'/'{pos_lag_col}' -> cannot compute pnl.")

    pnl = pos_lag * ret
    return pnl.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def equity_from_pnl(pnl: pd.Series, *, start_equity: float = 1.0) -> pd.Series:
    """Compounded equity curve from pnl (period returns)."""
    one_plus = (1.0 + pnl).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return float(start_equity) * one_plus.cumprod()


def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    """Drawdown series (negative), dd = equity/cummax - 1."""
    eq = _to_float(equity).replace([np.inf, -np.inf], np.nan)
    dd = eq / eq.cummax() - 1.0
    return dd


def rolling_sharpe(
    pnl: pd.Series,
    *,
    window: int = 63,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling Sharpe from period pnl (simple returns)."""
    x = _to_float(pnl).replace([np.inf, -np.inf], np.nan)
    mu = x.rolling(window, min_periods=max(5, window // 3)).mean()
    sd = x.rolling(window, min_periods=max(5, window // 3)).std(ddof=0)
    sr = (mu / sd) * np.sqrt(periods_per_year)
    return sr


def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    x = _to_float(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan

    downside = x[x < target] - float(target)
    if downside.empty:
        return np.nan

    downside_dev = float(downside.std(ddof=0))
    if not np.isfinite(downside_dev) or downside_dev <= 0.0:
        return np.nan

    numerator = float((x - float(target)).mean()) * periods_per_year
    denominator = downside_dev * np.sqrt(periods_per_year)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan

    out = numerator / denominator
    return float(out) if np.isfinite(out) else np.nan


def add_pnl(
    detail: pd.DataFrame,
    *,
    ret_col: str = "ret",
    pos_col: str = "pos",
    pos_lag_col: str = "pos_lag",
    pnl_col: str = "pnl",
    equity_col: str = "equity",
    cum_pnl_col: str = "cum_pnl",
    start_equity: float = 1.0,
) -> pd.DataFrame:
    """
    Add/overwrite pnl + equity + cum_pnl into a COPY of detail.
    cum_pnl is (equity - start_equity) for consistency with compounded returns.
    """
    out = detail.copy()

    # create pos_lag if needed
    if pos_lag_col not in out.columns and pos_col in out.columns:
        out[pos_lag_col] = _to_float(out[pos_col]).shift(1).fillna(0.0)

    pnl = compute_pnl_series(
        out,
        ret_col=ret_col,
        pos_col=pos_col,
        pos_lag_col=pos_lag_col,
        pnl_col=pnl_col,
        fill_missing_ret_with_zero=True,
    )
    out[pnl_col] = pnl

    eq = equity_from_pnl(pnl, start_equity=start_equity)
    out[equity_col] = eq
    out[cum_pnl_col] = eq - float(start_equity)
    return out


def add_fold_local_curves(
    test_detail: pd.DataFrame,
    *,
    fold_col: str = "fold_id",
    ret_col: str = "ret",
    pos_col: str = "pos",
    pos_lag_col: str = "pos_lag",
    pnl_col: str = "pnl",
    out_equity_col: str = "equity_local",
    out_dd_col: str = "dd_local",
) -> pd.DataFrame:
    """
    Fold-local curves (equity reset to 1.0 within each fold):
      - equity_local
      - dd_local
    Returns COPY with added columns.
    """
    if fold_col not in test_detail.columns:
        raise ValueError(f"Missing '{fold_col}' in test_detail.")

    out = test_detail.copy()
    out[out_equity_col] = np.nan
    out[out_dd_col] = np.nan

    for fold_id, g in out.groupby(fold_col, sort=True):
        pnl = compute_pnl_series(
            g,
            ret_col=ret_col,
            pos_col=pos_col,
            pos_lag_col=pos_lag_col,
            pnl_col=pnl_col,
            fill_missing_ret_with_zero=True,
        )
        eq = equity_from_pnl(pnl, start_equity=1.0)  # ✅ reset per fold
        dd = drawdown_from_equity(eq)
        out.loc[g.index, out_equity_col] = eq.values
        out.loc[g.index, out_dd_col] = dd.values

    return out


def stitched_curve(
    test_detail: pd.DataFrame,
    *,
    ret_col: str = "ret",
    pos_col: str = "pos",
    pos_lag_col: str = "pos_lag",
    pnl_col: str = "pnl",
    on_duplicate_index: str = "raise",  # "raise" | "sum"
    rolling_sharpe_window: int | None = 63,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    One stitched curve over the whole WF test timeline (equity reset to 1.0 once).

    If index has duplicates (shouldn't in your WF), you can:
      - on_duplicate_index="raise" (default)
      - on_duplicate_index="sum"   (sum pnl for same timestamp)
    """
    td = test_detail.copy()

    if not isinstance(td.index, pd.DatetimeIndex):
        td.index = pd.to_datetime(td.index)

    td = td.sort_index()

    if td.index.has_duplicates:
        if on_duplicate_index == "raise":
            dup = td.index[td.index.duplicated()].unique()
            raise ValueError(f"test_detail index has duplicates (example): {list(dup[:5])}")
        if on_duplicate_index == "sum":
            # aggregate pnl at same timestamp
            pnl_raw = compute_pnl_series(
                td,
                ret_col=ret_col,
                pos_col=pos_col,
                pos_lag_col=pos_lag_col,
                pnl_col=pnl_col,
                fill_missing_ret_with_zero=True,
            )
            pnl_raw.name = "pnl"
            pnl = pnl_raw.groupby(pnl_raw.index).sum()
        else:
            raise ValueError("on_duplicate_index must be 'raise' or 'sum'")
    else:
        pnl = compute_pnl_series(
            td,
            ret_col=ret_col,
            pos_col=pos_col,
            pos_lag_col=pos_lag_col,
            pnl_col=pnl_col,
            fill_missing_ret_with_zero=True,
        )
        pnl.name = "pnl"

    equity = equity_from_pnl(pnl, start_equity=1.0)
    dd = drawdown_from_equity(equity)

    out = pd.DataFrame({"pnl": pnl, "equity": equity, "drawdown": dd})

    if rolling_sharpe_window is not None and rolling_sharpe_window >= 5:
        out["rolling_sharpe"] = rolling_sharpe(
            pnl,
            window=rolling_sharpe_window,
            periods_per_year=periods_per_year,
        )

    return out

def perf_stats_from_pnl(
    pnl: pd.Series,
    *,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Simple performance stats from period returns (pnl)."""
    x = _to_float(pnl).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
        }

    eq = equity_from_pnl(x.fillna(0.0), start_equity=1.0)
    dd = drawdown_from_equity(eq)

    n = int(len(x))
    total_return = float(eq.iloc[-1] - 1.0) if n else np.nan
    cagr = (1.0 + total_return) ** (periods_per_year / n) - 1.0 if n and np.isfinite(total_return) else np.nan

    sd = float(x.std(ddof=0)) if n > 1 else np.nan
    mu = float(x.mean()) if n > 0 else np.nan
    ann_vol = sd * np.sqrt(periods_per_year) if np.isfinite(sd) else np.nan
    sharpe = (mu / sd) * np.sqrt(periods_per_year) if (np.isfinite(sd) and sd > 0) else np.nan
    sortino = sortino_ratio(x, target=0.0, periods_per_year=periods_per_year)

    max_dd = float(dd.min()) if not dd.empty else np.nan

    return {
        "total_return": total_return,
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "ann_vol": float(ann_vol) if np.isfinite(ann_vol) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "max_drawdown": max_dd,
    }


def buy_hold_stitched_curve(
    test_detail: pd.DataFrame,
    *,
    ret_col: str = "ret",
    rolling_sharpe_window: int | None = 63,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Buy & Hold benchmark stitched over the same test timeline.
    pnl_bh(t) = ret(t)
    """
    if ret_col not in test_detail.columns:
        raise ValueError(f"Missing '{ret_col}' in test_detail -> cannot build buy&hold benchmark.")

    td = test_detail.copy()
    td["pnl_bh"] = _to_float(td[ret_col]).fillna(0.0)

    stitched = stitched_curve(
        td,
        pnl_col="pnl_bh",
        rolling_sharpe_window=rolling_sharpe_window,
        periods_per_year=periods_per_year,
    )
    return stitched

def fold_summary(
    test_detail: pd.DataFrame,
    *,
    fold_col: str = "fold_id",
    ret_col: str = "ret",
    pos_col: str = "pos",
    pos_lag_col: str = "pos_lag",
    pnl_col: str = "pnl",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Per-fold summary computed on TEST rows, with equity RESET to 1.0 in each fold.
    """
    if fold_col not in test_detail.columns:
        raise ValueError(f"Missing fold column '{fold_col}' in test_detail.columns")

    rows: list[dict] = []

    for fold_id, g in test_detail.groupby(fold_col, sort=True):
        pnl = compute_pnl_series(
            g,
            ret_col=ret_col,
            pos_col=pos_col,
            pos_lag_col=pos_lag_col,
            pnl_col=pnl_col,
            fill_missing_ret_with_zero=True,
        )

        eq = equity_from_pnl(pnl, start_equity=1.0)  # ✅ reset each fold
        dd = drawdown_from_equity(eq)

        n = int(eq.shape[0])
        total_ret = float(eq.iloc[-1] - 1.0) if n else np.nan
        max_dd = float(dd.min()) if n else np.nan

        ann_ret = (1.0 + total_ret) ** (periods_per_year / n) - 1.0 if n and np.isfinite(total_ret) else np.nan

        pnl_clean = pnl.replace([np.inf, -np.inf], np.nan).dropna()
        if len(pnl_clean) > 1:
            mu = float(pnl_clean.mean())
            sd = float(pnl_clean.std(ddof=0))
            ann_vol = sd * np.sqrt(periods_per_year)
            sharpe = (mu / sd) * np.sqrt(periods_per_year) if sd > 0 else np.nan
            win_rate = float((pnl_clean > 0).mean())
        else:
            ann_vol = np.nan
            sharpe = np.nan
            win_rate = np.nan

        row = {
            "fold_id": fold_id,
            "n_test_rows": int(len(g)),
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "win_rate": win_rate,
        }

        for c in ["train_start", "train_end", "test_start", "test_end", "source_file"]:
            if c in g.columns:
                row[c] = g[c].iloc[0]

        rows.append(row)

    out = pd.DataFrame(rows)

    preferred = [
        "source_file",
        "fold_id",
        "train_start", "train_end", "test_start", "test_end",
        "n_test_rows",
        "total_return", "max_drawdown",
        "ann_return", "ann_vol", "sharpe", "win_rate",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols].sort_values("fold_id").reset_index(drop=True)


def daily_cross_sectional_ic(
    predictions: pd.DataFrame,
    *,
    date_col: str = "date",
    score_col: str = "score",
    forward_ret_col: str = "forward_ret",
    compute_rank_ic: bool = True,
) -> pd.DataFrame:
    required = [date_col, score_col, forward_ret_col]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions are missing required IC columns: {missing}")

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for trade_date, group in predictions.groupby(date_col, sort=True):
        valid = group[[score_col, forward_ret_col]].dropna()
        pearson_ic = np.nan
        rank_ic = np.nan

        if len(valid) >= 2:
            if valid[score_col].nunique() > 1 and valid[forward_ret_col].nunique() > 1:
                pearson_ic = float(valid[score_col].corr(valid[forward_ret_col], method="pearson"))
                if compute_rank_ic:
                    rank_ic = float(valid[score_col].corr(valid[forward_ret_col], method="spearman"))

        rows.append(
            {
                "date": pd.Timestamp(trade_date),
                "n_assets": int(len(valid)),
                "ic_pearson": pearson_ic,
                "ic_spearman": rank_ic,
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_ic(
    ic_daily: pd.DataFrame,
    *,
    ic_col: str = "ic_pearson",
) -> dict[str, float]:
    if ic_col not in ic_daily.columns:
        raise ValueError(f"IC daily data is missing '{ic_col}'")

    x = _to_float(ic_daily[ic_col]).dropna()
    if x.empty:
        return {
            "mean_ic": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "n_ic_days": 0,
        }

    mean_ic = float(x.mean())
    ic_std = float(x.std(ddof=0))
    ic_ir = mean_ic / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": float(ic_ir) if np.isfinite(ic_ir) else np.nan,
        "n_ic_days": int(len(x)),
    }


def daily_long_short_spread(
    portfolio_detail: pd.DataFrame,
    *,
    date_col: str = "date",
    weight_col: str = "weight",
    forward_ret_col: str = "forward_ret",
) -> pd.DataFrame:
    required = [date_col, weight_col, forward_ret_col]
    missing = [col for col in required if col not in portfolio_detail.columns]
    if missing:
        raise ValueError(f"Portfolio detail is missing spread columns: {missing}")

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for trade_date, group in portfolio_detail.groupby(date_col, sort=True):
        longs = group[(group[weight_col] > 0.0) & group[forward_ret_col].notna()]
        shorts = group[(group[weight_col] < 0.0) & group[forward_ret_col].notna()]

        long_mean = float(longs[forward_ret_col].mean()) if not longs.empty else np.nan
        short_mean = float(shorts[forward_ret_col].mean()) if not shorts.empty else np.nan
        spread = long_mean - short_mean if np.isfinite(long_mean) and np.isfinite(short_mean) else np.nan

        rows.append(
            {
                "date": pd.Timestamp(trade_date),
                "n_longs": int(len(longs)),
                "n_shorts": int(len(shorts)),
                "long_mean_ret": long_mean,
                "short_mean_ret": short_mean,
                "spread": spread,
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_spread(
    spread_daily: pd.DataFrame,
    *,
    spread_col: str = "spread",
) -> dict[str, float]:
    if spread_col not in spread_daily.columns:
        raise ValueError(f"Spread daily data is missing '{spread_col}'")

    x = _to_float(spread_daily[spread_col]).dropna()
    if x.empty:
        return {
            "mean_spread": np.nan,
            "spread_std": np.nan,
            "n_spread_days": 0,
        }

    return {
        "mean_spread": float(x.mean()),
        "spread_std": float(x.std(ddof=0)),
        "n_spread_days": int(len(x)),
    }


def portfolio_perf_summary(
    portfolio_daily: pd.DataFrame,
    *,
    gross_ret_col: str = "portfolio_ret_gross",
    net_ret_col: str = "portfolio_ret_net",
    turnover_col: str = "turnover",
    cost_col: str = "cost",
    gross_exposure_col: str = "gross_exposure",
    periods_per_year: int = 252,
) -> dict[str, float]:
    missing = [col for col in [gross_ret_col, net_ret_col, turnover_col, cost_col] if col not in portfolio_daily.columns]
    if missing:
        raise ValueError(f"Portfolio daily data is missing performance columns: {missing}")

    gross_stats = perf_stats_from_pnl(portfolio_daily[gross_ret_col], periods_per_year=periods_per_year)
    net_stats = perf_stats_from_pnl(portfolio_daily[net_ret_col], periods_per_year=periods_per_year)

    turnover = _to_float(portfolio_daily[turnover_col])
    cost = _to_float(portfolio_daily[cost_col])

    if gross_exposure_col in portfolio_daily.columns:
        gross_exposure = _to_float(portfolio_daily[gross_exposure_col]).fillna(0.0)
        n_traded_days = int((gross_exposure > 0.0).sum())
    else:
        n_traded_days = int(_to_float(portfolio_daily[net_ret_col]).notna().sum())

    return {
        "total_return_gross": gross_stats["total_return"],
        "total_return_net": net_stats["total_return"],
        "cagr_net": net_stats["cagr"],
        "ann_vol_net": net_stats["ann_vol"],
        "sharpe_net": net_stats["sharpe"],
        "sortino_net": net_stats["sortino"],
        "max_drawdown_net": net_stats["max_drawdown"],
        "mean_turnover": float(turnover.mean()) if not turnover.empty else np.nan,
        "mean_cost": float(cost.mean()) if not cost.empty else np.nan,
        "n_traded_days": n_traded_days,
    }
