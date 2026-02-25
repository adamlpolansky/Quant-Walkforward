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