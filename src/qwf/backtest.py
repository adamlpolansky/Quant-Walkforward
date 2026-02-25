from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, dataclass
import json
import numpy as np
import pandas as pd



@dataclass(frozen=True)
class ZScoreMRConfig:
    n: int = 20                 # rolling window for z-score
    K: float = 1.0              # max exposure (leverage), pos in [-K, K]
    step_frac: float = 0.25     # daily adjustment step as fraction of K (0.25 => Δ=0.25K)
    ddof: int = 0               # rolling std ddof
    min_periods: int | None = None  # if None -> n


def compute_zscore_mr_position(
    df: pd.DataFrame,
    price_col: str = "Close",
    cfg: ZScoreMRConfig = ZScoreMRConfig(),
) -> pd.DataFrame:
    """
    Compute z-score mean reversion signal and a position series with:
      - z = (Close - MA(n)) / STD(n), rolling, ddof=0
      - target exposure in rungs: ±{0.25K, 0.5K, 0.75K, 1.0K} based on |z|
      - gradual position adjustment: pos(t) = pos(t-1) + clip(target - pos(t-1), -Δ, +Δ)
        where Δ = step_frac * K

    Returns a DataFrame with columns: z, target_pos, pos
    Notes:
      - If z is NaN (warm-up or std=0), target_pos=0.
      - Single position variable pos ∈ [-K, K] => cannot be long and short simultaneously.
    """
    if cfg.n < 2:
        raise ValueError("cfg.n must be >= 2")
    if cfg.K <= 0:
        raise ValueError("cfg.K must be > 0")
    if not (0 < cfg.step_frac <= 1):
        raise ValueError("cfg.step_frac must be in (0, 1]")

    if price_col not in df.columns:
        raise ValueError(f"Missing price_col='{price_col}' in df.columns")

    x = pd.to_numeric(df[price_col], errors="coerce").astype(float)

    minp = cfg.n if cfg.min_periods is None else int(cfg.min_periods)
    ma = x.rolling(cfg.n, min_periods=minp).mean()
    sd = x.rolling(cfg.n, min_periods=minp).std(ddof=cfg.ddof)

    # Avoid division by 0; keep NaN where sd==0 or insufficient data.
    z = (x - ma) / sd.replace(0.0, np.nan)

    K = float(cfg.K)

    # Map z -> target position (rungs). NaN z => 0 target.
    absz = z.abs()

    # Build magnitude in {0, 0.25, 0.5, 0.75, 1.0}
    mag = pd.Series(0.0, index=df.index)
    mag = mag.where(~(absz >= 1.0), 0.25)
    mag = mag.where(~(absz >= 2.0), 0.50)
    mag = mag.where(~(absz >= 3.0), 0.75)
    mag = mag.where(~(absz >= 4.0), 1.00)

    # Direction: mean reversion => negative z => long, positive z => short
    sign = -np.sign(z)  # z<0 => +1, z>0 => -1, z==0 => 0
    target_pos = (sign * mag * K).fillna(0.0)

    # Gradual adjustment toward target
    delta = cfg.step_frac * K
    pos = pd.Series(0.0, index=df.index, dtype=float)
    prev = 0.0
    for i, t in enumerate(df.index):
        tgt = float(target_pos.iloc[i])
        step = np.clip(tgt - prev, -delta, +delta)
        prev = prev + step
        # Numerical safety clamp
        prev = float(np.clip(prev, -K, +K))
        pos.iloc[i] = prev

    out = pd.DataFrame(
        {
            "z": z.astype(float),
            "target_pos": target_pos.astype(float),
            "pos": pos.astype(float),
        },
        index=df.index,
    )
    return out


def backtest_from_position_logret(
    df: pd.DataFrame,
    pos: pd.Series,
    logret_col: str = "log_ret",
) -> pd.DataFrame:
    """
    Backtest using log returns with strict lag enforcement:
      pnl(t) = pos(t-1) * log_ret(t)

    Returns DataFrame with: pos, pos_lag, log_ret, pnl, cum_pnl, equity
    where equity = exp(cum_pnl).
    """
    if logret_col not in df.columns:
        raise ValueError(f"Missing logret_col='{logret_col}' in df.columns")

    log_ret = pd.to_numeric(df[logret_col], errors="coerce").astype(float)

    # Align indices
    pos = pos.reindex(df.index).astype(float)
    pos_lag = pos.shift(1)

    pnl = pos_lag * log_ret
    cum_pnl = pnl.cumsum(min_count=1)
    equity = np.exp(cum_pnl)

    out = pd.DataFrame(
        {
            "pos": pos,
            "pos_lag": pos_lag,
            "log_ret": log_ret,
            "pnl": pnl,
            "cum_pnl": cum_pnl,
            "equity": equity,
        },
        index=df.index,
    )
    return out

def backtest_from_position_ret(
    df: pd.DataFrame,
    pos: pd.Series,
    ret_col: str = "ret",
    fill_missing_ret_with_zero: bool = True,
) -> pd.DataFrame:
    """
    Backtest using SIMPLE returns with strict lag enforcement:
      pnl(t) = pos(t-1) * ret(t)

    Returns DataFrame with: pos, pos_lag, ret, pnl, cum_pnl, equity
    where:
      - pnl is per-period portfolio simple return
      - equity = (1 + pnl).cumprod()
      - cum_pnl = equity - 1
    """
    if ret_col not in df.columns:
        raise ValueError(f"Missing ret_col='{ret_col}' in df.columns")

    ret = pd.to_numeric(df[ret_col], errors="coerce").astype(float)

    # Align indices
    pos = pos.reindex(df.index).astype(float)
    pos_lag = pos.shift(1)

    if fill_missing_ret_with_zero:
        ret_used = ret.fillna(0.0)
    else:
        ret_used = ret

    pnl = pos_lag * ret_used  # portfolio simple return per period
    equity = (1.0 + pnl).cumprod()

    # If you prefer equity to start at 1 even if first rows are NaN-ish,
    # fill initial NaNs in pnl with 0 by turning on fill_missing_ret_with_zero.
    cum_pnl = equity - 1.0

    out = pd.DataFrame(
        {
            "pos": pos,
            "pos_lag": pos_lag,
            "ret": ret,
            "pnl": pnl,
            "cum_pnl": cum_pnl,
            "equity": equity,
        },
        index=df.index,
    )
    return out

def run_wf_backtest_ret(
    df: pd.DataFrame,
    plan: pd.DataFrame,
    *,
    price_col: str = "Close",
    ret_col: str = "ret",
    cfg: ZScoreMRConfig = ZScoreMRConfig(),
    source_file: str | None = None,
    fill_missing_ret_with_zero: bool = True,
) -> pd.DataFrame:
    """
    Pure walk-forward backtest (NO I/O):
      - df: price/returns DataFrame indexed by DatetimeIndex
      - plan: DataFrame with columns:
          fold_id, train_start, train_end, test_start, test_end
        (optionally also 'source_file' if you generated a multi-file plan)

    Returns:
      test_detail DataFrame indexed by date with columns:
        fold_id, train_start, train_end, test_start, test_end,
        z, target_pos, pos, pos_lag, ret, pnl, cum_pnl, equity
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # tz-naive, sorted unique index
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    required = {"fold_id", "train_start", "train_end", "test_start", "test_end"}
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"plan is missing columns: {sorted(missing)}")

    plan_used = plan.copy()

    # optional filter if plan contains multiple tickers/files
    if source_file is not None and "source_file" in plan_used.columns:
        plan_used = plan_used.loc[plan_used["source_file"] == source_file].copy()
        if plan_used.empty:
            raise ValueError(f"No folds left after filtering plan by source_file={source_file!r}")

    # ensure datetimes
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        plan_used[c] = pd.to_datetime(plan_used[c], errors="raise")

    # stable fold ordering
    plan_used = plan_used.sort_values(["fold_id"]).reset_index(drop=True)

    test_details: list[pd.DataFrame] = []

    for _, row in plan_used.iterrows():
        fold_id = row["fold_id"]
        train_start = pd.Timestamp(row["train_start"])
        train_end = pd.Timestamp(row["train_end"])
        test_start = pd.Timestamp(row["test_start"])
        test_end = pd.Timestamp(row["test_end"])

        # compute signal on train_start..test_end so the first test day has lag from train period
        sub = df.loc[train_start:test_end].copy()
        if sub.empty:
            raise ValueError(f"Fold {fold_id}: no data in range {train_start}..{test_end}")

        sig = compute_zscore_mr_position(sub, price_col=price_col, cfg=cfg)
        pos = sig["pos"]

        bt = backtest_from_position_ret(
            sub,
            pos=pos,
            ret_col=ret_col,
            fill_missing_ret_with_zero=fill_missing_ret_with_zero,
        )

        # add signal columns (avoid duplicating pos)
        bt = bt.join(sig[["z", "target_pos"]], how="left")

        test_bt = bt.loc[test_start:test_end].copy()
        if test_bt.empty:
            raise ValueError(f"Fold {fold_id}: no TEST rows in range {test_start}..{test_end}")

        # annotate fold metadata (handy for metrics later)
        test_bt["fold_id"] = fold_id
        test_bt["train_start"] = train_start
        test_bt["train_end"] = train_end
        test_bt["test_start"] = test_start
        test_bt["test_end"] = test_end
        if "source_file" in plan_used.columns:
            test_bt["source_file"] = row.get("source_file", np.nan)

        # Recompute fold-local equity/cum_pnl on TEST rows only
        # (pos_lag is already correct because signal/backtest were computed on train+test)
        pnl_local = pd.to_numeric(test_bt["pnl"], errors="coerce").astype(float)
        pnl_filled = pnl_local.fillna(0.0)

        test_bt["cum_pnl"] = pnl_filled.cumsum()
        test_bt["equity"] = (1.0 + pnl_filled).cumprod()

        test_details.append(test_bt)

    test_detail = pd.concat(test_details, axis=0)

    # keep chronological order (if folds overlap in time, it'll interleave; that's fine for later groupby fold_id)
    test_detail = test_detail.sort_index()

    # nice column order (optional)
    preferred = [
        "source_file",
        "fold_id", "train_start", "train_end", "test_start", "test_end",
        "z", "target_pos",
        "pos", "pos_lag",
        "ret", "pnl", "cum_pnl", "equity",
        "log_ret",
    ]
    cols = [c for c in preferred if c in test_detail.columns] + [c for c in test_detail.columns if c not in preferred]
    test_detail = test_detail[cols]

    return test_detail
