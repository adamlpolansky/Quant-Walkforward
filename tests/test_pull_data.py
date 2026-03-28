from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PULL_DATA_PATH = ROOT / "scripts" / "pull_data.py"


def _load_pull_data_module():
    spec = importlib.util.spec_from_file_location("pull_data_script", PULL_DATA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {PULL_DATA_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_default_etf_flag() -> None:
    pull_data = _load_pull_data_module()

    args = pull_data.parse_args(["--use-default-etf-universe", "--start-date", "2018-01-01"])

    assert args.use_default_etf_universe is True
    assert args.start_date == "2018-01-01"


def test_resolve_tickers_supports_default_universe_and_dedupes() -> None:
    pull_data = _load_pull_data_module()

    tickers = pull_data.resolve_tickers(
        ticker_args=["SPY", "XLF,QQQ"],
        tickers_csv="XLC,SPY",
        use_default_etf_universe=True,
    )

    assert tickers == list(pull_data.DEFAULT_ETF_UNIVERSE)


def test_save_price_csv_writes_loader_contract(tmp_path: Path) -> None:
    pull_data = _load_pull_data_module()
    out_path = tmp_path / "real_etf_xs" / "SPY.csv"
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [1_000_000, 1_100_000],
            "ret": [None, 0.01],
            "log_ret": [None, 0.00995],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )

    pull_data.save_price_csv(df, out_path)

    saved = pd.read_csv(out_path)
    assert list(saved.columns[:6]) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(saved) == 2
