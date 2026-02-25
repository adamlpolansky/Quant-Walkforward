from pathlib import Path
from qwf.data import load_ohlcv, add_returns

ROOT = Path(__file__).resolve().parents[1]

def main():
    ticker = "SPY"
    df = load_ohlcv(ticker, start="2018-01-01")
    df = add_returns(df)

    out_dir = ROOT / "scripts" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.csv"

    df.to_csv(out_path)
    print(f"Saved: {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()