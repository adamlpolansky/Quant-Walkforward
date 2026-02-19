from pathlib import Path
from qwf.data import load_ohlcv, add_returns

def main():
    ticker = "SPY"
    df = load_ohlcv(ticker, start="2018-01-01")
    df = add_returns(df)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{ticker}.csv"

    df.to_csv(out_path)
    print(f"Saved: {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
