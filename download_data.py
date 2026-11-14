"""Download 8 years of daily OHLCV data for the project stock.

Tries RELIANCE.NS first, then AAPL. Saves to data/stock.csv.
Run:  python download_data.py
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "stock.csv"
TICKERS = ["RELIANCE.NS", "AAPL"]
YEARS = 8


def fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for one ticker and normalise the frame."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance may return a MultiIndex (field, ticker) for single tickers.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename_axis("Date").reset_index()
    df.columns = [str(c).strip() for c in df.columns]
    keep = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE)
        print(f"{OUT_FILE} already exists ({len(existing)} rows) - not downloading.")
        return

    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=365 * YEARS)

    for ticker in TICKERS:
        print(f"Trying {ticker} ...")
        try:
            df = fetch_history(ticker, start.isoformat(), end.isoformat())
        except Exception as exc:  # network / API errors
            print(f"  failed: {exc}")
            continue
        if df.empty:
            print("  no data returned")
            continue
        if len(df) < 500:
            print(f"  only {len(df)} rows, skipping")
            continue
        df["Ticker"] = ticker
        df.to_csv(OUT_FILE, index=False)
        print(f"Saved {len(df)} rows for {ticker} -> {OUT_FILE}")
        print(df.head(3).to_string())
        print(df.tail(3).to_string())
        return

    raise SystemExit("Could not download data for any ticker. Save data/stock.csv manually.")


if __name__ == "__main__":
    main()
