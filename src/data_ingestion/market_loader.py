
from pathlib import Path

import pandas as pd
import yfinance as yf


# Locate the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_market_data(
    ticker: str = "AAPL",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download historical stock market data."""
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(
            f"No market data returned for ticker: {ticker}"
        )

    # Handle MultiIndex columns returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = "Date"
    return data


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = load_market_data()
    output_file = RAW_DATA_DIR / "market_raw.parquet"
    data.to_parquet(output_file)

    print(f"Saved raw market data to {output_file}")
    print(f"Rows downloaded: {len(data)}")
    print(data.tail())


if __name__ == "__main__":
    main()
