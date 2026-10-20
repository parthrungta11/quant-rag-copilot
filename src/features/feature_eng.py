
from pathlib import Path

import pandas as pd
import pandas_ta as ta


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "market_raw.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators from historical stock data."""
    df = data.copy()

    # Handle yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Daily percentage return
    df["Daily_Return"] = df["Close"].pct_change()

    # Moving averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Rolling volatility (20 trading days)
    df["Volatility_20"] = (
        df["Daily_Return"].rolling(window=20).std()
    )

    # Relative Strength Index
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # Drop initial rows where rolling indicators are unavailable
    df = df.dropna()

    return df


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw market data not found: {RAW_FILE}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = pd.read_parquet(RAW_FILE)
    features = build_features(raw_data)

    output_file = PROCESSED_DIR / "market_features.parquet"
    features.to_parquet(output_file)

    print(f"Saved processed features to {output_file}")
    print(f"Rows after feature engineering: {len(features)}")
    print(features.tail())


if __name__ == "__main__":
    main()
