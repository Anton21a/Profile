import pandas as pd
import requests
import time
import pyarrow
from binance.client import Client


def download_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:

    client = Client()

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_date,
        end_str=end_date
    )
    print(f"Download complete. Received {len(klines)} raw candles.")

    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ]

    df = pd.DataFrame(klines, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").astype("datetime64[ns]")
    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume", "trades",
                    "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df[["timestamp"] + numeric_cols].copy()
    return df


def main():
    data = download_klines(
        symbol='BTCUSDT',
        interval=Client.KLINE_INTERVAL_5MINUTE,
        start_date="1 Feb, 2021",
        end_date="4 Sept, 2026",
    )

    numeric_cols = ["open", "high", "low", "close", "volume", "trades",
                    "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]

    assert all(data[col].dtype == float for col in numeric_cols), "Not all numeric columns are float dtype"
    print("✅ validation passed: all necessary columns are float type")
    assert data.isna().sum().sum() == 0, "Missing values found in dataset"
    print("✅ validation passed: dataset doesn't contain null nor missing values")
    assert pd.api.types.is_datetime64_ns_dtype(data["timestamp"]), f"timestamp column is not datetime64[ns], got {data['timestamp'].dtype}"
    print("✅ validation passed: timestamp is datatime variable")

    print(data.head())

    data.to_parquet('Data/raw_data/data_binance.parquet', index=False)


if __name__ == "__main__":
    main()