import logging
import os
from pathlib import Path

import boto3
import pandas as pd

import pipeline

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

BUCKET = os.environ["BUCKET_NAME"]
RAW_KEY = "data/raw_data/data_binance.parquet"
RESAMPLED_KEY = "data/resampled_data/resampled_data_4h.parquet"
CLEANED_KEY = "data_for_analysis/cleaned_data.parquet"

TMP = Path("/tmp")
LOOKBACK = pd.Timedelta(days=45)
SYMBOL = "BTCUSDT"
INTERVAL = pipeline.Client.KLINE_INTERVAL_5MINUTE


def _download(key: str, local_name: str) -> pd.DataFrame:
    local_path = TMP / local_name
    s3.download_file(BUCKET, key, str(local_path))
    return pd.read_parquet(local_path)


def _upload(df: pd.DataFrame, key: str, local_name: str, index: bool = True):
    local_path = TMP / local_name
    df.to_parquet(local_path, index=index)
    s3.upload_file(str(local_path), BUCKET, key)


def lambda_handler(event, context):
    raw = _download(RAW_KEY, "raw.parquet")
    resampled_old = _download(RESAMPLED_KEY, "resampled.parquet")
    cleaned_old = _download(CLEANED_KEY, "cleaned.parquet")

    last_ts = raw["timestamp"].max()
    last_resampled_ts = resampled_old.index.max()
    last_cleaned_ts = cleaned_old.index.max()

    start_date = (last_ts + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    end_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S")

    new_data = pipeline.download_klines(SYMBOL, INTERVAL, start_date, end_date)

    if new_data.empty:
        logger.info("No new candles available — nothing to update.")
        return {"status": "no_update", "last_raw_ts": str(last_ts)}

    raw = (
        pd.concat([raw, new_data], ignore_index=True)
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    last_complete_bucket = raw["timestamp"].max().floor("4h")
    reprocess_from = min(last_cleaned_ts, last_resampled_ts) - LOOKBACK
    window_5min = raw[raw["timestamp"] >= reprocess_from].copy()

    data_4h_rv = pipeline.get_rv_4h_log(window_5min.set_index("timestamp"))

    data_4h = data_4h_rv.copy()
    data_4h["log_rv_plus_t"] = data_4h["log_RV"].shift(-1)
    data_4h = pipeline.add_har_components(data_4h)

    rq = pipeline.get_rq_4h_log(window_5min)
    data_4h = data_4h.join(rq, how="inner")

    data_4h = pipeline.add_jump_component(window_5min, data_4h, price_col="close")
    data_4h = pipeline.add_pump_risk(window_5min, data_4h)

    new_resampled = data_4h_rv.loc[
        (data_4h_rv.index > last_resampled_ts) & (data_4h_rv.index < last_complete_bucket)
    ]
    new_cleaned = data_4h.loc[
        (data_4h.index > last_cleaned_ts) & (data_4h.index < last_complete_bucket)
    ]

    resampled_new = pd.concat([resampled_old, new_resampled]).sort_index()
    cleaned_new = pd.concat([cleaned_old, new_cleaned]).sort_index()

    pipeline.validate_rv_data(resampled_new)

    _upload(raw, RAW_KEY, "raw.parquet", index=False)
    _upload(resampled_new, RESAMPLED_KEY, "resampled.parquet")
    _upload(cleaned_new, CLEANED_KEY, "cleaned.parquet")

    logger.info(f"Updated: +{len(new_resampled)} resampled rows, +{len(new_cleaned)} cleaned rows")

    return {
        "status": "updated",
        "new_resampled_rows": len(new_resampled),
        "new_cleaned_rows": len(new_cleaned),
        "last_cleaned_ts": str(cleaned_new.index.max()),
    }