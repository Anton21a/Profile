"""
Data/update_pipeline.py

Incrementally updates data_binance.parquet, resampled_data_4h.parquet
and cleaned_data.parquet with newly available Binance candles, instead
of re-downloading/re-processing the full history from scratch.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from binance.client import Client

ROOT = Path(__file__).resolve().parent

RAW_PATH = ROOT / "raw_data" / "data_binance.parquet"
RESAMPLED_PATH = ROOT / "resampled_data" / "resampled_data_4h.parquet"
CLEANED_PATH = ROOT / "data_for_analysis" / "cleaned_data.parquet"

LOOKBACK = pd.Timedelta(days=45)

SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_5MINUTE


def _load_module(name: str, path: Path):
    # importing this way only runs top-level defs (main() is guarded
    # by `if __name__ == "__main__"` in all three scripts), so it's
    # safe: no accidental parquet writes as a side effect of import.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetching = _load_module("data_binance_fetching", ROOT / "raw_data" / "data_binance_fetching.py")
resampling = _load_module("resampling", ROOT / "resampled_data" / "resampling.py")
preprocessing = _load_module("preprocessing", ROOT / "data_for_analysis" / "preprocessing.py")


def update_raw_data() -> tuple[pd.DataFrame, bool]:
    raw = pd.read_parquet(RAW_PATH)
    last_ts = raw["timestamp"].max()

    start_date = (last_ts + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    end_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S")

    new_data = fetching.download_klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        start_date=start_date,
        end_date=end_date,
    )

    if new_data.empty:
        print("No new candles available.")
        return raw, False

    combined = (
        pd.concat([raw, new_data], ignore_index=True)
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined.to_parquet(RAW_PATH, index=False)
    print(f"✅ raw data updated: +{len(new_data)} candles, {len(combined)} total")
    return combined, True


def build_features(window_5min: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_indexed = window_5min.set_index("timestamp")

    # kept separately: add_pump_risk() drops its own "volume" column at
    # the end, so the RV-only frame must be captured before that runs.
    data_4h_rv = resampling.get_rv_4h_log(window_indexed)

    data_4h = data_4h_rv.copy()
    data_4h["log_rv_plus_t"] = data_4h["log_RV"].shift(-1)
    data_4h = preprocessing.add_har_components(data_4h)

    rq = preprocessing.get_rq_4h_log(window_5min)
    data_4h = data_4h.join(rq, how="inner")

    data_4h = preprocessing.add_jump_component(window_5min, data_4h, price_col="close")
    data_4h = preprocessing.add_pump_risk(window_5min, data_4h)

    return data_4h_rv, data_4h


def main():
    cleaned_old = pd.read_parquet(CLEANED_PATH)
    resampled_old = pd.read_parquet(RESAMPLED_PATH)

    last_cleaned_ts = cleaned_old.index.max()
    last_resampled_ts = resampled_old.index.max()
    print(f"Last cleaned timestamp:    {last_cleaned_ts}")
    print(f"Last resampled timestamp:  {last_resampled_ts}")

    raw, has_new = update_raw_data()
    if not has_new:
        print("Nothing to update — parquet files are already current.")
        return

    # 4h-бакет, в который попадает последняя скачанная 5-мин свеча — потенциально
    # ещё не закрыт (не все его 5-минутки уже наступили), не сохраняем его как финальный
    last_complete_bucket = raw["timestamp"].max().floor("4h")

    reprocess_from = min(last_cleaned_ts, last_resampled_ts) - LOOKBACK
    window_5min = raw[raw["timestamp"] >= reprocess_from].copy()

    data_4h_rv, data_4h_window = build_features(window_5min)

    # resampled_data_4h.parquet только RV-only колонки
    new_resampled = data_4h_rv.loc[
        (data_4h_rv.index > last_resampled_ts) & (data_4h_rv.index < last_complete_bucket)
    ]
    resampled_new = pd.concat([resampled_old, new_resampled]).sort_index()
    resampling.validate_rv_data(resampled_new)
    resampled_new.to_parquet(RESAMPLED_PATH)
    print(f"✅ resampled_data_4h.parquet updated: +{len(new_resampled)} rows")

    # cleaned_data.parquet получает полный набор фич
    new_cleaned = data_4h_window.loc[
        (data_4h_window.index > last_cleaned_ts) & (data_4h_window.index < last_complete_bucket)
    ]
    cleaned_new = pd.concat([cleaned_old, new_cleaned]).sort_index()
    cleaned_new.to_parquet(CLEANED_PATH)
    print(f"✅ cleaned_data.parquet updated: +{len(new_cleaned)} rows")


if __name__ == "__main__":
    main()