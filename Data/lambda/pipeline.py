import numpy as np
import pandas as pd
from binance.client import Client


def download_klines(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    client = Client()
    klines = client.get_historical_klines(
        symbol=symbol, interval=interval, start_str=start_date, end_str=end_date
    )

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'trades',
               'taker_buy_base', 'taker_buy_quote', 'ignore']
    df = pd.DataFrame(klines, columns=columns)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").astype("datetime64[ns]")
    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume", "trades",
                     "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df[["timestamp"] + numeric_cols].copy()


def get_rv_4h_log(df):
    d = df.copy()
    d["log_ret"] = np.log(d["close"]).diff()
    d = d.dropna(subset=["log_ret"])

    data = np.sqrt(d["log_ret"].pow(2).resample("4h").sum()).to_frame(name="RV")
    data["log_RV"] = np.log(data["RV"])
    data["volume"] = d["volume"].resample("4h").sum()
    data["trades"] = d["trades"].resample("4h").sum()
    return data


def validate_rv_data(df):
    assert not df.empty, "Validation Error: The dataframe is empty!"
    assert df.isna().sum().sum() == 0, "Validation Error: NaN values found"
    assert not np.isinf(df).values.sum(), "Validation Error: infinite values found"
    assert df.index.is_monotonic_increasing, "Validation Error: index not sorted"
    assert df.index.is_unique, "Validation Error: duplicate timestamps"
    if len(df) > 1:
        deltas = pd.Series(df.index).diff().dropna()
        assert (deltas == pd.Timedelta("4h")).all(), "Validation Error: gap in 4h frequency"


def add_har_components(df):
    d = df.copy().sort_index()
    d["rv_daily"] = d["log_RV"].rolling(6).mean()
    d["rv_weekly"] = d["log_RV"].rolling(42).mean()
    d["rv_monthly"] = d["log_RV"].rolling(180).mean()
    return d


def get_rq_4h_log(df):
    d = df.copy().set_index("timestamp")
    d["log_ret"] = np.log(d["close"]).diff()
    d = d.dropna(subset=["log_ret"])

    resampled = d["log_ret"].resample("4h")
    n_obs = resampled.count()
    sum_r4 = resampled.apply(lambda x: (x**4).sum())
    rq_values = (n_obs / 3) * sum_r4

    data = rq_values.to_frame(name="RQ")
    data["log_RQ"] = np.log(data["RQ"])
    data["sqrt_RQ"] = np.sqrt(data["RQ"])
    data["log_sqrt_RQ"] = np.log(data["sqrt_RQ"])
    return data.replace([np.inf, -np.inf], np.nan).dropna()


def add_jump_component(df_5min, df_4hour, price_col='close'):
    d_5m = df_5min.copy().set_index('timestamp')
    d_4h = df_4hour.copy()

    d_5m['return'] = np.log(d_5m[price_col] / d_5m[price_col].shift(1))
    d_5m['abs_ret'] = d_5m['return'].abs()
    d_5m['abs_ret_lag'] = d_5m['abs_ret'].shift(1)
    d_5m['bv_component'] = d_5m['abs_ret'] * d_5m['abs_ret_lag']

    bv_4h = d_5m['bv_component'].resample('4h').sum() * (np.pi / 2)
    d_4h['BV'] = bv_4h
    d_4h['Jump'] = np.maximum(d_4h['RV'] - d_4h['BV'], 0)
    d_4h['log_Jump'] = np.log(d_4h['Jump'] + 1)
    return d_4h.drop(columns=['BV'])


def add_pump_risk(df_5min, df_4hour):
    d_5m = df_5min.copy().set_index('timestamp')
    d_4h = df_4hour.copy()

    if "log_ret" not in d_5m.columns:
        d_5m["log_ret"] = np.log(d_5m["close"] / d_5m["close"].shift(1))

    vol_4h = d_5m["volume"].resample("4h").sum()
    max_ret_4h = d_5m["log_ret"].resample("4h").max()

    d_4h["volume"] = vol_4h.reindex(d_4h.index)
    d_4h["max_ret"] = max_ret_4h.reindex(d_4h.index)

    vol_mean_180 = d_4h["volume"].shift(1).rolling(180, min_periods=180).mean()
    vol_std_180 = d_4h["volume"].shift(1).rolling(180, min_periods=180).std()
    d_4h["vol_z_30d"] = (d_4h["volume"] - vol_mean_180) / (vol_std_180 + 1e-8)

    ret_mean_180 = d_4h["max_ret"].shift(1).rolling(180, min_periods=180).mean()
    ret_std_180 = d_4h["max_ret"].shift(1).rolling(180, min_periods=180).std()
    d_4h["ret_z_30d"] = (d_4h["max_ret"] - ret_mean_180) / (ret_std_180 + 1e-8)

    d_4h["pump_risk"] = d_4h["vol_z_30d"] + d_4h["ret_z_30d"]
    return d_4h.drop(columns=["volume", "max_ret", "ret_z_30d"])