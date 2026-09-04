import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def add_har_components(df, target_col='log_RV'):
    d = df.copy()
    d = d.sort_index()

    d["rv_daily"] = d["log_RV"].rolling(6).mean()
    d["rv_weekly"] = d["log_RV"].rolling(42).mean()
    d["rv_monthly"] = d["log_RV"].rolling(180).mean()

    return d


def get_rq_4h_log(df):
    d = df.copy()
    d = d.set_index("timestamp")

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

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    return data


def add_jump_component(df_5min, df_4hour, price_col='close'):
    d_5m = df_5min.copy()
    d_4h = df_4hour.copy()

    d_5m = d_5m.set_index('timestamp')

    d_5m['return'] = np.log(d_5m[price_col] / d_5m[price_col].shift(1))

    d_5m['abs_ret'] = d_5m['return'].abs()
    d_5m['abs_ret_lag'] = d_5m['abs_ret'].shift(1)

    d_5m['bv_component'] = d_5m['abs_ret'] * d_5m['abs_ret_lag']

    pi_factor = np.pi / 2
    bv_4h = d_5m['bv_component'].resample('4h').sum() * pi_factor

    d_4h['BV'] = bv_4h

    d_4h['Jump'] = np.maximum(d_4h['RV'] - d_4h['BV'], 0)

    d_4h['log_Jump'] = np.log(d_4h['Jump'] + 1)

    d_4h = d_4h.drop(columns=['BV'])

    return d_4h


def add_pump_risk(df_5min, df_4hour):
    d_5m = df_5min.copy()
    d_4h = df_4hour.copy()

    d_5m = d_5m.set_index('timestamp')

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

    d_4h = d_4h.drop(columns=["volume", "max_ret", "ret_z_30d"])

    return d_4h


def main():
    data_5min = pd.read_parquet("../raw_data/data_binance.parquet")
    data = pd.read_parquet("../resampled_data/resampled_data_4h.parquet")

    data["log_rv_plus_t"] = data["log_RV"].shift(-1)

    data = add_har_components(data)

    rq = get_rq_4h_log(data_5min)
    data = data.join(rq, how="inner")

    data = add_jump_component(data_5min, data, price_col='close')

    data = add_pump_risk(data_5min, data)

    data.to_parquet("cleaned_data.parquet")

    return data


if __name__ == "__main__":
    main()