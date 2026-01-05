import numpy as np
import pandas as pd
import datetime

def timestamp_to_unix_milliseconds(x):      # Helper: converts timestamps to unix
    date_format = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    unix_seconds = datetime.datetime.timestamp(date_format)
    unix_milliseconds = int(unix_seconds * 1000)
    return unix_milliseconds

def estimate_sample_rate(df:pd.DataFrame, timestamp_colname:str, is_milli:bool=True):    # Helper: Estimates sample rate
    df = df.sort_values(timestamp_colname).reset_index(drop=True)
    t0 = df[timestamp_colname].iloc[0]
    t1 = df[timestamp_colname].iloc[-1]
    duration_s = (t1 - t0) / (1000 if is_milli else 1)
    fs = len(df) / duration_s
    return fs

def sparkline(values):
    ticks = "▁▂▃▄▅▆▇█"
    values = np.asarray(values, dtype=float)
    if values.max() == 0:
        return ""
    scaled = (values - values.min()) / (values.max() - values.min())
    return "".join(ticks[int(v * (len(ticks) - 1))] for v in scaled)