import numpy as np
import pandas as pd

# =======
# ENV VARIABLES
# =======

_HOW = ['last', 'first']

# =======
# FUNCTIONS
# =======

def sparkline(values) -> str:
    ticks = "▁▂▃▄▅▆▇█"
    values = np.asarray(values, dtype=float)
    if values.max() == 0:
        return ""
    scaled = (values - values.min()) / (values.max() - values.min())
    return "".join(ticks[int(v * (len(ticks) - 1))] for v in scaled)

def downsample(df:pd.DataFrame, timestamp_colname:str='unix_ms', how:str="last") -> pd.DataFrame:
    how = how.lower()
    assert how in _HOW, f"Param 'how' can only accept the following: {_HOW}"
    assert timestamp_colname in df.columns.tolist(), f"Param 'timestamp_colname' is not a column in the provided DataFrame"
    if how == 'last':   
        return df.groupby(timestamp_colname, as_index=False).last()
    return df.groupby(timestamp_colname, as_index=False).first()

def estimate_sample_freq(df:pd.DataFrame, timestamp_colname:str='unix_ms', is_milli:bool=True) -> float:
    df = df.sort_values(timestamp_colname).reset_index(drop=True)
    t0 = df[timestamp_colname].iloc[0]
    t1 = df[timestamp_colname].iloc[-1]
    duration_s = (t1 - t0) / (1000 if is_milli else 1)
    return len(df) / duration_s

def compare_sample_freqs(df1:pd.DataFrame, df2:pd.DataFrame, timestamp_colname:str='unix_ms', is_milli:bool=True) -> (float,float):
    # Analyze first df
    hz1 = estimate_sample_freq(df1, timestamp_colname, is_milli)
    c1 = df1.groupby(timestamp_colname).size()
    d1 = c1.value_counts().sort_index()
    n1 = len(df1.index)
    # Analyze the second df
    hz2 = estimate_sample_freq(df2, timestamp_colname, is_milli)
    c2 = df2.groupby(timestamp_colname).size()
    d2 = c2.value_counts().sort_index()
    n2 = len(df2.index)
    # Print analysis
    print(sparkline(d1.values), '-=>', sparkline(d2.values), " | ", n1, "-=>", n2)
    return (hz1, hz2)