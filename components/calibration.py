import pandas as pd
from . import config as CF

class Calibration:
    def __init__(self, df:pd.DataFrame, start_row:pd.Series, end_row:pd.Series, overlaps:pd.DataFrame):
        self.df = df
        self.start_row = start_row
        self.end_row = end_row
        self.overlaps = overlaps

def read_calibration(src:str, config:CF.Config):
    df = pd.read_csv(src)
    start_row = df.loc[df["event"] == "Start"].iloc[0]
    end_row = df.loc[df["event"] == "End"].iloc[0]
    overlaps = df[df['event'] == 'Overlap']
    if config.start_buffer is not None:    start_row['unix_ms'] += config.start_buffer
    if config.end_buffer is not None:      end_row['unix_ms'] -= config.end_buffer
    return Calibration(df, start_row, end_row, overlaps)