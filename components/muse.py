import numpy as np
import pandas as pd
import datetime
from . import downsampling

# =======
# ENV VARIABLES
# =======

_REMAPPINGS =      { 'RAW_TP9':'TP9', 
                        'RAW_TP10':'TP10', 
                        'RAW_AF7':'AF7', 
                        'RAW_AF8':'AF8', 
                        'Accelerometer_X':'accel_x', 
                        'Accelerometer_Y':'accel_y', 
                        'Accelerometer_Z':'accel_z', 
                        'Gyro_X':'gyro_x', 
                        'Gyro_Y':'gyro_y',
                        'Gyro_Z':'gyro_z'   }
_RAW_COLNAMES =    [ 'TP9', 'TP10', 'AF7', 'AF8'  ]
_IMU_COLNAMES =    [ 'accel_x', 'accel_y', 'accel_z', 'gyro_x',  'gyro_y', 'gyro_z' ]
_PROCESSED_COLNAMES =   [ 'Delta_TP9','Delta_TP10','Delta_AF7','Delta_AF8',
                            'Theta_TP9', 'Theta_TP10', 'Theta_AF7', 'Theta_AF8', 
                            'Alpha_TP9', 'Alpha_TP10', 'Alpha_AF7', 'Alpha_AF8',
                            'Beta_TP9', 'Beta_TP10', 'Beta_AF7', 'Beta_AF8',
                            'Gamma_TP9', 'Gamma_TP10', 'Gamma_AF7', 'Gamma_AF8'     ]

# =======
# CLASSES
# =======

class Muse:
    def __init__(self, src:str, 
                        df:pd.DataFrame, 
                        signals:pd.DataFrame, 
                        blinks:pd.DataFrame, 
                        raw:pd.DataFrame, 
                        processed:pd.DataFrame, 
                        imu:pd.DataFrame    ):
        self.src = src
        self.df = df
        self.signals = signals
        self.blinks = blinks
        self.raw = raw
        self.processed = processed
        self.imu = imu

def timestamp_to_unix_milliseconds(x) -> int:      # Helper: converts timestamps to unix
    date_format = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    unix_seconds = datetime.datetime.timestamp(date_format)
    unix_milliseconds = int(unix_seconds * 1000)
    return unix_milliseconds

def sparkline(values):
    ticks = "▁▂▃▄▅▆▇█"
    values = np.asarray(values, dtype=float)
    if values.max() == 0:
        return ""
    scaled = (values - values.min()) / (values.max() - values.min())
    return "".join(ticks[int(v * (len(ticks) - 1))] for v in scaled)

def read_muse(src:str) -> Muse:
    # Read DF
    df = pd.read_csv(src, dtype={'Elements':str})
    df['unix_ms'] = df['TimeStamp'].apply(timestamp_to_unix_milliseconds)
    df = df.rename(columns=_REMAPPINGS)
    df = df.sort_values('unix_ms')
    # Separate
    signals = df[df['Elements'].isna()]
    blinks = df[df['Elements']=='/muse/elements/blink']
    blinks = blinks[['TimeStamp', 'unix_ms']]
    raw_eeg = signals[['TimeStamp', 'unix_ms', *_RAW_COLNAMES]]
    processed_eeg = signals[['TimeStamp', 'unix_ms', *_PROCESSED_COLNAMES]]
    imu = signals[['TimeStamp', 'unix_ms', *_IMU_COLNAMES]]
    # let's try to measure overlaps in `TImeStamp`
    counts = df.groupby('TimeStamp').size()
    dist = counts.value_counts().sort_index()        
    print("\tGroup by `TimeStamp` counts:", sparkline(dist.values))
    # Generate new Muse type and return it
    return Muse(src, df, signals, blinks, raw_eeg, processed_eeg, imu)

def downsample_muse_by_df(muse:Muse, how:str='last') -> Muse:
    df = downsampling.downsample(muse.df, 'unix_ms', how=how)
    signals = df[df['Elements'].isna()]
    blinks = df[df['Elements']=='/muse/elements/blink']
    blinks = blinks[['TimeStamp', 'unix_ms']]    
    raw = signals[['unix_ms', *_RAW_COLNAMES]]
    processed= signals[['unix_ms', *_PROCESSED_COLNAMES]]
    imu = signals[['unix_ms', *_IMU_COLNAMES]]
    # Generate new Muse type and return it
    return Muse(muse.src, df, signals, blinks, raw, processed, imu)

def downsample_muse_by_components(muse:Muse, how:str='last') -> Muse:
    df = downsampling.downsample(muse.df, 'unix_ms', how=how)
    signals = downsampling.downsample(muse.signals, 'unix_ms', how=how)
    blinks = downsampling.downsample(muse.blinks, 'unix_ms', how=how)
    raw = downsampling.downsample(muse.raw, 'unix_ms', how=how)
    processed = downsampling.downsample(muse.processed, 'unix_ms', how=how)
    imu = downsampling.downsample(muse.imu, 'unix_ms', how=how)
    return Muse(muse.src, df, signals, blinks, raw, processed, imu)

def get_subset(muse:Muse, timestamp_colname:str, start:int|float, end:int|float) -> Muse:
    df = muse.df[muse.df[timestamp_colname].between(start,end)]
    signals = muse.signals[muse.signals[timestamp_colname].between(start,end)]
    blinks = muse.blinks[muse.blinks[timestamp_colname].between(start,end)]
    raw = muse.raw[muse.raw[timestamp_colname].between(start,end)]
    processed = muse.processed[muse.processed[timestamp_colname].between(start,end)]
    imu = muse.imu[muse.imu[timestamp_colname].between(start,end)]
    return Muse(muse.src, df, signals, blinks, raw, processed, imu)