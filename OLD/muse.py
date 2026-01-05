import pandas as pd
import datetime
import numpy as np

def sparkline(values):
    ticks = "▁▂▃▄▅▆▇█"
    values = np.asarray(values, dtype=float)
    if values.max() == 0:
        return ""
    scaled = (values - values.min()) / (values.max() - values.min())
    return "".join(ticks[int(v * (len(ticks) - 1))] for v in scaled)

_MUSE_REMAPPINGS =      { 'RAW_TP9':'TP9', 
                          'RAW_TP10':'TP10', 
                          'RAW_AF7':'AF7', 
                          'RAW_AF8':'AF8', 
                          'Accelerometer_X':'accel_x', 
                          'Accelerometer_Y':'accel_y', 
                          'Accelerometer_Z':'accel_z', 
                          'Gyro_X':'gyro_x', 
                          'Gyro_Y':'gyro_y',
                          'Gyro_Z':'gyro_z'     }
_MUSE_RAW_COLNAMES =    [ 'TP9', 'TP10', 
                          'AF7', 'AF8'  ]
_MUSE_IMU_COLNAMES =    [ 'accel_x', 'accel_y', 'accel_z', 
                          'gyro_x',  'gyro_y', 'gyro_z'     ]
_MUSE_PROCESSED_COLNAMES =   [ 'Delta_TP9','Delta_TP10','Delta_AF7','Delta_AF8',
                      'Theta_TP9', 'Theta_TP10', 'Theta_AF7', 'Theta_AF8', 
                      'Alpha_TP9', 'Alpha_TP10', 'Alpha_AF7', 'Alpha_AF8',
                      'Beta_TP9', 'Beta_TP10', 'Beta_AF7', 'Beta_AF8',
                      'Gamma_TP9', 'Gamma_TP10', 'Gamma_AF7', 'Gamma_AF8'     ]
class Muse:
    def __init__(self, parent, src:str):
        self.parent = parent
        self.src = src
        # These need to be filled out manually
        self.df = None
        self.blinks = None
        self.raw_eeg = None
        self.processed_eeg = None
        self.imu = None
        # Seven primary things we've effectively created and are caching:
        # - `self.parent`: Who owns this Muse data?
        # - `self.src`: The filename source
        # - `self.df`: Raw Pandas read from `self.src`
        # - `self.blinks`: Muse's detected blinks
        # - `self.raw_eeg`: Raw EEG (TP9, TP10, AF7, AF8) channels
        # - `self.processed_eeg`: The estimated PSD data calculated by the Muse system.
        # - `self.imu`: The raw IMU data.
    
    def initialize_from_src(self):
        df = pd.read_csv(self.src)
        df['unix_ms'] = df['TimeStamp'].apply(self.timestamp_to_unix_milliseconds)
        df['unix_sec'] = df['TimeStamp'].apply(self.timestamp_to_unix_seconds)    
        df['rel_sec'] = df['unix_sec'] - df['unix_sec'].iloc[0]
        df['rel_ms'] = df['unix_ms'] - df['unix_ms'].iloc[0] 
        self.df = df.rename(columns=_MUSE_REMAPPINGS)
        self.df = self.df.sort_values('unix_ms')

        # Downsampling & Sample Rate Estimation
        n = len(self.df.index)
        counts = df.groupby('unix_ms').size()
        dist = counts.value_counts().sort_index()        
        hz = self.estimate_sample_rate(self.df, 'unix_ms', is_milli=True)
        self.df = self.df.groupby('unix_ms', as_index=False).last()
        n2 = len(self.df.index)
    
        # Separation
        signals = self.df[self.df['Elements'].isna()]
        blinks = self.df[self.df['Elements']=='/muse/elements/blink']
        self.blinks = blinks[['TimeStamp', 'unix_sec', 'unix_ms', 'rel_sec', 'rel_ms']]    
        self.raw_eeg = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_MUSE_RAW_COLNAMES]]
        self.processed_eeg = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_MUSE_PROCESSED_COLNAMES]]
        self.imu = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_MUSE_IMU_COLNAMES]]

        # Printing Results
        print("\tGroup by `unix_ms` counts:", sparkline(dist.values))
        print(f"\tEstimated Sample Rate (Pre-Downsampling):", hz)
        print(f"\tDownsampling: {n} -> {n2} ({n2/n}% retention rate)")

    def get_subset(self, timestamp_colname:str, start, end, parent=None):
        # Make sure we've initialized
        assert self.df is not None, "This Muse has not been initialized yet"
        # Define the subset as the same type as the current muse, also defining a parent if necessary
        if parent is None: parent = self.parent
        subset = type(self)(parent, self.src)
        # Define the new values
        subset.df = self.df[self.df[timestamp_colname].between(start, end)]
        subset.blinks = self.blinks[self.blinks[timestamp_colname].between(start, end)]
        subset.raw_eeg = self.raw_eeg[self.raw_eeg[timestamp_colname].between(start, end)]
        subset.processed_eeg = self.processed_eeg[self.processed_eeg[timestamp_colname].between(start, end)]
        subset.imu = self.imu[self.imu[timestamp_colname].between(start, end)]
        # Return the subset muse
        return subset

    @staticmethod
    def timestamp_to_unix_seconds(x):
        date_format = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
        unix_seconds = datetime.datetime.timestamp(date_format)
        return unix_seconds

    @staticmethod
    def timestamp_to_unix_milliseconds(x):
        date_format = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
        unix_seconds = datetime.datetime.timestamp(date_format)
        unix_milliseconds = int(unix_seconds * 1000)
        return unix_milliseconds
    
    @staticmethod
    def remove_duplicates_from_subset(df, subset):
        mask = (df[subset] != df[subset].shift()).any(axis=1)
        df_no_consecutive_dupes = df[mask]
        return df_no_consecutive_dupes

    @staticmethod
    def estimate_sample_rate(df, timestamp_colname:str, is_milli:bool=True):
        df = df.sort_values(timestamp_colname).reset_index(drop=True)
        # Measure number of samples
        t0 = df[timestamp_colname].iloc[0]
        t1 = df[timestamp_colname].iloc[-1]
        duration_s = (t1 - t0) / (1000 if is_milli else 1)
        fs = len(df) / duration_s
        return fs
