import pandas as pd
import datetime

# Convert a timestamp to unix seconds
def timestamp_to_unix_seconds(x):
    date_format = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    unix_seconds = datetime.datetime.timestamp(date_format)
    return unix_seconds

# Convert a timestamp to unix milliseconds
def timestamp_to_unix_milliseconds(x):
    unix_seconds = timestamp_to_unix_seconds(x)
    unix_milliseconds = int(unix_seconds * 1000)
    return unix_milliseconds

# A reader to help with handling raw EEG files from the Mind Monitor application
def read(f:str):
    # Read the `eeg_rest.csv` as a DataFrame
    df = pd.read_csv(f)

    # Generate timestamp features
    df['unix_ms'] = df['TimeStamp'].apply(timestamp_to_unix_milliseconds)
    df['unix_sec'] = df['TimeStamp'].apply(timestamp_to_unix_seconds)    
    df['rel_sec'] = df['unix_sec'] - df['unix_sec'].iloc[0]
    df['rel_ms'] = df['unix_ms'] - df['unix_ms'].iloc[0]
    
    # separate blinks from raw data
    signals = df[df['Elements'].isna()]
    blinks = df[df['Elements']=='/muse/elements/blink']
    blinks = blinks[['TimeStamp', 'unix_sec', 'unix_ms', 'rel_sec', 'rel_ms']]    
    
    # return 
    return signals, blinks

# Separate columns, between raw and processed
def separate( df:pd.DataFrame, time_colnames=['unix_sec','unix_ms', 'rel_sec', 'rel_ms'] ):
    # Rename scheme for raw features
    raw_colnames = {
        'RAW_TP9':'TP9', 
        'RAW_TP10':'TP10', 
        'RAW_AF7':'AF7', 
        'RAW_AF8':'AF8'
    }

    # No need to rename processed features, but we need a list of them to extract them
    processed_colnames = [
        'Delta_TP9','Delta_TP10','Delta_AF7','Delta_AF8',
        'Theta_TP9', 'Theta_TP10', 'Theta_AF7', 'Theta_AF8',
        'Alpha_TP9', 'Alpha_TP10', 'Alpha_AF7', 'Alpha_AF8',
        'Beta_TP9', 'Beta_TP10', 'Beta_AF7', 'Beta_AF8',
        'Gamma_TP9', 'Gamma_TP10', 'Gamma_AF7', 'Gamma_AF8'
    ]
    
    # Get raw data, then rename the features
    raw_df = df[[*time_colnames, *list(raw_colnames.keys())]]
    raw_df.rename(columns=raw_colnames, inplace=True)
    
    # Get processed data
    processed_df = df[[*time_colnames, *processed_colnames]]
    
    # Return
    return raw_df, processed_df

def melt_raw_eeg( df:pd.DataFrame, timestamp_cols=['unix_ms'] ):

    processed_colnames = ['AF7', 'AF8', 'TP9', 'TP10']

    colnames = timestamp_cols + processed_colnames
    #processed_colnames.insert(0, timestamp_col)

    # Restrict the columns
    #restricted_df = df[processed_colnames]
    restricted_df = df[colnames]

    # Melt the dataframe
    long_df = restricted_df.melt(id_vars=timestamp_cols,
                      var_name='channel',
                      value_name='voltage')

    # return 
    return long_df

def melt_processed_eeg( df:pd.DataFrame, timestamp_cols=['unix_ms'] ):

    processed_colnames = [
        'Delta_TP9','Delta_TP10','Delta_AF7','Delta_AF8',
        'Theta_TP9', 'Theta_TP10', 'Theta_AF7', 'Theta_AF8',
        'Alpha_TP9', 'Alpha_TP10', 'Alpha_AF7', 'Alpha_AF8',
        'Beta_TP9', 'Beta_TP10', 'Beta_AF7', 'Beta_AF8',
        'Gamma_TP9', 'Gamma_TP10', 'Gamma_AF7', 'Gamma_AF8'
    ]

    colnames = timestamp_cols + processed_colnames
    #processed_colnames.insert(0, timestamp_col)

    # Restrict the columns
    #restricted_df = df[processed_colnames]
    restricted_df = df[colnames]

    # Melt the dataframe
    long_df = restricted_df.melt(id_vars=timestamp_cols,
                      var_name='band_channel',
                      value_name='power')
    
    # Split the combined "band_channel" into two separate columns
    long_df[['band', 'channel']] = long_df['band_channel'].str.split('_', expand=True)

    # Reorder the columns
    final_colnames = timestamp_cols + ['channel', 'band', 'power']
    long_df = long_df[final_colnames]

    # return 
    return long_df
