import os
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
import datetime
from scipy.stats import zscore
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Iterable

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.set_option('mode.chained_assignment', None)

_TRIAL_NAMES = [
    "Activation",
    "Trial-ApproachAudio Start", 
    "Trial-BehindAudio Start", 
    "Trial-Behind Start", 
    "Trial-AlleyRunnerAudio Start", 
    "Trial-AlleyRunner Start", 
    "Trial-Approach Start"
]
_CALIBRATION_COLUMNS = [
    'unix_ms', 
    'frame', 
    'rel_timestamp',
    'event', 
    'overlap_counter'
]
_EEG_RAW_COLNAMES = {
    'RAW_TP9':'TP9', 
    'RAW_TP10':'TP10', 
    'RAW_AF7':'AF7', 
    'RAW_AF8':'AF8'
}
_EEG_MUSE_COLNAMES = [
    'Delta_TP9','Delta_TP10','Delta_AF7','Delta_AF8',
    'Theta_TP9', 'Theta_TP10', 'Theta_AF7', 'Theta_AF8',
    'Alpha_TP9', 'Alpha_TP10', 'Alpha_AF7', 'Alpha_AF8',
    'Beta_TP9', 'Beta_TP10', 'Beta_AF7', 'Beta_AF8',
    'Gamma_TP9', 'Gamma_TP10', 'Gamma_AF7', 'Gamma_AF8'
]
_PREPEND_BUFFER = 100
_APPEND_BUFFER = 100


class Experiment:
    def __init__(self, root_dir:str, pdirs:Iterable[str], verbose:bool=False):
        self.root_dir = root_dir

        pbar = tqdm(pdirs)
        self.participants = []
        for pdir in pbar:
            pbar.set_description(os.path.basename(pdir))
            self.participants.append(Participant(self, pdir, verbose))


class Participant:
    def __init__(self, parent:Experiment, root_dir:str, verbose:bool=False):
        self.parent = parent
        self.root_dir = root_dir
        self.pid = os.path.basename(root_dir)
        # Load globals
        self.eye = pd.read_csv(os.path.join(self.root_dir, 'eye.csv'))
        self.eeg_rest_raw, self.eeg_rest_muse, self.eeg_rest_blinks = self.parse_eeg(os.path.join(self.root_dir, 'eeg_rest.csv'))
        self.eeg_vr_raw, self.eeg_vr_muse, self.eeg_vr_blinks = self.parse_eeg(os.path.join(self.root_dir, 'eeg_vr.csv'))
        # Initialize trials
        self.trial_order = self.identify_trials(verbose=verbose)    # Trial Definitions & Order
        self.trials = self.initialize_trials(verbose)       # Initialization of their trials
        self.offset_medians = self.get_offset_medians(verbose)
    def identify_trials(self, include_first_calibration:bool=False, verbose:bool=False):
        # Just in case
        if self.eye is None: self.eye = pd.read_csv(os.path.join(self.root_dir, 'eye.csv'))

        # Calibrations
        cdf = self.eye[self.eye['event'] == 'Calibration']
        cdf['tid'] = list(range(1,len(cdf.index)+1))
        cdf = cdf[['unix_ms','frame','tid']]
        cdf = cdf.rename(columns={'unix_ms':'trial_start_ms', 'frame':'trial_start_frame'})
        cdf['cal_start_ms'] = cdf['trial_start_ms']
        cdf['cal_start_frame'] = cdf['trial_start_frame']

        # Trials
        tdf = self.eye[self.eye["event"].isin(_TRIAL_NAMES)]
        tdf['tid'] = list(range(0,len(tdf.index)))
        tdf = tdf[['unix_ms','frame','tid','event']]
        tdf = tdf.rename(columns={'unix_ms':'sim_start_ms', 'frame':'sim_start_frame', 'event':'name'})
        tdf["name"] = tdf["name"].str.replace(r"^Trial-| Start$", "", regex=True)

        # Inner Join
        df = pd.merge(left=cdf, right=tdf, how='left', on='tid')
        df["trial_end_frame"] = df["trial_start_frame"].shift(-1)
        df["trial_end_ms"] = df["trial_start_ms"].shift(-1)
        
        # Read first calibration file, if it exists
        if include_first_calibration and os.path.exists(os.path.join(self.root_dir, 'calibration_0.csv')):
            fcal_df = pd.read_csv(os.path.join(self.root_dir, 'calibration_0.csv'))
            fcal_df = fcal_df.iloc[[0, -1]]
            fcal_df['tid'] = [0, 0]
            fcal_df = fcal_df.rename(columns={'unix_ms':'cal_start_ms', 'frame':'cal_start_frame', 'event':'name'})
            fcal_df['cal_end_ms'] = fcal_df['cal_start_ms'].shift(-1)
            fcal_df['cal_end_frame'] = fcal_df['cal_start_frame'].shift(-1)
            fdf = fcal_df.iloc[[0]]
            fdf['name'] = ['Calibration']
            fdf['sim_start_ms'] = fdf['cal_start_ms'] + 13000
            fdf['sim_start_frame'] = fdf['cal_start_frame']
            fdf['trial_start_ms'] = fdf['cal_start_ms']
            fdf['trial_start_frame'] = fdf['cal_start_frame']
            fdf['trial_end_ms'] = fdf['sim_start_ms']
            fdf['trial_end_frame'] = fdf['sim_start_frame']
            fdf = fdf[['tid','name','trial_start_ms','trial_start_frame','trial_end_ms','trial_end_frame','cal_start_ms', 'cal_start_frame', 'sim_start_ms', 'sim_start_frame']]
            df = pd.concat([fdf, df], ignore_index=True)

        # Correct any sim_start_ms NaNs by supplanting with 13 sec after the start of the calibration    
        df.loc[df['sim_start_ms'].isna(), 'sim_start_ms'] = (
            df.loc[df['sim_start_ms'].isna(), 'cal_start_ms'] + 13000   # Calibration is approx. 13 seconds
        )

        # Cleanup, typecasting
        df = df[['tid','name','trial_start_ms','trial_start_frame','trial_end_ms','trial_end_frame','cal_start_ms','cal_start_frame','sim_start_ms','sim_start_frame']]
        df = df[df['name'].notna()]
        df['tid'] = df['tid'].astype(int)
        df['trial_start_frame'] = df['trial_start_frame'].astype(int)
        df['trial_end_frame'] = df['trial_end_frame'].astype(int)
        df['cal_start_frame'] = df['cal_start_frame'].astype(int)
        df['sim_start_frame'] = df['sim_start_frame'].astype(int)

        # Cache the identified trials and their ordering
        if verbose: display(df)
        return df
    def initialize_trials(self, verbose:bool=False):
        assert self.trial_order is not None, f"Participant {self.pid} cannot generate trial dirs"
        trials = [Trial(self, row, verbose) for index, row in self.trial_order.iterrows()]
        return trials
    def get_offset_medians(self, verbose:bool=False):
        offsets = pd.concat([t.offsets for t in self.trials if t.offsets is not None], ignore_index=True)
        offset_medians = offsets.groupby(['channel'], as_index=False)['offset'].median()
        offset_medians.rename(columns={'offset':'offset_median'}, inplace=True)
        if verbose: display(offset_medians)
        return offset_medians
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
    @classmethod
    def parse_eeg(cls, filepath:str):
        # Assertion to check file existence
        assert os.path.exists(filepath), f'EEG file {filepath} does not exist...'
        # Initial read
        df = pd.read_csv(filepath)
        # Generate timestamp features
        df['unix_ms'] = df['TimeStamp'].apply(cls.timestamp_to_unix_milliseconds)
        df['unix_sec'] = df['TimeStamp'].apply(cls.timestamp_to_unix_seconds)    
        df['rel_sec'] = df['unix_sec'] - df['unix_sec'].iloc[0]
        df['rel_ms'] = df['unix_ms'] - df['unix_ms'].iloc[0] 
        # separate blinks from raw data
        signals = df[df['Elements'].isna()]
        blinks = df[df['Elements']=='/muse/elements/blink']
        blinks = blinks[['TimeStamp', 'unix_sec', 'unix_ms', 'rel_sec', 'rel_ms']]    
        # Get raw data, then rename the features
        raw_df = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *list(_EEG_RAW_COLNAMES.keys())]]
        raw_df.rename(columns=_EEG_RAW_COLNAMES, inplace=True)
        # Get processed data
        processed_df = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_EEG_MUSE_COLNAMES]]
        # return 
        return raw_df, processed_df, blinks


class Trial:
    def __init__(self, parent:Participant, details:pd.Series, verbose:bool=False):
        self.parent = parent
        self.details = details
        self.offsets = self.calculate_offsets(verbose)     # Calibrate based on eye and eeg differences
    def calculate_offsets(self, verbose:bool=False):
        # Get start and end timestamps. Account for the extra 5 seconds that occur at the beginning
        start_time = self.details['cal_start_ms'] + 5000
        end_time = self.details['sim_start_ms']

        # Load raw gaze csv, correct columns
        cals = pd.read_csv(os.path.join(self.parent.root_dir, f"calibration_{self.details['tid']}.csv"))
        cals = cals.iloc[:, :len(_CALIBRATION_COLUMNS)]
        cals.columns = _CALIBRATION_COLUMNS
        cals = cals[cals['event'] == 'Overlap']

        # Load EEG data, extract based on trial and blinks
        eeg_raw = self.parent.eeg_vr_raw[self.parent.eeg_vr_raw['unix_ms'].between(start_time, end_time)]
        eeg_blinks = self.parent.eeg_vr_blinks[self.parent.eeg_vr_blinks['unix_ms'].between(start_time, end_time)]
        muse_blinks = eeg_blinks.rename(columns={'unix_ms':'blink_unix_ms'})
        ranges = pd.merge_asof(
            cals, muse_blinks,
            left_on='unix_ms',
            right_on='blink_unix_ms',
            direction='forward'
        )
        ranges = ranges.drop(columns=['frame', 'rel_timestamp', 'event', 'TimeStamp', 'unix_sec', 'rel_sec', 'rel_ms'])
        ranges = ranges[['overlap_counter', 'unix_ms', 'blink_unix_ms']]
        ranges = ranges.rename(columns={'unix_ms':'overlap_unix_ms'})
        # Filter NaN rows or rows where the next overlap timestamp comes before the current blink timestamp
        filtered = ranges.dropna(subset=['blink_unix_ms'])
        next_overlap = filtered['overlap_unix_ms'].shift(-1)
        mask = (next_overlap >= filtered['blink_unix_ms']) | (next_overlap.isna())
        blink_ranges = filtered[mask]
        # We modify the unix milliseconds to account for a 100ms window before the overlap starts and after the blink is the detected
        blink_ranges['start_unix_ms'] = blink_ranges['overlap_unix_ms'] - _PREPEND_BUFFER
        blink_ranges['end_unix_ms'] = blink_ranges['blink_unix_ms'] + _APPEND_BUFFER
        blink_ranges['duration'] = blink_ranges['end_unix_ms'] - blink_ranges['start_unix_ms']

        # Extract peaks/valleys from eye and eeg data
        eye_df = self.parent.eye[self.parent.eye['unix_ms'].between(start_time, end_time)]
        vr_blinks = self.detect_vr_blinks(eye_df, blink_ranges)
        tp9_blinks, tp10_blinks = self.detect_eeg_blinks(eeg_raw, blink_ranges)

        # Exit early if tp9 and tp10 don't contain blinks
        if tp9_blinks is None and tp10_blinks is None:
            print(f"Warning: Could not extract TP9 and TP10 blinks from Participant {self.parent.pid,}, Trial {self.details['tid']}")
            return None

        # Merge the two three blink data
        vr_tp9 = self.merge_datasets(vr_blinks, tp9_blinks)
        vr_tp10 = self.merge_datasets(vr_blinks, tp9_blinks)
        vr_tp9['channel'] = 'tp9'
        vr_tp10['channel'] = 'tp10'
        combined = pd.concat([vr_tp9, vr_tp10], ignore_index=True)
        combined['channel'] = 'combined'
        offsets = pd.concat([vr_tp9, vr_tp10, combined]).sort_values(by=['overlap_counter'], ascending=True)
        offsets['tid'] = self.details['tid']
        offsets['pid'] = self.parent.pid
        offsets = offsets[['pid','tid','overlap_counter','channel','vr_x','eeg_x','offset']]
        return offsets
    @staticmethod
    def find_ascent_start(signal, peak_idx, rel_thresh=0.1, min_drop=1e-6):
        # Get the y-value of this signal
        peak_val = signal[peak_idx]
        # baseline threshold: stop once value <= threshold
        threshold = peak_val * (1 - rel_thresh)
        i = peak_idx
        # walk left while signal is >= threshold (and within bounds)
        while i > 0 and signal[i-1] >= threshold - min_drop:
            i -= 1
        return i
    @classmethod
    def peak_starts_by_scan(cls, signal, peaks, rel_thresh=0.1):
        starts = [cls.find_ascent_start(signal, p, rel_thresh=rel_thresh) for p in peaks]
        return np.array(starts)
    @classmethod
    def calculate_peaks(cls, _X, _Y, 
                    peak_height=0.5, 
                    peak_prominence=1, 
                    peak_width=0, 
                    valley_height=0.5, 
                    valley_prominence=1, 
                    valley_width=0):
        # Step 1: Using `scipy.stats.zscore()`, calculate the z-scores of this data. Get its inversion too.
        z = zscore(_Y)
        inv_z = [v*-1 for v in z]
        # Step 2: initialize empty arrays for these
        peaks, valleys, results = None, None, None
        # Step 3: Find peaks via `scipy.signal.find_peaks()`, then by `peak_starts_by_scan()`, then sort them in order, then aggregate
        peak_raws, _ = find_peaks(z, height=peak_height, width=peak_width, prominence=peak_prominence, plateau_size=True)
        # CHECK: do we even have detected peaks?
        if len(peak_raws) > 0:
            peak_indices = cls.peak_starts_by_scan(z, peak_raws)
            peak_indices.sort()
            peaks = [{'type':'peak', 'x':_X[i], 'y':z[i]} for i in peak_indices]
        # Step 3: Find valleys via `scipy.signal.find_peaks()` and inverted z-scores, then by `peak_starts_by_scan()`, then sort them in order, then aggregate
        valley_raws, _ = find_peaks(inv_z, height=valley_height, width=valley_width, prominence=valley_prominence, plateau_size=True)
        # CHECK: do we even have detected valleys
        if len(valley_raws) > 0:
            valley_indices = cls.peak_starts_by_scan(inv_z, valley_raws)
            valley_indices.sort()
            valleys = [{'type':'valley', 'x':_X[i], 'y':z[i]} for i in valley_indices]
        # Step 4: Combine them into a singular list
        combined = []
        if peaks is not None: combined.extend(peaks)
        if valleys is not None: combined.extend(valleys)
        # Step 5: Sort `combined` such that the peaks and valleys are ordered respective to both peaks and valleys
        # CHECK: do we even have any? Only proceed if we do have something
        if len(combined) > 0: 
            results = sorted(combined, key=lambda v: v['x']) 
        # Step 6: Return our findings
        return results, peaks, valleys, z
    @classmethod
    def detect_vr_blinks(cls,
                        eye_df:pd.DataFrame, 
                        blink_ranges:pd.DataFrame, 
                        peak_height=0.5, 
                        peak_prominence=1, 
                        peak_width=0, 
                        valley_height=0.5, 
                        valley_prominence=1, 
                        valley_width=0):
        # Initialize array to store all results
        all_results = []
        # Iterate through each blink range
        for _, row in blink_ranges.iterrows():
            # Get necessary timestamps
            overlap_counter = row['overlap_counter']
            range_start_ms = row['start_unix_ms']
            range_end_ms = row['end_unix_ms']
            _eye = eye_df[eye_df['unix_ms'].between(range_start_ms, range_end_ms)]
            # Get the relevant x and y data
            x = _eye['unix_ms'].to_list()
            y = _eye['gaze_target_screen_pos_y'].to_list()
            # Use `find_peaks()` we've created just above to detect combined peaks
            combined, peaks, valleys, z = cls.calculate_peaks(x, y, 
                                                       peak_height=peak_height, 
                                                       peak_prominence=peak_prominence, 
                                                       peak_width=peak_width, 
                                                       valley_height=valley_height, 
                                                       valley_prominence=valley_prominence, 
                                                       valley_width=valley_width)
            # Only contribute to `all_results` if `combined` is not None
            if combined is not None: 
                results = [{'overlap_counter':overlap_counter, **c} for c in combined]
                all_results.extend(results)
        # We'll combine all our results into a single dataframe
        df = pd.DataFrame(all_results)
        df = df.groupby('overlap_counter', group_keys=False).apply(lambda g: g.sort_values('x'))
        # We'll extract the first of each trial and overlap
        first_peaks = df.groupby('overlap_counter', as_index=False).first()
        # return the dfs
        return first_peaks
    @classmethod
    def detect_eeg_blinks(cls,
                          eeg_df:pd.DataFrame, 
                          blink_ranges:pd.DataFrame,
                          window_length=75,
                          polyorder=3,
                          mode='nearest',
                          peak_height=0.5, 
                          peak_prominence=1, 
                          peak_width=0, 
                          valley_height=0.5, 
                          valley_prominence=1, 
                          valley_width=0,
                          smooth_data:bool=True):
        # list of outputs
        all_results = {'TP9':[], 'TP10':[]}

        # Iterate through each blink range
        for _, row in blink_ranges.iterrows():
            # Get necessary timestamps
            overlap_counter = row['overlap_counter']
            range_start_ms = row['start_unix_ms']
            range_end_ms = row['end_unix_ms']
            _eeg = eeg_df[eeg_df['unix_ms'].between(range_start_ms, range_end_ms)]
            # Get the relevant x and y data
            x = _eeg['unix_ms'].to_list()
            tp9 = _eeg['TP9'].to_list()
            tp10 = _eeg['TP10'].to_list()
            # Smooth the data, if prompted
            if smooth_data:
                tp9 = savgol_filter(tp9, window_length=window_length, polyorder=polyorder, mode=mode)
                tp10 = savgol_filter(tp10, window_length=window_length, polyorder=polyorder, mode=mode)
            # Calculate the blinks
            _, _, tp9_valleys, tp9z = cls.calculate_peaks(
                x, tp9, 
                peak_height=peak_height, 
                peak_prominence=peak_prominence, 
                peak_width=peak_width, 
                valley_height=valley_height, 
                valley_prominence=valley_prominence, 
                valley_width=valley_width )
            _, _, tp10_valleys, tp10z = cls.calculate_peaks(
                x, tp10,
                peak_height=peak_height, 
                peak_prominence=peak_prominence, 
                peak_width=peak_width, 
                valley_height=valley_height, 
                valley_prominence=valley_prominence, 
                valley_width=valley_width )
            # Handle cases
            if tp9_valleys is not None: 
                results = [{'overlap_counter':overlap_counter, **c} for c in tp9_valleys]
                all_results['TP9'].extend(results)
            if tp10_valleys is not None: 
                results = [{'overlap_counter':overlap_counter, **c} for c in tp10_valleys]
                all_results['TP10'].extend(results)

        # We'll combine all our results into single dataframes for TP9 and TP10, then get the first blinks for each overlap
        if len(all_results['TP9']) > 0:
            tp9_df = pd.DataFrame(all_results['TP9'])
            tp9_df = tp9_df.groupby(['overlap_counter'], group_keys=False).apply(lambda g: g.sort_values('x'))
            first_valleys_tp9 = tp9_df.groupby(['overlap_counter'], as_index=False).first()
        else:
            first_valleys_tp9 = None
        if len(all_results['TP10']) > 0:
            tp10_df = pd.DataFrame(all_results['TP10'])
            tp10_df = tp10_df.groupby(['overlap_counter'], group_keys=False).apply(lambda g: g.sort_values('x'))
            first_valleys_tp10 = tp10_df.groupby(['overlap_counter'], as_index=False).first()
        else:
            first_valleys_tp10 = None

        # return the dfs
        return first_valleys_tp9, first_valleys_tp10
    @staticmethod
    def merge_datasets(vr,eeg):
        merged = pd.merge(
            left = vr[['overlap_counter', 'x']],
            right = eeg[['overlap_counter', 'x']],
            on=['overlap_counter'],
            how='inner'
        )
        merged.rename(columns={'x_x':'vr_x', 'x_y':'eeg_x'}, inplace=True)
        merged['offset'] = merged['eeg_x'] - merged['vr_x']
        return merged



def plot_calibration_timeline(pdirs:Iterable[str],  outpath:str=None):
    # Define the figure and subplots
    nrows = len(pdirs)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 2*nrows), constrained_layout=True)
    # Iterate through each provided participant directory
    pbar = tqdm(range(0,len(pdirs)))
    count = 0
    for pdir in pdirs:
        pbar.set_description(os.path.basename(pdir))
        # Read & plot eye data
        edf = pd.read_csv(os.path.join(pdir, 'eye.csv'))
        start_time = edf['unix_ms'].min()
        end_time = edf['unix_ms'].max()
        edf['rel_unix_ms'] = edf['unix_ms'] - start_time
        axes[count].plot(edf['rel_unix_ms'], [0]*len(edf), alpha=0.2, label="Eye Dataset Timeline")
        #axes[count].axvline(x=0, color='blue', linestyle='--', alpha=0.7)
        #axes[count].text(0, 0.1, f"Eye Start\n{start_time}", rotation=90, verticalalignment='bottom', fontsize=7.5)
        # Read and plot calibration data
        calibration_files = [cal_file for cal_file in sorted(glob(os.path.join(pdir, "calibration_*.csv")))]
        for f in calibration_files:
            cdf = pd.read_csv(f)
            if "unix_ms" not in cdf.columns: continue  # skip if missing
            first_timestamp = cdf["unix_ms"].iloc[0]
            rel_first_timestamp = first_timestamp - start_time
            cid = Path(f).stem.split('_')[1]
            ctitle = f"Trial {cid}\n{rel_first_timestamp}"
            axes[count].axvline(x=rel_first_timestamp, color='red', linestyle='--', alpha=0.7)
            axes[count].text(rel_first_timestamp+500, 0, ctitle, rotation=90, verticalalignment='bottom', fontsize=8)
        # Plot setup
        axes[count].set_xlim(0, end_time - start_time)
        axes[count].set_yticks([])
        axes[count].set_title(f"Participant: {os.path.basename(pdir)}",)
        # Update progress bar
        pbar.update(1)
        count+=1
    # Other plot stuff, save if needed, and render
    plt.suptitle(f"Per-Participant Timeline of Eye Data to Calibrations", 
                    y=0.99, fontsize=16, fontweight="bold")
    plt.xlabel("Relative unix time (ms)")
    plt.tight_layout()
    if outpath is not None: plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()

def plot_offsets(participants:Iterable[Participant], 
                    plot_type:str='box', 
                    hue_feature:str='channel',
                    outpath:str=None ):
    # Aggregate offsets across all trials
    offsets = pd.concat([t.offsets for p in participants for t in p.trials if t.offsets is not None])
    # Generate the plot
    plt.figure(figsize=(len(participants), 5))  # scale width with N
    # Plot the boxplots
    if plot_type == 'box':
        sns.boxplot(
            data=offsets,
            x="pid",   # horizontal orientation: y is category, x is value
            y="offset",
            hue=hue_feature,
            orient="v",
            showfliers=False,  # hide outlier dots (since we'll show raw data)
            width=0.6,
        )
    elif plot_type == 'violin':
        sns.violinplot(
            data=offsets,
            x="pid",
            y="offset",
            hue=hue_feature,
            orient="v",
            inner=None,
            cut=0,
            dodge=True,
            linewidth=1,
        )
    elif plot_type == 'split_violin':
        sns.violinplot(
            data=offsets,
            x="pid",
            y="offset",
            hue=hue_feature,
            split=True,   # 👈 key
            orient="v",
            inner="quartile",
            cut=0,
        )
    # Overlay jittered raw points
    sns.stripplot(
        data=offsets,
        x="pid",
        y="offset",
        hue="channel",
        dodge=True,
        orient="v",
        alpha=0.5,
        jitter=0.2,
        linewidth=1,
        edgecolor='gray',
        size=4,
    )
    # Other plot stuff
    plt.title("Diff Distributions per Participant (EEG - VR, TP9 & TP10)")
    plt.xlabel("Participant")
    plt.ylabel("Offset (ms)")
    # Add lines to indicate the 0-diff line
    plt.axhline(y=0, c='black', alpha=0.5)
    # Avoid duplicate legends from boxplot + stripplot
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[:len(offsets["channel"].unique())],
        labels[:len(offsets["channel"].unique())],
        title="EEG Channel",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    # Adjust the layout
    plt.tight_layout()
    # Save the figure if prompted, show if prompted
    if outpath is not None:     plt.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.show()



def plot_eye_calibration_timeline(filename:str):
    # --- Get overall time range ---
    if not os.path.exists(filename): 
        print(f"ERROR: {filename} does not exist")
        return
    df = pd.read_csv(filename)
    start_time = df["unix_ms"].min()
    end_time = df["unix_ms"].max()

    # Get calibrations from eye df
    cdf = df[df['event'] == 'Calibration']
    markers = cdf['unix_ms'].tolist()

    # --- Plot ---
    plt.figure(figsize=(12, 3))
    plt.plot(df["unix_ms"], [0]*len(df), alpha=0.2, label="Eye dataset timeline")

    # Add vertical markers for each smaller file
    for i in range(len(markers)):
        plt.axvline(x=markers[i], color='red', linestyle='--', alpha=0.7)
        plt.text(markers[i], 0.1, i, rotation=90, verticalalignment='bottom', fontsize=8)

    # Render other stuff
    plt.xlim(start_time, end_time)
    plt.xlabel("Unix time (ms)")
    plt.yticks([])
    plt.title(f"Timeline of Eye Data to Calibrations: {filename}")
    plt.legend()
    plt.tight_layout()
    plt.show()