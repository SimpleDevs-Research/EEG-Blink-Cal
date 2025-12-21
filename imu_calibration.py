import os
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime
from tqdm import tqdm
import argparse
import re
from collections.abc import Iterable
import statsmodels.formula.api as smf

pd.set_option('mode.chained_assignment', None)

parser = argparse.ArgumentParser(description="Given a list of trials and calbration files, manually calculate offsets between gaze dat and eeg data")
parser.add_argument("root_dir", help="The directory parent where all session directories are located.", type=str)
parser.add_argument("gaze_filename", help="The filename (including extension, e.g. `.csv`) that the gaze data is stored.", type=str)
parser.add_argument("gaze_colname", help="The column name in your gaze file that should be referenced as the y-position of the gaze direction", type=str)
parser.add_argument("eeg_filename", help="The filename (including extension, e.g. `.csv`) that the eeg data is stored.", type=str)
parser.add_argument("eeg_colname", help="The column name in your eeg file that should be referenced as the representative EEG channel.", type=str)
parser.add_argument("-sb", "--start_buffer", help="For each calibration, should the start millseconds be pushed further back with this buffer value?", type=int, default=0)
parser.add_argument("-eb", "--end_buffer", help="For each calibration, should the end millseconds be pushed later with this buffer value?", type=int, default=0)
args = parser.parse_args()

_EEG_RAW_COLNAMES =     { 'RAW_TP9':'TP9', 
                          'RAW_TP10':'TP10', 
                          'RAW_AF7':'AF7', 
                          'RAW_AF8':'AF8', 
                          'Accelerometer_X':'accel_x', 
                          'Accelerometer_Y':'accel_y', 
                          'Accelerometer_Z':'accel_z', 
                          'Gyro_X':'gyro_x', 
                          'Gyro_Y':'gyro_y',
                          'Gyro_Z':'gyro_z'     }
_EEG_MUSE_COLNAMES =    [ 'Delta_TP9','Delta_TP10','Delta_AF7','Delta_AF8',
                          'Theta_TP9', 'Theta_TP10', 'Theta_AF7', 'Theta_AF8', 
                          'Alpha_TP9', 'Alpha_TP10', 'Alpha_AF7', 'Alpha_AF8',
                          'Beta_TP9', 'Beta_TP10', 'Beta_AF7', 'Beta_AF8',
                          'Gamma_TP9', 'Gamma_TP10', 'Gamma_AF7', 'Gamma_AF8'     ]
#_CAL_FILES =            [ "calibration_test_1.csv", "calibration_test_2.csv", 
#                          "calibration_test_3.csv", "calibration_test_4.csv",
#                          "calibration_test_5.csv", "calibration_test_6.csv" ]
_CAL_FILES =            [ '0.25-72.csv', '0.25-90.csv',
                          '0.5-72.csv', '0.5-90.csv',
                          '0.75-72.csv', '0.75-90.csv'        ]
_CALIBRATION_COLUMNS =  [ 'unix_ms', 'frame', 'rel_timestamp', 
                          'event', 'overlap_counter' ]
_REJECTED_TRIALS =      [ '0.imu_test'  ]

class Experiment:
    def __init__(self, ename:str, root_dir:str, verbose:bool=False,):
        # Self-initialization, finding sessions
        self.ename = ename
        self.root_dir = root_dir
        self.subdirectories = self.find_subdirectories(self.root_dir)
        self.sessions = None
        self.offsets = None
        if verbose:
            print("IDENTIFIED SUBDIRECTORIES:")
            for f in self.subdirectories: 
                print('└──', f)
    
    def init_sessions(self, verbose:bool=False):
        if verbose: print("INITIALIZING SUBDIRECTORIES AS SESSIONS")
        pbar = tqdm(self.subdirectories)
        self.sessions = []
        for f in pbar:
            sname = os.path.basename(f)
            pbar.set_description(sname)
            s = Session(self, f, sname)
            s.init_calibrations(start_buffer=args.start_buffer, end_buffer=args.end_buffer)
            self.sessions.append(s)
        return self

    def plot_raw_calibrations(self, outname=None, show:bool=True):
        fig = plt.figure(figsize=(30, 5 * len(self.sessions)), layout=None)
        gs = fig.add_gridspec(
            len(self.sessions), 1,
            hspace=0.3,
            top=0.97
        )
        # Plot each session per row
        for row_index, session  in enumerate(self.sessions):
            session.plot_raw_calibrations(parent_spec=gs[row_index])
        # set the title
        fig.text( 0.5, 0.985, self.ename,
                    ha="center", va="top", fontsize=16, fontweight="bold" )
        # Save the figure if prompted, show if prompted
        if outname is not None: 
            if isinstance(outname, Iterable):
                for o in outname: plt.savefig(os.path.join(self.root_dir, o), bbox_inches='tight', dpi=300)
            elif isinstance(outname, str):
                plt.savefig(os.path.join(self.root_dir, outname), bbox_inches='tight', dpi=300)
            else:
                print("WARNING: supplied outname invalid type. Expects either str or Iterable[str]")
        if show:    plt.show()
        else:       plt.close()

        plt.show()

    def calculate_offsets(self, cache_offsets:bool=True, force_calc:bool=False):
        # Load in if we have an existing cache
        if not force_calc and os.path.exists(os.path.join(self.root_dir, self.ename+'_offsets.csv')):
            self.offsets = pd.read_csv(os.path.join(self.root_dir, self.ename+'_offsets.csv'))
            print(f"Loading experiment offsets from \"{self.ename}\"_offsets.csv")
            return self
        # Otherwise, calculate offsets
        offsets = []
        for s in self.sessions:
            if s.calculate_offsets(cache_offsets=cache_offsets, force_calc=force_calc):
                df = s.offsets
                if df is not None:
                    df.insert(loc=0, column='session_name', value=s.sname)
                    offsets.append(df)
        # Edge case: No sessions were valid
        if len(offsets) == 0:
            print(f"No sessions in experiment {self.ename}.")
            self.offsets = None
            return False
        self.offsets = pd.concat(offsets, ignore_index=True)
        # If cache, save the file
        if cache_offsets:
            self.offsets.to_csv(os.path.join(self.root_dir, self.ename+'_offsets.csv'), index=False)
        # Return TRUE to indicate successful run
        return True

    def plot_offsets(self, hue_feature:str, plot_type:str='box', jitter:float=0.02, outname=None, show:bool=True):
        # Assertion: self.offsets should not be None
        assert self.offsets is not None, "Cannot plot nonexistent offsets"
        # Aggregate offsets across all trials
        df = self.offsets.sort_values(
            by=["session_name", "calibration_name", hue_feature, "overlap_counter"],
            key=self.natural_sorting_by_key
        )
        ntrials = df['session_name'].nunique()
        hue_order = sorted(df[hue_feature].unique())
        # Generate the plot
        plt.figure(figsize=(ntrials*2, 5))  # scale width with N
        # Plot the boxplots
        if plot_type == 'box':
            sns.boxplot(
                data=df,
                x="session_name",   # horizontal orientation: y is category, x is value
                y="offset_eeg-gaze",
                hue=hue_feature,
                hue_order=hue_order,
                orient="v",
                showfliers=False,  # hide outlier dots (since we'll show raw data)
                width=0.6,
                boxprops=dict(alpha=.35)
            )
        elif plot_type == 'violin':
            sns.violinplot(
                data=df,
                x="session_name",
                y="offset_eeg-gaze",
                hue=hue_feature,
                hue_order=hue_order,
                orient="v",
                inner=None,
                cut=0,
                dodge=True,
                linewidth=1,
                alpha=0.35
            )
        elif plot_type == 'violinsplit':
            sns.violinplot(
                data=df,
                x="session_name",
                y="offset_eeg-gaze",
                hue=hue_feature,
                hue_order=hue_order,
                split=True,
                orient="v",
                inner="quartile",
                cut=0,
                alpha=0.35
            )
        # Overlay jittered raw points
        ax = sns.stripplot(
            data=df,
            x="session_name",
            y="offset_eeg-gaze",
            hue=hue_feature,
            hue_order=hue_order,
            dodge=True,
            orient="v",
            alpha=1.0,
            jitter=jitter,
            linewidth=1,
            edgecolor='gray',
            size=4,
        )
        
        # Other plot stuff
        plt.title("Diff Distributions per Session")
        plt.xlabel("Session names")
        plt.ylabel("Offset (EEG - VR, ms)")
        # Add lines to indicate the 0-diff line
        plt.axhline(y=0, c='black', alpha=0.5)
        # Avoid duplicate legends from boxplot + stripplot
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles[:len(hue_order)],
            labels[:len(hue_order)],
            title=hue_feature,
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )
        # Adjust the layout
        plt.tight_layout()
        # Save the figure if prompted, show if prompted
        if outname is not None: 
            if isinstance(outname, str):
                plt.savefig(os.path.join(self.root_dir, outname), bbox_inches='tight', dpi=300)
            elif isinstance(outname, Iterable):
                for o in outname: plt.savefig(os.path.join(self.root_dir, o), bbox_inches='tight', dpi=300)
            else:
                print("WARNING: supplied outname invalid type. Expects either str or Iterable[str]")
        if show:    plt.show()
        else:       plt.close()
        return df
    
    @staticmethod
    def find_subdirectories(dir:str):
        subdirs = [ f.path for f in os.scandir(dir) if f.is_dir() and f.name not in _REJECTED_TRIALS]
        return sorted(subdirs)
    @staticmethod
    def natural_sorting(s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))
        ]
    @staticmethod
    def natural_sorting_by_key(series: pd.Series) -> pd.Series:
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        def alphanum_key(s):
            return tuple(convert(c) for c in re.split(r'(\d+)', str(s)))
        return series.map(alphanum_key)

# ---------------------------------------------------------
# ---------------------------------------------------------

class Session:
    def __init__(self, parent:Experiment, root_dir:str, sname:str):
        self.parent = parent
        self.root_dir = root_dir
        self.sname = sname
        #self.device = self.sname.split(".")[1].split("_")[0]
        self.eeg_raw, self.eeg_muse, self.eeg_blinks = self.read_eeg(os.path.join(root_dir, args.eeg_filename))
        self.gaze_raw = pd.read_csv(os.path.join(root_dir, args.gaze_filename))
        self.calibrations_raw = [(Path(f).stem, self.correct_calibration_df(pd.read_csv(os.path.join(root_dir, f)))) for f in _CAL_FILES]
        self.offsets = None

    def init_calibrations(self, start_buffer:int=5000, end_buffer:int=1000):
        self.calibrations = []
        pbar = tqdm(self.calibrations_raw)
        for c in pbar:
            pbar.set_description(c[0])
            self.calibrations.append(Calibration(self, c[0], c[1], start_buffer=start_buffer, end_buffer=end_buffer))

    def calculate_offsets(self, cache_offsets:bool=True, force_calc:bool=False):
        # First, check if any offsets with the same name have been already cached
        if not force_calc and os.path.exists(os.path.join(self.root_dir, 'offsets.csv')):
            self.offsets = pd.read_csv((os.path.join(self.root_dir, 'offsets.csv')))
            print("Loaded session offsets from `offsets.csv`")
            return self
        # Otherwise, get offsets for all calibrations in this session
        offsets = []
        for c in self.calibrations:
            if c.calculate_offsets():
                df = c.offsets
                if df is not None:
                    df.insert(loc=0, column='calibration_name', value=c.cname)
                    offsets.append(df)
        # Edge case: No calibrations were valid. This is an ineffective trial
        if len(offsets) == 0:
            print(f"No valid calibrations for {self.sname}; invalid session.")
            self.offsets = None
            return False
        self.offsets = pd.concat(offsets, ignore_index=True)
        # Double-check: for any columns we want to add

        # If we want to cache, then we save
        if cache_offsets:
            self.offsets.to_csv(os.path.join(self.root_dir, 'offsets.csv'), index=False)
        # Return self
        return True

    def plot_raw_calibrations(self, parent_spec=None):
        if parent_spec is None:
            # This is a standalone plot. This means we need to generate our own parent_spec
            fig = plt.figure(figsize=(30, len(self.calibrations)))
            spec = fig.add_gridspec(1, 1)[0]
            fig.suptitle(self.sname)
            for i in range(len(self.calibrations)):
                c = self.calibrations[i]
                c.plot_raw_data(parent_spec=spec)
            plt.show()
        else:
            fig = parent_spec.get_gridspec().figure
            spec = parent_spec
        # Generate the grid spec, derived from parent_spec
        gs = GridSpecFromSubplotSpec(
            1, len(self.calibrations),
            subplot_spec=spec,
            wspace=0.2
        )
        # Embed your calibration plots into each column in `gs`
        for i, c in enumerate(self.calibrations):
            c.plot_raw_data(parent_spec=gs[i])
        # Plotting suptitle
        bbox = parent_spec.get_position(fig)
        suptitle_x = bbox.x0
        if parent_spec is None:
            suptitle_x += bbox.width / 2
        fig.text(
            suptitle_x,
            bbox.y1 + 0.0025,   # small vertical offset
            self.sname,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )
        # If this is a standalone, then we show the figure
        if parent_spec is None:
            plt.show()

    @staticmethod
    def correct_calibration_df(df:pd.DataFrame):
        fixed = df.iloc[:, :len(_CALIBRATION_COLUMNS)]
        fixed.columns = _CALIBRATION_COLUMNS
        return fixed
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
    def read_eeg(cls, filepath:str):
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

# ---------------------------------------------------------
# ---------------------------------------------------------

class Calibration:
    def __init__(self, parent:Session, cname:str, df:pd.DataFrame, start_buffer:int=5000, end_buffer:int=1000):
        self.parent = parent
        self.cname = cname
        self.start = df.loc[df["event"] == "Start"].iloc[0]
        self.end = df.loc[df["event"] == "End"].iloc[0]
        self.overlaps = df[df['event'] == 'Overlap']

        if start_buffer is not None:    self.start['unix_ms'] += start_buffer
        if end_buffer is not None:      self.end['unix_ms'] -= end_buffer
        self.eeg = parent.eeg_raw[parent.eeg_raw['unix_ms'].between(self.start['unix_ms'], self.end['unix_ms'])]
        self.gaze = parent.gaze_raw[parent.gaze_raw['unix_ms'].between(self.start['unix_ms'], self.end['unix_ms'])]

    def calculate_offsets(self):
        # Initialize our selections
        overlaps = []
        eegs = []
        gazes = []
        # Initiate our x and y data and vlines
        gx = self.gaze['unix_ms'].to_list()
        gy = self.gaze[args.gaze_colname].to_list()
        ex = self.eeg['unix_ms'].to_list()
        ey = self.eeg[args.eeg_colname].to_list()
        # Exit early if any missing data is detected
        if (len(gx) == 0 or len(gy) == 0 or len(ex) == 0 or len(ey) == 0):
            print("ERROR: No data associated with this calibration step. Skipping")
            self.offsets = None
            return self
        data = ((gx, gy, 'Gaze Y', 'red'), (ex, ey, 'TP9', 'blue'))
        vlines = [(row['unix_ms'], f"Start {row['overlap_counter']}") for row_index, row in self.overlaps.iterrows()]
        # Iterate through each overlap
        for _, row in self.overlaps.iterrows():
            # Get overlap coiunter
            overlap_counter = row['overlap_counter']
            title = f"{self.parent.sname} : {self.cname} : Overlap {overlap_counter}"
            # Define our selector for gaze
            selector = PointSelector(data, which_plot=0, vlines=vlines, previous=(gazes, eegs), title=title)
            plt.show()
            gaze_selection = selector.selected
            if gaze_selection is None: continue
            # Define our selector for eeg
            selector = PointSelector(data, which_plot=1, vlines=vlines, previous=(gazes+[gaze_selection], eegs), title=title) 
            plt.show()
            eeg_selection = selector.selected
            if eeg_selection is None: continue
            # Add to our selections
            overlaps.append(overlap_counter)
            gazes.append(gaze_selection)
            eegs.append(eeg_selection)
        # If any of the lists are empty, then this is an invalid calibration. We need to indicate that this is the case.
        if len(overlaps) == 0:
            # No calibrations allowed. Reject
            print("No calibration blinks, thus no offsets, detected in ", self.cname)
            self.offsets = None
            return False
        agg_selections = list(zip(overlaps, gazes, eegs))
        agg_entries = [{
            'overlap_counter':s[0], 
            'gaze_unix_ms':int(s[1][1]), 
            'eeg_unix_ms':int(s[2][1])} for s in agg_selections]
        # Calculate offsets, save the offsets
        df = pd.DataFrame(agg_entries)
        df['offset_eeg-gaze'] = df['eeg_unix_ms'] - df['gaze_unix_ms']
        self.offsets = df
        return True

    # CALLING THIS DOES NOT ACTUALLY PRODUCE A PLOT. CALL `Calibration.plot()`, THEN `plt.show()` IN THE NEXT LINE
    def plot_raw_data(self, parent_spec=None):
        # Define a grid spec that we'll render into. If standalone (e.g. no parent_spec), we create a spec ourselves
        if parent_spec is None:
            fig = plt.figure(figsize=(6, 3))
            parent_spec = fig.add_gridspec(1, 1)[0]
        else:
            fig = parent_spec.get_gridspec().figure
        # Define our subplot within our spec, which is two rows and one column subplots
        gs = GridSpecFromSubplotSpec( 2, 1, subplot_spec=parent_spec, hspace=0.1 )
        # Define our axes
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
        # Plot
        ax_top.plot(self.eeg['unix_ms'], self.eeg[args.eeg_colname], color='lightblue', label=f'RAW {args.eeg_colname}')
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False )
        ax_bottom.plot(self.gaze['unix_ms'], self.gaze[args.gaze_colname], color='red', label='L. Eye Y-pos')
        # Legend
        ax_top.legend()
        ax_bottom.legend()
        # Render vertical lines to represent calibration starts
        ax_top_trans = mtransforms.blended_transform_factory(
            ax_top.transData,   # x in data coordinates
            ax_top.transAxes    # y in axes coordinates
        )
        ax_bottom_trans = mtransforms.blended_transform_factory(
            ax_bottom.transData,   # x in data coordinates
            ax_bottom.transAxes    # y in axes coordinates
        )
        for row_index, row in self.overlaps.iterrows():
            ax_top.axvline(x=row['unix_ms'], color='black', alpha=0.75)
            ax_top.text( row['unix_ms'], 0.02, f"Start {row['overlap_counter']}",
                            transform=ax_top_trans, rotation=90, ha='left', va='bottom', fontsize=6, rotation_mode='anchor')
            ax_bottom.axvline(x=row['unix_ms'], color='black', alpha=0.75)
            ax_bottom.text(row['unix_ms'], 0.02, f"Start {row['overlap_counter']}", 
                            transform=ax_bottom_trans, rotation=90, ha='left', va='bottom', fontsize=6, rotation_mode='anchor')
        # Return
        return ax_top, ax_bottom

# ---------------------------------------------------------
# ---------------------------------------------------------

class PointSelector:
    def __init__(self, 
                    data,           # The raw data. Each element is a subplot. For each element, 0 = x-data, 1 = y-data 
                    which_plot=0,   # Which plot should the selection be extracted from?
                    figsize=None,   # How big should the plot be rendered as?
                    title:str=None, 
                    previous=None, 
                    vlines=None):
        # Assertions
        if previous is not None:
            assert len(data) == len(previous), 'If previous is provided, must be an array with the same length as the number of subplots rendered.'

        # Caching necessary data
        self.data = [{                  # Caching all data
            'x':np.asarray(d[0]),       # - x-data is remapped as a numpy array
            'y':np.asarray(d[1]),       # - y-data is remapped as a numpy array
            'label': d[2],              # - label is the 3rd item
            'color': d[3]               # - color is the 4th item 
                } for d in data] 
        self.figsize = (10, len(self.data)*3)   # How big (in inches x inches) should the plot be?
        if figsize is not None: self.figsize=figsize
        self.title = title                      # What should the title of the plot be?
        self.previous = previous                # Are there any previous markers?

        # Selection Variables
        self.which_plot = which_plot    # Which plot should the marker refer to?
        self.idx = 0                    # The index that the marker is assigned to
        self.selected = None            # Coordinates that were selected. Basically, the output.

        # Draw the plot(s)
        self.fig, self.axes = plt.subplots(len(data), 1, figsize=self.figsize, sharex=True)
        for i in range(len(self.data)):
            ax = self.axes[i]       # Currently-rendered axis
            x = self.data[i]['x']   # Currently-rendered x
            y = self.data[i]['y']   # Currently-renderec y
            label = self.data[i]['label']
            color = self.data[i]['color']
            ax.plot(x, y, "-o", color=color, label=label, alpha=0.3, markersize=1)
            ax.legend(loc="upper left")
            if vlines is not None:
                ax_transform = mtransforms.blended_transform_factory(
                    ax.transData,   # x in data coordinates
                    ax.transAxes    # y in axes coordinates
                )
                for marker in vlines:
                    ax.axvline( x=marker[0], color='black', alpha=0.5 )
                    ax.text( marker[0], 0.02, marker[1], 
                                transform=ax_transform, 
                                rotation=90, 
                                ha='left', 
                                va='bottom', 
                                fontsize=6, 
                                rotation_mode='anchor' )
            if self.previous is not None and len(self.previous[i]) > 0:
                pid, px, py = zip(*self.previous[i])
                ax.plot(px, py, "x", color="black", markersize=8, label="Previous", zorder=2 )
                self.text = ax.text( 1.02, 1.0, self.format_previous(),
                                    transform=ax.transAxes,
                                    va="top",
                                    ha="left",
                                    fontsize=9,
                                    family="monospace" )

        # movable cursor
        ax = self.axes[self.which_plot]
        x = self.data[self.which_plot]['x']
        y = self.data[self.which_plot]['y']
        self.marker, = ax.plot( x[self.idx], y[self.idx], "ro", markersize=10 )
        self.cid = self.fig.canvas.mpl_connect(
            "key_press_event", self.on_key
        )
        self.mid = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

        self.update()
    def format_previous(self, i:int=0):
        if not self.previous: return "Previous selections:\n(none)"
        prev = self.previous[i]
        lines = ["Previous selections:"]
        for i, (_i, x, y) in enumerate(prev, 1):   lines.append(f"{i:>2}: ({x:.3f}, {y:.3f})")
        return "\n".join(lines)
    def update(self):
        self.marker.set_data(
            [self.data[self.which_plot]['x'][self.idx]],
            [self.data[self.which_plot]['y'][self.idx]]
        )
        title_text = f"Index {self.idx} | x={self.data[self.which_plot]['x'][self.idx]:.3f}, y={self.data[self.which_plot]['y'][self.idx]:.3f}"
        if self.title is not None: title_text = self.title
        self.axes[self.which_plot].set_title(title_text + "\n← / → move | ENTER: select | ESC: skip")
        self.fig.canvas.draw_idle()
    def on_key(self, event):
        if event.key == "left":
            self.idx = max(0, self.idx - 1)
            self.update()
        elif event.key == "right":
            self.idx = min(len(self.data[self.which_plot]['x']) - 1, self.idx + 1)
            self.update()
        elif event.key == "enter":
            # Set Selected
            self.selected = (self.idx, self.data[self.which_plot]['x'][self.idx], self.data[self.which_plot]['y'][self.idx])
            self.close()
        elif event.key == "escape":
            # cancel
            self.selected = None
            self.close()
    def on_click(self, event):
        # ignore clicks outside the axes
        if event.inaxes != self.axes[self.which_plot]: return
        if event.xdata is None: return
        # find nearest x (fast + intuitive)
        self.idx = np.argmin(np.abs(self.data[self.which_plot]['x'] - event.xdata))
        self.update()
    def close(self):
        self.fig.canvas.mpl_disconnect(self.cid)
        plt.close(self.fig)







experiment = Experiment( 'street_sim_v2', args.root_dir, verbose=True )
experiment.init_sessions()
experiment.plot_raw_calibrations(outname=['raws.png','raws.pdf'], show=False)
experiment.calculate_offsets( cache_offsets=True, force_calc=False)
df = experiment.plot_offsets( 'calibration_name' , jitter=0, outname='street_sim_v2_offsets.png')
# We add extra modifications to dataframe
df[["conf_threshold", "fps"]] = df["calibration_name"].str.split("-", expand=True)
session_name_pattern = (
    r'^(?P<session_order>\d+)\.'      # number before "."
    r'vr=(?P<vr_status>[^-]+)-'
    r'eeg=(?P<eeg_status>[^-]+)-'
    r'distance=(?P<distance>.+)$'
)
df[["session_order", "vr_status", "eeg_status", "distance"]] = (
    df["session_name"]
    .str.extract(session_name_pattern)
)
df['session_order'] = df["session_order"].astype(int)
df.to_csv(os.path.join(args.root_dir, 'offsets.csv'), index=False)
print(df.to_string())

# employing multiple linear regression (OLS)
ols_model = smf.ols(
    formula="""
    Q("offset_eeg-gaze")
    ~ conf_threshold
    + fps
    + C(vr_status)
    + C(eeg_status)
    + C(distance)
    """,
    data=df
).fit()

print(ols_model.summary())

# Employing mixed effects model
mixed_model = smf.mixedlm(
    'Q("offset_eeg-gaze") ~ conf_threshold + fps + C(vr_status) + C(eeg_status) + C(distance)',
    df,
    groups=df['session_name']
)
result = mixed_model.fit()
print(result.summary())



from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy

y, X = patsy.dmatrices(
    'Q("offset_eeg-gaze") ~ conf_threshold + fps + C(vr_status) + C(eeg_status) + C(distance)',
    df,
    return_type='dataframe'
)

vif = pd.DataFrame({
    "variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print(vif)


#for s in experiment.sessions:
#    s.plot_raw_calibrations()