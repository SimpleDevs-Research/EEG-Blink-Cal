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

pd.set_option('mode.chained_assignment', None)

_EEG_RAW_COLNAMES =     { 'RAW_TP9':'TP9', 
                          'RAW_TP10':'TP10', 
                          'RAW_AF7':'AF7', 
                          'RAW_AF8':'AF8'         }
_EEG_MUSE_COLNAMES =    [ 'Delta_TP9','Delta_TP10','Delta_AF7','Delta_AF8',
                          'Theta_TP9', 'Theta_TP10', 'Theta_AF7', 'Theta_AF8', 
                          'Alpha_TP9', 'Alpha_TP10', 'Alpha_AF7', 'Alpha_AF8',
                          'Beta_TP9', 'Beta_TP10', 'Beta_AF7', 'Beta_AF8',
                          'Gamma_TP9', 'Gamma_TP10', 'Gamma_AF7', 'Gamma_AF8'     ]
_CAL_FILES =            [ '0.25-72.csv', '0.25-90.csv',
                          '0.5-72.csv', '0.5-90.csv',
                          '0.75-72.csv', '0.75-90.csv'        ]

class Experiment:
    def __init__(self, tname:str, root_dir:str, verbose:bool=False,):
        # Self-initialization, finding sessions
        self.tname = tname
        self.root_dir = root_dir
        self.subdirectories = self.find_subdirectories(self.root_dir)
        if verbose:
            print("IDENTIFIED SUBDIRECTORIES:")
            for f in self.subdirectories: 
                print('└──', f)
    
    def init_sessions(self, eeg_filename:str, gaze_filename:str, calibration_filenames, verbose:bool=False, start_buffer:int=5000, end_buffer:int=1000):
        if verbose: print("INITIALIZING SUBDIRECTORIES AS SESSIONS")
        pbar = tqdm(self.subdirectories)
        self.sessions = []
        for f in pbar:
            sname = os.path.basename(f)
            pbar.set_description(sname)
            s = Session(self, f, sname, eeg_filename, gaze_filename, calibration_filenames)
            s.init_calibrations(start_buffer=start_buffer, end_buffer=end_buffer)
            self.sessions.append(s)
        return self
   
    def calculate_offsets(self, cache_offsets:bool=True, force_calc:bool=False):
        # Load in if we have an existing cache
        if not force_calc and os.path.exists(os.path.join(self.root_dir, self.tname+'_offsets.csv')):
            self.offsets = pd.read_csv(os.path.join(self.root_dir, self.tname+'_offsets.csv'))
            print(f"Loading experiment offsets from \"{self.tname}\"_offsets.csv")
            return self
        # Otherwise, calculate offsets
        offsets = []
        for s in self.sessions:
            s.calculate_offsets(cache_offsets=cache_offsets, force_calc=force_calc)
            df = s.offsets
            df.insert(loc=0, column='session_name', value=s.sname)
            df.insert(loc=1, column='device', value=s.device)
            offsets.append(df)
        self.offsets = pd.concat(offsets, ignore_index=True)
        # If cache, save the file
        if cache_offsets:
            self.offsets.to_csv(os.path.join(self.root_dir, self.tname+'_offsets.csv'), index=False)
        # Return self
        return self

    def plot_offsets(self, hue_feature:str, plot_type:str='box', outname=None, show:bool=True):
        # Aggregate offsets across all trials
        df = self.offsets.sort_values(["session_name", "calibration_name", "overlap_counter"])
        ntrials = df['session_name'].nunique()
        # Generate the plot
        plt.figure(figsize=(ntrials, 5))  # scale width with N
        # Plot the boxplots
        if plot_type == 'box':
            sns.boxplot(
                data=df,
                x="session_name",   # horizontal orientation: y is category, x is value
                y="offset_eeg-gaze",
                hue=hue_feature,
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
                split=True,   # 👈 key
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
            dodge=True,
            orient="v",
            alpha=1.0,
            jitter=0.2,
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
            handles[:len(df[hue_feature].unique())],
            labels[:len(df[hue_feature].unique())],
            title=hue_feature,
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )
        # Adjust the layout
        plt.tight_layout()
        # Save the figure if prompted, show if prompted
        if outname is not None: 
            if isinstance(outname, Iterable):
                for o in outname: plt.savefig(o, bbox_inches='tight', dpi=300)
            elif isinstance(outname, str):
                plt.savefig(outpath, bbox_inches='tight', dpi=300)
            else:
                print("WARNING: supplied outname invalid type. Expects either str or Iterable[str]")
        if show:    plt.show()
        else:       plt.close()
        # Return offsets
        return offsets
    
    @staticmethod
    def find_subdirectories(dir:str):
        subdirs = [ f.path for f in os.scandir(dir) if f.is_dir() ]
        return sorted(subdirs)

# ---------------------------------------------------------
# ---------------------------------------------------------

class Session:
    def __init__(self, parent:Experiment, root_dir:str, sname:str, eeg_filename:str, gaze_filename:str, calibration_filenames):
        self.parent = parent
        self.root_dir = root_dir
        self.sname = sname
        self.device = self.sname.split(".")[1].split("_")[0]
        self.eeg_raw, self.eeg_muse, self.eeg_blinks = self.read_eeg(os.path.join(root_dir, eeg_filename))
        self.gaze_raw = pd.read_csv(os.path.join(root_dir, gaze_filename))
        self.calibrations_raw = [(Path(f).stem, pd.read_csv(os.path.join(root_dir, f))) for f in calibration_filenames]
   
    def init_calibrations(self, start_buffer:int=5000, end_buffer:int=1000):
        self.calibrations = []
        pbar = tqdm(self.calibrations_raw)
        for c in pbar:
            pbar.set_description(c[0])
            self.calibrations.append(Calibration(self, c[0], c[1], start_buffer=start_buffer, end_buffer=end_buffer))
    
    def calculate_offsets(self, cache_offsets:bool=True, force_calc:bool=False):
        # First, check if any offsets with the same name have been already cached
        if not force_calc and os.path.exists(os.path.join(self.root_dir, self.sname+'_offsets.csv')):
            self.offsets = pd.read_csv((os.path.join(self.root_dir, self.sname+'_offsets.csv')))
            print(f"Loading session offsets from \"{self.sname}\"_offsets.csv")
            return self
        # Otherwise, get offsets for all calibrations in this session
        offsets = []
        for c in self.calibrations:
            c.calculate_offsets()
            df = c.offsets
            if df is not None:
                df.insert(loc=0, column='calibration_name', value=c.cname)
                offsets.append(df)
        self.offsets = pd.concat(offsets, ignore_index=True)
        # If we want to cache, then we save
        if cache_offsets:
            self.offsets.to_csv(os.path.join(self.root_dir, self.sname+'_offsets.csv'), index=False)
        # Return self
        return self

    def plot_raw_calibrations(self):
        fig = plt.figure(figsize=(30, len(self.calibrations)))
        subplots = fig.add_gridspec(1, len(self.calibrations))
        # Embed your 2-row plot in the bottom-right quadrant
        for i in range(len(self.calibrations)):
            c = self.calibrations[i]
            c.plot_raw_data(parent_spec=subplots[i])
        plt.show()

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
        ipds = []
        conf_thresholds = []
        raw_fps = []
        smooth_fps = []
        eegs = []
        gazes = []
        # Initiate our x and y data and vlines
        gx = self.gaze['unix_ms'].to_list()
        gy = self.gaze['left_screen_pos_y'].to_list()
        ex = self.eeg['unix_ms'].to_list()
        ey = self.eeg['TP9'].to_list()
        # Exit early if any missing data is detected
        if (len(gx) == 0 or len(gy) == 0 or len(ex) == 0 or len(ey) == 0):
            print("ERROR: No data associated with this calibration step. Skipping")
            self.offsets = None
            return self
        vlines = [(row['unix_ms'], f"Start {row['overlap_counter']}") for row_index, row in self.overlaps.iterrows()]
        # Iterate through each overlap
        for _, row in self.overlaps.iterrows():
            # Get overlap coiunter
            overlap_counter = row['overlap_counter']
            title = f"Overlap {overlap_counter}"
            # Define our selector for gaze
            selector = PointSelector(gx, gy, (10,3), color='red', vlines=vlines, previous=gazes, title=title)
            plt.show()
            gaze_selection = selector.selected
            if gaze_selection is None: continue
            # Define our selector for eeg
            selector = PointSelector(ex, ey, (10,3), color='blue', vlines=vlines, previous=eegs, title=title) 
            plt.show()
            eeg_selection = selector.selected
            if eeg_selection is None: continue
            # Add to our selections
            overlaps.append(overlap_counter)
            ipds.append(row['ipd'])
            raw_fps.append(row['raw_fps'])
            smooth_fps.append(row['smooth_fps'])
            conf_thresholds.append(row['conf_threshold'])
            gazes.append(gaze_selection)
            eegs.append(eeg_selection)
        agg_selections = list(zip(overlaps, ipds, conf_thresholds, raw_fps, smooth_fps, gazes, eegs))
        agg_entries = [{
            'overlap_counter':s[0], 
            'ipd': int(s[1]),
            'conf_threshold': float(s[2]),
            'raw_fps': float(s[3]),
            'smooth_fps': float(s[4]),
            'gaze_unix_ms':int(s[5][1]), 
            'eeg_unix_ms':int(s[6][1])} for s in agg_selections]
        # Calculate offsets, save the offsets
        df = pd.DataFrame(agg_entries)
        df['offset_eeg-gaze'] = df['eeg_unix_ms'] - df['gaze_unix_ms']
        self.offsets = df
        return self

    # CALLING THIS DOES NOT ACTUALLY PRODUCE A PLOT. CALL `Calibration.plot()`, THEN `plt.show()` IN THE NEXT LINE
    def plot_raw_data(self, parent_spec=None):
        # Define a grid spec that we'll render into. If standalone (e.g. no parent_spec), we create a spec ourselves
        if parent_spec is None:
            fig = plt.figure(figsize=(6, 3))
            parent_spec = fig.add_gridspec(1, 1)[0]
        else:
            fig = plt.gcf()
        # Define our subplot within our spec, which is two rows and one column subplots
        gs = GridSpecFromSubplotSpec( 2, 1, subplot_spec=parent_spec, hspace=0.1 )
        # Define our axes
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
        # Plot
        ax_top.plot(self.eeg['unix_ms'], self.eeg['TP9'], color='lightblue', label='RAW TP9')
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False )
        ax_bottom.plot(self.gaze['unix_ms'], self.gaze['left_screen_pos_y'], color='red', label='L. Eye Y-pos')
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
    def __init__(self, x, y, figsize, color:str='gray', title:str=None, vlines=None, previous=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.figsize = figsize
        self.idx = 0
        self.selected = None
        self.previous = None
        self.title = title

        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.plot(self.x, self.y, "-o", color=color, alpha=0.3, markersize=1)
        if vlines is not None:
            ax_transform = mtransforms.blended_transform_factory(
                self.ax.transData,   # x in data coordinates
                self.ax.transAxes    # y in axes coordinates
            )
            for marker in vlines:
                self.ax.axvline( x=marker[0], color='black', alpha=0.5 )
                self.ax.text( marker[0], 0.02, marker[1], 
                            transform=ax_transform, 
                            rotation=90, 
                            ha='left', 
                            va='bottom', 
                            fontsize=6, 
                            rotation_mode='anchor' )
        if previous is not None and len(previous) > 0:
            self.previous = previous
            pid, px, py = zip(*self.previous)
            self.ax.plot(px, py, "x", color="black", markersize=8, label="Previous", zorder=2 )

        # movable cursor
        self.marker, = self.ax.plot(
            self.x[self.idx],
            self.y[self.idx],
            "ro",
            markersize=10
        )

        self.cid = self.fig.canvas.mpl_connect(
            "key_press_event", self.on_key
        )
        self.mid = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

        # legend
        self.ax.legend(loc="upper left")
        if self.previous is not None and len(self.previous) > 0:
            # text panel
            self.text = self.ax.text(
                1.02,
                1.0,
                self.format_previous(),
                transform=self.ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
            )

        self.update()
    def format_previous(self):
        if not self.previous: return "Previous selections:\n(none)"
        lines = ["Previous selections:"]
        for i, (_i, x, y) in enumerate(self.previous, 1):   lines.append(f"{i:>2}: ({x:.3f}, {y:.3f})")
        return "\n".join(lines)
    def update(self):
        self.marker.set_data(
            [self.x[self.idx]],
            [self.y[self.idx]]
        )
        title_text = f"Index {self.idx} | x={self.x[self.idx]:.3f}, y={self.y[self.idx]:.3f}\n"
        if self.title is not None: title_text = self.title
        self.ax.set_title(title_text + "← / → move   Enter select")
        self.fig.canvas.draw_idle()
    def on_key(self, event):
        if event.key == "left":
            self.idx = max(0, self.idx - 1)
            self.update()
        elif event.key == "right":
            self.idx = min(len(self.x) - 1, self.idx + 1)
            self.update()
        elif event.key == "enter":
            # Set Selected
            self.selected = (self.idx, self.x[self.idx], self.y[self.idx])
            self.close()
        elif event.key == "escape":
            # cancel
            self.selected = None
            self.close()
    def on_click(self, event):
        # ignore clicks outside the axes
        if event.inaxes != self.ax: return
        if event.xdata is None: return
        # find nearest x (fast + intuitive)
        self.idx = np.argmin(np.abs(self.x - event.xdata))
        self.update()
    def close(self):
        self.fig.canvas.mpl_disconnect(self.cid)
        plt.close(self.fig)




experiment = Experiment(    'P1', 
                            os.path.join('.','samples','blink_calibration'), 
                            verbose=True    )
experiment.init_sessions(   'eeg.csv', 
                            'left_eye_gaze.csv', 
                            _CAL_FILES, 
                            start_buffer=4000, 
                            end_buffer=0    )
experiment.calculate_offsets(   cache_offsets=True,
                                force_calc=False)
experiment.plot_offsets('conf_threshold')
#for s in experiment.sessions:
#    s.plot_raw_calibrations()

"""
x1 = np.linspace(0, 10, 50)
y1 = np.sin(x1)
x2 = np.linspace(0, 10, 50)
y2 = np.cos(x2)

selections = []
def print_selection(x, y, index, selector):
    print(f"Selected point {index}: ({x:.3f}, {y:.3f})")
    selections.append((index, x, y))

selector1 = PointSelector(x1, y1, (3,3), on_select=print_selection)
plt.show()
selector2 = PointSelector(x2, y2, (3,3), on_select=print_selection)
plt.show()
print(selector1.selected, selector2.selected)
#print("Final selection:", selector.selected)
"""