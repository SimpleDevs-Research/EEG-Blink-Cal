# === IMPORT PACKAGES ===
import os
import argparse
import pandas as pd
from tqdm import tqdm


# === EXPLANATION / DOCUMENTATION ===
# The expected file structure of this `_ROOT_DIR` is as follows:
# ------------------------------------------------
# - _ROOT_DIR/
#   - config.json
#   - Participant(s)
#       - Sessions
#           - Src Files, Trial Files
# ------------------------------------------------
# So if you have multiple participants (or multiple samples collected from a single participant),
#   You simply add them as individual subdirectories directly under `_ROOT_DIR` (e.g. `P1/`, `P2/`)
# Each Participant directory has subdirectories of their own - each representing a Session
#   And each Session contains 1) "Src" files (data files from Muse and Meta Quest Pro) and 2) "Trial" files
#   Each trial file is in of itself its own calibration stage.
#   When we refer to "Calbiration" or "Trial", they are referring to the same thing.
# ------------------------------------------------
# The `_CONFIG_FILENAME` file provides the EXPECTED sessions and their various parameters, for each participant.
#   It's expected that this is a JSON array file
# ------------------------------------------------
# Each Session expects the existence of certain files. This includes 1) "Src" files and 2) "Trial" files
#   Both "Src" and "Trial" files are always expected. Any one is missing, and we reject the session.
#   "Src" files are raw data provided by Muse 2 / Mind Monitor and the Meta Quest Pro
#   "Trial" files are the list of files that contain timestamps for each calibration.
#   

# ===== STEP 0: COMMAND-LINE ARGUMENTS =====
parser = argparse.ArgumentParser(description="Please provide a root directory and expected filename for the config file")
parser.add_argument("root_dir", help="Root directory where all participant data is stored.", type=str)
parser.add_argument("-c", "--config", help="Inside root dir, the expected config JSON file to read.", type=str, default="config.json")
args = parser.parse_args()


# ===== STEP 1: READING CONFIGURATION ======
import os
from components import config as CF
# ----------
config_src = os.path.join(args.root_dir, args.config)
config = CF.Config(config_src)


# ===== STEP 2: READ AND VALIDATE PARTICIPANT FILES =====
from components import participants as PS
# ----------
pbar = tqdm(config.participants)
participants = []
for p in pbar:
    _p = PS.validate_participant(os.path.join(args.root_dir, p), p, config)
    if _p: participants.append(_p)
assert len(participants)>0, "NO VALID PARTICIPANTS. EXITING EARLY"
print(f"{len(participants)} VALID PARTICIPANT(S) FOUND")


# ===== STEP 3: TRIALS FROM PARTICIPANTS =====
from components import trials as TS
pbar = tqdm(participants)
participant_trials = []
for p in pbar:
    participant_trials.append(TS.extract_trials_from_participant(p, config))

# ===== INTERMISSION 1: PLOTTING RAWS FOR VALIDATION
from components import plots as PL
pbar = tqdm(participant_trials)
for p in pbar:
    PL.plot_participant(p, config, outname="raws.png")

exit()



# ===== STEP 3: EXTRACT TRIALS FROM PARTICIPANTS =====
# Purely a raw data container
class Trial:
    def __init__(self, start, end, calibration, muse_eeg, muse_imu, vr_eye, vr_imu):
        self.start = start
        self.end = end
        self.calibration = calibration
        self.muse_eeg = muse_eeg
        self.muse_imu = muse_imu
        self.vr_eye = vr_eye
        self.vr_imu = vr_imu

class Muse2:
    def __init__(self, src:str, initialize_from_src:bool=True):
        self.src = src
        if initialize_from_src: 
            self.initialize_from_src()
    def initialize_from_src(self):
        df = pd.read_csv(src)
        df['unix_ms'] = df['TimeStamp'].apply(h.timestamp_to_unix_milliseconds)
        df = df.rename(columns=_MUSE_REMAPPINGS)
        self.initialize_from_df(df)
    def initialize_from_df(self, df):
        self.df = df.sort_values('unix_ms')
        signals = df[df['Elements'].isna()]
        blinks = df[df['Elements']=='/muse/elements/blink']
        self.blinks = blinks[['TimeStamp', 'unix_sec', 'unix_ms', 'rel_sec', 'rel_ms']]    
        self.raw_eeg = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_MUSE_RAW_COLNAMES]]
        self.processed_eeg = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_MUSE_PROCESSED_COLNAMES]]
        self.imu = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_MUSE_IMU_COLNAMES]]
    def downsample_from_df(self, analyze:bool=True):
        # Downsample
        df = self.df.groupby('unix_ms', as_index=False).last()
        # Analysis
        if analyze:
            counts = self.df.groupby('unix_ms').size()
            dist = counts.value_counts().sort_index()        
            hz = self.estimate_sample_rate(self.df, 'unix_ms', is_milli=True)
            n = len(raw_df.index)
            n2 = len(df.index)
            print("\tGroup by `unix_ms` counts:", sparkline(dist.values))
            print(f"\tEstimated Sample Rate (Pre-Downsampling):", hz)
            print(f"\tDownsampling: {n} -> {n2} ({n2/n}% retention rate)")
        # Generate
        downsampled = (type(self))(self.src)
        downsampled.initialize_from_df(df)
        

    @staticmethod
    def estimate_sample_rate(df, verbose:bool=True):
        counts = raw_df.groupby('unix_ms').size()
        dist = counts.value_counts().sort_index()        
        hz = estimate_sample_rate(raw_df, 'unix_ms', is_milli=True)
        if verbose:
            print(f"\tEstimated Sample Rate (Pre-Downsampling):", hz)
        return hz, dist

def read_muse(src:str):
    # Reading and Relabling
    df = pd.read_csv(src)
    df['unix_ms'] = df['TimeStamp'].apply(h.timestamp_to_unix_milliseconds)
    df = df.rename(columns=_MUSE_REMAPPINGS)
    df = df.sort_values('unix_ms')
    
    # Separation
    signals = df[df['Elements'].isna()]
    blinks = df[df['Elements']=='/muse/elements/blink']
    blinks = blinks[['TimeStamp', 'unix_sec', 'unix_ms', 'rel_sec', 'rel_ms']]    
    raw_eeg = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_MUSE_RAW_COLNAMES]]
    processed_eeg = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_MUSE_PROCESSED_COLNAMES]]
    imu = signals[['unix_sec','unix_ms', 'rel_sec', 'rel_ms', *_MUSE_IMU_COLNAMES]]

    # Downsampling & Sample Rate Estimation
    counts = raw_df.groupby('unix_ms').size()
    dist = counts.value_counts().sort_index()        
    hz = estimate_sample_rate(raw_df, 'unix_ms', is_milli=True)
    df = raw_df.groupby('unix_ms', as_index=False).last()
    n = len(raw_df.index)
    n2 = len(df.index)
    print("\tGroup by `unix_ms` counts:", sparkline(dist.values))
    print(f"\tEstimated Sample Rate (Pre-Downsampling):", hz)
    print(f"\tDownsampling: {n} -> {n2} ({n2/n}% retention rate)")

for p in participants:
    for s in p['sessions']:
        spath = os.path.join(p['src'], s)
        # already validated the existence of necesary files
        muse_df = pd.read_csv(os.path.join(spath, config.files['muse']['filename']))

    print(p)

#print(participants)
exit()







_TRIAL_FILES = [        # The expected files in each session representing a trial
    "0.25-72.csv",
    "0.25-90.csv",
    "0.75-72.csv",
    "0.75-90.csv"
]
_TRIAL_BOUNDS = {
    "start_buffer":5000,    # How much buffer do you want to ignore at the beginning of a calibration trial?
    "end_buffer":1000       # How much buffer do you want to ignore at the end of a calibration trial?
}

# ARG PARSER
parser = argparse.ArgumentParser()
parser.add_argument("root_dir", help="The root directory where all experiment files are located", type=str)
parser.add_argument("-cf", "--config_filename", help="The name of the config.json file used to configurate our experiment.", type=str, default="config.json")
args = parser.parse_args()



class Experiment:
    def __init__(self, root_dir:str, config_filename:str):

        self.root_dir = root_dir

        # 




        config_filepath = os.path.join(self.root_dir, config_filename)
        assert os.path.exists(config_filepath), "Configuration file not found."
        with open(config_filepath) as f:
            self.config = json.load(f)
        
        # validate each directory.
        self.sessions = []
        for session in self.config['sessions']:
            for d in session['dirs']:
                _d = os.path.join(self.root_dir, d)
                assert os.path.exists(_d), f"\"{d}\" doesn't exist."
                for t in self.config['trials']:     assert os.path.exists(os.path.join(_d, t)), f"Trial \"{t}\" doesn't exist in \"{d}\""
                for e in self.config['expected']:   assert os.path.exists(os.path.join(_d, e)), f"File \"{e}\" doesn't exist in \"{d}\""
                s = Session(self, _d, d, session['params'])
                self.sessions.append(s)
            print(f"Session \"{session['name']}\" is valid")
        for s in self.sessions: print(s.root_dir)

class Session:
    def __init__(self, parent:Experiment, root_dir:str, sname:str, session_params):
        self.parent = parent
        self.root_dir = root_dir
        self.sname = sname
        self.params = session_params
        # Read EEG data
        self.muse = Muse(self, os.path.join(self.root_dir, self.parent.config['args']['muse']))
        self.muse.initialize_from_src()
        #self.muse.remove_duplicates(inplace=True)
        self.vr = VR(self, 
                    os.path.join(self.root_dir, self.parent.config['args']['eye']),
                    os.path.join(self.root_dir, self.parent.config['args']['head_imu']) 
        )
        self.vr.initialize_from_src()
        self.trials = [Trial(self, Path(t).stem, os.path.join(self.root_dir, t)) for t in self.parent.config['trials']]
        #for t in self.trials: 
        #    print(t.muse.get_sample_rates(rolling=True))
            #print(t.tname, t.params, t.muse.get_sample_rates(rolling=True), len(t.eye.df))
    def get_buffers(self):
        start_buffer, end_buffer = 5000, 1000
        if "start_buffer" in self.parent.config['args']: start_buffer = self.parent.config['args']['start_buffer']
        if "end_buffer" in self.parent.config['args']: end_buffer = self.parent.config['args']['end_buffer']
        return start_buffer, end_buffer

class Trial:
    def __init__(self, parent:Session, tname:str, src:str):
        self.parent = parent
        self.tname = tname
        self.src = src
        # Derive params from both parent and from tname
        self.params = dict(self.parent.params)
        tp = self.tname.split("-")
        self.params['conf_threshold'] = tp[0]
        self.params['fps'] = tp[1]
        self.df = self.correct_calibration(pd.read_csv(self.src), self.parent.parent.config['calibration_columns'])
        self.start = self.df.loc[self.df["event"] == "Start"].iloc[0]
        self.end = self.df.loc[self.df["event"] == "End"].iloc[0]
        self.overlaps = self.df[self.df['event'] == 'Overlap']
        # Modify start and end times based on buffers received form parent session
        start_buffer, end_buffer = self.parent.get_buffers()
        self.start['unix_ms'] += start_buffer
        self.end['unix_ms'] -= end_buffer
        # Get subsets of the Muse and Eye data. For now, we set the parent to this trial <* CHECK LATER *>
        self.muse = self.parent.muse.get_subset('unix_ms', self.start['unix_ms'], self.end['unix_ms'], self)
        self.vr = self.parent.vr.get_subset('unix_ms', self.start['unix_ms'], self.end['unix_ms'], self)

    def plot(self, parent_spec=None):
        if parent_spec is None:
            # This is a standalone plot. Generate our own spec
            fig = plt.figure(figsize(30,3))
            spec = fig.add_gridspec(1,1)[0]
        else:
            fig = parent_spec.get_gridspec().figure
            spec = parent_spec
        # Define our subplot within our spec, which is four rows and one column subplots
        gs = GridSpecFromSubplotSpec(4, 1, subplot_spec=spec, hspace=0.1 )
        # Define our axes
        ax_muse_eeg = fig.add_subplot(gs[0], sharex=True)
        ax_muse_imu = fig.add_subplot(gs[1], sharex=True)
        ax_vr_eye = fig.add_subplot(gs[2], sharex=True)
        ax_vr_imu = fig.add_subplot(gs[3], sharex=True)
        # Plot
        muse_eeg_colname = self.parent.parent.config['args']['muse_eeg_colname']
        ax_muse_eeg.plot(self.muse.raw_eeg['unix_ms'], self.muse.raw_eeg[muse_eeg_colname], color='lightblue', label=f"Muse {muse_eeg_colname}")
        ax_muse_eeg.tick_params(axis="x", which="both", bottom=False, labelbottom=False )
        ax_muse_eeg.legend()
        muse_imu_colname = self.parent.parent.config['args']['muse_imu_colname']
        ax_muse_imu.plot(self.muse.imu['unix_ms'], self.muse.imu[muse_imu_colname], color='blue', label=f"Muse {muse_imu_colname}")
        ax_muse_imu.tick_params(axis="x", which="both", bottom=False, labelbottom=False )
        ax_muse_imu.legend()
        vr_eye_colname = self.parent.parent.config['args']['eye_colname']
        ax_vr_eye.plot(self.vr.eye['unix_ms'], self.vr.eye[vr_eye_colname], color='red', label=f"VR {vr_eye_colname}")
        ax_vr_eye.tick_params(axis="x", which="both", bottom=False, labelbottom=False )
        ax_vr_eye.legend()
        vr_imu_colname = self.parent.parent.config['args']['head_imu_colname']
        ax_vr_imu.plot(self.vr.imu['unix_ms'], self.vr.imu[vr_imu_colname], color='black', label=f"VR {vr_imu_colname}")
        ax_vr_imu.legend()


    @staticmethod
    def correct_calibration(df:pd.DataFrame, colnames):
        fixed = df.iloc[:, :len(colnames)]
        fixed.columns = colnames
        return fixed

experiment = Experiment(args.root_dir, args.config_filename)