# IMPORTS
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import shutil
from glob import glob
from tqdm import tqdm
from . import file_handling as fh

# IDENTIFICATION OF VALID PARTICIPANTS
# - Params: 
#   - `root_dir`: Directory path that contains all trial directories as immediate child folders.
#   - `expected_files`: List of filenames expected within each child folder
# - Returns:
#   - `valids`: list of directories that had all expected files
#   - `invalids`: list of directories that did not contain one or more expected files
def identify_valid_participants( root_dir:str, expected:Iterable[str] ):
    root_path = Path(root_dir)  # Initialize paths
    expected = set(expected)
    with_all:List[str] = []     # Initialize outputs
    missing:List[str] = []

    for subdir in root_path.iterdir():      # Iterating through subdirectories

        if not subdir.is_dir(): continue    # Ignore if subdirectory is not a directory
        present = {                         # Get present files in the subdirectory
            p.name for p in subdir.iterdir() 
            if p.is_file()
        }
        missing_files = sorted(expected - present)  # Check if files are within subset of expected
        if expected.issubset(present):  with_all.append(str(subdir))
        else:                           missing.append((str(subdir), len(missing_files), missing_files))

    # Return both
    return with_all, missing


# COPY AND RENAME VALID PARTICIPANT FILES
# - Params:
#   - src_subdirs: a List of participant directories
#   - rename_map: a Dictionary dictating how to rename files
#   - dest_root: the parent directory where participant directories should be saved
# - Returns
#   - List of output participant directories
def copy_and_rename( 
        src_subdirs: Iterable[str],
        rename_map: Dict[str, str],
        dest_root: str ) -> List[str]:

    # Initialize destination directory
    fh.mkdirs(dest_root)
    dest_root = Path(dest_root)

    # Initialize new subdirs list
    new_subdirs:List[str] = []

    # Iterate through existing source dirs
    for src in src_subdirs:
        # Prep copied dir
        src_dir = Path(src)
        dest_dir = dest_root / src_dir.name
        fh.mkdirs(str(dest_dir))
        # Iterate through each file in each source directory
        for item in src_dir.iterdir():
            # Ignore any that are not files
            if not item.is_file(): continue
            # Remaps
            new_name = rename_map.get(item.name, item.name)
            dest_path = dest_dir / new_name
            # Copy
            shutil.copy2(item, dest_path)
        # save new output directory
        new_subdirs.append(str(dest_dir))
    # Return \
    return new_subdirs

def identify_trial_order(pdir:str, trial_names:Iterable[str]):
    df = pd.read_csv(os.path.join(pdir, 'eye.csv'))
    df = df[df["event"].isin(trial_names)]
    df = df[['unix_ms','rel_timestamp','frame','event']]
    trials = df.to_dict(orient="records")
    return trials
def identify_calibration_order(pdir:str):
    df = pd.read_csv(os.path.join(pdir, 'eye.csv'))
    df = df[df["event"] == 'Calibration']
    df = df[['unix_ms','rel_timestamp','frame','event']]
    trials = df.to_dict(orient="records")
    return trials

def plot_calibration_timeline(pdirs:Iterable[str], show:bool=True, outpath:str=None):

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
    if show:                plt.show()