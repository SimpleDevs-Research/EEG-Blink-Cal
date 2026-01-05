import os
import matplotlib.pyplot as plt
import pandas as pd

def eye_calibration_timeline(_PID, _EYE_F, _CALIBRATION_FS, show:bool=True, outpath:str=None):
    
    # Get the eye dataframe
    eye_df = pd.read_csv(_EYE_F)

    # --- Get overall time range ---
    start_time = eye_df["unix_ms"].min()
    end_time = eye_df["unix_ms"].max()

    # --- Collect first timestamps from small files ---
    markers = []
    for f in _CALIBRATION_FS:
        df = pd.read_csv(f)
        if "unix_ms" not in df.columns:
            continue  # skip if missing
        first_time = df["unix_ms"].iloc[0]
        markers.append((f"{os.path.basename(f)}\n{first_time}", first_time))

    # --- Plot ---
    plt.figure(figsize=(12, 4))
    plt.plot(eye_df["unix_ms"], [0]*len(eye_df), alpha=0.2, label="Eye dataset timeline")

    # Add vertical markers for each smaller file
    for name, t in markers:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.7)
        plt.text(t, 0.1, name, rotation=90, verticalalignment='bottom', fontsize=8)

    # Add a vertical marker for the start and ends too.
    plt.axvline(x=start_time, color='blue', linestyle='--', alpha=0.7)
    plt.text(start_time, 0.1, f"Eye Start\n{start_time}", rotation=90, verticalalignment='bottom', fontsize=7.5)

    # Render other stuff
    plt.xlim(start_time, end_time)
    plt.xlabel("Unix time (ms)")
    plt.yticks([])
    plt.title(f"Participant {_PID}: Timeline of Eye Data to Calibrations:")
    plt.legend()
    plt.tight_layout()
    if outpath is not None: plt.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:                plt.show()