# ========================
# This script focuses on identifying session and trial files
# ========================

# === PACKAGES ===
import os

# === CONFIGURATION ===
# NOTE: YOU CAN EDIT THESE FOR YOUR OWN PREFERENCES
_EXPECTED = [               # We expect these to be inside each individual session
    "head_imu.csv",
    "left_eye_gaze.csv",
    "right_eye_gaze.csv",
    "muse.csv",
    "0.25-72.csv",
    "0.25-90.csv",
    "0.75-72.csv",
    "0.75-90.csv"
]
_MAPPINGS = {
    "muse": "muse.csv",
    "vr_imu": "head_imu.csv",
    "vr_eye": "left_eye_gaze.csv",
    "trials": [
        "0.25-72.csv",
        "0.25-90.csv",
        "0.75-72.csv",
        "0.75-90.csv"
    ]
}

# Given a root directory, we check if all the expected files are present or not
class FileChecker:
    def __init__(self, root_dir:str):
        self.root_dir = root_dir
        # validate existence of files
        for f in _EXPECTED: 
            assert os.path.exists(os.path.join(self.root_dir, f)), f"{os.path.basename(self.root_dir)}: \"{f}\" not found"
        # Upon validation, apply mapping for easy reference
        for key in _MAPPINGS:
            self.__dict__[key] = _MAPPINGS[key]