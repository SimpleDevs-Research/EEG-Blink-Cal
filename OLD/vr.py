import pandas as pd

class VR:
    def __init__(self, parent, eye_src:str, imu_src:str):
        self.parent = parent
        self.eye_src = eye_src
        self.imu_src = imu_src
    
    def initialize_from_src(self):
        self.eye = pd.read_csv(self.eye_src)
        self.imu = pd.read_csv(self.imu_src)
    
    def get_subset(self, timestamp_colname:str, start, end, parent=None):
        # Define the subset as the same type as the current muse, also defining a parent if necessary
        if parent is None: parent = self.parent
        subset = type(self)(parent, self.eye_src, self.imu_src)
        # Define the new values
        subset.eye = self.eye[self.eye[timestamp_colname].between(start, end)]
        subset.imu = self.imu[self.imu[timestamp_colname].between(start, end)]
        # Return the subset
        return subset
