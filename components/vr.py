import pandas as pd

class VR:
    def __init__(self, imu_src:str, imu:pd.DataFrame, eye_src:str, eye:pd.DataFrame):
        self.imu_src = imu_src
        self.imu = imu
        self.eye_src = eye_src
        self.eye = eye
    
def read_vr(imu_src:str, eye_src:str) -> VR:
    imu = pd.read_csv(imu_src)
    eye = pd.read_csv(eye_src)
    return VR(imu_src, imu, eye_src, eye)

def get_subset(vr:VR, timestamp_colname:str, start:int|float, end:int|float) -> VR:
    imu = vr.imu[vr.imu[timestamp_colname].between(start,end)]
    eye = vr.eye[vr.eye[timestamp_colname].between(start,end)]
    return VR(vr.imu_src, imu, vr.eye_src, eye)