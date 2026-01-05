import os
from . import config as CF
from . import muse as MS
from . import vr as VR

class Session:
    def __init__(self, src:str, params, muse:MS.Muse, vr:VR.VR):
        self.src = src
        self.params = params
        self.muse = muse
        self.vr = vr
        self.trials = None

def read_session(src:str, p_name:str, s_name:str, config:CF.Config) -> Session:
    # Prepare params
    params = config.sessions[s_name]
    params['participant'] = p_name
    params['session'] = s_name
    # Read muse
    muse_src = os.path.join(src, config.files['muse']['filename'])
    muse = MS.read_muse(muse_src)
    # Read VR
    vr_imu_src, vr_eye_src = os.path.join(src, config.files['vr_imu']['filename']), os.path.join(src, config.files['vr_eye']['filename'])
    vr = VR.read_vr(vr_imu_src, vr_eye_src)
    # Return a new Session
    return Session(src, params, muse, vr)