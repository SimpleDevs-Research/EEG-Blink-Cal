import os
from . import config as CF
from . import muse as MS
from . import vr as VR
from . import calibration as CB
from . import participants as PS


class Trial:
    def __init__(self, src:str, params, calibration:CB.Calibration, muse:MS.Muse, vr:VR.VR):
        self.src = src
        self.params = params
        self.calibration = calibration
        self.muse = muse
        self.vr = vr

def extract_trials_from_participant(p:PS.Participant, config:CF.Config):
    # Participant class has `sessions` property
    #   This `sessions` prop is a list of Participant type class
    #       Each contains a directory path `src` that that includes the experiment's src directory
    #           AKA: No no neeed to os.path.join(). Can use `src` prop from each session directly
    for s in p.sessions:
        trials = []
        # Each session is comprised of individual Trials. You can consider a Trial as the smallest denominator across all sessions and participants
        for t in config.trials:
            # Get elements of this trial
            t_src = os.path.join(s.src, t)
            t_params = {**s.params, 'trial':t, **config.trials[t]}
            # Read the calibration file
            calibration = CB.read_calibration(t_src, config)
            # Based on calibration start and end timestamps, extract the muse and vr data
            #   pertinent only to this trial
            muse = MS.get_subset(s.muse, 'unix_ms', calibration.start_row['unix_ms'], calibration.end_row['unix_ms'])
            vr = VR.get_subset(s.vr, 'unix_ms', calibration.start_row['unix_ms'], calibration.end_row['unix_ms'])
            # Generate trial and append it to `trials`
            trials.append(Trial(t_src, t_params, calibration, muse, vr))
        s.trials = trials
    # Return participant
    return p


