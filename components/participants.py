import os
from . import config as CF
from . import muse as MS
from . import vr as VR
from . import session as SS

class Participant:
    def __init__(self, src:str, sessions):
        self.src = src
        self.sessions = sessions

def validate_participant(src:str, p_name:str, config:CF.Config) -> Participant|bool:
    # Given a src for a participant, identify 1) if all sessions are present, and 2) for each, that they have the expected files
    if not os.path.exists(src):
        print(f"Participant \"{p_name}\" ({src}) doesn't exist...")
        return False
    sessions = []
    for s_name in config.expected_sessions:
        s_path = os.path.join(src, s_name)
        if os.path.exists(s_path) and all([os.path.exists(os.path.join(s_path, f)) for f in config.expected_files]):
            # valid session. Prep and cache
            session = SS.read_session(s_path, p_name, s_name, config)
            sessions.append(session)
    valid = len(sessions) > 0
    if not valid: return False
    return Participant(src, sessions)