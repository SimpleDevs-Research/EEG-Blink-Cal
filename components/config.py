import json

class Config:
    def __init__(self, src:str):
        # read config
        with open(src) as f:
            config = json.load(f)
            # Save immediate keys in config as classs properties
            for key in config:
                self.__dict__[key] = config[key]
            if "start_buffer" not in config:
                self.start_buffer = None
            if "end_buffer" not in config:
                self.end_buffer = None
            # Extract expected files
            self.expected_sessions = list(self.sessions.keys())
            self.expected_files = [self.files[key]['filename'] for key in self.files] + list(self.trials.keys())