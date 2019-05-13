

class BaseDistrib(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, size):
        raise NotImplementedError
