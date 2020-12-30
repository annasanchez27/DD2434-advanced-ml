import numpy as np


class np_seed:
    def __init__(self, seed):
        self.seed = seed
    

    def __enter__(self):
        self.old_state = np.random.get_state()
        np.random.seed(self.seed)
    

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.old_state)
