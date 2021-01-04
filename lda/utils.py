import numpy as np
from scipy.special import polygamma
from scipy.special import digamma


class np_seed:
    def __init__(self, seed):
        self.seed = seed
    

    def __enter__(self):
        self.old_state = np.random.get_state()
        np.random.seed(self.seed)
    

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.old_state)


def guarded_polygamma(x):
    '''Computes polygamma(1, x), but makes sure that x isn't numerically bonkers'''
    assert np.all(x > 0)
    return polygamma(1, x)

def guarded_digamma(x):
    '''Computes polygamma(1, x), but makes sure that x isn't numerically bonkers'''
    assert np.all(x > 0)
    return digamma(x)
