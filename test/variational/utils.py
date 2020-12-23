import numpy as np


def assert_numerically_ok(array):
    assert np.all(~np.isnan(array))
    assert np.all(~np.isinf(array))
