import numpy as np
from lda.utils import np_seed


def test_np_seed():
    with np_seed(0):
        assert np.random.randint(0, 256) == 172
    with np_seed(0):
        with np_seed(123):
            assert np.random.randint(0, 256) == 254
            assert np.random.randint(0, 256) == 109
        assert np.random.randint(0, 256) == 172
