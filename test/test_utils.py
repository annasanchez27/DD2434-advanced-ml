import numpy as np
from lda.utils import np_seed, normalize


def test_np_seed():
    with np_seed(0):
        assert np.random.randint(0, 256) == 172
    with np_seed(0):
        with np_seed(123):
            assert np.random.randint(0, 256) == 254
            assert np.random.randint(0, 256) == 109
        assert np.random.randint(0, 256) == 172


def test_normalize():
    assert np.all(np.isclose(
        normalize(np.array([0, 5, 3, 2])),
        np.array([0, .5, .3, .2])
    ))
    assert np.all(np.isclose(
        normalize(np.array([[0, 1], [2, 3]])),
        np.array([[0, 1], [.4, .6]])
    ))
    assert np.all(np.isclose(
        normalize(np.array([[0, 1], [2, 3]]), axis=0),
        np.array([[0, .25], [1, .75]])
    ))
