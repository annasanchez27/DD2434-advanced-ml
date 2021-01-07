import numpy as np
from lda import lda
from lda.data.corpus import Corpus
from lda.data.document import Document
from lda.data.word import Word
from utils import assert_numerically_ok
import lda.utils as utils


def test_lda():
    with utils.np_seed(62):
        out = lda(corpus, num_topics=2, num_iterations=32)
        params = out['params']
        lower_bound_evol = out['lower_bound_evol']
    assert set(params.keys()) == {'alpha', 'beta', 'phis', 'gammas'}
    
    assert params['alpha'].shape == (2,)
    assert_numerically_ok(params['alpha'])
    assert np.all(params['alpha'] > 0)

    assert params['beta'].shape == (2, 4)
    assert_numerically_ok(params['beta'])
    assert np.all(np.isclose(params['beta'].sum(axis=1), 1))
    
    assert len(params['phis']) == 3
    assert set(params['phis'].keys()) == {doc1, doc2, doc3}
    for document, phi in params['phis'].items():
        assert phi.shape == (len(document), 2)
        assert_numerically_ok(phi)
        assert np.all(np.isclose(phi.sum(axis=1), 1))
    
    assert len(params['gammas']) == 3
    assert set(params['gammas'].keys()) == {doc1, doc2, doc3}
    for document, gama in params['gammas'].items():
        assert gama.shape == (2,)
        assert_numerically_ok(gama)
        assert np.all(gama > 0)
        assert np.all(gama >= params['alpha'])
    
    assert np.all(lower_bound_evol[1:] - lower_bound_evol[:-1] >= 0)


please = Word('Please', 'please', include=True)
crash = Word('crash', 'crash', include=True)
grandma = Word('grandma', 'grandma', include=True)
sun = Word('sun', 'sun', include=True)
doc1 = Document(
    words=[
        please,
        Word('do', 'do', include=False),
        Word('not', 'not', include=False),
        crash,
        Word('.', '.', include=False),
    ]
)
doc2 = Document(
    words=[
        please,
        Word('do', 'do', include=False),
        Word('not', 'not', include=False),
        grandma,
        Word('.', '.', include=False),
    ]
)
doc3 = Document(
    words=[
        grandma,
        grandma,
        sun,
    ]
)
corpus = Corpus(documents=[doc1, doc2, doc3])
