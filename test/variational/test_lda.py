import numpy as np
from lda import lda
from lda.data.corpus import Corpus
from lda.data.document import Document
from lda.data.word import Word
from utils import assert_numerically_ok


def test_lda():
    params, lower_bound_evol = lda(corpus, num_topics=2, num_iterations=32)
    assert set(params.keys()) == {'alpha', 'beta', 'phis', 'gammas'}
    
    assert params['alpha'].shape == (2,)
    assert_numerically_ok(params['alpha'])
    assert np.all(params['alpha'] > 0)

    assert len(params['beta']) == 2
    for beta_row in params['beta']:
        assert len(beta_row) == 4
        assert_numerically_ok(list(beta_row.values()))
        assert set(beta_row.keys()) == {please, crash, grandma, sun}
        assert np.isclose(sum(beta_row.values()), 1)
    
    assert len(params['phis']) == 3
    assert set(params['phis'].keys()) == {doc1, doc2, doc3}
    for document, phi_row in params['phis'].items():
        assert phi_row.shape == (len(document), 2)
        assert_numerically_ok(phi_row)
        assert np.all(np.isclose(phi_row.sum(axis=1), 1))
    
    assert len(params['gammas']) == 3
    assert set(params['gammas'].keys()) == {doc1, doc2, doc3}
    for document, gamma_row in params['gammas'].items():
        assert gamma_row.shape == (2,)
        assert_numerically_ok(gamma_row)
        assert np.all(gamma_row > 0)
        assert np.all(gamma_row > params['alpha'])


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
