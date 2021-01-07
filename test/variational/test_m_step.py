import numpy as np
from lda.data.corpus import Corpus
from lda.data.document import Document
from lda.data.word import Word
from lda.variational import m_step
from lda.variational.m_step.alpha_update import alpha_update
from lda.variational.m_step.beta_update import beta_update
from utils import assert_numerically_ok


def test_m_step():
    parameters = m_step(corpus=corpus, alpha=alpha_initial, phis=phis, gammas=gammas)
    assert set(parameters.keys()) == {'alpha', 'beta'}


def test_alpha():
    alpha = alpha_update(
        alpha=alpha_initial,
        gammas=gammas
    )
    assert alpha.shape == alpha_initial.shape
    assert_numerically_ok(alpha)
    assert np.all(alpha > 0)


def test_beta():
    beta = beta_update(corpus=corpus, phis=phis)
    assert beta.shape == (2, 3)
    assert np.all(np.isclose(
        beta,
        [
            [1, 0, 0],
            [0, .5, .5],
        ]
    ))


please = Word('Please', 'please', include=True)
crash = Word('crash', 'crash', include=True)
grandma = Word('grandma', 'grandma', include=True)
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
corpus = Corpus(documents=[doc1, doc2])
alpha_initial = np.array([0.7, 1.2])
phis = {
    doc1: np.array([[1, 0], [0, 1]]),
    doc2: np.array([[1, 0], [0, 1]]),
}
gammas = {
    doc1: np.array([0.9, 1.8]),
    doc2: np.array([1.5, 1.3]),
}
