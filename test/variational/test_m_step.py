import numpy as np
from lda.data.corpus import Corpus
from lda.data.document import Document
from lda.data.word import Word
from lda.variational.m_step import beta_update


def test_beta():
    beta = beta_update(corpus=corpus, phis=phis)
    assert len(beta) == 2
    assert beta == [
        {please: 1, crash: 0, grandma: 0},
        {please: 0, crash: .5, grandma: .5},
    ]


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
phis = {
    doc1: np.array([[1, 0], [0, 1]]),
    doc2: np.array([[1, 0], [0, 1]]),
}
