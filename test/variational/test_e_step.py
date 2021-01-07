from lda.data.corpus import Corpus
from lda.data.document import Document
from lda.data.word import Word
from lda.variational.e_step import e_step, document_e_step
import numpy as np


def test_document_e_step():
    alpha = np.array([2, 0.7])
    beta = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2],
    ])
    parameters = document_e_step(corpus=corpus, document=document, alpha=alpha, beta=beta)
    assert parameters['phi'].shape == (2, 2)
    assert np.all(np.isclose(
        parameters['phi'],
        np.array([[0.9349707, 0.0650293], [0.577935 , 0.422065]])
    ))
    
    assert parameters['gamma'].shape == (2,)
    assert np.all(np.isclose(
        parameters['gamma'],
        np.array([3.51289496, 1.18710504])
    ))


please = Word('Please', 'please', include=True)
crash = Word('crash', 'crash', include=True)
document = Document(
    words=[
        please,
        Word('do', 'do', include=False),
        Word('not', 'not', include=False),
        crash,
        Word('.', '.', include=False),
    ]
)
corpus = Corpus(documents=[document])
