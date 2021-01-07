<<<<<<< HEAD
=======
from lda.data.corpus import Corpus
>>>>>>> 745375531ad5b7ae49dcd52d443f79d110aee2b9
from lda.data.document import Document
from lda.data.word import Word
from lda.variational.e_step import e_step, document_e_step
import numpy as np


<<<<<<< HEAD
def test_document_output_shape():
    please = Word('Please', 'please', include=True)
    crash = Word('crash', 'crash', include=True)
    from_another_doc = Word('grandma', 'grandma', include=True)
    document = Document(
        words=[
            please,
            Word('do', 'do', include=False),
            Word('not', 'not', include=False),
            crash,
            Word('.', '.', include=False),
        ]
    )
    alpha = np.array([2, 0.7])
    beta = [
        {please: 0.7, crash: 0.2, from_another_doc: 0.1},
        {please: 0.2, crash: 0.6, from_another_doc: 0.2},
    ]
    parameters = document_e_step(document=document, alpha=alpha, beta=beta)
    assert parameters['phi'].shape == (2, 2)
    assert parameters['gamma'].shape == (2,)
=======
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
>>>>>>> 745375531ad5b7ae49dcd52d443f79d110aee2b9
