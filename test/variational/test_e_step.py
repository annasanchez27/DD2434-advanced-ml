from lda.data.document import Document
from lda.data.word import Word
from lda.variational import e_step
import numpy as np


def test_output_shape():
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
    parameters = e_step(document=document, alpha=alpha, beta=beta)
    assert parameters['phi'].shape == (2, 2)
    assert parameters['gamma'].shape == (2,)
