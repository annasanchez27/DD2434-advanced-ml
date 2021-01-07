from typing import Dict
import numpy as np
from lda.data.corpus import Corpus
from lda.data.document import Document
from .alpha_update import alpha_update
from .beta_update import beta_update


def m_step(corpus: Corpus, alpha: np.ndarray, phis: Dict[Document, np.ndarray], gammas: Dict[Document, np.ndarray]):
    '''
    Parameters:
    * corpus: a Corpus object
    * alpha: initial guess for alpha - we return an updated version
    * phis: {document: array of size (document_length, num_topics)}
        (document is a Document object, so phis is a dictionary)
    * gammas: {document: array of size (num_topics,)}
        (document is a Document object, so gammas is a dictionary)
    Returns: {
        'alpha': array of size (num_topics,)
<<<<<<< HEAD
        'beta': beta[topic_id][word] = probability of word in topic
            (word is a Word object, so beta[topic_id] is a dictionary)
=======
        'beta': beta[topic_id][word_id] = probability of word in topic
>>>>>>> 745375531ad5b7ae49dcd52d443f79d110aee2b9
    }
    '''
    return {
        'alpha': alpha_update(alpha=alpha, gammas=gammas),
        'beta': beta_update(corpus=corpus, phis=phis),
    }


