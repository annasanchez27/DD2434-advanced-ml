import itertools
import numpy as np
from scipy.special import digamma
from lda.data.corpus import Corpus
from lda.data.document import Document
from .constants import E_STEP_MAX_ITERS


def e_step(corpus: Corpus, alpha: np.ndarray, beta):
    '''
    Parameters:
    * corpus: a Corpus object
    * alpha: array of size (num_topics,)
    * beta: beta[topic_id][word] = probability of word in topic
        (word is a Word object here, so beta[topic_id] is a dictionary)
    Returns: {
        'phis': {document: array of size (document_length, num_topics)},
        'gamma': {document: array of size (num_topics,)},
    }
    '''
    params = {
        document: document_e_step(
            document=document,
            alpha=alpha,
            beta=beta,
        )
        for document in corpus.documents
    } # {doc1: {'phi': 3, 'gamma': 2}, doc2: {...}}
    return {
        'phis': {
            doc: doc_params['phi']
            for doc, doc_params in params.items()
        },
        'gammas': {
            doc: doc_params['gamma']
            for doc, doc_params in params.items()
        }
    }


def document_e_step(document: Document, alpha: np.ndarray, beta):
    '''
    Parameters:
    * document: a Document object
    * alpha: array of size (num_topics,)
    * beta: beta[topic_id][word] = probability of word in topic
        (word is a Word object here, so beta[topic_id] is a dictionary)
    Returns: {
        'phi': array of size (document_length, num_topics),
        'gamma': array of size (num_topics,),
    }
    '''
    num_topics = len(beta)
    phi = np.ones(shape=(len(document), num_topics)) / num_topics
    gamma = alpha + len(document) / num_topics

    for step in range(E_STEP_MAX_ITERS):
        for word_idx, word in enumerate(document.included_words):
            for topic in range(num_topics):
                phi[word_idx, topic] = (
                    beta[topic][word]
                    * np.exp(digamma(gamma[topic]))
                )
            phi[word_idx] /= phi[word_idx].sum()
        new_gamma = alpha + phi.sum(axis=0)
        if np.all(np.isclose(gamma, new_gamma)):
            break
        gamma = new_gamma
    
    return {
        'phi': phi,
        'gamma': gamma,
    }
