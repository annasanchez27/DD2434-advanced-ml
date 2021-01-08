import itertools
import numpy as np
from lda.utils import normalize, guarded_digamma
from lda.data.corpus import Corpus
from lda.data.document import Document
from .constants import E_STEP_MAX_ITERS


def e_step(corpus: Corpus, alpha: np.ndarray, beta):
    '''
    Parameters:
    * corpus: a Corpus object
    * alpha: array of size (num_topics,)
    * beta: beta[topic_id][word_id] = probability of word in topic
    Returns: {
        'phis': {document: array of size (document_length, num_topics)},
        'gamma': {document: array of size (num_topics,)},
    }
    '''
    params = {
        document: document_e_step(
            corpus=corpus,
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


def document_e_step(corpus: Corpus, document: Document, alpha: np.ndarray, beta):
    '''
    Parameters:
    * document: a Document object
    * alpha: array of size (num_topics,)
    * beta: beta[topic_id][word_id] = probability of word in topic
    Returns: {
        'phi': array of size (document_length, num_topics),
        'gamma': array of size (num_topics,),
    }
    '''
    num_topics = len(beta)
    phi = np.ones(shape=(len(document), num_topics)) / num_topics
    gamma = alpha + len(document) / num_topics

    for step in range(E_STEP_MAX_ITERS):
        phi = (
            beta[:, corpus.vocabulary_indices[document]].transpose() # (document_length, num_topics)
            * np.exp(guarded_digamma(gamma)) # (num_topics,)
        )
        assert np.all(~np.isnan(phi))
        phi = normalize(phi, axis=1)
        assert np.all(np.isclose(phi.sum(axis=1), 1))
        new_gamma = alpha + phi.sum(axis=0)
        if np.all(np.isclose(gamma, new_gamma)):
            break
        gamma = new_gamma
    
    return {
        'phi': phi,
        'gamma': gamma,
    }
