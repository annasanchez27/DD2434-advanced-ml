import itertools
import numpy as np
from scipy.special import digamma
from lda.data.corpus import Corpus
from lda.data.document import Document


def calculate_e_step(corpus: Corpus, alpha: np.ndarray, beta):
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
        document: document_phi_gamma(
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


def document_phi_gamma(document: Document, alpha: np.ndarray, beta):
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

    for step in itertools.count():
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

#We need to calculate phis and gammas until convergence.
#Once we have found the optimal parameters, lambda can be calculated.
def calculate_lambdas(corpus: Corpus,alpha: np.ndarray, beta,eta: float):
    '''
    Parameters:
    * Corpus: a Corpus object
    * alpha: array of size (num_topics,)
    * beta: beta[topic_id][word] = probability of word in topic
        (word is a Word object here, so beta[topic_id] is a dictionary)
    * eta: float

    Returns: {
        'lambda': array of size (num_topics, vocab),
    }
    '''


    num_topics = len(beta)
    vocab = corpus.vocabulary
    lambda_vi = np.zeros(shape=(num_topics, vocab))
    params = calculate_e_step(corpus,alpha,beta)
    for i in range(num_topics):
        for j in range(len(vocab)):
            update_lambda(lambda_vi,i,j,params,eta)
    return lambda_vi


def update_lambda(lambda_vi, i, j,corpus,params,eta):
    vocab = corpus.vocabulary
    for document_idx, document in enumerate(corpus.documents):
        for word_idx, word in enumerate(document.included_words):
            if word == vocab[j]:
                lambda_vi[i][j] += params['phis'][document][word_idx][i]
    lambda_vi[i][j] += eta
    return lambda_vi