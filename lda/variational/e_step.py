import numpy as np
from scipy.special import digamma
from lda.data.document import Document


def e_step(document: Document, alpha: np.ndarray, beta):
    '''
    Parameters:
    * document: a Document object
    * alpha: array of size (num_topics,)
    * beta: beta[topic_id][word] = probability of word in topic
        (word is a Word object here, so beta[topic_id] is a dictionary)
    Returns: {
        'phi': array of size (document_length, num_topics)
        'gamma': array of size (num_topics,),
    }
    '''
    num_topics = len(beta)
    phi = np.ones(shape=(len(document), num_topics)) / num_topics
    gamma = alpha + len(document) / num_topics

    # the paper claims that we can iterate for ~len(document) steps
    for step in range(len(document)):
        for word_idx, word in enumerate(document.included_words):
            for topic in range(num_topics):
                phi[word_idx, topic] = (
                    beta[topic][word]
                    * np.exp(digamma(gamma[topic]))
                )
            phi[word_idx] /= phi[word_idx].sum()
        gamma = alpha + phi.sum(axis=0)
    
    return {
        'phi': phi,
        'gamma': gamma,
    }
