import numpy as np
from tqdm.auto import trange
from lda.data.corpus import Corpus
from .e_step import e_step
from .m_step import m_step
from scipy.special import loggamma, digamma

def lda(corpus: Corpus, num_topics=64, num_iterations=1024):
    '''
    Parameters:
    * corpus: a Corpus object
    * num_topics: number of topics :)
    * num_iterations: go see a doctor
    Returns: {
        'alpha': array of size (num_topics,)
        'beta': beta[topic_id][word] = probability of word in topic
            (word is a Word object, so beta[topic_id] is a dictionary)
        'phis': {document: array of size (document_length, num_topics)}
            (document is a Document object, so phis is a dictionary)
        'gammas': {document: array of size (num_topics,)}
            (document is a Document object, so gammas is a dictionary)
    }
    '''
    lower_bound_evol = []
    vocab = corpus.vocabulary
    params = {
        'alpha': np.random.uniform(size=num_topics),
        'beta': [
            {
                word: prob
                for word, prob in zip(vocab, random_categorical_distribution(len(vocab)))
            }
            for topic in range(num_topics)
        ]
    }
    for iteration in trange(num_iterations):
        variational_parameters = {
            document: e_step(
                document=document,
                alpha=params['alpha'],
                beta=params['beta']
            )
            for document in corpus.documents
        } # {doc1: {'phi': 3, 'gamma': 2}, doc2: {...}}
        params.update({
            'phis': {
                document: variational['phi']
                for document, variational in variational_parameters.items()
            },
            'gammas': {
                document: variational['gamma']
                for document, variational in variational_parameters.items()
            }
        })
        params.update(m_step(
            corpus=corpus,
            alpha=params['alpha'],
            phis=params['phis'],
            gammas=params['gammas']
        ))
        lower_bound_evol.append(corpus_lower_bound(
            corpus=corpus, 
            alpha=params["alpha"],
            beta=params["beta"],
            phis=params['phis'], 
            gammas=params["gammas"]
        ))
    return params, np.array(lower_bound_evol)


def corpus_lower_bound(corpus, alpha, beta, phis, gammas):
    return sum(
        document_lower_bound(
            corpus=corpus,
            document=document,
            alpha=alpha,
            beta=beta,
            phi=phis[document],
            gamma=gammas[document]
        )
        for document in corpus.documents
    )


def document_lower_bound(corpus, document, alpha, beta, phi, gamma):
    '''Eq. 15 on the paper. Lower bound to maximize for a document'''
    return (
        loggamma(np.sum(alpha)) - np.sum(loggamma(alpha))
        + np.sum((alpha-1)*(digamma(gamma)-digamma(np.sum(gamma))))
        + sum(
            np.sum(phi[n, :] * (digamma(gamma)-digamma(np.sum(gamma))))
            for n in range(phi.shape[0])
        )
        + sum(
            phi[word_idx, topic] * np.log(beta[topic][vocab_word])
            for word_idx, document_word in enumerate(document.included_words)
            for topic in range(alpha.shape[0])
            for vocab_word in corpus.vocabulary
            if document_word == vocab_word
        )
        - loggamma(np.sum(gamma)) + np.sum(loggamma(gamma))
        - np.sum((gamma-1)*(digamma(gamma)-digamma(np.sum(gamma))))
        - np.sum(phi * np.log(phi))
    )


def random_categorical_distribution(num_choices):
    unnormalized = np.random.uniform(size=num_choices)
    return unnormalized / np.sum(unnormalized)
