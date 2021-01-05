import numpy as np
from tqdm.auto import trange
from lda.data.corpus import Corpus
from .e_step import e_step
from .m_step import m_step
from scipy.special import loggamma, digamma
from lda.utils import normalize


def lda(corpus: Corpus, num_topics=64, num_iterations=1024, max_attempts=1024):
    '''
    Parameters:
    * corpus: a Corpus object
    * num_topics: number of topics :)
    * num_iterations: go see a doctor
    Returns: {
        'alpha': array of size (num_topics,)
        'beta': beta[topic_id][word_id] = probability of word in topic
        'phis': {document: array of size (document_length, num_topics)}
        'gammas': {document: array of size (num_topics,)}
    }
    '''
    # TODO: maybe gammas should be a numpy array in its entirety?
    for attempt in range(max_attempts):
        return lda_single_attempt(
            corpus=corpus,
            attempt_number=attempt,
            num_topics=num_topics,
            num_iterations=num_iterations
        )


def lda_single_attempt(corpus: Corpus, attempt_number, num_topics=64, num_iterations=1024):
    lower_bound_evol = []
    vocab = corpus.vocabulary
    params = {
        'alpha': np.random.uniform(size=num_topics),
        'beta': normalize(np.random.uniform(size=(num_topics, len(vocab))), axis=1)
    }
    params.update(e_step(
        corpus=corpus,
        alpha=params['alpha'],
        beta=params['beta']
    ))
    for iteration in trange(num_iterations, desc=f'Attempt {attempt_number}'):
        out = lda_step(corpus=corpus, params=params)
        params = out['params']
        assert not np.isnan(out['lower_bound'])
        if len(lower_bound_evol) > 0:
            assert out['lower_bound'] >= lower_bound_evol[-1]
        lower_bound_evol.append(out['lower_bound'])
    return {
        'params': params,
        'lower_bound_evol': np.array(lower_bound_evol)
    }


def lda_step(corpus: Corpus, params: dict):
        params = {
            **params,
            **m_step(
                corpus=corpus,
                alpha=params['alpha'],
                phis=params['phis'],
                gammas=params['gammas']
            )
        }
        params = {
            **params,
            **e_step(
                corpus=corpus,
                alpha=params['alpha'],
                beta=params['beta']
            )
        }
        return {
            'params': params,
            'lower_bound': corpus_lower_bound(
                corpus=corpus, 
                alpha=params['alpha'],
                beta=params['beta'],
                phis=params['phis'],
                gammas=params['gammas']
            )
        }


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
    supergamma = digamma(gamma) - digamma(gamma.sum())
    return (
        loggamma(alpha.sum()) - loggamma(alpha).sum()
        + np.sum((alpha - 1) * supergamma)
        + np.sum(phi * supergamma)
        + np.sum(
            phi.transpose([1, 0]) * np.log(beta[:, corpus.vocabulary_indices[document]])
        ) # we pray that this works
        - loggamma(gamma.sum()) + loggamma(gamma).sum()
        - np.sum((gamma - 1) * supergamma)
        - np.sum(
            phi[phi > 0] * np.log(phi[phi > 0])
        )
    )
