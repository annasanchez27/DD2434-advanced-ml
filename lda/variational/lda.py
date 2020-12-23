import numpy as np
from lda.data.corpus import Corpus
from .e_step import e_step
from .m_step import m_step


def lda(corpus: Corpus, num_topics=64, num_iterations=1024):
    vocab = corpus.vocabulary
    print(vocab)
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
    print(params['beta'])
    for iteration in range(num_iterations):
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
    return params


def random_categorical_distribution(num_choices):
    unnormalized = np.random.uniform(size=num_choices)
    return unnormalized / np.sum(unnormalized)
