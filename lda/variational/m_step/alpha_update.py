from typing import Dict
import numpy as np
from scipy.special import loggamma, digamma
from lda.data.document import Document
from lda.utils import guarded_polygamma


def alpha_update(alpha: np.ndarray, gammas: Dict[Document, np.ndarray], num_iterations=1024):
    for iteration in range(num_iterations):
        alpha = step(alpha=alpha, gammas=gammas)
    return alpha


def step(alpha: np.ndarray, gammas: Dict[Document, np.ndarray]):
    h = hessian_diagonal(alpha=alpha, gammas=gammas)
    c = magic_constant(alpha=alpha, gammas=gammas, h=h)
    return alpha - np.array([
        (
            gradient(alpha=alpha, gammas=gammas, topic=topic)
            - c
        ) / h[topic]
        for topic in range(alpha.shape[0])
    ])


def magic_constant(alpha: np.ndarray, gammas: Dict[Document, np.ndarray], h: np.ndarray):
    '''Referred to as c in the paper.'''
    return (
        sum(
            gradient(
                alpha=alpha,
                gammas=gammas,
                topic=topic
            ) / h[topic]
            for topic in range(alpha.shape[0])
        )
        / (
            1 / hessian_constant(alpha=alpha, gammas=gammas)
            + (1 / h).sum()
        )
    )


def hessian_constant(alpha: np.ndarray, gammas: Dict[Document, np.ndarray]):
    '''Scalar. Referred to as z in the paper.'''
    return len(gammas) * guarded_polygamma(np.sum(alpha))


def hessian_diagonal(alpha: np.ndarray, gammas: Dict[Document, np.ndarray]):
    '''Vector of shape (num_topics,). Referred to as h in the paper.'''
    return -len(gammas) * guarded_polygamma(alpha)


def gradient(alpha: np.ndarray, gammas: Dict[Document, np.ndarray], topic: int):
    '''Scalar - gradient of a single topic in alpha'''
    return (
        len(gammas) * (digamma(alpha.sum()) - digamma(alpha[topic]))
        + sum(
            digamma(gamma[topic]) - digamma(gamma.sum())
            for document, gamma in gammas.items()
        )
    )
