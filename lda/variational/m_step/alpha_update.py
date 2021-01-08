from typing import Dict
import numpy as np
from scipy.special import loggamma, digamma, polygamma
from lda.data.document import Document
from lda.utils import guarded_polygamma
from lda.utils import guarded_digamma


def alpha_update(alpha: np.ndarray, gammas: Dict[Document, np.ndarray], num_iterations=32):
    converged = False
    alpha_old = alpha.copy()
    for iteration in range(num_iterations):
        gradients = all_gradients(alpha_old, gammas)
        alpha_new = step(alpha=alpha_old, gammas=gammas, gradients=gradients)
        if np.any(alpha_new < 0):
            return alpha_old
        if maximum_found(alpha_new, gammas, gradients):
            converged = True
            break
        alpha_old = alpha_new
    assert converged
    return alpha_new


def step(alpha: np.ndarray, gammas: Dict[Document, np.ndarray], gradients):
    h = hessian_diagonal(alpha=alpha, gammas=gammas)
    c = magic_constant(alpha=alpha, gammas=gammas, h=h, gradients=gradients)
    return alpha - np.array([
        (
            gradients[topic]
            - c
        ) / h[topic]
        for topic in range(alpha.shape[0])
    ])


def magic_constant(alpha: np.ndarray, gammas: Dict[Document, np.ndarray], h: np.ndarray, gradients):
    '''Referred to as c in the paper.'''
    return (
        sum(
            gradients[topic] / h[topic]
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


def all_gradients(alpha: np.ndarray, gammas: Dict[Document, np.ndarray]):
    '''List - gradient for each topic in alpha'''
    return [
        (
            len(gammas) * (guarded_digamma(alpha.sum()) - guarded_digamma(alpha[topic]))
            + sum(
                guarded_digamma(gamma[topic]) - guarded_digamma(gamma.sum())
                for document, gamma in gammas.items())
        )
        for topic in range(alpha.shape[0])
    ]

# checks if the gradient of the lower bound alpha function is close to zero
# (last page of the appendix)
def maximum_found(alpha: np.ndarray, gammas: Dict[Document, np.ndarray], gradients, thres=1e-10):
    max_found = True
    for i in range(len(alpha)):
        max_found = max_found and abs(gradients[i]) < thres
    return max_found
