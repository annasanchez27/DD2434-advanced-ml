from typing import Dict
import numpy as np
from lda.data.corpus import Corpus
from lda.data.document import Document
from lda.data.word import Word


def m_step(corpus: Corpus, phis: Dict[Document, np.ndarray], gammas: Dict[Document, np.ndarray]):
    '''
    Parameters:
    * corpus: a Corpus object
    * phis: {document: array of size (document_length, num_topics)}
        (document is a Document object, so phis is a dictionary)
    * gammas: {document: array of size (num_topics,)}
        (document is a Document object, so gammas is a dictionary)
    Returns: {
        'alpha': array of size (num_topics,)
        'beta': beta[topic_id][word] = probability of word in topic
            (word is a Word object, so beta[topic_id] is a dictionary)
    }
    '''
    return {
        'alpha': alpha_update(),
        'beta': beta_update(corpus=corpus, phis=phis),
    }


def alpha_update():
    pass # TODO


def beta_update(corpus: Corpus, phis: Dict[Document, np.ndarray]):
    num_topics = next(iter(phis.values())).shape[1]
    return [
        normalize_dict({
            vocab_word: topic_word_frequency(
                corpus=corpus,
                phis=phis,
                topic=topic,
                vocab_word=vocab_word
            )
            for vocab_word in corpus.vocabulary
        })
        for topic in range(num_topics)
    ]


def topic_word_frequency(
    corpus: Corpus,
    phis: Dict[Document, np.ndarray],
    topic: int,
    vocab_word: Word
):
    return sum(
        phis[document][word_idx, topic] * int(word == vocab_word)
        for document in corpus.documents
        for word_idx, word in enumerate(document.included_words)
    )


def normalize_dict(to_normalize: Dict):
    normalization = sum(to_normalize.values())
    return {
        key: value / normalization
        for key, value in to_normalize.items()
    }
