from typing import Dict
import numpy as np
from lda.utils import normalize
from lda.data.corpus import Corpus
from lda.data.document import Document
from lda.data.word import Word


def beta_update(corpus: Corpus, phis: Dict[Document, np.ndarray]):
    num_topics = next(iter(phis.values())).shape[1]
    return normalize(
        (
            np.array([
                topic_scores(corpus=corpus, phis=phis, idx_in_vocab=idx_in_vocab)
                for idx_in_vocab, vocab_word in enumerate(corpus.vocabulary)
            ]) # (vocab_size, num_topics)
            .transpose() # (num_topics, vocab_size)
        ),
        axis=1,
    ) # (num_topics, vocab_size)


def topic_scores(corpus: Corpus, phis: Dict[Document, np.ndarray], idx_in_vocab: int):
    '''Array of size (num_topics,) representing scores for each topic given a vocabulary word'''
    return np.sum(
        [
            (
                phis
                [document] # (document_length, num_topics)
                [np.array(corpus.vocabulary_indices[document]) == idx_in_vocab] # (num_matches, num_topics)
                .sum(axis=0) # (num_topics,)
            )
            for document in corpus.documents
        ], # (num_documents, num_topics)
        axis=0
    )
