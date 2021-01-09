
import numpy as np
from lda.data.corpus import Corpus


def beta_update(corpus, phis):
    num_topics = next(iter(phis.values())).shape[1]
    vocab_size = len(corpus.vocabulary)
    beta = np.zeros(shape=(num_topics, vocab_size))
    for j in range(vocab_size):
        compute_beta_column(corpus, beta, phis, j)
    beta = normalize_beta(beta)
    return beta
        

def normalize_beta(beta):
    num_topics = beta.shape[0]
    for i in range(num_topics):
        beta[i, :] = beta[i, :]  / np.sum(beta[i, :]) 
    return beta


# we will compute beta_ij for all values of i given a fixed j
def compute_beta_column(corpus, beta, phis, j_vocab):
    num_topics = beta.shape[0]
    w_sparse = compute_sparse_w(corpus, j_vocab)
    for i in range(num_topics):
        sum_d = 0
        for d_idx, document in enumerate(corpus.documents):
            sum_d += np.sum(phis[d_idx][:, i] * w_sparse[d_idx])
        beta[i, j_vocab] = sum_d
    
    return 
    

# See Eq.9 on the pdf
# for a fixed value of j, we will compute w^j
# w^j will be a list of size M=num_docs. 
# For each document index d, w^j_d will contain a sparse array of size Nd 
#(number of words in the document) 


# we just build the variable and we init it with zeros
def create_sparse_w_shape(corpus: Corpus):
    sparse_w = []
    for document in corpus.documents:
        num_words = len(document.included_words)
        sparse_w.append(np.zeros(num_words, np.int8))
    return sparse_w

#See Eq.9
# For a fixed j, we find those d and n indexes where w_{dn}^j = 1  
def compute_sparse_w(corpus: Corpus, j_vocab):
    sparse_w = create_sparse_w_shape(corpus)
    for d_idx, document in enumerate(corpus.documents):
        for n_idx, word in enumerate(document.included_words):
            if word == corpus.vocabulary[j_vocab]:
                sparse_w[d_idx, n_idx] = 1
    return sparse_w
            
        



"""
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
"""