import collections
import numpy as np
import argparse
import os
from lda import data

def unigram(documents):
    model = collections.defaultdict(lambda: 1e-15)
    for doc in documents:
        wordcounts = doc.word_count
        for word in doc.included_words:
            if word in model:
                model[word] += wordcounts[word]
            else:
                model[word] = wordcounts[word]
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/N
    return model


def unigram_perplexity(model, documents):
    perpl_sum = 0
    perplexity = 1
    n = 0
    for doc in documents:
        p_wd = 1
        wordcounts = doc.word_count
        for word in doc.included_words:
            count = wordcounts[word]
            n += count
            for _ in range(count):
                p_wd *= model[word]
        perpl_sum += np.log(p_wd)
    return np.exp(-perpl_sum/n)

def main():
    reuters_corpus = data.reuters
    all_documents = reuters_corpus.documents
    train_set_ratio = 0.95
    elements = len(all_documents)
    middle = int(elements * train_set_ratio)
    X_train, X_test = [all_documents[:middle], all_documents[middle:]]
    
    smoothed_unigram = unigram(X_train)
    print(unigram_perplexity(smoothed_unigram, X_test))


if __name__ == "__main__":
    main()
