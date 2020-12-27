import collections
import numpy as np
import argparse
import os
from lda import data
from lda.data.word import Word
from lda.data.document import Document


def unigram(documents):
    wordcount_dict = {}
    for doc in documents:
        wordcounts = doc.word_count
        for word, count in wordcounts.items():
            if word in wordcount_dict:
                wordcount_dict[word] += count
            else:
                wordcount_dict[word] = count

    model = collections.defaultdict(
        lambda: 1e-10,
        {
            word: count
            for word, count in wordcount_dict.items()
            # if count > 1
        })
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/N
    return model


def unigram_perplexity(model, documents):
    word_sum = 0
    for doc in documents:
        word_sum += sum(doc.word_count.values())

    perpl_sum = 0
    for doc in documents:
        for word in doc.included_words:
            perpl_sum += np.log(model[word]) / word_sum
    return np.exp(-perpl_sum)


def main():
    reuters_corpus = data.reuters
    all_documents = reuters_corpus.documents
    train_set_ratio = 0.9
    elements = len(all_documents)
    middle = int(elements * train_set_ratio)
    X_train, X_test = [all_documents[:middle], all_documents[middle:]]

    # X_train = [
    # Document.from_text('Have a jolly christmas christmas!'),
    # Document.from_text('Jens Lagergen is the best professor at university.'),
    # Document.from_text('Universities are important learning centers. Especially because of Jens Lagergen.'),
    # ]
    # X_test = [Document.from_text('merry christmas!'), Document.from_text('What even is a university? Experts disagree.')]

    smoothed_unigram = unigram(X_train)
    uni_perplexity = unigram_perplexity(smoothed_unigram, X_test)
    print(uni_perplexity)


if __name__ == "__main__":
    main()
