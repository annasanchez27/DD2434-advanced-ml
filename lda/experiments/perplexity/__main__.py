
import sys 
sys.path.append("../../../")
import collections
import numpy as np
import argparse
import os
from lda import data
from lda.data.word import Word
from lda.data.document import Document
from plsa.algorithms.conditional_plsa import ConditionalPLSA
from plsa.corpus import Corpus
from plsa.pipeline import Pipeline
from plsa.preprocessors import tokenize
from plsa.pipeline import DEFAULT_PIPELINE
from lda.experiments.perplexity.utils import docs_to_strings
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__),
                        '..', 'data', 'perplexity')


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


def plsa_probability(docs_num, topics_num, doc, word_given_topic, topic_given_doc):
    new_word_prob = 1e-10
    doc_probability = 0
    for doc_iter in range(docs_num):
        train_doc_probability = 1
        for word in doc:
            topic_sum = 0
            for topic_iter in range(topics_num):
                if word not in word_given_topic[topic_iter]:
                    topic_sum += new_word_prob * \
                        topic_given_doc[doc_iter, topic_iter]
                else:
                    topic_sum += word_given_topic[topic_iter][word] * \
                        topic_given_doc[doc_iter, topic_iter]
            train_doc_probability *= topic_sum
        doc_probability += train_doc_probability  # / docs_num THIS SHOULD BE UNCOMMENTED once train_doc_probability is not zero
    print(doc_probability)

"""
def main():
    reuters_corpus = data.reuters
    all_documents = reuters_corpus.documents
    train_set_ratio = 0.2
    elements = len(all_documents)
    middle = int(elements * train_set_ratio)
    end = int(elements * 0.4)
    X_train, X_test = [all_documents[:middle], all_documents[middle:end]]

    X_train_strings = docs_to_strings(X_train)
    # X_test_strings = docs_to_strings(X_test)

    docs_num = len(X_train_strings)
    topics_num = 2

    plsa_corpus = Corpus(X_train_strings, Pipeline(tokenize))
    print(plsa_corpus)
    plsa = ConditionalPLSA(corpus=plsa_corpus, n_topics=topics_num)
    result = plsa.fit(max_iter=10)
    print(result.topic_given_doc.shape)
    print(result.topic)

    word_given_topic = np.array([
        {
            word_probability[0]: word_probability[1]
            for word_probability in topic_tuple
        }
        for topic_tuple in result.word_given_topic
    ])

    doc = tuple(str(X_train_strings[0]).split())
    plsa_probability(docs_num, topics_num, doc,
                     word_given_topic, result.topic_given_doc)

    # X_train = [
    # Document.from_text('Have a jolly christmas christmas!'),
    # Document.from_text('Jens Lagergen is the best professor at university.'),
    # Document.from_text('Universities are important learning centers. Especially because of Jens Lagergen.'),
    # ]
    # X_test = [Document.from_text('merry christmas!'), Document.from_text('What even is a university? Experts disagree.')]

    # smoothed_unigram = unigram(X_train)
    # uni_perplexity = unigram_perplexity(smoothed_unigram, X_test)
    # print(uni_perplexity)


if __name__ == "__main__":
    main()
"""