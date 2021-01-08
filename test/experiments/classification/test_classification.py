from lda.data.document import Document
from lda.data.corpus import Corpus
from lda.data.word import Word
from lda import data
import experiments.classification.utils as utils
import experiments.classification.classification as classification
import numpy as np
from sklearn.model_selection import train_test_split

test_data_fractions = [0.99, 0.95, 0.9, 0.8]


def test_classification_output_grain():
    expected_accuracies = {
        0.99: 94.44228148724937,
        0.95: 94.53291408864996,
        0.9: 95.62098501070663,
        0.8: 97.37412671645387,
    }
    reuters_corpus = data.reuters
    corpus = utils.corpus_to_documents_with_topics(reuters_corpus)
    labels = utils.topic_to_labels("grain", corpus)

    for test_frac in test_data_fractions:
        X_train_docs, X_test_docs, y_train, y_test = train_test_split(
            corpus, labels, test_size=test_frac, random_state=42)
        accuracy = classification.classification_for_label(
            X_train_docs, X_test_docs, y_train, y_test, test_size=test_frac)
        assert accuracy == expected_accuracies[test_frac]


def test_classification_output_earn():
    expected_accuracies = {
        0.99: 91.91162156900916,
        0.95: 96.6528045440714,
        0.9: 97.44111349036403,
        0.8: 97.71139484461575,
    }
    reuters_corpus = data.reuters
    corpus = utils.corpus_to_documents_with_topics(reuters_corpus)
    labels = utils.topic_to_labels("earn", corpus)

    for test_frac in test_data_fractions:
        X_train_docs, X_test_docs, y_train, y_test = train_test_split(
            corpus, labels, test_size=test_frac, random_state=42)
        accuracy = classification.classification_for_label(
            X_train_docs, X_test_docs, y_train, y_test, test_size=test_frac)
        assert accuracy == expected_accuracies[test_frac]
