import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler


def create_docs_for_label(label, docs):
    topic_docs = []
    non_topic_docs = []
    for doc in docs:
        if label in doc.topics:
            topic_docs.append(doc)
        else:
            non_topic_docs.append(doc)

    return topic_docs[:50] + non_topic_docs[:50]


def corpus_to_documents_with_topics(corpus):
    """ Returns list of documents which contain at least one topic. """
    return [
        document
        for document in corpus.documents
        if document.topics
    ]


def topic_to_labels(label_name, documents):
    labels = []
    for doc in documents:
        if label_name in doc.topics:
            labels.append(1)
        else:
            labels.append(0)

    return np.array(labels)


def docs_to_wordcount_dicts(documents):
    all_wordcounts = []
    for doc in documents:
        doc_wordcount = {}
        wordcount = doc.word_count
        for word in doc.included_words:
            doc_wordcount[word.lda_form] = wordcount[word]
        all_wordcounts.append(doc_wordcount)

    return all_wordcounts


def docs_to_features(X_train_docs, X_test_docs):
    vec = DictVectorizer()
    tt = TfidfTransformer(use_idf=False)

    wordcount_dicts = docs_to_wordcount_dicts(X_train_docs)
    wordcounts_matrix = vec.fit_transform(wordcount_dicts).toarray()
    X_train = tt.fit_transform(wordcounts_matrix)
    wordcount_dicts = docs_to_wordcount_dicts(X_test_docs)
    wordcounts_matrix = vec.transform(wordcount_dicts).toarray()
    X_test = tt.transform(wordcounts_matrix)

    return X_train, X_test


def phis_docs_to_topiccounts(phis, docs):
    topiccounts = []

    for doc in docs:
        topiccounts.append(np.sum(phis[doc], axis=0))

    return topiccounts


def gammas_docs_to_ordered_gammas(gammas, docs):
    """ Return gammas in the same order as in docs. """
    ordered_gammas = []

    for doc in docs:
        ordered_gammas.append(gammas[doc])

    return ordered_gammas


def gammas_to_features(gammas, X_train_docs, X_test_docs):
    gammas_list = gammas_docs_to_ordered_gammas(gammas, X_train_docs) \
        + gammas_docs_to_ordered_gammas(gammas, X_test_docs)
    scaled_gammas = StandardScaler().fit_transform(gammas_list)
    return scaled_gammas[:len(X_train_docs)], scaled_gammas[-len(X_test_docs):]


def phis_to_features(phis, X_train_docs, X_test_docs):
    tt = TfidfTransformer(use_idf=False)

    train_topiccounts = phis_docs_to_topiccounts(phis, X_train_docs)
    test_topiccounts = phis_docs_to_topiccounts(phis, X_test_docs)
    X_train = tt.fit_transform(train_topiccounts)
    X_test = tt.transform(test_topiccounts)
    return X_train, X_test
