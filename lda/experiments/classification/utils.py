import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


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


def docs_to_features(X_train_docs, X_test_docs, test_size):
    vec = DictVectorizer()
    tt = TfidfTransformer(use_idf=False)

    wordcount_dicts = docs_to_wordcount_dicts(X_train_docs)
    wordcounts_matrix = vec.fit_transform(wordcount_dicts).toarray()
    X_train = tt.fit_transform(wordcounts_matrix)
    wordcount_dicts = docs_to_wordcount_dicts(X_test_docs)
    wordcounts_matrix = vec.transform(wordcount_dicts).toarray()
    X_test = tt.transform(wordcounts_matrix)

    return X_train, X_test
