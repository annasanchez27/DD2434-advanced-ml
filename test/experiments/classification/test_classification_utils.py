from lda.data.document import Document
from lda.data.corpus import Corpus
from lda.data.word import Word
from lda import data
import experiments.classification.utils as utils
import experiments.classification.classification as classification
import numpy as np


def create_word(original, include=True):
    return Word(original, original.lower(), include=include)


def create_mary_corpus():
    mary = create_word("Mary")
    had = create_word("had", include=False)
    a = create_word("a", include=False)
    little = create_word("little")
    Little = create_word("Little")
    lamb = create_word("lamb")
    return Document([mary, had, a, little, lamb, Little, lamb, little, lamb], topics=['nursery_rhyme', 'lamb'])


def create_itsy_corpus():
    Itsy = create_word("Itsy")
    bitsy = create_word("bitsy")
    spider = create_word("spider")
    return Document([Itsy, bitsy, spider], topics=['nursery_rhyme'])


def create_lorem_corpus():
    Lorem = create_word("Lorem")
    ipsum = create_word("ipsum")
    dot = create_word(".", include=False)
    return Document([Lorem, ipsum, dot], topics=['placeholder'])


def create_corpus_without_topics():
    word1 = create_word("word1")
    word2 = create_word("word2")
    return Document([word1, word1, word2])


def create_corpus_with_empty_topics():
    word1 = create_word("word1")
    word2 = create_word("word2")
    return Document([word1, word1, word2], topics=[])


def create_corpus():
    doc1 = create_mary_corpus()
    doc2 = create_itsy_corpus()
    doc3 = create_lorem_corpus()
    doc_without_topic = create_corpus_without_topics()
    doc_with_empty_topic = create_corpus_with_empty_topics()
    return Corpus([doc1, doc2, doc3, doc_without_topic, doc_with_empty_topic])


def create_corpus_with_topics():
    doc1 = create_mary_corpus()
    doc2 = create_itsy_corpus()
    doc3 = create_lorem_corpus()
    return Corpus([doc1, doc2, doc3])


def test_corpus_with_topics():
    corpus = create_corpus()
    expected_docs_with_topics = create_corpus_with_topics().documents

    docs_with_topics = utils.corpus_to_documents_with_topics(corpus)

    assert len(docs_with_topics) == len(expected_docs_with_topics)
    for i, doc in enumerate(docs_with_topics):
        assert doc == expected_docs_with_topics[i]


def test_topic_to_labels():
    docs_with_topics = create_corpus_with_topics().documents
    labels = utils.topic_to_labels("nursery_rhyme", docs_with_topics)
    assert np.array_equal(np.array([1, 1, 0]), labels)


def test_docs_to_wordcount_dicts():
    expected = [{'mary': 1, 'little': 3, 'lamb': 3}, {
        'itsy': 1, 'bitsy': 1, 'spider': 1}, {'lorem': 1, 'ipsum': 1}]
    docs_with_topics = create_corpus_with_topics().documents
    wordcount_dicts = utils.docs_to_wordcount_dicts(docs_with_topics)
    assert expected == wordcount_dicts
