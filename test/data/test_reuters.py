import string
from lda import data


def test_word_cleanness():
    for document in data.reuters.documents:
        for word in document.words:
            if word.include:
                assert len(word.lda_form) > 0
                characters = set(word.original_form.lower())
                assert characters & set(string.ascii_lowercase)


def test_lengths():
    assert len(data.reuters) > 0
    for document in data.reuters.documents:
        assert len(document) > 0


def test_word_count():
    num_included_in_corpus = 0
    for document in data.reuters.documents:
        word_counts = document.word_count.values()
        included_in_document = [
            word
            for word in document.words
            if word.include
        ]
        num_included_in_corpus += len(included_in_document)
        assert sum(word_counts) == len(document)
        assert 1<= min(word_counts) <= max(word_counts)
        assert len(document) == len(included_in_document)
    
    assert sum(data.reuters.word_count.values()) == num_included_in_corpus
    assert min(data.reuters.word_count.values()) == 1
    assert max(data.reuters.word_count.values()) > 1
