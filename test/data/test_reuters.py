import string
from lda import data


def test_reuters():
    num_included_words = 0
    assert len(data.reuters) > 0
    for document in data.reuters.documents:
        assert len(document) > 0
        assert sum(document.word_count.values()) == len(document)
        for word in document.words:
            if word.include:
                assert len(word.lda_form) > 0
                characters = set(word.original_form.lower())
                assert characters & set(string.ascii_lowercase)
                num_included_words += 1
    assert sum(data.reuters.word_count.values()) == num_included_words
