import string
from lda import data


def test_reuters():
    assert len(data.reuters) > 0
    for document in data.reuters.documents:
        assert len(document) > 0
        for word in document.words:
            if word.include:
                assert len(word.lda_form) > 0
                characters = set(word.original_form.lower())
                assert characters & set(string.ascii_lowercase)
