import pytest
from lda.data.document import Document
from lda.data.word import Word


def test_from_text():
    doc = Document.from_text('Mary had a little lamb.')
    assert len(doc) == 3 # "had" and "a" are stop words
    assert (
        [word.original_form for word in doc.words]
        == ['Mary', 'had', 'a', 'little', 'lamb', '.']
    )
    assert (
        [word.original_form for word in doc.included_words]
        == ['Mary', 'little', 'lamb']
    )
    Document.from_text('Let\'s see if nlp model cache works')


def test_to_text():
    doc = Document(words=[
        Word('Mary', 'mary', include=True),
        Word('had', 'had', include=False),
        Word('a', 'a', include=False),
        Word('little', 'little', include=True),
        Word('lamb', 'lamb', include=True),
        Word('.', '.', include=False),
    ])
    assert doc.to_text().replace('\x1b[0m', '') == 'Mary had a little lamb .'
    with pytest.raises(AssertionError):
        doc.to_text(colors=['red'])
    doc.to_text(colors=['red', None, 'green', None, None, 'blue'])
