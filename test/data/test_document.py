from lda.data.document import Document


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
