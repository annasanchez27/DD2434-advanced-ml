from lda.data.corpus import Corpus
from lda.data.document import Document
from lda.data.word import Word


<<<<<<< HEAD
def test_vocabulary():
    assert corpus.vocabulary == {
        please,
        crash,
        grandma
    }
=======
def test_vocabulary_indices():
    assert corpus.vocabulary_indices == {
        doc1: [0, 1],
        doc2: [0, 2],
    }


def test_vocabulary():
    assert corpus.vocabulary == [
        please,
        crash,
        grandma
    ]
>>>>>>> 745375531ad5b7ae49dcd52d443f79d110aee2b9


def test_word_count():
    assert corpus.word_count == {
        please: 2,
        crash: 1,
        grandma: 1,
    }


please = Word('Please', 'please', include=True)
crash = Word('crash', 'crash', include=True)
grandma = Word('grandma', 'grandma', include=True)
<<<<<<< HEAD
corpus = Corpus(documents=[
    Document(
        words=[
            please,
            Word('do', 'do', include=False),
            Word('not', 'not', include=False),
            crash,
            Word('.', '.', include=False),
        ]
    ),
    Document(
        words=[
            please,
            Word('do', 'do', include=False),
            Word('not', 'not', include=False),
            grandma,
            Word('.', '.', include=False),
        ]
    )
])
=======
doc1 = Document(
    words=[
        please,
        Word('do', 'do', include=False),
        Word('not', 'not', include=False),
        crash,
        Word('.', '.', include=False),
    ]
)
doc2 = Document(
    words=[
        please,
        Word('do', 'do', include=False),
        Word('not', 'not', include=False),
        grandma,
        Word('.', '.', include=False),
    ]
)
corpus = Corpus(documents=[doc1, doc2])
>>>>>>> 745375531ad5b7ae49dcd52d443f79d110aee2b9