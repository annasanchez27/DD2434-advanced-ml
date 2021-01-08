from functools import reduce, cached_property
from collections import Counter


class Corpus:
    def __init__(self, documents):
        self.documents = documents
    

    @classmethod
    def merge(cls, corpora):
        return cls(
            documents=reduce(
                lambda docs1, docs2: docs1 + docs2,
                [
                    corpus.documents
                    for corpus in corpora
                ]
            )
        )
    

    @cached_property
    def vocabulary_indices(self):
        '''{document: list of vocabulary indices of the document's included words}'''
        return {
            document: [
                self.vocabulary.index(word)
                for word in document.included_words
            ]
            for document in self.documents
        }
    

    @cached_property
    def vocabulary(self):
        return list(self.word_count)
    

    @cached_property
    def word_count(self):
        return Counter(
            word
            for document in self.documents
            for word in document.included_words
        )


    def __eq__(self, other):
        return hash(self) == hash(other)
    

    def __len__(self):
        return len(self.documents)
    

    def __hash__(self):
        return hash(self.documents)

    
    def __repr__(self):
        return f'{type(self).__name__} with {len(self)} documents'
