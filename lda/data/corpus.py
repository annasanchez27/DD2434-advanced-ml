from functools import reduce


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
    

    def __len__(self):
        return len(self.documents)

    
    def __repr__(self):
        return f'{type(self).__name__} with {len(self)} documents'
