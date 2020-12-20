from collections import Counter
import spacy
nlp = spacy.load('en_core_web_md')
from .word import Word


class Document:
    def __init__(self, words, *, title=None, topics=None):
        self.words = words
        self.title = title
        self.topics = topics

    
    @classmethod
    def from_text(cls, text, *, title=None, topics=None):
        doc = nlp(text)
        return cls(
            words=[
                Word.from_token(token)
                for token in doc
            ],
            title=title,
            topics=topics,
        )
    

    @property
    def included_words(self):
        return [
            word
            for word in self.words
            if word.include
        ]
    

    @property
    def word_count(self):
        return Counter(self.included_words)


    def __eq__(self, other):
        return hash(self) == hash(other)
    

    def __len__(self):
        return len(self.words)
    

    def __hash__(self):
        return hash((self.words, self.title, self.topics))
    

    def __repr__(self):
        return f'{type(self).__name__} with {len(self)} words'
