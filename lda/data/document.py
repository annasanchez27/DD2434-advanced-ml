from typing import List
from collections import Counter
from termcolor import colored
from .word import Word
from .nlp import nlp


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
    

    def to_text(self, colors: List=None):
        '''The reconstructed text of the document, with optional coloring'''
        if colors is None:
            colors = [None for word in self.words]
        assert len(colors) == len(self.words)
        return ' '.join(
            colored(word.original_form, color)
            for word, color in zip(self.words, colors)
        )


    def __eq__(self, other):
        return hash(self) == hash(other)
    

    def __len__(self):
        return len(self.included_words)
    

    def __hash__(self):
        return hash((
            tuple(self.words),
            self.title,
            None if self.topics is None else tuple(self.topics)
        ))
    

    def __repr__(self):
        return f'{type(self).__name__} including {len(self)} words'
