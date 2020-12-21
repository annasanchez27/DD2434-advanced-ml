import string
from spacy.tokens import Token


class Word:
    def __init__(self, original_form, lda_form, *, include=True):
        self.original_form = original_form
        self.lda_form = lda_form
        self.include = include
    

    @classmethod
    def from_token(cls, token: Token):
        return cls(
            original_form=token.text,
            lda_form=token.lemma_.lower(),
            include=(
                (not token.is_stop)
                and (token.pos_ not in ['PUNCT', 'SPACE', 'SYM', 'NUM', 'X'])
                and len(set(token.text) & set(string.ascii_letters)) > 0
            )
        )


    def __eq__(self, other):
        return hash(self) == hash(other)
    

    def __hash__(self):
        return hash((self.lda_form, self.include))
    

    def __repr__(self):
        return (
            f'{type(self).__name__}('
                f'original_form={repr(self.original_form)}, '
                f'lda_form={repr(self.lda_form)}, '
                f'include={self.include}'
            f')'
        )
