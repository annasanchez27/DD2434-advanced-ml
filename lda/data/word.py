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
            lda_form=token.text.lower(),
            include=(
                (not token.is_stop)
                and (token.pos_ not in ['PUNCT', 'SPACE', 'SYM', 'NUM', 'X'])
                and (set(token.text) & set(string.ascii_letters))
            )
        )
    

    def __repr__(self):
        return (
            f'{type(self).__name__}('
                f'original_form={repr(self.original_form)}, '
                f'lda_form={repr(self.lda_form)}, '
                f'include={self.include}'
            f')'
        )
