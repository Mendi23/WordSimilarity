from collections import namedtuple

_base = namedtuple('SimpleFilter',
                   ['word2count',
                    'wordThreshold',
                    'contextThreshold',
                    'coThreshold'])

# set defaults for: wordThreshold, contextThreshold, coThreshold
_base.__new__.__defaults__ = (100, 10, 25)

class SimpleFilter(_base):
    """
    Simple filter derived from _base.
    The "self" fields are listed above in the list passed into namedtuple ctor.
    Also above, __new__.__defaults__ function is setting defaults for the LAST params.
    """

    def __new__(cls, *args, **kwargs):
        return super(SimpleFilter, cls).__new__(cls, *args, **kwargs)

    def filter(self, word, context, count) -> bool:
        if (self.word2count[word] < self.wordThreshold or
                self.word2count[context] < self.contextThreshold or
                count < self.coThreshold):
            return True  # Yes! filter this
        return False  # Nah... let him through
