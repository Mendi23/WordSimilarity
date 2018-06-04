import re

import numpy as np

from ContextProcessors.BaseContext import BaseContext

SENTENCE_OUT = "sentence.data.out"

class SentenceContext(BaseContext):
    def __init__(self, ):
        super().__init__(SENTENCE_OUT)
        self.regex = re.compile("[a-z]", re.I)

    def _sentence_words(self, sentence_words):
        return np.array([w for w in sentence_words if self.regex.search(w)])

    def process(self, index, cur):
        # take only column 2 and delete myself from the sentence
        sentence_except_me = np.delete(cur[:, 2], index)
        word = cur[index][2]
        if self.regex.search(word):
            return word, self._sentence_words(sentence_except_me)

        return None


