import re

import numpy as np

from ContextProcessors.BaseContext import BaseContext

SKIPGRAM_OUT = "skipgram.data.out"

class SkipGram(BaseContext):
    FunctionWordsFilePath = "functionWords.data"
    WINDOW_SIZE = 2

    def __init__(self):
        super().__init__(SKIPGRAM_OUT)
        self.regex = re.compile("[a-z]", re.I)

        funcWordsLines = (w.split('#', 1)[0].strip()
                          for w in open(self.FunctionWordsFilePath).readlines())
        self.excludeWords = [w for w in funcWordsLines if w]

        self.excludeTags = ["DT", "IN", "PRP$", "WP$", "$", "CC", "PRP"]

    def _isfunction(self, w):
        return w[2] in self.excludeWords \
               or w[3] in self.excludeTags \
                or not self.regex.search(w[2])

    def _getwindow(self, i, cur):
        left = []
        l = i - 1
        while l >= 0 and len(left) < self.WINDOW_SIZE:
            if not self._isfunction(cur[l]):
                left.append(cur[l])
            l -= 1
        left.reverse()  # because we inserted the items from right to left

        right = []
        r = i + 1
        while r < len(cur) and len(right) < self.WINDOW_SIZE:
            if not self._isfunction(cur[r]):
                right.append(cur[l])
            r += 1

        return np.array(left + right)

    def process(self, index, cur):
        window = self._getwindow(index, cur)
        if self._isfunction(cur[index]) or len(window) == 0: return None

        return cur[index][2], window[:, 2]
