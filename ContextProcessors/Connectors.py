import numpy as np

from ContextProcessors.BaseContext import BaseContext

CONNECTORS_OUT = "connectors.data.out"

class Connectors(BaseContext):
    FunctionWordsFilePath = "functionWords.data"

    def __init__(self):
        super().__init__(CONNECTORS_OUT)

        funcWordsLines = (w.split('#', 1)[0].strip()
                          for w in open(self.FunctionWordsFilePath).readlines())
        self.excludeWords = [w for w in funcWordsLines if w]

        self.excludeTags = ["DT", "IN", "PRP$", "WP$", "$", "CC", "PRP"]

    def _isfunction(self, w):
        return w[2] in self.excludeWords \
               or w[3] in self.excludeTags

    def process(self, index, cur):
        window = self._getwindow(index, cur)
        if self._isfunction(cur[index]) or len(window) == 0: return None

        return cur[index][2], window[:, 2]
