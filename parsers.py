from collections import defaultdict, Counter
import itertools

import numpy as np

class InputParser:
    def __init__(self, inputFile):
        self.inputFile = inputFile

    def iter_cols(self, cols):
        index = np.array(cols)
        retVal = []
        with open(self.inputFile, encoding="utf8") as f:
            for line in f:
                if line.strip():
                    retVal.append(np.array(line.split())[index])
                else:
                    yield retVal
                    retVal = []

    def create_bank_set(self, colNum):
        return frozenset(itertools.chain.from_iterable(self.iter_cols(colNum)))

    # def create_taggs_counter(self):
    #     tagsCount = defaultdict(Counter)
    #     for (i, j) in self.iter_cols((2,3)):
    #         tagsCount[j][i] += 1
    #     return tagsCount
    #
    # def get_tagged_examples(self, k=3):
    #     tagsCount = self.create_taggs_counter()
    #     for tag, count in tagsCount.items():
    #         yield (tag, map(lambda x: x[0], count.most_common(k)))


def store_list(listToStore, filePath):
    with open(filePath, "w", encoding="utf8") as f:
        for i in listToStore:
            f.write(i+'\n')


def load_list(filePath):
    with open(filePath, encoding="utf8") as f:
        return [line.strip() for line in f]

def store_cooccurrence(filePath, counts):
    with open(filePath, "w", encoding="utf8") as f:
        for word, contextCounts in counts.items():
            for context, count in contextCounts.items():
                f.write(f"{word} {context} {count}\n")