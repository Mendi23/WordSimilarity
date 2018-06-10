import itertools
import numpy as np

DELIM = '\t'

class InputParser:
    INPUT_FILE = "wikipedia.sample.trees.lemmatized"
    # INPUT_FILE = "wikipedia.tinysample.trees.lemmatized.txt"

    def __init__(self):
        self.inputFile = InputParser.INPUT_FILE

    def iter_cols(self, cols, includePredicate = None):
        if not includePredicate:
            includePredicate = lambda x: True
        if not cols:
            cols = tuple(range(8))

        index = np.array(cols)
        retVal = []
        with open(self.inputFile, encoding="utf8") as f:
            for line in f:
                if line.strip():
                    cur = line.split()
                    if includePredicate(cur):
                        retVal.append(np.array(cur)[index])
                else:
                    yield retVal
                    retVal = []

    def iter_all(self, colNum):
        return itertools.chain.from_iterable(self.iter_cols(colNum))

    # def create_bank_set(self, colNum):
    #     return frozenset(self.iter_all(colNum))

def store_list(filePath, listToStore):
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
                f.write(f"{word}{DELIM}{context}{DELIM}{count}\n")

