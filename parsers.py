import itertools

import numpy as np

# INPUT_FILE = sys.argv[1]
INPUT_FILE = "wikipedia.sample.trees.lemmatized"
# INPUT_FILE = "wikipedia.tinysample.trees.lemmatized.txt"

DELIM = '\t'

class InputParser:

    def __init__(self, inputFile=INPUT_FILE):
        self.inputFile = inputFile

    def iter_cols(self, cols):
        if not cols:
            cols = tuple(range(8))
        index = np.array(cols)
        retVal = []
        with open(self.inputFile, encoding="utf8") as f:
            for line in f:
                if line.strip():
                    retVal.append(np.array(line.split())[index])
                else:
                    yield retVal
                    retVal = []

    def iter_all(self, colNum):
        return itertools.chain.from_iterable(self.iter_cols(colNum))

    def create_bank_set(self, colNum):
        return frozenset(self.iter_all(colNum))

def store_dict(dictToStore:{}, filePath):
    with open(filePath, "w", encoding="utf8") as f:
        for key, val in dictToStore.items():
            f.write(f"{key}{DELIM}{val}\n")


def load_dict(filePath, modifier=None):
    if not modifier:
        modifier = lambda x: x
    with open(filePath, "r", encoding="utf8") as f:
        lines = (line.strip().split(DELIM) for line in f)
        return {pair[0]: modifier(pair[1]) for pair in lines if pair}

def store_list(listToStore:[], filePath, mode="w"):
    with open(filePath, mode, encoding="utf8") as f:
        for i in listToStore:
            f.write(i+'\n')

def load_list(filePath):
    with open(filePath, encoding="utf8") as f:
        return [line.strip() for line in f]

def store_cooccurrence(filePath, counts, filter=None):
    with open(filePath, "w", encoding="utf8") as f:
        for word, contextCounts in counts.items():
            for context, count in contextCounts.items():
                if filter and filter(word, context, count): continue
                f.write(f"{word}{DELIM}{context}{DELIM}{count}\n")
