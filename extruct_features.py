from collections import defaultdict, Counter
from itertools import product, chain

from helpers.measuretime import measure
from parsers import InputParser, store_list, store_cooccurrence

SENTENCE_OUT = "sentence.out"
SENTENCE_ROWS = "sentence.rows"
SENTENCE_COLS = "sentence.cols"
SKIPGRAM_OUT = "skipgram.out"
SKIPGRAM_ROWS = "skipgram.rows"
SKIPGRAM_COLS = "skipgram.cols"
CONNECTORS_OUT = "connect.out"
CONNECTORS_ROWS = "connect.rows"
CONNECTORS_COLS = "connect.cols"

LEMMA_THRESHOLD = 100
FEATURE_THRESHOLD = 100
COOCCURRENCE_THRESHOLD = 5


def get_cooccurrence_from_iter(iterPairs):
    input_parsed = InputParser()
    wordCounts = Counter(input_parsed.iter_all(2))
    tempCooDict = defaultdict(Counter)
    for word, context in iterPairs:
        if wordCounts[word] > LEMMA_THRESHOLD and \
                wordCounts[context.rsplit("|", 1)[-1]] > FEATURE_THRESHOLD:
            tempCooDict[word][context] += 1

    cooccurrences = defaultdict(dict)
    for word, contextWords in tempCooDict.items():
        for context, val in contextWords.items():
            if val > COOCCURRENCE_THRESHOLD:
                cooccurrences[word][context] = val
    return cooccurrences, cooccurrences.keys(), \
           frozenset(chain.from_iterable(keyVal.keys() for keyVal in cooccurrences.values()))


@measure
def create_store_space_params(vectorsFile, rowsFile, colsFile, contextIterator):
    input_parsed = InputParser()
    cooccurrence, rows, cols = get_cooccurrence_from_iter(contextIterator(input_parsed))
    store_cooccurrence(vectorsFile, cooccurrence)
    store_list(rowsFile, rows)
    store_list(colsFile, cols)


class SentenceContext:
    def __init__(self, input_parsed):
        self.iter = iter(input_parsed.iter_cols(2)).__iter__()
        self.cur = []
        self.index = 0

    def _get_contexts(self, sentence):
        self.cur = []
        for i in range(len(sentence)):
            self.cur.extend(product([sentence[i]], sentence[:i] + sentence[i + 1:]))

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.cur):
            self._get_contexts(self.iter.__next__())
            self.index = 0
        self.index += 1
        return self.cur[self.index - 1]


class SkipGram:

    def __init__(self, input_parsed):
        self.excludeTags = ["DT", "IN", "PRP$", "WP$", "$", "CC", "PRP"]
        include_predicate = lambda w: w[3] not in self.excludeTags
        self.iter = iter(input_parsed.iter_cols(2, include_predicate)).__iter__()
        self.cur = []
        self.index = 0

    def _get_contexts(self, sentence):
        self.cur = []
        for i in range(len(sentence)):
            lval = max(i - 2, 0)
            self.cur.extend(product([sentence[i]], sentence[lval:i] + sentence[i + 1:i + 3]))

    def __iter__(self):
        return self

    def __next__(self):
        while self.index == len(self.cur):
            self._get_contexts(self.iter.__next__())
            self.index = 0
        self.index += 1
        return self.cur[self.index - 1]


class Connectors:

    def __init__(self, input_parsed):
        self.iter = iter(input_parsed.iter_cols(None)).__iter__()
        self.cur = []
        self.index = 0
        self.prepTags = ["IN"]
        self.NounTags = ["NN", "NNP", "NNPS", "NNS"]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.cur):
            self._get_contexts(self.iter.__next__())
            self.index = 0
        self.index += 1
        return self.cur[self.index - 1]

    def _get_contexts(self, sentence):
        self.cur = []
        wordId = {wordFeatures[0]: wordFeatures for wordFeatures in sentence}
        wordId["0"] = None

        for wordFeatures in sentence:
            tag = wordFeatures[3]
            if tag not in self.prepTags:
                word = wordFeatures[2]
                dependency = wordFeatures[7]
                parent = wordId[wordFeatures[6]]
                if parent is not None:
                    if parent[3] in self.prepTags:
                        dependency = "_".join([parent[7], parent[2]])
                        parent = wordId[parent[6]]
                        if parent is not None:
                            self.cur.append((word, "|".join([dependency, "c", parent[2]])))
                            if tag in self.NounTags:
                                self.cur.append((parent[2], "|".join([dependency, "p", word])))

                    else:
                        self.cur.append((word, "|".join([dependency, "c", parent[2]])))
                        self.cur.append((parent[2], "|".join([dependency, "p", word])))

# ==================================================================================================

def main():
    create_store_space_params(SKIPGRAM_OUT, SKIPGRAM_ROWS, SKIPGRAM_COLS, SkipGram)
    create_store_space_params(CONNECTORS_OUT, CONNECTORS_ROWS, CONNECTORS_COLS, Connectors)
    create_store_space_params(SENTENCE_OUT, SENTENCE_ROWS, SENTENCE_COLS, SentenceContext)


if __name__ == '__main__':
    main()
