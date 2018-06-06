from collections import defaultdict, Counter
from functools import lru_cache
from itertools import product, chain, filterfalse, islice

from hashing import MagicHash
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


@lru_cache(maxsize=1)
def getWrodCount():
    return Counter(InputParser().iter_all(2))

def get_cooccurrence_from_iter(iterPairs):
    wordCounts = getWrodCount()
    mhash = MagicHash()
    tempCooDict = defaultdict(Counter)
    for word, context in iterPairs:
        if wordCounts[word] > LEMMA_THRESHOLD and \
                wordCounts[context.rsplit("|", 1)[-1]] > FEATURE_THRESHOLD:
            tempCooDict[mhash[word]][mhash[context]] += 1

    cooccurrences = defaultdict(dict)
    for word, contextWords in tempCooDict.items():
        for context, val in contextWords.items():
            if val > COOCCURRENCE_THRESHOLD:
                cooccurrences[mhash[word]][mhash[context]] = val
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
    WINDOW = 2

    def __init__(self, input_parsed):
        self.excludeTags = ["DT", "IN", "PRP$", "WP$", "$", "CC", "PRP"]
        self.iter = iter(input_parsed.iter_cols((2,3))).__iter__()
        self.cur = []
        self.index = 0

    def _is_function(self, w):
        return w[1] in self.excludeTags

    def _filter(self, sen):
        return islice(filterfalse(self._is_function, sen), self.WINDOW)

    def _get_contexts(self, sentence):
        self.cur = []
        for i in range(len(sentence)):
            target = sentence[i][0]
            left = [w[0] for w in self._filter(sentence[:i:-1])]
            right = [w[0] for w in self._filter(sentence[i+1:])]
            self.cur.extend(product([target], left + right))

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
            word = wordFeatures[2]
            dependency = wordFeatures[7]
            parent = wordId[wordFeatures[6]]
            tag = wordFeatures[3]

            save = True
            save_reverse = tag not in self.prepTags

            if parent is None:
                continue

            if parent[3] in self.prepTags:
                p_parent = wordId[parent[6]]
                if p_parent is None:
                    save = False
                else:
                    dependency = "_".join([parent[7], parent[2]])
                    parent = p_parent
                    save_reverse = tag in self.NounTags

            if save:
                self.cur.append((word, "|".join([dependency, "c", parent[2]])))
            if save_reverse:
                self.cur.append((parent[2], "|".join([dependency, "p", word])))


# ==================================================================================================

def main():
    create_store_space_params(SKIPGRAM_OUT, SKIPGRAM_ROWS, SKIPGRAM_COLS, SkipGram)
    create_store_space_params(CONNECTORS_OUT, CONNECTORS_ROWS, CONNECTORS_COLS, Connectors)
    create_store_space_params(SENTENCE_OUT, SENTENCE_ROWS, SENTENCE_COLS, SentenceContext)

    print(getWrodCount.cache_info())


if __name__ == '__main__':
    main()
