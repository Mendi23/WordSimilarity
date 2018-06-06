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


# def get_cooccurrence_from_iter(iterPairs):
#     cooccurrence = defaultdict(Counter)
#     for word, contextWords in iterPairs:
#         for context in contextWords:
#             cooccurrence[word][context] += 1
#     return filter_cooccurrence(cooccurrence)
#
#
# def filter_cooccurrence(coo):
#     input_parsed = InputParser()
#     wordCounts = Counter(input_parsed.iter_all(2))
#     filteredCoo = defaultdict(dict)
#     for word, contextWords in coo.items():
#         if wordCounts[word] > LEMMA_THRESHOLD:
#             for context, val in contextWords.items():
#                 if wordCounts[context] > LEMMA_THRESHOLD and val > COOCCURRENCE_THRESHOLD:
#                     filteredCoo[word][context] = val
#     return filteredCoo


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
        wordId = {wordFeatures[0]: wordFeatures[2] for wordFeatures in sentence}
        wordId["0"] = None

        for wordFeatures in sentence:
            if wordFeatures[3] not in self.prepTags:
                word = wordFeatures[2]
                dependency = wordFeatures[7]
                parent = wordId[wordFeatures[6]]
                if parent is not None:
                    self.cur.append((word, "|".join([dependency, "c", parent])))
                    self.cur.append((parent, "|".join([dependency, "p", word])))

    # def _handle_preposition(self, feature, is_child):
    #     if is_child:
    #         nouns = list(filter(lambda x: x[3] in self.NounTags, self._get_children(feature)))
    #         if len(nouns) > 1:
    #             print(f"~~~ ERRORRR: Mendi was right - what do we do? cur:")
    #             pprint(self.cur)
    #         elif len(nouns) > 0:
    #             noun = nouns[0]
    #             return noun[2]
    #     else:
    #         pp_parent = self._get_parent(feature)
    #         if pp_parent is not None:
    #             return pp_parent[2]
    #     return None
    #
    # def _extract_feature_details(self, word, feature, feature_is_child):
    #     label = feature[7] if feature_is_child else word[7]
    #     direction = "c" if feature_is_child else "p"
    #     feature_name = feature[2]
    #     if feature[3] in self.prepTags:
    #         prep_data = self._handle_preposition(feature, feature_is_child)
    #
    #     return "|".join([label, feature_name, direction])
    #
    # def _get_connected(self):
    #     current_word = self.cur[self.index - 1]
    #     parent = self._get_parent(current_word)
    #     children = self._get_children(current_word)
    #
    #     res = []
    #
    #     if parent is not None:
    #         res.append(self._extract_feature_details(current_word, parent, False))
    #
    #     for c in children:
    #         res.append(self._extract_feature_details(current_word, c, True))
    #
    #     return res
    #
    # def _get_parent(self, word):
    #     index = int(word[6])
    #     for p in reversed(self.cur[:index]):
    #         if int(p[0]) == index:
    #             return p
    #     return None
    #
    # def _get_children(self, word):
    #     index = word[0]
    #     return [w for w in self.cur if w[6] == index]

    # def __next__(self):
    #     if self.index == len(self.cur):
    #         self.cur = self.iter.__next__()
    #         self.index = 0
    #
    #     self.index += 1
    #     return self.cur[self.index - 1][2], self._get_connected()


# ====================================================

def main():
    #create_store_space_params(SENTENCE_OUT, SENTENCE_ROWS, SENTENCE_COLS, SentenceContext)
    create_store_space_params(SKIPGRAM_OUT, SKIPGRAM_ROWS, SKIPGRAM_COLS, SkipGram)
    create_store_space_params(CONNECTORS_OUT, CONNECTORS_ROWS, CONNECTORS_COLS, Connectors)


if __name__ == '__main__':
    main()
