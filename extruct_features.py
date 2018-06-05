from collections import defaultdict, Counter
import numpy as np
from helpers.measuretime import measure
from parsers import InputParser, store_list, store_cooccurrence

SENTENCE_OUT = "sentence.out"
SENTENCE_VOC = "sentence.voc"
SKIPGRAM_OUT = "skipgram.out"
SKIPGRAM_VOC = "skipgram.voc"

LEMMA_THRESHOLD = 100
# FEATURE_THRESHOLD = 50
COOCCURRENCE_THRESHOLD = 5


def get_cooccurrence_from_iter(iterPairs):
    cooccurrence = defaultdict(Counter)
    wordCounts = Counter()
    for word, contextWords in iterPairs:
        wordCounts[word] += 1
        for context in contextWords:
            cooccurrence[word][context] += 1
    return filter_cooccurrence(cooccurrence, wordCounts)


def filter_cooccurrence(coo, wordCounts):
    filteredCoo = defaultdict(dict)
    for word, contextWords in coo.items():
        if wordCounts[word] > LEMMA_THRESHOLD:
            for context, val in contextWords.items():
                if wordCounts[context] > LEMMA_THRESHOLD and val > COOCCURRENCE_THRESHOLD:
                    filteredCoo[word][context] = val
    return filteredCoo


def create_store_space_params(vectorsFile, vocabularyFile, contextIterator):
    input_parsed = InputParser()
    cooccurrence = get_cooccurrence_from_iter(contextIterator(input_parsed))
    store_cooccurrence(vectorsFile, cooccurrence)
    store_list(vocabularyFile, cooccurrence.keys())


class SentenceContext:
    def __init__(self, input_parsed):
        self.iter = iter(input_parsed.iter_cols(2)).__iter__()
        self.cur = []
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.cur):
            self.cur = self.iter.__next__()
            self.index = 0
        self.index += 1
        return self.cur[self.index - 1], self.cur[:self.index]+self.cur[self.index+1:]


class SkipGram:

    def __init__(self, input_parsed):
        self.excludeTags = ["DT", "IN", "PRP$", "WP$", "$", "CC", "PRP"]
        self.iter = iter(input_parsed.iter_cols((2, 3), self.filterFunctionWords)).__iter__()
        self.cur = []
        self.filterFunctionWords = lambda x: x[3] in self.excludeTags
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.cur):
            self.cur = self.iter.__next__()
            self.index = 0

        self.index += 1
        lval = max(0, self.index - 3)
        return self.cur[self.index - 1], self.cur[lval:self.index + 2]


# ====================================================
@measure
def main():
    create_store_space_params(SENTENCE_OUT, SENTENCE_VOC, SentenceContext)
    create_store_space_params(SKIPGRAM_OUT, SKIPGRAM_VOC, SkipGram)


if __name__ == '__main__':
    main()
