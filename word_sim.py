import sys
from collections import defaultdict, Counter

from filters import SimpleFilter
from helpers.measuretime import measure
from parsers import InputParser, store_list, store_cooccurrence

SENTENCE_OUT = "sentence.out"
SKIPGRAM_OUT = "skipgram.out"

def get_cooccurrence_from_iter(iterPairs):
    counts = defaultdict(Counter)
    for word, contextWords in iterPairs:
        for context in contextWords:
            counts[word][context] += 1
    return counts


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
        return self.cur[self.index - 1], self.cur


class SkipGram:
    FunctionWordsFilePath = "functionWords.data"

    def __init__(self, input_parsed):
        funcWordsLines = (w.split('#', 1)[0].strip()
                          for w in open(self.FunctionWordsFilePath).readlines())
        self.excludeWords = [w for w in funcWordsLines if w]

        self.excludeTags = ["DT", "IN", "PRP$", "WP$", "$", "CC", "PRP"]

        self.iter = iter(input_parsed.iter_cols((2, 3))).__iter__()
        self.cur = []
        self.index = 0

    def __iter__(self):
        return self

    def _filter_function_words(self, sentence):
        return [pair[0] for pair in sentence
                if pair[0] not in self.excludeWords
                and pair[1] not in self.excludeTags]

    def __next__(self):
        if self.index == len(self.cur):
            self.cur = self._filter_function_words(self.iter.__next__())
            self.index = 0

        self.index += 1
        lval = max(0, self.index - 3)
        return self.cur[self.index - 1], self.cur[lval:self.index + 2]


# ====================================================
@measure
def main():
    input_parsed = InputParser()

    # filterClass = SimpleFilter(Counter(input_parsed.iter_all(2)))

    cooccurrence = get_cooccurrence_from_iter(SentenceContext(input_parsed))
    store_cooccurrence(SENTENCE_OUT, cooccurrence)  # , filterClass.filter)

    cooccurrence = get_cooccurrence_from_iter(SkipGram(input_parsed))
    store_cooccurrence(SKIPGRAM_OUT, cooccurrence)  # , filterClass.filter)

    unique_words = sorted(input_parsed.create_bank_set(2))
    store_list(unique_words, "words.out")

if __name__ == '__main__':
    main()
