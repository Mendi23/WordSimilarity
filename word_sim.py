import sys
from collections import defaultdict, Counter
from parsers import InputParser, store_list, store_cooccurrence

input_file = sys.argv[1]
input_parsed = InputParser(input_file)

def get_cooccurrence_from_iter(iterPairs):
    counts = defaultdict(Counter)
    for word, contextWords in iterPairs:
        for context in contextWords:
            counts[word][context] += 1
    return counts

class SentenceContext:
    def __init__(self):
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
    def __init__(self, functionTags):
        self.excludeTags = functionTags
        # TODO: is it better to use col 3 or col 7?
        self.iter = iter(input_parsed.iter_cols((2, 3))).__iter__()
        self.cur = []
        self.index = 0
        # TODO: can function word be target word or are we skipping them altogether? right now
        # skipping, but it's possible we need to save original string and add second insdex
        # one for the target word (all sentence) and one for the skipgram window (skip function)

    def __iter__(self):
        return self

    def _filter_function_words(self, sentence):
        _filter = filter(lambda x: x[1] not in self.excludeTags, sentence)
        return list(map(lambda x: x[0], _filter))

    def __next__(self):
        if self.index == len(self.cur):
            self.cur = self._filter_function_words(self.iter.__next__())
            self.index = 0

        self.index += 1
        lval = max(0, self.index - 3)
        return self.cur[self.index - 1], self.cur[lval:self.index + 2]


cooccurrence = get_cooccurrence_from_iter(SentenceContext())
store_cooccurrence("sentence.out", cooccurrence)

# TODO: this is just a demo, need a better function words classification
cooccurrence = get_cooccurrence_from_iter(SkipGram(['DT', 'IN', 'JJ']))
store_cooccurrence("skipgram.out", cooccurrence)

unique_words = input_parsed.create_bank_set(2)
store_list(unique_words, "words.out")
