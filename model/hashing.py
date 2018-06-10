from collections import defaultdict, UserDict
from itertools import count

from parsers import store_list, load_list

WORDS_INDEX_PATH = "words2index.data.out"
WORDS_COUNT_PATH = "words_count.data.out"


class MagicHash(UserDict):
    def __init__(self):
        self.data = defaultdict(count().__next__)
        self.id2word = {}
        self.lastid = -1

    @classmethod
    def create_from_keys(cls, keys, freezed=True):
        ret = cls()
        length = len(keys)
        generator = zip(keys, range(length))
        ret.data = dict(generator) if freezed else defaultdict(count(length).__next__, generator)
        ret.id2word = dict(zip(ret.data.values(), ret.data.keys()))
        return ret

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.id2word[item]
        else:
            i = self.data[item]
            if i > self.lastid:
                self.id2word[i] = item
                self.lastid = i
            return i

    def freeze(self):
        self.data.default_factory = None


def transform_line(line, wordDict):
    return [wordDict[word] for word in line]


# generator
def get_transform_sentences(wordsDict, input_parsed):
    for line in input_parsed.iter_cols(2):
        yield transform_line(line, wordsDict)


def load_words2index(filePath):
    words = load_list(filePath)
    return MagicHash.create_from_keys(words)


def store_words2index(filePath, hashingDict):
    store_list(filePath, hashingDict.keys())

