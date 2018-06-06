from collections import defaultdict, UserDict

from helpers.measuretime import measure
from parsers import store_list, load_list

WORDS_INDEX_PATH = "words2index.data.out"
WORDS_COUNT_PATH = "words_count.data.out"

class MagicHash(UserDict):
    def __init__(self):
        self.data = defaultdict(self._nextId)
        self.id2word = {}
        self.id = -1

    @classmethod
    def create_from_keys(cls, keys, freezed=True):
        ret = cls()
        ret.id = len(keys)
        generator = zip(keys, range(ret.id))
        ret.data = dict(generator) if freezed else defaultdict(ret._nextId, generator)
        ret.id2word = dict(zip(ret.data.values(), ret.data.keys()))
        return ret

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.id2word[item]
        else:
            i = self.data[item]
            self.id2word[i] = item
            return i

    def _nextId(self):
        self.id += 1
        return self.id

    def freeze(self):
        self.data.default_factory = None

def transform_line(line, wordDict):
    return [wordDict[word] for word in line]

# generator
def get_transform_sentences(wordsDict, input_parsed):
    for line in input_parsed.iter_cols(2):
        yield transform_line(line, wordsDict)

@measure
def load_words2index(filePath):
    words = load_list(filePath)
    return MagicHash.create_from_keys(words)

def store_words2index(filePath, hashingDict):
    store_list(filePath, hashingDict.keys())

@measure
def main():
    pass

if __name__ == '__main__':
    main()

