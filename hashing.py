import operator
from collections import Counter

from helpers.measuretime import measure
from parsers import InputParser, store_dict, load_dict, store_list, load_list

WORDS_INDEX_PATH = "words2index.data.out"
WORDS_COUNT_PATH = "words_count.data.out"
THRESHOLD = 100

def _create_words2index(keys):
    words_dict = dict(zip(keys, range(len(keys))))
    return words_dict

def create_r_dict(words_dict):
    reverse_words_dict = dict(zip(words_dict.values(), words_dict.keys()))
    return reverse_words_dict

def transform_line(line, wordDict):
    return [wordDict[word] for word in line]

# generator
def get_transform_sentences(wordsDict, input_parsed):
    for line in input_parsed.iter_cols(2):
        yield transform_line(line, wordsDict)

@measure
def load_words2index():
    words = load_list(WORDS_INDEX_PATH)
    return _create_words2index(words)
    # return load_dict(WORDS_INDEX_PATH, int)

@measure
def load_words_count():
    return Counter(load_dict(WORDS_COUNT_PATH, int))

@measure
def main():
    input_parsed = InputParser()
    words_count = Counter({key:val
                           for key, val in Counter(input_parsed.iter_all(2)).items()
                           if val > THRESHOLD})

    store_dict(words_count, WORDS_COUNT_PATH)
    store_list(words_count.keys(), WORDS_INDEX_PATH)

if __name__ == '__main__':
    main()

    # _word2index, _index2word = create_dicts()

    # data = list(get_transform_sentences(_word2index))
