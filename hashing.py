import operator

from helpers.measuretime import measure
from parsers import InputParser, store_dict, load_dict

WORDS_INDEX_PATH = "words2index.data"

def _create_words_dict(input_parsed):
    unique_words = input_parsed.create_bank_set(2)
    words_dict = dict(zip(unique_words, range(len(unique_words))))
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
def load_words2dict():
    return load_dict(WORDS_INDEX_PATH, int)

@measure
def main():
    input_parsed = InputParser()
    store_dict(_create_words_dict(input_parsed), WORDS_INDEX_PATH)

if __name__ == '__main__':
    main()

    # _word2index, _index2word = create_dicts()

    # data = list(get_transform_sentences(_word2index))
