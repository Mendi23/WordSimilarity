from helpers.measuretime import measure
from parsers import store_list, load_list

WORDS_INDEX_PATH = "words2index.data.out"
WORDS_COUNT_PATH = "words_count.data.out"

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
def load_words2index(filePath):
    words = load_list(filePath)
    return _create_words2index(words)

def store_words2index(filePath, hashingDict):
    store_list(filePath, hashingDict.keys())

@measure
def main():
    pass

if __name__ == '__main__':
    main()

