from word_sim import input_parsed

#TODO: if we want to load all file to memory, use hashing and replae file in parser with "data"

def create_dicts():
    unique_words = input_parsed.create_bank_set(2)
    words_dict = dict(zip(unique_words, range(len(unique_words))))
    reverse_words_dict = dict(zip(words_dict.values(), words_dict.keys()))
    return words_dict, reverse_words_dict

_dict, _rdict = create_dicts()

def transform_line(line, wordDict):
    return [wordDict[word] for word in line]

# generator
def get_transform_sentences(wordsDict):
    for line in input_parsed.iter_cols(2):
        yield transform_line(line, wordsDict)

data = list(get_transform_sentences(_dict))
