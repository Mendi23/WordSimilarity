from itertools import product

from ContextProcessors.SkipGram import SkipGram
from ContextProcessors.SentenceContext import SentenceContext
from helpers.measuretime import measure
from parsers import InputParser, DELIM
import numpy as np

def append_to_file(fileobj, processed_pair):
    if not processed_pair or len(processed_pair) == 0: return

    word, features = processed_pair
    for feature in features:
        fileobj.write(f"{word}{DELIM}{feature}\n")

@measure
def main():
    input_parsed = InputParser()

    with SentenceContext() as c1, SkipGram() as c2:
        for line in input_parsed.iter_cols(None):
            line = np.array(line)
            for i, context in product(range(len(line)), [c1, c2]):
                append_to_file(context.fileobj, context.process(i, line))


if __name__ == '__main__':
    main()
