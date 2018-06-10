# Word_similarity

## Python version and libraries:

 * We used Python 3.6
 * Libraries:
   * numpy
   * sklearn
   * scipy

## Run sequence:

 * `python extract_features.py input_file_name` <br/>
   In order to create features files (for each context). <br/>
   *input_file_name* is "wikipedia.sample.trees.lemmatized" if not argument is given.<br/>
   Output files:

        *.cols  -  context words
        *.rows  -  target words
        *.out   -  the actual data (counts for each pair)

 * `python calculate.py` <br/>
   In order to load the data into the matrix and calculate the similarities. <br/>
   Output files:

        sim_2ndOrder.res   -  second order results
        sim_1stOrder.res   -  first order (contexts) results

 * `python word2vec/word2vec_examples.py` <br/>
   In order to run and calculate the output for word2vec. <br/>
   **Attention:** word2vec expects those files to be at the root folder:
   *deps.words, deps.contexts, bow5.words, bow5.contexts* <br/>
   Output files:

        word2vec_words.res
        word2vec_contexts.res


