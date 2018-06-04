# Word_similarity

> WTF? start:
>
> that's bad english right here....
> we did a quick check, and all sentences end with the pos-tag '.' which appear only in the end of a sentence. so we use it as stop word
>
> WTF? end

## Run sequence:

 * `python hashing.py` in order to create word2index.data.out, words_count.data.out
 * `python extract_features.py` in order to create features files (for each context)
 * `sort skipgram.data.out | uniq -c | awk '{ print $2 "\t" $3 "\t" $1}' > skipgram_uniq.data.out`
 *
