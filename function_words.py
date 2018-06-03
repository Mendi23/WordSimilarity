# TODO: how to decide which CPOSTAG/DEPREL are function-words? the teacher answer in the piaza was
#  not understood

# don't ask...
# insane idea that didn't really worked...
def define_function_tags_interactive(taggedList):
    function_tags = []
    for tag, words in taggedList:
        i = 0
        for w in words:
            res = None
            while not res:
                res = input(f"is this word a function word? [y|n]   {w}")
                if res == "y": i += 1
                elif res == "n": i -= 1
                else:
                    print("please type 'y' or 'n' keys only")
                    res = None
        if i > 0: function_tags.append(tag)
    return function_tags
