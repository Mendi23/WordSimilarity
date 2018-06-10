from collections import Counter


output_file = "sim_1stOrder.res"


def read_word(wordLines):
    d = Counter()
    for i in range(12):
        word = wordLines[25 * i:25 * (i + 1)]
        print(word[1].strip())
        t = [[p.strip()] for p in word[5].split("|")]
        for w in word[6:]:
            for x,y in enumerate(w.split("|", 2)):
                t[x].append(y.strip())


        t[2] = [x.split("|")[-1] for x in t[2]]
        a = len(set(t[0]).intersection(t[1]))
        b = len(set(t[1]).intersection(t[2]))
        c = len(set(t[2]).intersection(t[0]))
        h = len(set(t[2]).intersection(t[0]).intersection(t[1]))
        print(a,b,c, h)


        d["Sentence - Skipgram"] += a
        d["Skipgram - Dependency"] += b
        d["Dependency - Sentence"] +=c
        d["All"] +=h

    for x,y in d.items():
        print(x, y, y/12)


with open(output_file, encoding="utf8") as f:
    read_word(f.read().split("\n"))

