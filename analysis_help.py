from collections import Counter

output_file = "sim_2ndOrder.res"
word2vec = "word2vec/word2vec_words.res"
car = """PROPELLER	1	0
WHEEL	1	0
SPEED	1	0
PASSENGER	1	0
HORSE	1	1
TRUCK	1	1
LOCOMOTIVE	1	1
CAR	1	1
AIRCRAFT	1	1
AUTOMOBILE	1	1
AUTO	1	0
NEURON	0	0
MODEL	1	0
CAB	1	1
THOROUGHBRED	0	0
ENGINE	1	0
RACE	1	0
DRIVE	1	0
MOTORCYCLE	1	1
LAP	1	0
DRIVER	1	0
TRAILER	1	0
RACING	1	0
FORD	1	0
ITEM	0	0
VEHICLE	1	1
ROAD	1	0
FORMULA	1	0
MOTOR	1	0
BOAT	1	1
STOCK	0	0
BUS	1	1
TRAIN	1	1
NASCAR	1	0
LIMOUSINE	1	1
MOTORCAR	1	1
CARS	1	1
MOTORHOME	1	0
MOPED	1	1
JEEP	1	1
BIKE	1	1
LIMO	1	1
MID-ENGINED	1	0
MERCEDES-BENZ	1	0
MOTORBIKE	1	1
RACECAR	1	1
MINIVAN	1	1
MINIBUS	1	1
SPEEDBOAT	1	1
TAXICAB	1	1
REAR-ENGINED	1	0
ROADSTER	1	1
SUV	1	1
LORRY	1	1
FRONT-ENGINED	1	0"""

piano = """KEYBOARD	1	1
PERCUSSION	1	0
SAXOPHONE	1	1
INSTRUMENT	1	1
VIOLIN	1	1
INSTRUMENTAL	1	0
VIOLA	1	1
CHORUS	1	0
MUSIC	1	0
TUNE	1	0
BASS	1	0
SYMPHONY	1	0
SONATA	1	0
PIANO	1	1
SHOOTING	0	0
DECREE	0	0
STRING	1	0
ACOUSTIC	1	0
DRUM	1	1
GUITAR	1	1
FLUTE	1	1
STAIRCASE	0	0
POKER	0	0
SLAM	0	0
ORCHESTRA	1	0
OP	0	0
VOCAL	1	0
QUARTET	1	0
TRUMPET	1	1
PHILHARMONIC	1	0
MELODY	1	0
CONCERTO	1	0
CELLO	1	1
TENOR	1	0
SOLO	1	0
CLAVICHORD	1	1
HARPSICHORD	1	1
OBOE	1	1
BASSOON	1	1
VIOLONCELLO	1	1
MARIMBA	1	1
CLAVINET	1	1
HARMONICA	1	1
HARP	1	1
TROMBONE	1	1
FORTEPIANO	1	1
SONATAS	1	0
VIBRAPHONE		
MANDOLIN	1	1
ACCORDION	1	1
PIANOFORTE	1	1
CLARINET	1	1"""


def read_word(wordLines):
    d = Counter()
    for i, j in zip((0,), (car,)):
        word = wordLines[25 * i:25 * (i + 1)]
        print(word[1].strip())
        t = [[p.strip()] for p in word[5].split("|")]
        for w in word[6:]:
            for x, y in enumerate(w.split("|")):
                t[x].append(y.strip())

        t[-1] = [x.split("|")[-1] for x in t[-1]]
        x = list(i.split("\t") for i in j.split("\n"))
        topical = list(map(lambda y: y[0].lower(), filter(lambda z: z[1] == '1', x)))
        semantic = list(map(lambda y: y[0].lower(), filter(lambda z: z[2] == '1', x)))
        top_score = [0, 0]
        sem_score = [0, 0]
        for l in range(2):
            for w in t[l]:
                if w in topical: top_score[l] += 1
                if w in semantic: sem_score[l] += 1
        print(top_score)
        print(sem_score)


    #     for w in set(t[0] + t[1] + t[2]):
    #         print(w)
    #     print("\n")
    # #



    #     a = len(set(t[0]).intersection(t[1]))
    #     b = len(set(t[1]).intersection(t[2]))
    #     c = len(set(t[2]).intersection(t[0]))
    #     h = len(set(t[2]).intersection(t[0]).intersection(t[1]))
    #     print(a, b, c, h)
    #
    #
    #     d["Sentence - Skipgram"] += a
    #     d["Skipgram - Dependency"] += b
    #     d["Dependency - Sentence"] += c
    #     d["All"] += h
    #
    # for x, y in d.items():
    #     print(x, y, y / 12)


with open(word2vec, encoding="utf8") as g:
    read_word(g.read().split("\n"))
