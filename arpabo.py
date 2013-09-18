import re
import gzip
import math
import locale
from common_tools import meta_open

arpabo_rx = re.compile(r"""
^(?P<preamble>.*?)
^BBOARD_BEGIN\s*
(?P<bboard>.*?)
^BBOARD_END\s*
\\data\\
(?P<data>.*?)
#ngram \d+\=\d+)+\s+)
(?P<grams>\\\d+-grams:
.*)
\\end\\
\s*
""", re.X | re.M | re.S)


def compare(x, y):
    if x == y:
        return 0
    elif x.startswith(y) and x[len(y)] == "'":
        return -1
    elif y.startswith(x) and y[len(x)] == "'":
        return 1
    else:
        return locale.strcoll(x, y)



def format_grams(grams):
    to_join = []
    for toks, prob, bow in sorted(grams):
        if bow == None:
            #to_join.append("%s %s" % (prob, " ".join(toks)))
            to_join.append("%f %s" % (prob, " ".join(toks)))
        else:
            #to_join.append("%s %s %s" % (prob, " ".join(toks), bow))
            to_join.append("%f %s %f" % (prob, " ".join(toks), bow))
    return "\n".join(to_join)

def unpack_grams(n, s):
    retval = []
    for e in s.strip().split("\n"):
        toks = e.split()
        if len(toks) == n:
            raise Exception("line \"%s\" is too short" % e)
        elif len(toks) == n + 1:
            retval.append((toks[1:], float(toks[0]), None))
            #retval.append((toks[1:], toks[0], None))
        elif len(toks) == n + 2:
            retval.append((toks[1:-1], float(toks[0]), float(toks[-1])))
            #retval.append((toks[1:-1], toks[0], toks[-1]))
    return retval

class Arpabo():
    def __init__(self, file_handle):
        m = arpabo_rx.match(file_handle.read())
        self.preamble = m.group("preamble").strip()
        self.bboard = m.group("bboard").strip()
        #self.data = [(int(x.group(1)), int(x.group(2))) for x in re.finditer(r"ngram (\d+)\=(\d+)", m.group("data"))]
        m2 = re.finditer(r"\\(?P<n>\d+)-grams:(?P<vals>.*?)\n\n", m.group("grams"), re.S)
        self.grams = {int(x.group("n")) : unpack_grams(int(x.group("n")), x.group("vals")) for x in m2}
        self.words = set([x[0][0] for x in self.grams[1]])
    def __str__(self):
        return "%d words" % (len(self.get_words()))

    def unigram_sum(self, n):
        return sum([math.pow(10, p) for t, p, b in self.grams[n]])

    def get_words(self):
        return self.words

    def add_unigrams(self, words, weight, new_bow=.000000001):
        words = [w for w in words if w not in self.get_words()]
        for w in words:
            self.words.add(w)
        scale = 1.0 - weight
        new_prob = math.log10(weight / len(words))
        self.grams[1] = sorted([(t, math.log10(math.pow(10, p) * scale), b) for t, p, b in self.grams[1]] + [([w], new_prob, new_bow) for w in words])

    def format(self):
        return """%s

BBOARD_BEGIN
%s
BBOARD_END

\\data\\
%s

%s
\\end\\

""" % (self.preamble, 
       self.bboard, 
       "\n".join(["ngram %d=%d" % (n, len(grams)) for n, grams in sorted(self.grams.iteritems())]), 
       "\n".join(sorted(["\\%d-grams:\n%s\n" % (k, format_grams(v)) for k, v in self.grams.iteritems()])))

class Dictionary():
    def __init__(self, file_handle):
        self.entries = {}
        for w, n, p in [re.match(r"^(\S+?)(\(\d+\))? (.*)$", l).groups() for l in file_handle]:
            if n:
                n = int(n[1:-1])
            else:
                n = len(self.entries.get(w, {})) + 1
            self.entries[w] = self.entries.get(w, {})
            self.entries[w][int(n)] = p.replace("[ wb ]", "").split()

    def __str__(self):
        return "%d words" % (len(self.get_words()))

    def add_entries(self, other):
        for w, ps in sorted(other.entries.iteritems()):
            if w not in self.entries:
                self.entries[w] = ps

    def get_words(self):
        return set(self.entries.keys())

    def format_vocabulary(self):
        retval = []
        for w, ps in sorted(self.entries.iteritems(), lambda x, y : compare(x[0], y[0])):
            for n, p in sorted(ps.iteritems()):
                if "REJ" not in p:
                    if w == "~SIL":
                        retval.append("%s(%.2d) VOCAB_NIL_WORD 1.0" % (w, n))
                    else:
                        retval.append("%s(%.2d) %s" % (w, n, w))

        return "\n".join(retval)

    def format_dictionary(self):
        retval = []
        for w, ps in sorted(self.entries.iteritems(), lambda x, y : compare(x[0], y[0])):
            for n, p in sorted(ps.iteritems()):
                if "REJ" not in p:
                    if len(p) == 1:
                        pp = " ".join(p + ["[ wb ]"])
                    else:
                        pp = " ".join([p[0]] + ["[ wb ]"] + p[1:] + ["[ wb ]"])
                retval.append("%s(%.2d) %s" % (w, n, pp))
        return "\n".join(retval)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--dict_input", dest="dict_input")
    parser.add_argument("-A", "--dict_added", dest="dict_added")
    parser.add_argument("-d", "--dictionary", dest="dictionary")
    parser.add_argument("-v", "--vocabulary", dest="vocabulary")

    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("-u", "--unigrams", dest="unigrams")
    parser.add_argument("-w", "--weight", dest="weight", type=float, default=.1)
    parser.add_argument("-l", "--language_model", dest="language_model")
    options = parser.parse_args()

    if options.dict_input:
        with meta_open(options.dict_input) as fd, meta_open(options.dict_added) as afd:
            old_dict = Dictionary(fd)
            print len(old_dict.entries)
            added_dict = Dictionary(afd)
            old_dict.add_entries(added_dict)
            print len(old_dict.entries)
        with meta_open(options.dictionary, "w") as new_dict, meta_open(options.vocabulary, "w") as new_vocabulary:
            new_dict.write(old_dict.format_dictionary())
            new_vocabulary.write(old_dict.format_vocabulary())        
    else:

        with meta_open(options.input) as fd, meta_open(options.unigrams) as ufd:
            unis = [x.strip() for x in ufd]
            arpabo = Arpabo(fd)

        print arpabo.unigram_sum(1), len(arpabo.grams[1])
        arpabo.add_unigrams(unis, options.weight)
        print arpabo.unigram_sum(1), len(arpabo.grams[1])

        with meta_open(options.language_model, "w") as lm, meta_open(options.dictionary, "w") as dictionary, meta_open(options.vocabulary, "w") as vocab:
            lm.write(arpabo.language_model())
            dictionary.write(arpabo.dictionary())
            vocab.write(arpabo.vocabulary())

