import re
import gzip
import math
import locale
import operator
from common_tools import meta_open, Probability


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
            to_join.append("%f %s" % (prob.log10(), " ".join(toks)))
        else:
            to_join.append("%f %s %f" % (prob.log10(), " ".join(toks), bow))
    return "\n".join(to_join)


def unpack_grams(n, s):
    retval = []
    for e in s.strip().split("\n"):
        toks = e.split()
        if len(toks) == n:
            raise Exception("line \"%s\" is too short" % e)
        elif len(toks) == n + 1:
            retval.append((toks[1:], Probability(log10prob=float(toks[0])), None))
        elif len(toks) == n + 2:
            retval.append((toks[1:-1], Probability(log10prob=float(toks[0])), float(toks[-1])))
    return retval


class ProbabilityList(dict):
    """
    A dictionary of unigrams and probabilities in *negative* natural log space.
    """

    def __init__(self):
        self.special = set()

    @staticmethod
    def from_stream(s, special=["~SIL"]):
        retval = ProbabilityList()
        retval.special = set(special)
        for line in s:
            word, prob = line.split()
            retval[word] = Probability(neglogprob=float(prob))
        return retval

    @staticmethod
    def from_dict(d, special=["~SIL"]):
        retval = ProbabilityList()
        retval.special = set(special)
        for k, v in d.iteritems():
            retval[k] = v
        return retval

    def get_words(self):
        return set(self.keys())

    def filter_by(self, other):
        other_words = other.get_words()
        for k in self.keys():
            if k not in other_words and k not in self.special:
                del self[k]

    def get_top_n(self, n):
        retval = ProbabilityList()
        for w, p in sorted(self.iteritems(), lambda x, y : cmp(y[1], x[1]))[0:n]:
            retval[w] = p
        return retval
    
    def format(self):
        return "\n".join(["%s\t%f" % (k, -v.log()) for k, v in sorted(self.iteritems(), lambda x, y : cmp(y[1], x[1]))]) + "\n"

    def __str__(self):
        return "%d words, %s total probability" % (len(self), reduce(operator.add, self.values()))


class Arpabo():
    """
    The Arpabo format stores n-gram probabilities in log-10 space.
    """

    def __init__(self, file_handle, special=["~SIL"]):
        self.special = set(special)
        m = arpabo_rx.match(file_handle.read())
        self.preamble = m.group("preamble").strip()
        self.bboard = m.group("bboard").strip()
        m2 = re.finditer(r"\\(?P<n>\d+)-grams:(?P<vals>.*?)\n\n", m.group("grams"), re.S)
        self.grams = {int(x.group("n")) : unpack_grams(int(x.group("n")), x.group("vals")) for x in m2}
        self.words = set([x[0][0] for x in self.grams[1]])
        self.normalize()

    def __str__(self):
        return "%d words, %s total unigram probability" % (len(self.get_words()), self.unigram_sum())

    def unigram_sum(self):
        return reduce(operator.add, [p for g, p, b in self.grams[1]])

    def get_words(self):
        return self.words

    def get_probability_of_words(self, words):
        return reduce(operator.add, [p for w, p, b in self.grams[1] if w[0] in words])

    def get_probability_of_not_words(self, words):
        return reduce(operator.add, [p for w, p, b in self.grams[1] if w[0] not in words])

    def filter_by(self, other_vocab):
        to_keep = other_vocab.get_words()
        [to_keep.add("~SIL") for x in self.special]
        to_drop = set([p for w, p, b in self.grams[1] if w[0] not in to_keep] )
        not_drop = [p for w, p, b in self.grams[1] if w[0] in to_keep] 
        removed_prob = reduce(operator.add, [p for w, p, b in self.grams[1] if w[0] not in to_keep])
        self.grams[1] = [(w, p, b) for w, p, b in self.grams[1] if w[0] not in to_drop or w[0] in self.special]
        self.words = self.words & to_keep
        #[self.words.add(x) for x in self.special]
        self.normalize()
        return None

    def normalize(self):
        total = self.unigram_sum()
        scale = Probability(prob=1.0) / total
        self.grams[1] = [(t, p * scale, b) for t, p, b in self.grams[1]]
        return None

    def add_unigrams(self, words, weight, new_bow=.000000001):
        words = [w for w in words if w not in self.get_words()]
        for w in words:
            self.words.add(w)
        scale = Probability(prob=1.0 - weight)
        new_prob = Probability(prob=weight / len(words))
        self.grams[1] = sorted([(t, p * scale, b) for t, p, b in self.grams[1]] + [([w], new_prob, new_bow) for w in words])
        self.normalize()

    def scale_unigrams(self, scale):
        self.grams[1] = [(w, p * scale, b) for w, p, b in self.grams[1]]

    def add_unigrams_with_probs(self, words, weight=None, new_bow=.000000001):
        total_new_prob = reduce(operator.add, words.values())
        if weight != None:
            self.scale_unigrams(Probability(prob=1.0 - weight))
            weight = Probability(prob=weight)            
            words = {w : (p / total_new_prob) * weight for w, p in words.iteritems() if w not in self.get_words()}
        else:
            words = {w : p for w, p in words.iteritems() if w not in self.get_words()}
        for w in words.keys():
            self.words.add(w)

        self.grams[1] = sorted([(t, p, b) for t, p, b in self.grams[1]] + 
                                   [([w], words[w], new_bow) for w in words])

        self.normalize()

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


class Pronunciations(dict):
    """
    """
    def __init__(self, source={}, special=["~SIL"], keep_rejects=False):
        self.special = set(special)
        self.keep_rejects = keep_rejects
        if isinstance(source, list):
            for f in source:
                with meta_open(f) as ifd:
                    for k, v in source.iteritems():
                        self[k] = v
        elif isinstance(source, (file, gzip.GzipFile)):            
            for w, n, p in [re.match(r"^(\S+?)(\(\d+\))? (.*)$", l).groups() for l in source]:
                if "REJ" in p and not self.keep_rejects:
                    continue
                if n:
                    n = int(n[1:-1])
                else:
                    n = len(self.get(w, {})) + 1
                self[w] = self.get(w, {})
                self[w][int(n)] = p.replace("[ wb ]", "").split()
    
    def __str__(self):
        return "%d words, %d pronunciations" % (len(self.get_words()), sum([len(x) for x in self.values()]))

    def phones(self):
        return set(sum(sum([x.values() for x in self.values()], []), []))
    
    def add_entries(self, other):
        for w, ps in sorted(other.iteritems()):
            if w not in self and not w.startswith("<"):
                self[w] = ps
    
    def filter_by(self, other_vocab):
        other_words = other_vocab.get_words()
        new_entries = {k : self[k] for k in self.get_words() if k in other_words or k in self.special}
        self.clear()
        for k, v in new_entries.iteritems():
            self[k] = v
        return None
    
    def replace_by(self, other_prons):
        for w in other_prons.keys():
            if w in self:
                self[w] = other_prons[w]
        return None
    
    def get_words(self):
        return set(self.keys())
    
    def format_vocabulary(self, print_rejects=False):
        retval = []
        for w, ps in sorted(self.iteritems(), lambda x, y : compare(x[0], y[0])):

            for n, p in sorted(ps.iteritems()):
                if "REJ" not in p or print_rejects:
                    if w in self.special:
                        retval.append("%s(%.2d) VOCAB_NIL_WORD 1.0" % (w, n))
                    else:
                        retval.append("%s(%.2d) %s" % (w, n, w))    
        return "\n".join(retval) + "\n"
    
    def format(self, print_rejects=False):
        retval = []
        for w, ps in sorted(self.iteritems(), lambda x, y : compare(x[0], y[0])):
            for n, p in sorted(ps.iteritems()):
                if "REJ" not in p or print_rejects:
                    if len(p) == 1:
                        pp = " ".join(p + ["[ wb ]"])
                    else:
                        pp = " ".join([p[0]] + ["[ wb ]"] + p[1:] + ["[ wb ]"])
                retval.append("%s(%.2d) %s" % (w, n, pp))
        return "\n".join(retval) + "\n"


class Vocabulary(dict):

    def __init__(self, source={}, special=["~SIL"]):
        self.special = special
        if isinstance(source, list):
            for f in source:
                with meta_open(f) as ifd:
                    for k, v in source.iteritems():
                        self[k] = v
        elif isinstance(source, (file, gzip.GzipFile)):            
            for w1, n, w2 in [re.match(r"^(\S+?)\((\d+)\)? (.*)$", l).groups() for l in source]:
                if w1 != w2 and w1 not in self.special:
                    raise Exception("%s != %s" % (w1, w2))
                self[w1] = self.get(w1, []) + [int(n)]
            for w, c in [x.split() for x in source]:
                self[w] = int(c)
        elif isinstance(source, dict):
            for k, v in source.iteritems():
                self[k] = v

    @staticmethod
    def from_set(s):
        retval = Vocabulary()
        for i, x in enumerate(s):
            retval[x] = [i]
        return retval

    def get_words(self):
        return set(self.keys())

    def filter_by(self, other_vocab):
        other_words = other_vocab.get_words()
        #self = {k : self[k] for k in self.get_words() if k in other_words or k in self.special}
        return Vocabulary.from_set([k for k in self.get_words() if k in other_words or k in self.special])

    def format(self):
        retval = []
        for w, ns in sorted(self.iteritems()):            
            for n in sorted(ns):
                if w == "~SIL":
                    retval.append("%s(%.2d) VOCAB_NIL_WORD 1.0" % (w, n))
                else:
                    retval.append("%s(%.2d) %s" % (w, n, w))
        return "\n".join(retval) + "\n"

    def __str__(self):
        return "%d words, %d forms" % (len(self.get_words()), sum([len(x) for x in self.values()]))


class FrequencyList(dict):

    def __init__(self, source={}):
        if isinstance(source, list):
            for f in source:
                with meta_open(f) as ifd:
                    for k, v in source.iteritems():
                        self[k] = v
        elif isinstance(source, (file, gzip.GzipFile)):            
            for w, c in [x.split() for x in source]:
                self[w] = int(c)
        elif isinstance(source, dict):
            for k, v in source.iteritems():
                self[k] = v

    def join(self, other):
        new_counts = {}
        for word, count in self.iteritems():
            new_counts[word] = count
        for word, count in other.iteritems():
            new_counts[word] = new_counts.get(word, 0) + count
        return FrequencyList(new_counts)

    def format(self):
        return "\n".join(["%s %d" % (k, v) for k, v in self.iteritems()]) + "\n"

    def make_conservative(self):
        new_counts = {}
        for word, count in self.iteritems():
            if all([not word.startswith(x) for x in ["[", "<", "("]]) and all([not word.endswith(x) for x in ["]", ">", ")"]]) and all([not x in word for x in ["_", "*", "+"]]):
                cleaned_word = re.sub(r"\+|\*|\-", "", word.lower())
                new_counts[cleaned_word] = new_counts.get(cleaned_word, 0) + count
        return FrequencyList(new_counts)


class MorfessorOutput(dict):
    def __init__(self, fd):  
        self.morphs = {"PRE" : set(),
                       "STM" : set(),
                       "SUF" : set(),
                       }
        for l in fd:
            count, rest = re.match(r"^(\d+) (.*)$", l).groups()
            key = tuple([re.match(r"^(.*)\/(PRE|STM|SUF)$", x).groups() for x in rest.split(" + ")])
            self[key] = self.get(key, 0) + int(count)
            for morph, type in key:
                self.morphs[type].add(morph)

    def format(self):
        return "\n".join(["%d %s" % (count, " + ".join(morphs)) for morphs, count in self.iteritems()]) + "\n"


class ASRResults():
    def __init__(self, fd):
        agg, nsents, nwords, corr, sub, deleted, inserted, error, sentence_error = [l for l in fd if "aggregated" in l][0].replace("|", " ").strip().split()
        self.vals = {"error" : float(error),
                     "substitutions" : float(sub),
                     "deletions" : float(deleted),
                     "insertions" : float(inserted),
                     }
    def get(self, name):
        return self.vals[name] / 100.0


class KWSResults():
    def __init__(self, fd):
        toks = [l for l in fd if "Occurrence" in l][-1].replace("|", " ").strip().split()
        self.vals = {"pmiss" : float(toks[14]),
                     "mtwv" : float(toks[15]),
                     }
    def get(self, name):
        return self.vals[name]


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--dict_input", dest="dict_input")
    parser.add_argument("-A", "--dict_added", dest="dict_added")
    parser.add_argument("-p", "--pronunciations", dest="pronunciations")
    parser.add_argument("-v", "--vocabulary", dest="vocabulary")

    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("-u", "--unigrams", dest="unigrams")
    parser.add_argument("-w", "--weight", dest="weight", type=float, default=.1)
    parser.add_argument("-l", "--language_model", dest="language_model")
    options = parser.parse_args()

    if options.dict_input:
        with meta_open(options.dict_input) as fd, meta_open(options.dict_added) as afd:
            old_dict = Pronunciations(fd)
            print len(old_dict.entries)
            added_dict = Pronunciations(afd)
            old_dict.add_entries(added_dict)
            print len(old_dict.entries)
        with meta_open(options.pronunciations, "w") as new_dict, meta_open(options.vocabulary, "w") as new_vocabulary:
            new_dict.write(old_dict.format_pronunciations())
            new_vocabulary.write(old_dict.format_vocabulary())        
    else:

        with meta_open(options.input) as fd, meta_open(options.unigrams) as ufd:
            unis = [x.strip() for x in ufd]
            arpabo = Arpabo(fd)

        print arpabo.unigram_sum(), len(arpabo.grams[1])
        arpabo.add_unigrams(unis, options.weight)
        print arpabo.unigram_sum(), len(arpabo.grams[1])

        with meta_open(options.language_model, "w") as lm, meta_open(options.pronunciations, "w") as pronunciations, meta_open(options.vocabulary, "w") as vocab:
            lm.write(arpabo.language_model())
            pronunciations.write(arpabo.pronunciations())
            vocab.write(arpabo.vocabulary())

