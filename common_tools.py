from subprocess import Popen, PIPE
import re
import os
import gzip
import random
import glob
import cPickle
import csv
import codecs
import tempfile
import math
import tempfile
import shutil
import contextlib
import bz2
from os.path import basename, splitext
from math import log, sqrt, pow, pi, e, sinh
import tarfile
import subprocess
import shlex
#import numpy
#from scipy.special import gamma, gammainc
#from scipy import sparse
#from scipy.misc import comb

#from rpy2.robjects.conversion import ri2py
#from rpy2.robjects.numpy2ri import numpy2ri
from unicodedata import name
import logging
try:
    import lxml.etree as et
except:
    import xml.etree.ElementTree as et


def strip_file(f):
    return os.path.splitext(os.path.basename(f))[0]

    
def regular_word(w):
    return not any(w.endswith(x) for x in ["-", "*", ">", "~"]) and "(" not in w and "_" not in w


def list_to_tuples(xs, n=2):    
    return [[xs[i * n + j] for j in range(n)] for i in range(len(xs) / n)]


def pairs(xs, n):
    return [xs[i * n : (i + 1) * n] for i in range(len(xs) / n)]

def parse_location(l):
    word_id = int(l.find("word").get("id"))
    try:
        tag_id = int(l.find("tag").get("id"))    
    except:
        tag_id = None
    analysis_ids = [int(x.get("id")) for x in l.getiterator("analysis")]
    return (word_id, tag_id, analysis_ids)

class DataSet():
    def __init__(self, sentences, indexToWord, indexToTag, indexToAnalysis):
        assert(all([isinstance(x, int) for x in indexToWord.keys()]))
        assert(all([isinstance(x, basestring) for x in indexToWord.values()]))
        assert(all([isinstance(x, int) for x in indexToTag.keys()]))
        assert(all([isinstance(x, basestring) for x in indexToTag.values()]))
        assert(all([isinstance(x, int) for x in indexToAnalysis.keys()]))
        assert(all([isinstance(x, tuple) for x in indexToAnalysis.values()]))
        assert(all([all([isinstance(y, basestring) for y in x]) for x in indexToAnalysis.values()]))        
        self.sentences = sentences
        self.indexToWord = indexToWord
        self.indexToTag = indexToTag
        self.indexToAnalysis = indexToAnalysis
        wordToIndex = {v : k for k, v in self.indexToWord.iteritems()}        
        self.analysisIndexToWordIndex = {a_id : wordToIndex.get("".join([x for x in a]), None) for a_id, a in self.indexToAnalysis.iteritems()}

    @staticmethod
    def from_sentences(sentences):
        wordToIndex = {}
        tagToIndex = {}
        analysisToIndex = {}
        for s in sentences:
            for w, t, aa in s:
                wordToIndex[w] = wordToIndex.get(w, len(wordToIndex))
                if t != None:
                    tagToIndex[t] = tagToIndex.get(t, len(tagToIndex))
                for a in map(tuple, aa):
                    try:
                        analysisToIndex[a] = analysisToIndex.get(a, len(analysisToIndex))
                    except:
                        print a
                        sys.exit()
        return DataSet([[(wordToIndex[w], tagToIndex.get(t, None), [analysisToIndex[tuple(a)] for a in aa]) for w, t, aa in s] for s in sentences], 
                       {v : k for k, v in wordToIndex.iteritems()}, 
                       {v : k for k, v in tagToIndex.iteritems()},                       
                       {v : k for k, v in analysisToIndex.iteritems()})

    def words(self):
        return set(self.indexToWord.values())

    def analyses(self, word):
        return []
    
    @staticmethod
    def from_xml(xml):
        indexToWord = {int(x.get("id")) : unicode(x.text) for x in xml.findall("preamble/word_inventory/entry")}
        indexToTag = {int(x.get("id")) : x.text for x in xml.findall("preamble/tag_inventory/entry")}
        indexToAnalysis = {int(x.get("id")) : tuple([y.text for y in x.getiterator("morph")]) for x in xml.findall("preamble/analysis_inventory/entry")}
        sentences = [[parse_location(l) for l in s.getiterator("location")] for s in xml.getiterator("sentence")]
        return DataSet(sentences, indexToWord, indexToTag, indexToAnalysis)

    @staticmethod
    def from_stream(stream):
        xml = et.parse(stream)
        return [DataSet.from_xml(x) for x in xml.getiterator("dataset")]

    @staticmethod
    def from_analyses(analyses):
        wordToIndex = {}
        analysisToIndex = {}
        for c, analysis in analyses:
            c = int(c)
            analysis = tuple(analysis)
            word = "".join(analysis)
            wordToIndex[word] = wordToIndex.get(word, len(wordToIndex))
            analysisToIndex[analysis] = analysisToIndex.get(analysis, len(analysisToIndex))
        indexToWord = {v : k for k, v in wordToIndex.iteritems()}
        indexToAnalysis = {v : k for k, v in analysisToIndex.iteritems()}
        indexToTag = {}
        return DataSet([], indexToWord, indexToTag, indexToAnalysis)

    @staticmethod
    def empty():
        return DataSet([], {}, {}, {})

    def get_subset(self, indices):
        sentences = [self.sentences[i] for i in indices]
        all_words, all_analyses, all_tags = set(), set(), set()
        for w, t, aa in sum(sentences, []):
            all_words.add(w)
            if t != None:
                all_tags.add(t)
            for a in aa:
                all_analyses.add(a)
        return DataSet(sentences, 
                       {k : v for k, v in self.indexToWord.iteritems() if k in all_words},
                       {k : v for k, v in self.indexToTag.iteritems() if k in all_tags},
                       {k : v for k, v in self.indexToAnalysis.iteritems() if k in all_analyses}
                       )

    def get_analyses(self):
        retval = {}
        for a_id, w_id in self.analysisIndexToWordIndex.iteritems():            
            word = self.indexToWord.get(w_id, None)
            if not word:
                continue
            analysis = self.indexToAnalysis[a_id]
            retval[word] = retval.get(word, set())
            retval[word].add(analysis)
        return retval

    def write(self, fd):
        xml = et.TreeBuilder()
        xml.start("xml", {})
        xml.start("dataset", {})
        xml.start("preamble", {})
        xml.start("analysis_inventory", {})
        for i, a in self.indexToAnalysis.iteritems():
            xml.start("entry", {"id" : str(i)})            
            properties = {}
            for morph in a:
                xml.start("morph", properties)
                try:
                    xml.data(morph.decode("utf-8"))
                except:
                    xml.data(morph)
                xml.end("morph")
            xml.end("entry")
        xml.end("analysis_inventory")
        xml.start("tag_inventory", {})
        for i, t in self.indexToTag.iteritems():
            xml.start("entry", {"id" : str(i)}), xml.data(t), xml.end("entry")
        xml.end("tag_inventory")
        xml.start("word_inventory", {})
        for i, w in self.indexToWord.iteritems():            
            xml.start("entry", {"id" : str(i)}) 
            try:
                xml.data(w.decode("utf-8"))
            except:
                xml.data(w)
            xml.end("entry")
        xml.end("word_inventory")
        xml.end("preamble")
        xml.start("sentences", {})
        for i, s in enumerate(self.sentences):
            xml.start("sentence", {"n" : str(i + 1)})
            for j, (w, t, aa) in enumerate(s):
                xml.start("location", {"n" : str(j + 1)})
                xml.start("word", {"id" : str(w)}), xml.end("word")
                if t != None:
                    xml.start("tag", {"id" : str(t)}), xml.end("tag")
                xml.start("analyses", {})
                for a in aa:
                    xml.start("analysis", {"id" : str(a)}), xml.end("analysis")
                xml.end("analyses")
                xml.end("location")
            xml.end("sentence")
        xml.end("sentences")
        xml.end("dataset")
        xml.end("xml")
        fd.write(et.tounicode(xml.close(), pretty_print=True).encode("utf-8"))


@contextlib.contextmanager
def temp_dir(prefix="tmp", remove=True):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    yield temp_dir
    if remove:
        shutil.rmtree(temp_dir)

@contextlib.contextmanager
def temp_file(suffix="", prefix="tmp", remove=True, dir=None):
    (fid, temp_file) = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir)
    yield temp_file
    if remove:
        os.remove(temp_file)

# def run_command(cmd, env={}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, torque=False, data=None):
#     """
#     Simple convenience wrapper for running commands (not an actual Builder).
#     """
#     if torque:
#         pass
#     else:
#         logging.info("Running local command: %s", cmd)    
#         process = subprocess.Popen(shlex.split(cmd), env=env, stdin=stdin, stdout=stdout, stderr=stderr)
#         if data:
#             out, err = process.communicate(data)
#         else:
#             out, err = process.communicate()
#         return out, err, process.returncode == 0

def Exp(lam):
    return -log(random.random()) / lam

def gamma(x):
    return sqrt(2 * pi / x) * pow((x / e) * sqrt(x * sinh(1.0 / x) + 1.0 / (810 * pow(x, 6))), x)

class Probability():

    def __init__(self, prob=None, logprob=None, log2prob=None, log10prob=None, neglogprob=None, neglog2prob=None, neglog10prob=None):
        vals = [prob, logprob, log2prob, log10prob, neglogprob, neglog2prob, neglog10prob]
        if all([x == None for x in vals]):
            raise Exception("no value was specified for Probability object!")
        elif len([x for x in vals if x != None]) > 1:
            raise Exception("only one type of value may be specified for Probability object!")
        elif prob != None:
            self.logprob = math.log(prob)
        elif logprob != None:
            self.logprob = logprob
        elif log2prob != None:
            self.logprob = log2prob / math.log2(math.e)
        elif log10prob != None:
            self.logprob = log10prob / math.log10(math.e)
        elif neglogprob != None:
            self.logprob = -neglogprob
        elif neglog2prob != None:
            self.logprob = (-neglog2prob) / math.log2(math.e)
        elif neglog10prob != None:
            self.logprob = (-neglog10prob) / math.log10(math.e)

    def __str__(self):
        return "%f/%f" % (self.prob(), self.logprob)

    def prob(self):
        return math.exp(self.logprob)

    def log(self):
        return self.logprob

    def log2(self):
        return self.logprob / math.log(2, math.e)

    def log10(self):
        return self.logprob / math.log(10, math.e)

    def __mul__(self, other):
        return Probability(logprob=self.logprob + other.logprob)

    def __div__(self, other):
        return Probability(logprob=self.logprob - other.logprob)

    def __add__(self, other):        
        logprobs = sorted([self.logprob, other.logprob])        
        pi = logprobs[-1]
        return Probability(logprob=pi + math.log(sum([math.exp(log_p - pi) for log_p in logprobs])))

    def __cmp__(self, other):
        return cmp(self.log(), other.log())
    #def __sub__(self, other):
    #    return self + Probability(prob=-other.prob())
        #logprobs = sorted([self.logprob, other.logprob])        
        #pi = logprobs[-1]
        #return Probability(logprob=pi + math.log(sum([math.exp(log_p - pi) for log_p in logprobs])))

def jags_to_numpy(samples):
    samples = ri2py(samples)
    dimensions = {}
    data = {}
    values = []
    for chain_num, chain in enumerate(samples):
        chain = numpy.asarray(chain)
        nodes = [x for x in samples[chain_num].dimnames[1]]
        nsamples = chain.shape[0]
        for i, node in enumerate(nodes):
            name, indices = re.match(r"^([^\[]*)(\[.*\])?$", node).groups()
            for sample_num in range(nsamples):
                findices = [chain_num + 1, sample_num + 1] + [int(x) for x in indices.strip("[]").split(",")]
                dimensions[name] = dimensions.get(name, findices)
                for j, val in enumerate(findices):
                    if val > dimensions[name][j]:
                        dimensions[name][j] = val
                values.append((name, [[k - 1] for k in findices], chain[sample_num, i]))
    for name, shape in dimensions.iteritems():
        data[name] = numpy.zeros(shape=shape)
    for name, indices, value in values:
        data[name][indices] = value
    return data

def Gamma(k=1, theta=1):
    if k == 1 and theta == 1:
        return -log(random.random())
    elif k == 1:
        return theta * Gamma()
    else:
        return 1.0

def beta(x, y):
    return gamma(x) * gamma(y) / gamma(x + y)

def Beta(alpha, beta):
    X = Gamma(alpha, 1)
    Y = Gamma(beta, 1)
    return X / (X + Y)

def adjusted_rand(clusterA, clusterB):
    retval = 0.0
    N00, N11, N01, N10 = 0, 0, 0, 0
    mat = numpy.zeros(shape=(max(clusterA) + 1, max(clusterB) + 1))
    for a, b in zip(clusterA, clusterB):
        mat[a, b] += 1
    A = mat.sum(1)
    B = mat.sum(0)
    retval = (comb(mat, 2).sum() - (comb(A, 2).sum() * comb(B, 2).sum() / comb(len(clusterA), 2))) / \
        (.5 * (comb(A, 2).sum() + comb(B, 2).sum()) - (comb(A, 2).sum() * comb(B, 2).sum()) / comb(len(clusterA), 2))
    return retval


class Discrete():
    def __init__(self, probs, vals):
        self.probs = probs
        self.vals = vals
    def sample(self):
        val = random.random()
        cur = self.probs[0]
        i = 0
        while cur < val:
            i += 1
            cur += self.vals[i]
            pass
        return self.vals[i]
        pass

class Gaussian():
    def __init__(self, mean, standard_deviation):
        self.mean = mean
        self.standard_deviation = standard_deviation
    def sample(self):
        return random.gauss(self.mean, self.standard_deviation)

class DirichletProcess():
    def __init__(self, base_distribution, alpha):
        self.base_distribution = base_distribution
        self.alpha = alpha
    def sample(self, max_classes):
        betas = [random.betavariate(1.0, self.alpha) for i in range(max_classes)]
        probs = []
        vals = []
        rem = 1.0
        for x in betas:
            probs.append(rem * x)
            rem = rem - probs[-1]
            vals.append(self.base_distribution.sample())
        return Discrete(probs, vals)
    def collapsed_sample(self):
        pass

def soft_rand(test, gold):
    retval = 0.0
    if isinstance(test[0], (int,)):
        A = numpy.zeros(shape=(len(test), max(test) + 1))
        for i, x in enumerate(test):
            A[i, x] = 1.0
    else:
        A = numpy.asarray(test)
    if isinstance(gold[0], (int,)):
        B = numpy.zeros(shape=(len(gold), max(gold) + 1))
        for i, x in enumerate(gold):
            B[i, x] = 1.0
    else:
        B = numpy.asarray(gold)
    A = numpy.asmatrix(A)
    B = numpy.asmatrix(B)
    N = numpy.asarray(A.T * B)

    #phi = A.shape[0] / N.sum(1)

    #print N, phi

    NN = N * N
    #print NN
    jsum = numpy.sum([N.sum(1)[i] * N.sum(1)[i] for i in range(N.shape[0])])
    isum = numpy.sum([N.sum(0)[j] * N.sum(0)[j] for j in range(N.shape[1])])

    a = numpy.sum(N * (N - 1)) / 2.0
    b = (isum - numpy.sum(NN)) / 2.0
    c = (jsum - numpy.sum(NN)) / 2.0
    d = ((N.sum() * N.sum()) + numpy.sum(NN) - (isum + jsum)) / 2.0

    # 33, 75, 0, -85
    return (a - ((a + c) * (a + b) / (a + b + c + d))) / ((((a + c) + (a + b)) / 2) - (((a + c) * (a + b))/ (a + b + c + d)))


def adjusted_soft_rand(x, y):
    return soft_rand(x, y) / max(soft_rand(x, x), soft_rand(y, y))

def modified_purity(assignments, gold, mcs=1):
    retval = 0.0
    prevs = {}
    for a, b in zip(assignments, gold):
        prevs[a] = prevs.get(a, {})
        prevs[a][b] = prevs[a].get(b, 0) + 1
    total = 0
    correct = 0
    for key in prevs.keys():
        num, best = sorted([(x, i) for i, x in prevs[key].iteritems()])[-1]
        if num >= mcs:
            correct += num
            total += sum(prevs[key].values())
    if total == 0:
        return float(0)
    return float(correct) / float(total)

def harmonic_mean(precision, recall):
    return (2 * precision * recall) / (precision + recall)

def unpoint(token):
    return u"".join([x for x in token.decode('utf-8') if "LETTER" in name(x, "UNKNOWN")])

def meta_splitext(fname, max_depth=None, acc=[]):
    name, ext = os.path.splitext(fname)
    if max_depth <= 0 or ext == "":
        return (name, "".join(acc + [ext]))
    elif max_depth == None:
        return meta_splitext(name, max_depth, acc + [ext])
    else:
        return meta_splitext(name, max_depth - 1, acc + [ext])


def meta_basename(fname, mdepth=1):
    if mdepth == 0:
        return basename(fname).split(".")[0]
    else:
        temp_bname = basename(fname).split(".")
        return ".".join(temp_bname[0 : max(1, len(temp_bname) - mdepth)])
    if fname.endswith(".bz") or fname.endswith(".gz"):
        return splitext(splitext(basename(fname))[0])[0]
    else:
        return splitext(basename(fname))[0]


def meta_open(fname, mode="r", enc="utf-8"):
    if fname.endswith(".tgz"):
        return tarfile.open(fname, "%s:gz" % mode)
    elif fname.endswith(".tbz2"):
        return tarfile.open(fname, "%s:bz2" % mode)
    elif fname.endswith(".tar"):
        return tarfile.open(fname, "%s" % mode)
    elif fname.endswith(".gz"):
        g = gzip.open(fname, mode)
        if enc:
            reader = codecs.getreader(enc)
            return reader(g)
        else:
            return g    
    elif fname.endswith(".bz2"):
        return bz2.BZ2File(fname, mode)
    else:
        if enc:
            return codecs.open(fname, mode, enc)
        else:
            return open(fname, mode)


def generic_emitter_old(target, source, env, name="", ext="", remove=None, extra_targets=[], targets=[]):
    """
    Adjusts the first target to include the bound values "name" and "ext",
    as well as the key-value pairs of the dictionary passed as the final
    source.
    """
    if not any([hasattr(x, "read") for x in source]):
        source.append(env.Value({}))
    #if target[0].rstr() != source[0].rstr():
        #print target[0].rstr(), source[0].rstr()
    #    return target, source
    if remove:
        target[0] = re.sub(remove, "", source[0].rstr())
    else:
        target[0] = source[0].rstr()
    target[0] = os.path.join(env.get("DIR", os.path.dirname(target[0])),
                             meta_basename(target[0], True))
    if name != "":
        target[0] = target[0] + "_%s" % name
    if env.get("ADD_DICT", True):
        env["ARGS"] = source[-1].read()
        newdict = {}
        for k, v in env["ARGS"].iteritems():
            if callable(v):
                if hasattr(v, "func"):
                    target[0] = target[0] + "_%s=%s" % (k, v.func.func_name)
                    newdict[k] = v.func.func_name
                else:
                    target[0] = target[0] + "_%s=%s" % (k, v.func_name)
                    newdict[k] = v.func_name
            elif isinstance(v, list):
                target[0] = target[0] + "_%s=%s" % (k, "".join([str(x) for x in v]))
                newdict[k] = v
            else:
                target[0] = target[0] + "_%s=%s" % (k, v)
                newdict[k] = v
        source[-1] = env.Value(newdict)
    for (xname, xext) in extra_targets:
        if xname != "":
            xname = "_" + xname
        target.append("%s%s.%s" % (target[0], xname, xext))
    if ext != "":
        target[0] = "%s.%s" % (target[0], ext)
    if name:
        env.Alias(name, target)
    return target, source


def generic_emitter(target, source, env, remove=None, targets=[]):
    """
    Adjusts the first target to include the bound values "name" and "ext",
    as well as the key-value pairs of the dictionary passed as the final
    source.
    """
    btarget = target
    new_targets = []
    if not any([hasattr(x, "read") for x in source]):
        source.append(env.Value({}))
    if remove:
        newtarg = re.sub(remove, "", source[0].rstr())
    else:
        newtarg = source[0].rstr()

    newtarg = os.path.join(env.get("DIR", os.path.dirname(newtarg)),
                           meta_basename(newtarg, 1))
    if env.get("ADD_DICT", True):
        env["ARGS"] = source[-1].read()        
        newdict = {}
        for k, v in env["ARGS"].iteritems():
            if callable(v):
                #newdict[k.lstrip("_")] = v
                if hasattr(v, "func"):
                    if not k.startswith("_"):
                        newtarg = newtarg + "_%s=%s" % (k, v.func.func_name)
                    newdict[k.lstrip("_")] = v.func.func_name
                else:
                    if not k.startswith("_"):
                        newtarg = newtarg + "_%s=%s" % (k, v.func_name)
                    newdict[k.lstrip("_")] = v.func_name
            elif isinstance(v, list):
                if not k.startswith("_"):
                    newtarg = newtarg + "_%s=%s" % (k, "".join([str(x) for x in v]))                
                newdict[k.lstrip("_")] = v
            else:
                if not k.startswith("_"):
                    newtarg = newtarg + "_%s=%s" % (k, v)
                newdict[k.lstrip("_")] = v
    else:
        newdict = source[-1].read()
    env["ARGS"] = dict([(k.lstrip("_"), v) for k, v in env["ARGS"].iteritems()])
    source[-1] = env.Value(newdict)
    for name, ext in targets:
        if name:
            env.Alias(name, target)
            name = "_%s" % name
        new_targets.append("%s%s.%s" % (newtarg, name, ext))
    if isinstance(target[0], basestring):
        t = target[0]
    else:
        t = target[0].rstr()
    if isinstance(source[0], basestring):
        s = source[0]
    else:
        s = source[0].rstr()
    if re.match("^%s\.\S+$" % t, s):
        return new_targets, source
    else:
        return target, source


def unpack_numpy(fname, dense=False, labelname=None, featurename=None, oldstyle=False, env={}):
    if not oldstyle:
        pass
    if fname.endswith("pkl.gz"):
        mat, labels, features = cPickle.load(meta_open(fname))
    else:
        try:
            fd = numpy.load(meta_open(fname))
            mat, labels, features = fd["data"], fd["labels"], fd["features"]
        except:
            mat, labels, features = numpy.load(meta_open(fname))
    if "TRANSFORM_LABELS" in env:
        labels = [env["TRANSFORM_LABELS"](l) for l in labels]
    if mat.shape == ():
        mat = mat.tolist()
    if sparse.issparse(mat) and dense==True:
        mat = numpy.asarray(mat.todense())      
    if len(labels[0]) == 1 and "_NAME" not in labels[0]:
        for l in labels:
            l["_NAME"] = l[l.keys()[0]].split(".")[0]
    if len(features[0]) == 1 and "_NAME" not in features[0]:
        for l in features:
            l["_NAME"] = l[l.keys()[0]].split(".")[0]
    logging.info("read %s of shape %s, %d labels and %d features", type(mat), mat.shape, len(labels), len(features))
    return mat, labels, features


def pack_numpy(fname, data=[], labels=[], features=[]):
    """
    data is a numpy array or scipy matrix
    labels and features are lists of either strings, dicts or tuples
    """
    if isinstance(labels[0], basestring):
        labels = [{"_NAME" : x} for x in labels]
    elif isinstance(labels[0], tuple) or isinstance(labels[0], list):
        labels = [dict(x) for x in labels]
    if isinstance(features[0], basestring):
        features = [{"_NAME" : x} for x in features]
    elif isinstance(features[0], tuple) or isinstance(features[0], list):
        features = [dict(x) for x in features]
    numpy.savez(meta_open(fname, "w"), data=data, labels=labels, features=features)
    logging.info("wrote %s of shape %s", type(data), data.shape)



def kullback_leibler_divergence(dist1, dist2):
    s1 = sum(dist1)
    s2 = sum(dist2)
    ret = 0.0
    for i in range(len(dist1)):
        if dist1[i] == 0 or dist2[i] == 0:
            continue
        p1 = dist1[i] / float(s1)
        p2 = dist2[i] / float(s2)
        try:
            ret += p1 * (log(p1 / p2))
        except:
            pass
    return ret


def jensen_shannon_divergence(dists, counts=False, total=False):
    """
    dists is a matrix where the rows are observations of the object over its possible values (the columns)

    Return JSD and p-value (probability of drawing the data from the same distribution by chance) for
    each combination of rows
    """
    #print dists
    dists = numpy.asarray(dists, dtype=float)
    shape = dists.shape
    nonzeros = [i for i, x in enumerate(dists.sum(1)) if x.sum() > 0]
    zeros = [i for i, x in enumerate(dists.sum(1)) if not x.sum() > 0 and i <= len(nonzeros)]
    zeros += [len(nonzeros) for i, x in enumerate(dists.sum(1)) if not x.sum() > 0 and i > len(nonzeros)]
    npydists = numpy.asarray(dists[nonzeros, :])
    #npydists = numpy.asarray(dists)
    #logging.info("distro matrix of shape %s has %s non-zero columns", dists.shape, len(nonzeros))
    if counts:
        weights = npydists.sum(1) / npydists.sum()
    else:
        weights = [1.0 / npydists.shape[0] for x in npydists]
    #print type(npydists)
    norm_dists = numpy.transpose(npydists.T / npydists.sum(1))
    #wdists = norm_dists * weights
    if total:
        pass
    else:
        entropies = entropy(norm_dists, 1)
        jsds = [[entropy(norm_dists[[i, j], :].sum(0) / 2.0) - (entropies[[i, j], :].sum() / 2.0)
                 for j in range(entropies.shape[0])] for i in range(entropies.shape[0])]
    #jsd = entropy(wdists.sum(1)) - sum([entropy(norm_dists[:, i]) * weights[i] for i in range(len(weights))]).sum()
    #if counts:
    #    return jsd, jsd_s(jsd, k=npydists.shape[1], N=npydists.sum(), m=npydists.shape[0])
    #else:
    #    return jsd, None
    #full_jsds = [[x for x in item] for item in ]
    jsds = numpy.insert(jsds, zeros, 0.0, axis=0)
    jsds = numpy.insert(jsds, zeros, 0.0, axis=1)
    return numpy.asarray(jsds) #, len(jsds)

def jensen_shannon_divergence_old(dists, counts=False):
    """
    dists is a matrix where the rows are observations of the object over its possible values (the columns)

    Return JSD and p-value (probability of drawing the data from the same distribution by chance) for
    each combination of rows
    """
    dists = numpy.asfarray(dists)
    nonzeros = [i for i, x in enumerate(dists.sum(1)) if x.sum() > 0]
    npydists = dists[nonzeros, :]
    #logging.info("distro matrix of shape %s has %s non-zero columns", dists.shape, len(nonzeros))
    if counts:
        weights = npydists.sum(1) / npydists.sum()
    else:
        weights = [1.0 / npydists.shape[0] for x in npydists]
    norm_dists = npydists.T / npydists.sum(1)
    wdists = norm_dists * weights
    jsd = entropy(wdists.sum(1)) - sum([entropy(norm_dists[:, i]) * weights[i] for i in range(len(weights))]).sum()
    if counts:
        return jsd, jsd_s(jsd, k=npydists.shape[1], N=npydists.sum(), m=npydists.shape[0])
    else:
        return jsd, None



def jsd_s(x, k, N, m):
    #print "x=%f k=%f N=%f m=%f" % (x, k, N, m)
    v = (k - 1.0) * (m - 1.0)
    return gammainc(v / 2.0, N * log(2.0) * x) / gamma(v / 2.0)


def log_likelihood(counts):
    actual_counts = numpy.asfarray(counts)
    actual_totals = actual_counts.sum(0)
    r_sums = actual_counts.sum(1)
    expected_proportions = actual_totals / actual_totals.sum()
    expected_counts = numpy.asfarray([x * expected_proportions for x in r_sums])
    retval = numpy.asfarray(numpy.zeros(shape=actual_counts.shape))
    for i, (this, this_expected) in enumerate(zip(actual_counts, expected_counts)):
        that = actual_totals - this
        that_expected = actual_totals - this_expected
        res = (2.0 * ((this * numpy.log(this / this_expected)) + (that * numpy.log(that / that_expected))))
        retval[i] = (2.0 * ((this * numpy.log(this / this_expected)) + (that * numpy.log(that / that_expected))))
        signs = numpy.ones(shape=(retval[i].shape))
        for j in range(signs.shape[0]):
            if this_expected[j] > this[j]:
                signs[j] = -1
        retval[i] *= signs
    return retval


def _log_likelihood(counts, totals, sign=False):
    """
    Calculate log-likelihood

    Takes 1xN vectors of counts and totals, returns log-likelihood

    when sign is True, a negative ll means underuse in first corpus
    """
    counts = numpy.asarray(counts)
    totals = numpy.asarray(totals)
    ret = numpy.empty(shape=totals.shape)
    exp_freq = counts.astype(float).sum() / totals.sum()
    exp_counts = exp_freq * totals
    ll = 2.0 * numpy.sum(counts * numpy.log(counts / exp_counts))
    if sign and counts[0] < exp_counts[0]:
        return -ll
    else:
        return ll


def _log_likelihood_wrapper(counts, totals):
    counts = numpy.asarray(counts)
    totals = numpy.asarray(totals)
    ttotal = totals.sum()
    tcount = counts.sum()
    return [log_likelihood([c, tcount - c], [t, ttotal - t], sign=True) for (c, t) in zip(counts, totals)]


def old_pearson(dists):
    newA = [x for i, x in enumerate(dists[0]) if x > 0.0 and dists[1][i] > 0.0]
    newB = [x for i, x in enumerate(dists[1]) if x > 0.0 and dists[0][i] > 0.0]
    return pearsonr(newA, newB)[0]


def old_spearman(dists):
    newA = [x for i, x in enumerate(dists[0]) if x > 0.0 and dists[1][i] > 0.0]
    newB = [x for i, x in enumerate(dists[1]) if x > 0.0 and dists[0][i] > 0.0]
    return spearmanr(newA, newB)[0]


def multidist(distA, distB, func=None, distweights=None):
    dists = {}    
    for f in set.intersection(set(distA.keys()), set(distB.keys())):
        if isinstance(distA[f], float):
            try:
                k = f.split("_=_")[0].split("-")[1]
            except:
                k = "SINGLE"
            if k not in dists:
                dists[k] = []
            dists[k].append((distA[f], distB[f]))
    vals = []
    for k, v in [x for x in dists.iteritems()]:
        res = func([x for x, y in v], [y for x, y in v])
        if isinstance(res, float):
            score = res
        else:
            score = res[0]
        if distweights == None:
            vals.append((k, score))
        elif distweights.get(k, 0.0) > 0.0:
            #print k, score, distweights.get(k, 0.0)
            vals.append((k, distweights.get(k, 0.0) * score))
    if distweights:        
        return sum([y for x, y in vals])
    else:
        return sum([y for x, y in vals]) / float(len(vals))


def singledist(distA, distB, func=None, distweights=None):
    A = []
    B = []
    for f in set.intersection(set(distA.keys()), set(distB.keys())):
        if isinstance(distA[f], float):
            A.append(distA[f])
            B.append(distB[f])
    res = func(A, B)
    if isinstance(res, float):
        return res
    else:
        return res[0]


def tei_splitter(fname, level="chapter"):
    text = et.parse(meta_open(fname))
    items = text.xpath("//div[@type='%s']" % level)
    return [(item, dict([(x.get("type", "UNKNOWN"), x.get("n", "UNKNOWN")) for x in item.xpath("ancestor-or-self::div")])) for item in items]


def file_splitter(fname):
    return [(meta_open(fname).read(), {"LABEL_filename" : os.path.basename(fname)})]


gs_rx = re.compile(r"^(\S+)\s*?\n\n(^\S+ \S+.*?)\n\n", re.S|re.M)
def lexicon_loader(fname):
    data = {}
    for m in gs_rx.finditer(meta_open(fname).read()):
        data[m.group(1)] = {}
        for scf, val in [x.split() for x in m.group(2).split("\n")]:
            data[m.group(1)][scf] = float(val)
    if len(data) == 0:
        for l in meta_open(fname):
            verb, scf, freq, count = l.strip().split()
            data[verb] = data.get(verb, {})
            data[verb][scf] = float(freq)
    return data

def gs_loader(fname):
    clusters = set()
    data = {}
    for m in re.finditer(r"^(\w+)\n(.*?)\n\s*\n", meta_open(fname).read(), re.M|re.S):
        verb = m.group(1)
        try:
            data[verb] = dict([(cluster, float(freq)) for cluster, freq in [x.split() for x in m.group(2).strip().split("\n")]])
            for k in data[verb].keys():
                clusters.add(k)
        except:
            pass
    clusters = sorted(clusters)
    verbs = sorted(data.keys())
    retval = numpy.zeros(shape=(len(data), len(clusters)))
    for row in range(len(verbs)):
        for col in range(len(clusters)):
            retval[row, col] = data[verbs[row]].get(clusters[col], 0)
    return verbs, clusters, retval

class GoldStandard():
    item_names = []
    class_names = []
    data = None
    def __init__(self, fname):
        clusters = set()
        tdata = {}
        for m in re.finditer(r"^(\w+)\n\n?(.*?)\n\s*\n", meta_open(fname).read(), re.M|re.S):
            verb = m.group(1)
            try:
                tdata[verb] = dict([(cluster.lstrip("0"), float(freq)) for cluster, freq in [x.split() for x in m.group(2).strip().split("\n")]])
                for k in tdata[verb].keys():
                    clusters.add(k)
            except:
                pass
        self.class_names = sorted(clusters)
        self.item_names = sorted(tdata.keys())
        self.data = numpy.zeros(shape=(len(tdata), len(clusters)))
        for row in range(len(self.item_names)):
            for col in range(len(self.class_names)):
                self.data[row, col] = tdata[self.item_names[row]].get(self.class_names[col], 0)

def soft_clustering_jsd(clusterA, clusterB, clusterC=None):
    common_items = sorted(set(clusterA.item_names).intersection(set(clusterB.item_names)))
    if clusterC:
        common_items = sorted(set(common_items).intersection(set(clusterC.item_names)))
    common_classes = sorted(set(clusterA.class_names).intersection(set(clusterB.class_names)))
    data = numpy.zeros(shape=(len(common_items), len(common_classes), 2))
    AIidx = [clusterA.item_names.index(x) for x in common_items]
    BIidx = [clusterB.item_names.index(x) for x in common_items]
    ACidx = [clusterA.class_names.index(x) for x in common_classes]
    BCidx = [clusterB.class_names.index(x) for x in common_classes]
    data[:, :, 0] = clusterA.data[AIidx, :][:, ACidx]
    data[:, :, 1] = clusterB.data[BIidx, :][:, BCidx]
    total = 0.0
    for i in range(data.shape[0]):
        jsd = jensen_shannon_divergence(data[i, :, :].T)[0, 1]
        total += jsd
        print common_items[i], jsd
    return total / data.shape[0]

def read_cpp(fname):
    retval = {}
    xml = et.fromstring(meta_open(fname).read())

    for matrix in xml.xpath("descendant::matrices/item"):
        data = []
        name = matrix.xpath("first/text()")[0]
        for row in matrix.xpath("second/data/item"):
            data.append([float(x) for x in row.xpath("item/text()")])
        rows = dict([(int(x.xpath("first/text()")[0]), "".join(x.xpath("second/text()"))) 
                     for x in matrix.xpath("descendant::row_to_name/item")])
        cols = dict([(int(x.xpath("first/text()")[0]), "".join(x.xpath("second/text()"))) 
                     for x in matrix.xpath("descendant::col_to_name/item")])

        for i in rows.keys():
            if rows[i] == '':
                rows[i] = str(i + 1)
                
        for i in cols.keys():
            if cols[i] == '':                
                cols[i] = str(i + 1)

        rows = [rows[i] for i in range(len(rows))]
        cols = [cols[i] for i in range(len(cols))]
        retval[name] = rows, cols, numpy.asarray(data, dtype=float)
    return retval

def write_cpp(vals, fname):
    ofd = meta_open(fname, "w")
    for name, (rows, cols, data) in vals.iteritems():
        ofd.write("%s\n" % name)
        ofd.write("\t".join(rows) + "\n")
        ofd.write("\t".join(cols) + "\n")
        for row in data:
            ofd.write("\t".join(["%f" % x for x in row]) + "\n")

def entropy(dist, axis=None):
    return -numpy.sum(dist * numpy.nan_to_num(numpy.log(dist)), axis)

def seq_entropy(A):
    return -sum([y * log(y) for y in [A.count(x) / float(len(A)) for x in set(A)]])

def mutual_information(A, B):
    retval = 0.0
    combined = zip(A, B)
    N = float(len(A))
    for i in set(A):
        for j in set(B):
            nijn = combined.count((i, j)) / N
            if nijn > 0:
                retval += nijn * log(nijn / ((A.count(i) * B.count(j)) / pow(N, 2)))
    return retval

def normalized_information_distance(A, B):
    return 1.0 - mutual_information(A, B) / max(seq_entropy(A), seq_entropy(B))

def v_measure(clusts, gold, beta=1.0):
    def _entropy(assignments):
        poss = {v : i for i, v in enumerate(set(assignments))}
        dist = numpy.zeros(shape=(len(poss)))
        for a in assignments:
            dist[poss[a]] += 1.0
        dist /= dist.sum()
        return -(dist * numpy.log2(dist)).sum()
    def _conditional_entropy(assignmentsA, assignmentsB):
        poss = {v : i for i, v in enumerate(set(assignmentsA))}
        distA = numpy.zeros(shape=(len(poss)))
        for a in assignmentsA:
            distA[poss[a]] += 1.0
        distA /= distA.sum()
        possB = {v : i for i, v in enumerate(set(assignmentsB))}
        distB = numpy.zeros(shape=(len(possB)))
        for a in assignmentsB:
            distB[possB[a]] += 1.0
        distB /= distB.sum()
        poss = {v : i for i, v in enumerate(set(zip(assignmentsA, assignmentsB)))}
        distAB = numpy.zeros(shape=len(poss))
        for a, b in zip(assignmentsA, assignmentsB):
            distAB[poss[(a, b)]] += 1.0
        distAB /= distAB.sum()
        return (distAB * numpy.log2(numpy.array([distB[possB[b]] / distAB[i] for (a, b), i in poss.iteritems()]))).sum()
    h = 1.0 - (_conditional_entropy(gold, clusts) / _entropy(gold))
    c = 1.0 - (_conditional_entropy(clusts, gold) / _entropy(clusts))
    return ((1.0 + beta) * h * c) / (beta * h + c)

def parse_rasp_file(text):
    return [parse_rasp_sentence(x) for x in re.split(r"\n\n", text)]

def parse_rasp_word(text):
    try:
        word, number, tag = re.match(r"^\|(\S+)\:(\d+)_(\S+)\|$", text).groups()
    except:
        raise BaseException(text)
    return word, number, tag

def parse_rasp_header(text):
    return []

def parse_rasp_dependency(text):
    toks = [x for x in text[1:-1].split() if x not in ["_", "|;|"]]
    try:
        rel, head, dep = toks
        head = parse_rasp_word(head)
        dep = parse_rasp_word(dep)
        return head, dep
    except:
        return ()

def parse_rasp_sentence(text):
    lines = text.strip().split("\n")
    words = parse_rasp_header(lines[0])
    deps = filter(lambda x : x != (), [parse_rasp_dependency(x) for x in lines[1:]])
    return deps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("-w", "--word", dest="word")
    options = parser.parse_args()

    with meta_open(options.input) as ifd:
        d = DataSet.from_stream(ifd)[0]
        w2i = {v : k for k, v in d.indexToWord.iteritems()}
        a2i = {v : k for k, v in d.indexToAnalysis.iteritems()}
        wi2ai = {v : k for k, v in d.analysisIndexToWordIndex.iteritems()}
        print d.indexToAnalysis[wi2ai[w2i[options.word]]]
