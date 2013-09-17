from SCons.Builder import Builder
from SCons.Script import FindFile
from SCons.Node.FS import Dir, File
from SCons.Node.Alias import Alias
import re
import logging
import pickle
import gzip
import random
from glob import glob
import os.path
import numpy
from rpy2.robjects.packages import importr
from rpy2.robjects import ListVector, IntVector, StrVector
from rpy2.robjects.conversion import ri2py
from rpy2.robjects.numpy2ri import numpy2ri
from common_tools import meta_open, meta_basename, jags_to_numpy
import sys
import math

#r.options(warn=-1)
tempout = sys.stdout
sys.stdout = open("/dev/null", "w")
r_MixSim = importr("MixSim")
r_clv = importr("clv")
r_clues = importr("clues")
rjags = importr("rjags")
coda = importr("coda")
clues = importr("clues")
stats = importr("stats")
[rjags.load_module(x) for x in ["glm", "dic", "mix", "msm", "nonparametrics", "lecuyer"]]
sys.stdout = tempout

def evaluate_clustering(target, source, env):
    cluster_names = {}
    gdata = {}
    for verb, cluster_name in [x.strip().split() for x in meta_open(source[1].rstr())]:
        if cluster_name not in cluster_names:
            cluster_names[cluster_name] = len(cluster_names)
        gdata[verb] = cluster_names[cluster_name] + 1

    data = {}
    for i, cluster in enumerate([x.split() for x in meta_open(source[0].rstr())]):
        for verb in cluster:
            data[verb] = i + 1

    kverbs = sorted(set(data.keys()).intersection(gdata.keys()))
    tcand, tgold = [data[x] for x in kverbs], [gdata[x] for x in kverbs]

    gmapping = dict([(x, i + 1) for i, x in enumerate(set(tgold))])
    cmapping = dict([(x, i + 1) for i, x in enumerate(set(tcand))])
    gold = [gmapping[x] for x in tgold]
    cand = [cmapping[x] for x in tcand]

    #print harmonic_mean(modified_purity(cand, gold, mcs=2), modified_purity(gold, cand, mcs=1))
    open(target[0].rstr(), "w").write(str(r_clues.adjustedRand(cand, gold)))
    return None


def cluster_verbs(target, source, env):
    args = source[-1].read()
    verbs, samples = pickle.load(meta_open(source[0].rstr()))
    samples = samples.sum(2)
    data = numpy.transpose(samples.T / samples.sum(1))
    res = stats.kmeans(numpy2ri(data), centers=args.get("clusters", 20)) #data[args["matrix"]].shape[0] / 10)
    ofd = meta_open(target[0].rstr(), "w")
    for c in set(res.rx2("cluster")):
        ofd.write(" ".join([verbs[i] for i, a in enumerate(res.rx2("cluster")) if a == c]) + "\n")
    return None


def run_jags(target, source, env):
    args = source[-1].read()
    data, lookups = pickle.load(meta_open(source[1].rstr()))
    for k, v in args.get("parameters", {}).iteritems():
        data[k] = v
    model = rjags.jags_model(source[0].rstr(), data=ListVector(data), n_chains=args.get("chains", 2), n_adapt=args.get("adapt", 1000))
    for fname, i in zip(target, range(1 + args.get("samples") / args.get("save", 100))):
        count = min((i + 1) * args.get("save", 100), args.get("samples")) - i * args.get("save", 100)
        samples = rjags.coda_samples(model, variable_names=StrVector(args.get("monitor", [])), n_iter=count, by=5, thin=args.get("thin", 1))
        pickle.dump((samples, lookups), meta_open(fname.rstr(), "w"))
    return None


def cluster_by_grs(target, source, env):
    args = source[-1].read()
    data, lookups = pickle.load(meta_open(source[0].rstr()))
    retval = numpy.zeros(shape=(data["verbs"], data["grs"]))
    for i in range(len(data["instance_starts"])):
        verb = data["instance_verbs"][i]
        start = data["instance_starts"][i]
        length = data["instance_lengths"][i]
        for gr in data["instance_grs"][start - 1 : start + length - 1]:
            retval[verb - 1, gr - 1] += 1
    retval = numpy.transpose(retval.T / retval.sum(1))
    print retval.shape
    res = stats.kmeans(numpy2ri(retval), centers=args.get("clusters", 20)) #data[args["matrix"]].shape[0] / 10)
    rlookup = dict([(v - 1, k) for k, v in lookups["%s_to_id" % args["lookup"]].iteritems()])
    ofd = meta_open(target[0].rstr(), "w")
    for c in set(res.rx2("cluster")):
        ofd.write(" ".join([rlookup[i] for i, a in enumerate(res.rx2("cluster")) if a == c]) + "\n")    
    return None

sentence_rx = re.compile(r"\n\s*\n")
fpos = ["I", "P", "C", "D", "W", "R", "T", "U"]

class Word():
    def __init__(self, parent, toks):
        self.parent = parent
        self.head_index = int(toks[6]) - 1
        self.gr = toks[7]
        self.lemma = toks[2]
        self.token = toks[1]
        self.pos = toks[3]
        self.index = int(toks[0]) - 1
    def __str__(self):
        return self.token
    def head(self):
        if self.head_index >= 0:
            return self.parent[self.head_index]
        else:
            return None


class Sentence(list):
    def __init__(self, lines):
        tok_lines = [line.split() for line in lines.strip().split("\n")]
        for toks in sorted(tok_lines, lambda x, y : cmp(int(x[0]), int(y[0]))):
            self.append(Word(self, toks))

    def __str__(self):
        return " ".join([str(x) for x in self])


def conll_to_data(target, source, env):
    args = source[-1].read()
    if args.get("verbs"):
        keep_verbs = [x.split()[0] for x in meta_open(args["verbs"])]
    else:
        keep_verbs = []
    lookups = dict([(x, {}) for x in ["verb_to_id", "gr_to_id", "lemma_to_id"]])
    data = dict([(x, []) for x in ["instance_starts", "instance_verbs", "instance_lengths", "instance_grs"]])
    for fname in args.get("inputs", []):
        this_verb = None
        if "0parsed" in fname:
            this_verb = re.match(r".*0parsed\.(.*?)\..*", fname).group(1)
            if len(keep_verbs) > 0 and this_verb not in keep_verbs:
                continue
        fd = meta_open(fname)
        text = fd.read()
        fd.close()
        for stext in sentence_rx.split(text.strip()):
            try:
                sent = Sentence(stext)
            except:
                continue
            for verb in [x for x in sent if x.pos.startswith("V") and x.gr not in ["auxpass", "cop"] and (not keep_verbs or x.lemma in keep_verbs) and (not this_verb or this_verb == x.lemma)]:
                try:
                    if verb.head():
                        if verb.head().pos[0] in fpos:
                            grs = ["%s(%s-%s, %s)" % (verb.gr, verb.head().pos, verb.head().lemma, verb.pos)]
                        else:
                            grs = ["%s(%s, %s)" % (verb.gr, verb.head().pos, verb.pos)]
                        lemmas = [verb.head().lemma]
                    else:
                        grs = []
                        lemmas = []
                except:
                    continue
                for tok in sent:
                    if tok.head_index == verb.index:
                        if tok.pos[0] in fpos:
                            grs.append("%s(%s, %s-%s)" % (tok.gr, verb.pos, tok.pos, tok.lemma))
                        else:
                            grs.append("%s(%s, %s)" % (tok.gr, verb.pos, tok.pos))
                        lemmas.append(tok.lemma)
                if len(grs) == 0:
                    continue
                lookups["verb_to_id"][verb.lemma] = lookups["verb_to_id"].get(verb.lemma, len(lookups["verb_to_id"]) + 1)
                data["instance_starts"].append(len(data["instance_grs"]) + 1)
                data["instance_verbs"].append(lookups["verb_to_id"][verb.lemma])
                data["instance_lengths"].append(len(grs))
                for gr in grs:
                    lookups["gr_to_id"][gr] = lookups["gr_to_id"].get(gr, len(lookups["gr_to_id"]) + 1)
                    data["instance_grs"].append(lookups["gr_to_id"][gr])
    data["verbs"] = len(lookups["verb_to_id"])
    data["grs"] = len(lookups["gr_to_id"])
    data["ns"] = max(data["instance_lengths"])
    pickle.dump((data, lookups), meta_open(target[0].rstr(), "w"))
    return None


def conll_to_tagging_data(target, source, env):
    args = source[-1].read()
    data = {"instance_tokens" : [], "instance_lengths" : [], "instance_starts" : []}
    lookups = {"token_to_id" : {}}
    for fname in args.get("inputs", []):
        fd = meta_open(fname)
        text = fd.read().lower()
        fd.close()
        for stext in sentence_rx.split(text.strip()):
            try:
                sent = Sentence(stext)
            except:
                continue
            toks = [x.token for x in sent]
            data["instance_starts"] = len(data["instance_tokens"]) + 1
            data["instance_lengths"] = len(toks)
            for tok in toks:
                lookups["token_to_id"][tok] = lookups["token_to_id"].get(tok, len(lookups["token_to_id"]) + 1)                
                data["instance_tokens"].append(lookups["token_to_id"][tok])
    pickle.dump((data, lookups), meta_open(target[0].rstr(), "w"))            
    return None


def run_jags_emitter(target, source, env):
    try:
        args = source[-1].read()
    except:
        args = {}
    new_targets = []
    basename = meta_basename(target[0].rstr())
    for i in range(int(math.ceil(args.get("samples") / float(args.get("save", 100))))):
        new_targets.append(os.path.join(env["DIR"], "%s_%s-%s.pkl.gz" % (basename, i * args.get("save", 100) + 1, min((i + 1) * args.get("save", 100), args.get("samples")))))
    return new_targets, source


def TOOLS_ADD(env):
    env.Append(BUILDERS = {
            "ConllToData" : Builder(action=conll_to_data),
            "RunJags" : Builder(action=run_jags, emitter=run_jags_emitter),
            "ClusterVerbs" : Builder(action=cluster_verbs),
            "EvaluateClustering" : Builder(action=evaluate_clustering),
            "ConllToTaggingData" : Builder(action=conll_to_tagging_data),
            "ClusterByGRs" : Builder(action=cluster_by_grs),
        })
