from SCons.Builder import Builder
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
from common_tools import meta_open, DataSet, Probability
import cPickle as pickle
import numpy
import math
import xml.etree.ElementTree as et
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

def plot_reduction(target, source, env):
    data = {"types" : {}} #, "tokens" : {}}
    try:
        first = source[0].read()
    except:
        first = -1
    for props, fname in env["REDUCTIONS"].iteritems():
        with meta_open(fname[0].rstr()) as ifd:
            data["types"][props] = []
            #data["tokens"][props] = []
            for vals in [map(int, toks) for toks in [l.split() for l in ifd]][:first]:
                total_oov_tokens, total_oov_types, bucket_total_tokens, bucket_total_types, bucket_total_oov_tokens, bucket_total_oov_types = [float(x) for x in vals]                
                token_precision = bucket_total_oov_tokens / bucket_total_tokens
                token_recall = bucket_total_oov_tokens / total_oov_tokens
                token_fscore = (2 * token_precision * token_recall) / (token_precision + token_recall)
                type_precision = bucket_total_oov_types / bucket_total_types
                type_recall = bucket_total_oov_types / total_oov_types
                type_fscore = (2 * type_precision * type_recall) / (type_precision + type_recall)
                #data["tokens"][props].append((token_precision, token_recall, token_fscore))
                data["types"][props].append((type_precision, type_recall, type_fscore))


    pyplot.figure(figsize=(7 * 3, 7 * 1))
    pyplot.title(env.subst("${LANGUAGE}"))
    legend_args = {"fontsize" : 6, "loc" : "upper right"}
    for i, (over, meths) in enumerate(data.iteritems()):        
        buckets = len(meths.values()[0])
        bucket_size = bucket_total_types / buckets
        loc_size = buckets / 5
        locs = range(0, buckets, loc_size)
        labels = [str(int((x + 1) * bucket_size)) for x in locs]
        pyplot.subplot(1, 3, 1 + i * 3)
        for k, v in meths.iteritems():
            name = " ".join(k)
            pyplot.plot([x[0] for x in v], label=name)
            pyplot.title("%s - Precision" % over.title())
            pyplot.legend(**legend_args)
            pyplot.xticks(locs, labels)
        pyplot.subplot(1, 3, 2 + i * 3)
        for k, v in meths.iteritems():
            name = " ".join(k)
            pyplot.plot([0.0] + [x[1] for x in v], label=name)
            pyplot.title("%s - Recall" % over.title())
            pyplot.legend(**legend_args)
            pyplot.xticks(locs, labels)
        pyplot.subplot(1, 3, 3 + i * 3)
        for k, v in meths.iteritems():
            name = " ".join(k)
            pyplot.plot([0.0] + [x[2] for x in v], label=name)
            pyplot.title("%s - FScore" % over.title())
            pyplot.legend(**legend_args)
            pyplot.xticks(locs, labels)
    pyplot.savefig(target[0].rstr())
    pyplot.close()
    return None


def plot_reduction_emitter(target, source, env):
    args = source[-1].read()
    new_targets = pjoin(os.path.dirname(source[0].rstr()), "%d_reduction.png" % (args["bins"]))
    return new_targets, source


def variation_of_information(target, source, env):
    return None

def split_expansion(target, source, env):
    if len(source) == 2:
        limit = source[1].read()
    else:
        limit = 0
    words = {}
    with meta_open(source[0].rstr()) as ifd:
        for l in ifd:
            toks = l.split("\t")
            assert(len(toks) == len(target) + 1)
            words[toks[0]] = [Probability(neglogprob=float(x)) for x in toks[1:]]
    for i, f in enumerate(target):
        with meta_open(f.rstr(), "w") as ofd:
            vals = [(z[0], -z[1][i].log()) for z in sorted(words.iteritems(), lambda x, y : cmp(y[1][i].log(), x[1][i].log()))]
            if limit > 0:
                vals = vals[0:limit]
            ofd.write("\n".join(["%s\t%f" % (w, p) for w, p in vals]))
    return None

def top_words(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as words_ifd, meta_open(source[1].rstr()) as pron_ifd:
        top = ProbabilityList(words_ifd).get_top_n(args["COUNT"])
        prons = Pronunciations(pron_ifd)
        prons.filter_by(top)
    with meta_open(target[0].rstr(), "w") as words_ofd, meta_open(target[1].rstr(), "w") as pron_ofd:
        words_ofd.write(top.format())
        pron_ofd.write(prons.format())
    return None

def oov_reduction(target, source, env):
    """
    split expansions into buckets
    for each bucket 0 to N, output line of format:
      OOV_TOTAL_TOKENS, OOV_TOTAL_TYPES, BUCKET_TOTAL_TOKENS, BUCKET_TOTAL_TYPES, BUCKET_TOTAL_OOV_TOKENS, BUCKET_TOTAL_OOV_TYPES
    """
    if len(source) == 3:
        bucket_size = 1000
    else:
        bucket_size = source[3].read()
    training_fname, expansion_fname, oov_fname = [x.rstr() for x in source[0:3]]
    with meta_open(training_fname) as training_ifd, meta_open(expansion_fname) as expansion_ifd, meta_open(oov_fname) as oov_ifd:
        training = set(DataSet.from_stream(training_ifd).indexToWord.values())
        expansion = [(w, math.exp(-float(lp))) for w, lp in [l.split() for l in expansion_ifd] if w not in training]
        oov = {w : int(c) for c, w in [l.strip().split() for l in oov_ifd]}

    total_prob = sum([x[1] for x in expansion])
    expansion = [(w, p / total_prob) for w, p in expansion]
    values = [[0, 0, 0, 0]]
    buckets = len(expansion) / bucket_size
    total_oov_tokens = sum(oov.values())
    total_oov_types = len(oov)
    oov = {w : c for w, c in oov.iteritems() if w not in training}
    for bucket in range(buckets):
        bucket_total_tokens, bucket_total_types, bucket_total_oov_tokens, bucket_total_oov_types = values[bucket]
        for w, p in expansion[bucket * bucket_size : (bucket + 1) * bucket_size]:
            predicted_oov_count = total_oov_tokens * p
            bucket_total_tokens += predicted_oov_count
            bucket_total_types += 1
            if w in oov:
                bucket_total_oov_tokens += oov[w]
                bucket_total_oov_types += 1
        values.append([bucket_total_tokens, bucket_total_types, bucket_total_oov_tokens, bucket_total_oov_types])
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["\t".join([str(int(y)) for y in [total_oov_tokens, total_oov_types] + x]) for x in values[1:]]))
    return None

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
            "PlotReduction" : Builder(action=plot_reduction),
            "TopWords" : Builder(action=top_words),
            "SplitExpansions" : Builder(action=split_expansion),
            "VariationOfInformation" : Builder(action=variation_of_information),
            "OOVReduction" : Builder(action=oov_reduction),
            })
