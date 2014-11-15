from SCons.Builder import Builder
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
from common_tools import meta_open, DataSet, Probability, adjusted_rand, harmonic_mean, modified_purity, normalized_information_distance, v_measure
import cPickle as pickle
import numpy
from random import randint
import math
import xml.etree.ElementTree as et
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

def plot_reduction(target, source, env):
    data = {} #{"types" : {}} #, "tokens" : {}}
    try:
        first = source[0].read()
    except:
        first = -1
    for props, fname in env["REDUCTIONS"].iteritems():
        with meta_open(fname[0].rstr()) as ifd:
            data[props] = []
            header = ifd.readline()
            #data["tokens"][props] = []
            for vals in [map(int, toks) for toks in [l.split() for l in ifd]][:first]:
                total_oov_tokens, total_oov_types, bucket_total_tokens, bucket_total_types, bucket_total_oov_tokens, bucket_total_oov_types, bucket_accepted_types = [float(x) for x in vals]
                token_precision = bucket_total_oov_tokens / bucket_total_tokens
                token_recall = bucket_total_oov_tokens / total_oov_tokens
                token_fscore = (2 * token_precision * token_recall) / (token_precision + token_recall)
                type_precision = bucket_total_oov_types / bucket_total_types
                type_recall = bucket_total_oov_types / total_oov_types
                type_fscore = (2 * type_precision * type_recall) / (type_precision + type_recall)
                accepted_precision = bucket_accepted_types / bucket_total_types
                fscore = (2 * accepted_precision * token_recall) / (accepted_precision + token_recall)
                #data["tokens"][props].append((token_precision, token_recall, token_fscore))
                #data["types"][props].append((type_precision, type_recall, type_fscore))
                data[props].append((accepted_precision, token_recall, fscore))
    pyplot.figure(figsize=(7 * 3, 7 * 1))
    pyplot.title(env.subst("${LANGUAGE}"))
    legend_args = {"fontsize" : 6, "loc" : "upper right"}
    for i, (props, vals) in enumerate(data.iteritems()):        
        if props[0] == "bbg":
            name = "Mofessor"
        elif props[0] == "joint":
            name = "Joint"
        elif props[0] == "adaptor":
            name = "Adaptors"
        else:
            name = "Infinite adaptors"
        buckets = len(vals) #meths.values()[0])
        bucket_size = bucket_total_types / buckets
        loc_size = buckets / 5
        locs = [10 + (10 * j) for j in range(9)] #, buckets, loc_size)
        labels = ["%dK" % x for x in locs]
        pyplot.subplot(1, 3, 1)
        #for k, v in meths.iteritems():
        #name = " ".join(k)
        pyplot.plot([x[0] for x in vals], label=name)
        pyplot.title("Precision")
        pyplot.legend(**legend_args)
        pyplot.xticks(locs, labels)

        pyplot.subplot(1, 3, 2)
        #for k, v in meths.iteritems():
        #name = " ".join(k)
        pyplot.plot([x[1] for x in vals], label=name)
        pyplot.title("Recall")
        pyplot.legend(**legend_args)
        pyplot.xticks(locs, labels)

        pyplot.subplot(1, 3, 3)
        #for k, v in meths.iteritems():
        #name = " ".join(k)
        pyplot.plot([x[2] for x in vals], label=name)
        pyplot.title("FScore")
        pyplot.legend(**legend_args)
        pyplot.xticks(locs, labels)
    pyplot.savefig(target[0].rstr())
    pyplot.close()
    return None

def plot_reduction_emitter(target, source, env):
    args = source[-1].read()
    new_targets = pjoin(os.path.dirname(source[0].rstr()), "%d_reduction.png" % (args["bins"]))
    return new_targets, source

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
      OOV_TOTAL_TOKENS, OOV_TOTAL_TYPES, BUCKET_TOTAL_TOKENS, BUCKET_TOTAL_TYPES, BUCKET_TOTAL_OOV_TOKENS, BUCKET_TOTAL_OOV_TYPES, BUCKET_ACCEPTED_TYPES
    """
    if len(source) == 4:
        bucket_size = 1000
    else:
        bucket_size = source[4].read()
    training_fname, expansion_fname, oov_fname, accepted_fname = [x.rstr() for x in source[0:4]]
    with meta_open(training_fname) as training_ifd, meta_open(expansion_fname) as expansion_ifd, meta_open(oov_fname) as oov_ifd, meta_open(accepted_fname) as accepted_ifd:
        training = set(DataSet.from_stream(training_ifd)[-1].indexToWord.values())
        expansion = [(w, math.exp(-float(lp))) for w, lp in [l.split() for l in expansion_ifd] if w not in training]
        oov = {w : int(c) for c, w in [l.strip().split() for l in oov_ifd if w not in training]}
        accepted = set([x.strip() for x in accepted_ifd])
    total_prob = sum([x[1] for x in expansion])
    expansion = [(w, p / total_prob) for w, p in expansion]
    values = [[0, 0, 0, 0, 0]]
    buckets = len(expansion) / bucket_size
    #oov = {w : c for w, c in oov.iteritems() if w not in training}
    total_oov_tokens = sum(oov.values())
    total_oov_types = len(oov)    
    for bucket in range(buckets):
        bucket_total_tokens, bucket_total_types, bucket_total_oov_tokens, bucket_total_oov_types, accepted_types = values[bucket]
        for w, p in expansion[bucket * bucket_size : (bucket + 1) * bucket_size]:
            predicted_oov_count = total_oov_tokens * p
            bucket_total_tokens += predicted_oov_count
            bucket_total_types += 1
            if w in accepted:
                accepted_types += 1
            if w in oov:
                bucket_total_oov_tokens += oov[w]
                bucket_total_oov_types += 1
        values.append([bucket_total_tokens, bucket_total_types, bucket_total_oov_tokens, bucket_total_oov_types, accepted_types])
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\t".join(["Total OOV Tokens", "Total OOV Types", "Bucket Total Types", "Bucket OOV Tokens", "Bucket Total OOV Types", "Bucket Accepted Types"]) + "\n")
        ofd.write("\n".join(["\t".join([str(int(y)) for y in [total_oov_tokens, total_oov_types] + x]) for x in values[1:]]) + "\n")
    return None

def evaluate_tagging_vm(target, source, env):
    with meta_open(source[0].rstr()) as gold_fd, meta_open(source[1].rstr()) as pred_fd:
        gold = DataSet.from_stream(gold_fd)[-1]
        preds = DataSet.from_stream(pred_fd)
        #assert(len(gold.sentences) == len(pred.sentences))
    scores = []
    for pred in preds:
        for gold_sentence, pred_sentence in zip(gold.sentences, pred.sentences):
            assert(len(gold_sentence) == len(pred_sentence))
        gold_tags = sum([[l[1] for l in s] for s in gold.sentences], [])
        pred_tags = sum([[l[1] for l in s] for s in pred.sentences], [])
        #scores = {}
    #scores["TRand"] = adjusted_rand(gold_tags, pred_tags)
    #scores["TmPur"] = harmonic_mean(modified_purity(gold_tags, pred_tags), modified_purity(pred_tags, gold_tags, 2))
    #scores["TNIS"] = 1.0 - normalized_information_distance(gold_tags, pred_tags)
        scores.append({"VM" : v_measure(pred_tags, gold_tags)})
        #names = scores.keys()
    names = ["VM"]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\t".join(names) + "\n")
        for score in scores:
            ofd.write("\t".join(["%.3f" % score[x] for x in names]) + "\n")
    return None

def evaluate_morphology(target, source, env):
    with meta_open(source[0].rstr()) as gold_fd, meta_open(source[1].rstr()) as pred_fd:
        gold = DataSet.from_stream(gold_fd)
        pred = DataSet.from_stream(pred_fd)
        gold_analyses = gold.get_analyses()
        pred_analyses = pred.get_analyses()
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("%f\n" % 1.0)
    return None

def collate_oov_quality(target, source, env):
    data = {}
    for (language, size), [original, good, bad] in env["FILES"].iteritems():
        with meta_open(original.rstr()) as ifd:
            original_size = len([l for l in ifd])
        with meta_open(good.rstr()) as ifd:
            good_size = len([l for l in ifd])
        with meta_open(bad.rstr()) as ifd:
            bad_size = len([l for l in ifd])
        data[(language, size)] = good_size / float(original_size)
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\t".join(["Language", "Size", "% valid words"]) + "\n")
        for (l, s), v in sorted(data.iteritems()):
            ofd.write("\t".join([l, s, "%.3f" % v]) + "\n")
    return None

def random_segmentations(target, source, env):
    def get_random_segmentation(w):
        stem_length = randint(1, len(w))
        prefix_length = randint(0, len(w) - stem_length)
        suffix_length = randint(0, len(w) - (stem_length + prefix_length))
        prefix = w[:prefix_length]
        stem = w[prefix_length:prefix_length + stem_length]
        suffix = w[prefix_length + stem_length:]
        return tuple([x for x in [prefix, stem, suffix] if len(x) > 0])
    style_name = source[1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[-1]
    if style_name == "type-based":
        wordIndexToAnalysis = {}
        for i, w in data.indexToWord.iteritems():
            wordIndexToAnalysis[i] = get_random_segmentation(w)
        sentences = [[(data.indexToWord[w], data.indexToTag.get(t, None), [wordIndexToAnalysis[w]]) for w, t, aa in s] for s in data.sentences]
    else:        
        sentences = [[(data.indexToWord[w], data.indexToTag.get(t, None), [get_random_segmentation(data.indexToWord[w])]) for w, t, aa in s] for s in data.sentences]
    new_data = DataSet.from_sentences(sentences)
    with meta_open(target[0].rstr(), "w") as ofd:
        new_data.write(ofd)
    return None

def random_tags(target, source, env):
    num_tags = env["NUM_TAGS"]
    style_name = source[1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[-1]
    if style_name == "type-based":
        wordIndexToTag = {i : randint(0, num_tags) for i in data.indexToWord.keys()}
        new_data = DataSet.from_sentences([[(data.indexToWord[w], str(wordIndexToTag[w]), [data.indexToAnalysis[a] for a in aa]) for w, t, aa in s] for s in data.sentences])
    else:
        new_data = DataSet.from_sentences([[(data.indexToWord[w], str(randint(0, num_tags)), [data.indexToAnalysis[a] for a in aa]) for w, t, aa in s] for s in data.sentences])
    with meta_open(target[0].rstr(), "w") as ofd:
        new_data.write(ofd)
    return None

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "EvaluateTaggingVM" : Builder(action=evaluate_tagging_vm),
        #"EvaluateMorphology" : Builder(action=evaluate_morphology),
        "PlotReduction" : Builder(action=plot_reduction),
        "TopWords" : Builder(action=top_words),
        "SplitExpansions" : Builder(action=split_expansion),
        "OOVReduction" : Builder(action=oov_reduction),
        "CollateOOVQuality" : Builder(action=collate_oov_quality),
        "RandomSegmentations" : Builder(action=random_segmentations),
        "RandomTags" : Builder(action=random_tags),
    })
