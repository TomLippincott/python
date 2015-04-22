from SCons.Builder import Builder
import codecs
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
from common_tools import meta_open, DataSet, regular_word
import cPickle as pickle
import numpy
import math
import xml.etree.ElementTree as et
from morfessor import BaselineModel, MorfessorIO, get_default_argparser
import tarfile

def apply_morfessor(target, source, env):
    parser = get_default_argparser()
    args = parser.parse_args([])
    io = MorfessorIO(encoding=args.encoding,
                     compound_separator=args.cseparator,
                     atom_separator=args.separator)
    model = io.read_binary_model_file(source[0].rstr())
    words = []
    terms = {}
    for fname in source[1:]:
        try:
            with meta_open(fname.rstr(), enc=None) as ifd:
                for t in et.parse(ifd).getiterator("kw"):
                    text = list(t.getiterator("kwtext"))[0].text
                    words += text.strip().split()
        except:
            with meta_open(fname.rstr()) as ifd:
                words = [l.strip() for l in ifd]
    words = set(sum([w.strip("-").split("-") for w in words if "_" not in w], []))
    #            kwid = t.get("kwid")
    #            terms[kwid] = (text, [])
    #            for w in words:
    #
    for w in words:
        toks, score = model.viterbi_segment(w)
        if len(toks) >= 2:
            toks = ["%s+" % toks[0]] + ["+%s+" % t for t in toks[1:-1]] + ["+%s" % toks[-1]]
        terms[w] = toks
            #terms[kwid] = (text, terms[kwid][1] + toks)
    #lines = []
    #for i, (t, s) in terms.iteritems():
    #    lines.append(" ".join(s))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(("\n".join(sorted(["%s" % (" ".join(v)) for k, v in terms.iteritems()]))) + "\n")
    return None

def train_morfessor(target, source, env):
    parser = get_default_argparser()
    args = parser.parse_args([])
    dampfunc = lambda x : x
    rx_str = "((\\S(%(ALT)s))|(^(%(ALT)s)\\S))" % {"ALT" : "|".join([unichr(int(x, base=16)) for x in env.get("NON_ACOUSTIC_GRAPHEMES")])}
    model = BaselineModel(forcesplit_list=env.get("FORCE_SPLIT", []),
                          corpusweight=1.0,
                          use_skips=False,
                          nosplit_re=rx_str)
    io = MorfessorIO(encoding=args.encoding,
                     compound_separator=args.cseparator,
                     atom_separator=args.separator)
    words = {}
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            toks = line.strip().split()
            for word in toks[0].split("-"):
                if len(toks) == 1:
                    words[word] = 1
                elif len(toks) == 2:                
                    words[word] = words.get(word, 0) + int(toks[1])
                else:
                    return "malformed vocabulary line: %s" % (line.strip())

    words = {w : c for w, c in words.iteritems() if not re.match(env.get("NON_WORD_PATTERN", "^$"), w)}
    #words = set(sum([w.strip("-").split("-") for w in words.keys() if "_" not in w], []))
    model.load_data([(c, w, (w)) for w, c in words.iteritems()], args.freqthreshold, dampfunc, args.splitprob)
    algparams = ()
    develannots = None
    e, c = model.train_batch(args.algorithm, algparams, develannots,
                             args.finish_threshold, args.maxepochs)
    with meta_open(target[0].rstr(), "w") as ofd:
        for n, morphs in model.get_segmentations():
            if len(morphs) >= 2:
                morphs = ["%s+" % morphs[0]] + ["+%s+" % t for t in morphs[1:-1]] + ["+%s" % morphs[-1]]
            ofd.write(" ".join(morphs) + "\n")
    io.write_binary_model_file(target[1].rstr(), model)
    return None

def normalize_morfessor_output(target, source, env):
    segs = []
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            toks = line.strip().split()
            segs.append(("".join([x.strip("+") for x in toks]), toks))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s\t%s" % (w, " ".join(ts)) for w, ts in segs]) + "\n")        
    return None

def morfessor_data_builder(target, source, env):    
    xml = et.parse(open(source[0].rstr()))
    data = {}
    for word in xml.getiterator("word"):
        tok = word.get("form", "").lower()
        if len(tok) > 1 and tok.isalpha():
            data[tok] = data.get(tok, 0) + 1
    open(target[0].rstr(), "w").write("\n".join(["%d %s" % (v, k) for k, v in sorted(data.iteritems())]))
    return None

def morfessor_run_generator(target, source, env, for_signature):
    if len(source) == 1:
        return "${MORFESSOR}/bin/morfessor1.0.pl -trace 3 -data ${SOURCE} > ${TARGET}"
    elif len(source) == 2:
        return "${MORFESSOR}/bin/morfessor1.0.pl -trace 3 -data ${SOURCES[0]} -load ${SOURCES[1]} > ${TARGET}"

def morfessor_display_counts_generator(target, source, env, for_signature):
    return "cat ${SOURCES[0]} | sed 's/^[0-9]* //' | ${MORFESSOR}/bin/display_morph_counts.pl ${SOURCES[1]} > ${TARGET}"

def morfessor_estimate_probs_generator(target, source, env, for_signature):
    return "cat ${SOURCES[0]} | ${MORFESSOR}/bin/estimateprobs.pl -pplthresh ${SOURCES[1].read()} > ${TARGET}"

def morfessor_viterbitag_generator(target, source, env, for_signature):
    return "cat ${SOURCES[0]} | ${MORFESSOR}/bin/viterbitag.pl ${SOURCES[1]} > ${TARGET}"

def morfessor_align_segmentations(target, source, env, for_signature):
    # gold, to_eval
    return "${HUTMEGS}/bin/align_segmentations.pl ${SOURCES[0]} < ${SOURCES[1]} > ${TARGET}"

def morfessor_evaluate_tags(target, source, env, for_signature):
    # categories, alignment
    return "${HUTMEGS}/bin/evaluate_tags.pl ${SOURCES[0]} < ${SOURCES[1]} > $TARGET"

def morfessor_latin_gold(target, source, env):
    xml = et.parse(source[0].rstr())
    data = {}
    for w in xml.getiterator("word"):
        form = w.get("form")
        head = w.get("head")
        relation = w.get("relation")
        postag = w.get("postag")
        lemma = re.match(r"^(.*?)(\d+)?$", w.get("lemma")).group(1)
        data[(lemma, form)] = data.get((form, lemma), 0) + 1
    open(target[0].rstr(), "w").write("\n".join(["%s %s %d" % (l, f, c) for (l, f), c in sorted(data.iteritems())]))
    return None

def evaluate_morfessor(target, source, env):
    return None

def asr_test(target, source, env):
    return None

def unsegment(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        words = ["".join([x.strip("+") for x in l.strip().split()]) for l in ifd]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(words))
    return None

def lines_to_vocabulary(target, source, env):
    words = {}
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            for word in line.strip().split():
                words[word] = words.get(word, 0) + 1            
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s\t%d" % (w, c) for w, c in sorted(words.iteritems())]) + "\n")
    return None

# def morfessor_babel_experiment(env, target_base, training, non_acoustic, dev_keywords=None, eval_keywords=None):
#     if env.get("RUN_SEGMENTATION", True):
#         temp_segs, model = env.TrainMorfessor(["work/morfessor/%s.txt" % (target_base),
#                                                "work/morfessor/%s.model" % (target_base)], training, NON_ACOUSTIC_GRAPHEMES=non_acoustic)
#         segmented = [temp_segs]
#         if dev_keywords:
#             dev_kw_segs = env.ApplyMorfessor("work/morfessor/%s_dev_keywords.txt" % (target_base),
#                                              [model, dev_keywords])
#             segmented.append(dev_kw_segs)
#         if eval_keywords:
#             eval_kw_segs = env.ApplyMorfessor("work/morfessor/%s_eval_keywords.txt" % (target_base),
#                                               [model, eval_keywords])
#             segmented.append(eval_kw_segs)
#     return segmented

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "TrainMorfessor" : Builder(action=train_morfessor),
        "ApplyMorfessor" : Builder(action=apply_morfessor),
        "NormalizeMorfessorOutput" : Builder(action=normalize_morfessor_output),
        "Unsegment" : Builder(action=unsegment),
        "LinesToVocabulary" : Builder(action=lines_to_vocabulary),
        # 'MorfessorData' : Builder(action=morfessor_data_builder),
        # 'MorfessorRun' : Builder(generator=morfessor_run_generator),
        # 'MorfessorDisplayCounts' : Builder(generator=morfessor_display_counts_generator),
        # 'MorfessorEstimateProbs' : Builder(generator=morfessor_estimate_probs_generator),
        # 'MorfessorViterbiTag' : Builder(generator=morfessor_viterbitag_generator),
        # 'MorfessorAlignSegmentations' : Builder(generator=morfessor_align_segmentations),
        # 'MorfessorEvaluateTags' : Builder(generator=morfessor_evaluate_tags),
        # 'MorfessorLatinGold' : Builder(action=morfessor_latin_gold),
    })

