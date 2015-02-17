from SCons.Builder import Builder
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

def apply_morfessor(target, source, env):
    parser = get_default_argparser()
    args = parser.parse_args([])
    io = MorfessorIO(encoding=args.encoding,
                     compound_separator=args.cseparator,
                     atom_separator=args.separator)
    model = io.read_binary_model_file(source[0].rstr())
    terms = {}
    for fname in source[1:]:
        with meta_open(fname.rstr()) as ifd:
            for t in et.parse(ifd).getiterator("kw"):
                text = list(t.getiterator("kwtext"))[0].text.lower()
                words = text.strip().split()
                kwid = t.get("kwid")
                terms[kwid] = (text, [])
                for w in words:
                    toks, score = model.viterbi_segment(w.encode("utf-8"))
                    if len(toks) >= 2:
                        toks = ["%s+" % toks[0]] + ["+%s+" % t for t in toks[1:-1]] + ["+%s" % toks[-1]]                        
                    terms[kwid] = (text, terms[kwid][1] + toks)
    lines = []
    for i, (t, s) in terms.iteritems():
        lines.append("%s %s" % (i, " ".join(s)))
    try:
        with meta_open(target[0].rstr(), "w") as ofd:
            ofd.write(("\n".join(lines)).encode("utf-8"))
    except:
        with meta_open(target[0].rstr(), "w") as ofd:
            ofd.write("\n".join(lines))

    return None

def train_morfessor(target, source, env):
    parser = get_default_argparser()
    args = parser.parse_args([])
    dampfunc = lambda x : x
    model = BaselineModel(forcesplit_list=[],
                          corpusweight=1.0,
                          use_skips=False)
    io = MorfessorIO(encoding=args.encoding,
                     compound_separator=args.cseparator,
                     atom_separator=args.separator)
    words = {}
    try:
        with meta_open(source[0].rstr()) as ifd:
            dataset = DataSet.from_stream(ifd)[-1]
            for sentence in dataset.sentences:
                for word_id, tag_id, analysis_ids in sentence:
                    word = dataset.indexToWord[word_id].lower()
                    if regular_word(word):
                        words[word] = words.get(word, 0) + 1
    except:
        with meta_open(source[0].rstr()) as ifd:
            for line in ifd:
                for w in line.lower().split():
                    if regular_word(w):
                        words[w] = words.get(w, 0) + 1
            #words = {w : int(n) for n, w in [x.strip().split() for x in ifd]}
    model.load_data([(1, w, (w)) for w, c in words.iteritems()], args.freqthreshold, dampfunc, args.splitprob)
    algparams = ()
    develannots = None
    e, c = model.train_batch(args.algorithm, algparams, develannots,
                             args.finish_threshold, args.maxepochs)
    
    #d = DataSet.from_analyses([x for x in model.get_segmentations()])
    with meta_open(target[0].rstr(), "w") as ofd:
        for n, morphs in model.get_segmentations():
            if len(morphs) >= 2:
                morphs = ["%s+" % morphs[0]] + ["+%s+" % t for t in morphs[1:-1]] + ["+%s" % morphs[-1]]
            ofd.write(" ".join(morphs) + "\n")
    io.write_binary_model_file(target[1].rstr(), model)
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

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "TrainMorfessor" : Builder(action=train_morfessor),
        "ApplyMorfessor" : Builder(action=apply_morfessor),
        # 'MorfessorData' : Builder(action=morfessor_data_builder),
        # 'MorfessorRun' : Builder(generator=morfessor_run_generator),
        # 'MorfessorDisplayCounts' : Builder(generator=morfessor_display_counts_generator),
        # 'MorfessorEstimateProbs' : Builder(generator=morfessor_estimate_probs_generator),
        # 'MorfessorViterbiTag' : Builder(generator=morfessor_viterbitag_generator),
        # 'MorfessorAlignSegmentations' : Builder(generator=morfessor_align_segmentations),
        # 'MorfessorEvaluateTags' : Builder(generator=morfessor_evaluate_tags),
        # 'MorfessorLatinGold' : Builder(action=morfessor_latin_gold),
    })
