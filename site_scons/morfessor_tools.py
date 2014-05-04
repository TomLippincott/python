from SCons.Builder import Builder
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
from common_tools import meta_open, DataSet
import cPickle as pickle
import numpy
import math
import xml.etree.ElementTree as et
from morfessor import BaselineModel, AnnotatedCorpusEncoding, AnnotationsModelUpdate, LexiconEncoding, CorpusEncoding, Encoding, MorfessorException, MorfessorIO, get_default_argparser

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
    with meta_open(source[0].rstr()) as ifd:
        dataset = DataSet.from_stream(ifd)
        for sentence in dataset.sentences:
            for word_id, tag_id, analysis_ids in sentence:
                word = dataset.indexToWord[word_id]
                words[word] = words.get(word, 0) + 1
    model.load_data([(c, w, (w)) for w, c in words.iteritems()], args.freqthreshold, dampfunc, args.splitprob)
    algparams = ()
    develannots = None
    e, c = model.train_batch(args.algorithm, algparams, develannots,
                             args.finish_threshold, args.maxepochs)
    d = DataSet.from_analyses([x for x in model.get_segmentations()])
    with meta_open(target[0].rstr(), "w") as ofd:
        d.write(ofd)
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

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
            "TrainMorfessor" : Builder(action=train_morfessor),
            'MorfessorData' : Builder(action=morfessor_data_builder),
            'MorfessorRun' : Builder(generator=morfessor_run_generator),
            'MorfessorDisplayCounts' : Builder(generator=morfessor_display_counts_generator),
            'MorfessorEstimateProbs' : Builder(generator=morfessor_estimate_probs_generator),
            'MorfessorViterbiTag' : Builder(generator=morfessor_viterbitag_generator),
            'MorfessorAlignSegmentations' : Builder(generator=morfessor_align_segmentations),
            'MorfessorEvaluateTags' : Builder(generator=morfessor_evaluate_tags),
            'MorfessorLatinGold' : Builder(action=morfessor_latin_gold),
            })
