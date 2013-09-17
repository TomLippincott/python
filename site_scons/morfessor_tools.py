from SCons.Builder import Builder
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
from common_tools import meta_open
import cPickle as pickle
import numpy
import math
import xml.etree.ElementTree as et

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
            'MorfessorData' : Builder(action=morfessor_data_builder),
            'MorfessorRun' : Builder(generator=morfessor_run_generator),
            'MorfessorDisplayCounts' : Builder(generator=morfessor_display_counts_generator),
            'MorfessorEstimateProbs' : Builder(generator=morfessor_estimate_probs_generator),
            'MorfessorViterbiTag' : Builder(generator=morfessor_viterbitag_generator),
            'MorfessorAlignSegmentations' : Builder(generator=morfessor_align_segmentations),
            'MorfessorEvaluateTags' : Builder(generator=morfessor_evaluate_tags),
            'MorfessorLatinGold' : Builder(action=morfessor_latin_gold),
            })
