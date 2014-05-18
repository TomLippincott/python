from SCons.Builder import Builder
from SCons.Script import *
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import cPickle as pickle
import numpy
import math
import lxml.etree as et
import xml.sax
import sys
import gzip
from os.path import join as pjoin
from os import listdir
import tarfile
import operator
from random import randint
from common_tools import DataSet, meta_open
import fst

def segmentation_to_fst(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)
    counts = {"prefix" : {}, "stem" : {}, "suffix" : {}}
    #for sentence in data.sentences:
    #    for word, tag, analyses in sentence:
    #        partial_count = 1.0 / len(analyses)
    if len(data.sentences) == 0:
        for analysis in data.indexToAnalysis.values():
            for t, v in analysis:
                counts[t][v] = counts[t].get(v, 0) + 1.0
    f = fst.Acceptor()
    for t, morphs in counts.iteritems():
        total = sum(morphs.values())
        if t == "prefix":
            start = 0
        elif t == "stem":
            start = 1
        elif t == "suffix":
            start = 2
        for m, c in morphs.iteritems():
            #f.add_arc(start, start + 1, m, c / total)
            f.add_arc(start, start + 1, m, math.log(c / total))
    f[3].final = True
    f.write(target[0].rstr())
    f.isyms.write(target[1].rstr())
    return None

def generate_words(target, source, env):
    f = fst.read(source[0].rstr())
    s = fst.read_symbols(source[1].rstr())
    words = []
    for path in f.shortest_path(source[2].read()).paths():
        word = "+".join([s.find(a.olabel) for a in path if a.olabel != 0])
        prob = reduce(operator.add, (arc.weight for arc in path))
        words.append((word, prob))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s %f" % (w, p) for w, p in words]) + "\n")
    return None

def rerank_by_ngrams(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        pass
    with meta_open(target[0].rstr(), "w") as ofd:
        pass
    return None

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
            "SegmentationToFST" : Builder(action=segmentation_to_fst),
            "GenerateWords" : Builder(action=generate_words),
            "RerankByNgrams" : Builder(action=rerank_by_ngrams),
            })
