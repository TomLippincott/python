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
import math
import xml.etree.ElementTree as et
from morfessor import BaselineModel, MorfessorIO, get_default_argparser
import tarfile


def train_morfessor(target, source, env):
    """Train a Morfessor model using a word list as input.

    This builder is largely based on the code in the new Python version of Morfessor.
    Note that it prevents splitting that would create a morph composed just of
    non-acoustic graphemes.

    Sources: word list file
    Targets: segmented word list file, morfessor model file
    """
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


def apply_morfessor(target, source, env):
    """Applies a trained Morfessor model to an unseen word list.

    Sources: morfessor model file, word list file
    Targets: segmented word list
    """
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
                words = [l.strip().split()[0] for l in ifd]
    words = set(sum([w.strip("-").split("-") for w in words if "_" not in w], []))
    for w in words:
        toks, score = model.viterbi_segment(w)
        if len(toks) >= 2:
            toks = ["%s+" % toks[0]] + ["+%s+" % t for t in toks[1:-1]] + ["+%s" % toks[-1]]
        terms[w] = toks
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(("\n".join(sorted(["%s" % (" ".join(v)) for k, v in terms.iteritems()]))) + "\n")
    return None


def normalize_morfessor_output(target, source, env):
    """Turns Morfessor parsed output into something more human-readable.

    Sources: segmented py-cfg output file
    Targets: reformatted segmentations file
    """
    segs = []
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            toks = line.strip().split()
            segs.append(("".join([x.strip("+") for x in toks]), toks))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s\t%s" % (w, " ".join(ts)) for w, ts in segs]) + "\n")        
    return None


def TOOLS_ADD(env):
    """Conventional way to add builders to an SCons environment."""
    env.Append(BUILDERS = {
        "TrainMorfessor" : Builder(action=train_morfessor),
        "ApplyMorfessor" : Builder(action=apply_morfessor),
        "NormalizeMorfessorOutput" : Builder(action=normalize_morfessor_output),
    })

