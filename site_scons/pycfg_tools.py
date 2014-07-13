from SCons.Builder import Builder
from SCons.Script import *
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import cPickle as pickle
import math
import sys
import gzip
from os.path import join as pjoin
from os import listdir
import tarfile
import operator
from random import randint
from subprocess import Popen, PIPE
from common_tools import meta_open, DataSet
from itertools import product
import numpy
from torque_tools import submit_job

def morphology_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        characters = set(sum([[c for c in w] for w in words], []))
    num_syntactic_classes = args.get("num_syntactic_classes", 1)
    rules = []
    if num_syntactic_classes == 1:
        rules += [("Word", ["Stem"]),                  
                  ("Word", ["Prefix", "Stem"]),
                  ("Word", ["Stem", "Suffix"]),
                  ("Word", ["Prefix", "Stem", "Suffix"]),
                  ("Prefix", ["^^^", "Chars"]),
                  ("Prefix", ["^^^"]),
                  ("Stem", ["Chars"]),
                  ("Suffix", ["Chars", "$$$"]),
                  ("Suffix", ["$$$"]),
                  ]
    else:
        for syntactic_class in range(num_syntactic_classes):
            rules += [("Word", ["Word%d" % syntactic_class]),
                      ("Word%d" % syntactic_class, ["Stem"]),
                      ("Word%d" % syntactic_class, ["Prefix", "Stem"]),
                      ("Word%d" % syntactic_class, ["Stem", "Suffix"]),
                      ("Word%d" % syntactic_class, ["Prefix", "Stem", "Suffix"]),
                      ("Prefix", ["^^^", "Chars"]),
                      ("Stem", ["Chars"]),
                      ("Suffix", ["Chars", "$$$"]),
                      ]
    rules += [("Chars", ["Char"]),
              ("Chars", ["Char", "Chars"]),
              ]
    rules += [("Char", [character]) for character in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
         ofd.write("\n".join(["%s --> %s" % (k, " ".join(v)) for k, v in rules]))
    with meta_open(target[1].rstr(), "w") as ofd:
        ofd.write("\n".join(["^^^ %s $$$" % (" ".join([c for c in w])) for w in words]))
    return None

def tagging_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()        
    num_tags = args.get("num_tags", 10)
    markov = args.get("markov", 2)    
    histories = product(range(num_tags), repeat=markov)
    rules = []
    for tag in range(num_tags):
        rules.append(("Sentence", ["Tag%d" % tag]))
        for word in words:
            rules.append(("Tag%d" % tag, [word]))
            for next_tag in range(num_tags):
                rules.append(("Tag%d" % tag, [word, "Tag%d" % next_tag]))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s --> %s" % (k, " ".join(v)) for k, v in rules]))
    with meta_open(target[1].rstr(), "w") as ofd:
        sentences = [" ".join([data.indexToWord[w] for w, t, m in s]) for s in data.sentences if len(s) < 40]
        ofd.write("\n".join([s for s in sentences]))
    return None

def joint_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        characters = set(sum([[c for c in w] for w in words], []))
    num_tags = args.get("num_tags", 10)
    markov = args.get("markov", 2)    
    histories = product(range(num_tags), repeat=markov)
    rules = []

    for tag in range(num_tags):
        rules += [
            ("Sentence", ["Tag%d" % tag]),
            ("Prefix%d" % tag, ("Prefix%d" % tag, "Chars")),
            ("Prefix%d" % tag, ["^^^"]),
            ("Stem%d" % tag, ("Stem%d" % tag, "Chars")),
            ("Stem%d" % tag, ["Chars"]),
            ("Suffix%d" % tag, ("Chars", "Suffix%d" % tag)),
            ("Suffix%d" % tag, ["$$$"]),
            ]
        for word in words:
            rules += [
                ("Tag%d" % tag, ["%s%d" % (word, tag)]),
                ("%s%d" % (word, tag), ["Prefix%d" % tag, "Stem%d" % tag, "Suffix%d" % tag]),
                ]
            for next_tag in range(num_tags):
                rules += [
                    ("Tag%d" % tag, ["%s%d" % (word, tag), "Tag%d" % next_tag]),
                    ]

    rules += [
        ("Chars", ("Chars", "Char")),
        ("Chars", ["Char"]),
        ]
    rules += [("Char", [character]) for character in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s --> %s" % (k, " ".join(v)) for k, v in rules]))
    with meta_open(target[1].rstr(), "w") as ofd:
        ofd.write("\n".join([" ".join([" ".join(["^^^"] + [c for c in data.indexToWord[w]] + ["$$$"]) for w, t, m in s]) for s in data.sentences]))
    return None

def gold_segmentations_joint_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        characters = set(sum([[c for c in w] for w in words], []))
        sentences = [[(data.indexToWord[w], [data.indexToAnalysis[x] for x in xs]) for w, t, xs in s] for s in data.sentences]

    segmentations = {}
    for w, xs in sum(sentences, []):
        if len(xs) == 0 or len(xs[0]) == 1:
            segmentations[w] = ("", w, "")
        elif len(xs[0]) > 2:
            segmentations[w] = (xs[0], "".join(xs[1:-1]), xs[-1])
        else:
            if len(xs[0][0]) > len(xs[0][1]):
                segmentations[w] = ("", xs[0][0], xs[0][1])
            else:
                segmentations[w] = (xs[0][0], xs[0][1], "")

    sentences = [[(w, segmentations[w]) for w, xs in s] for s in sentences]
    num_tags = args.get("num_tags", 10)

    rules = []
    for tag in range(num_tags):
        rules += [
            ("Sentence", ["Tag%d" % tag]),
            ("Prefix%d" % tag, ("Prefix%d" % tag, "Chars")),
            ("Prefix%d" % tag, ["^^^"]),
            ("Stem%d" % tag, ("@@@", "Chars", "@@@")),
            ("Stem%d" % tag, ["Chars"]),
            ("Suffix%d" % tag, ("Chars", "Suffix%d" % tag)),
            ("Suffix%d" % tag, ["$$$"]),
            ]
        for word in words:
            rules += [
                ("Tag%d" % tag, ["%s%d" % (word, tag)]),
                ("%s%d" % (word, tag), ["Prefix%d" % tag, "Stem%d" % tag, "Suffix%d" % tag]),
                ]
            for next_tag in range(num_tags):
                rules += [
                    ("Tag%d" % tag, ["%s%d" % (word, tag), "Tag%d" % next_tag]),
                    ]

    rules += [
        ("Chars", ("Chars", "Char")),
        ("Chars", ["Char"]),
        ]
    rules += [("Char", [character]) for character in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s --> %s" % (k, " ".join(v)) for k, v in rules]))
    with meta_open(target[1].rstr(), "w") as ofd:
        ofd.write("\n".join([" ".join([" ".join(["^^^"] + [c for c in pre] + ["@@@"] + [c for c in stm] + ["@@@"] + [c for c in suf] + ["$$$"]) for w, (pre, stm, suf) in s]) for s in sentences]))

    return None

def gold_tags_joint_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        characters = set(sum([[c for c in w] for w in words], []))
    sentences = [[(data.indexToWord[w], data.indexToTag[t]) for w, t, xx in s] for s in data.sentences]
    tags = set([t for w, t in sum(sentences, [])])
    rules = []
    for tag in tags:
        rules += [
            ("Sentence", ["Tag%s" % tag]),
            ("Prefix%s" % tag, ("Prefix%s" % tag, "Chars")),
            ("Prefix%s" % tag, ["^^^%s" % tag]),
            ("Stem%s" % tag, ("Stem%s" % tag, "Chars")),
            ("Stem%s" % tag, ["Chars"]),
            ("Suffix%s" % tag, ("Chars", "Suffix%s" % tag)),
            ("Suffix%s" % tag, ["$$$%s" % tag]),
            ]
        for word in words:
            rules += [
                ("Tag%s" % tag, ["%s%s" % (word, tag)]),
                ("%s%s" % (word, tag), ["Prefix%s" % tag, "Stem%s" % tag, "Suffix%s" % tag]),
                ]
            for next_tag in tags:
                rules += [
                    ("Tag%s" % tag, ["%s%s" % (word, tag), "Tag%s" % next_tag]),
                    ]
    rules += [
        ("Chars", ("Chars", "Char")),
        ("Chars", ["Char"]),
        ]
    rules += [("Char", [character]) for character in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\n".join(["%s --> %s" % (k, " ".join(v)) for k, v in rules]))
    with meta_open(target[1].rstr(), "w") as ofd:
        ofd.write("\n".join([" ".join([" ".join(["^^^%s" % t] + [c for c in w] + ["$$$%s" % t]) for w, t in s]) for s in sentences]))
    return None

def run_pycfg(target, source, env, for_signature):
    return env.subst("zcat ${SOURCES[1]}|${PYCFG_PATH}/py-cfg-mp ${SOURCES[0]} -w 0.1 -A ${TARGETS[0]} -N 1 -d 100 -E -n ${NUM_BURNINS} -e 1 -f 1 -g 10 -h 0.1 > ${TARGETS[1]}", target=target, source=source)

def list_to_tuples(xs):
    return [(xs[i * 2], xs[i * 2 + 1]) for i in range(len(xs) / 2)]

def run_pycfg_torque(target, source, env):
    """
    Pairs of inputs and outputs
    """
    for (cfg, data), (out, log) in zip(list_to_tuples(source), list_to_tuples(target)):
        cmd = env.subst("zcat ${SOURCES[1]}|${PYCFG_PATH}/py-cfg ${SOURCES[0]} -A ${TARGETS[0]} -d 100 -E -n ${NUM_BURNINS} -e 1 -f 1 -g 10 -h 0.1 > ${TARGETS[1]}", 
                        target=[out, log], 
                        source=[cfg, data])
    
    return None

def collate_pycfg_output(target, source, env):
    counts = {}
    wordToIndex = {}
    with meta_open(source[0].rstr()) as ifd:
        for m in re.finditer(r"Tag(\d+)(#\d+)? (\S+)", ifd.read()):
            tag, customers, word = m.groups()
            word = word.rstrip(")")
            tag = int(tag)
            wordToIndex[word] = wordToIndex.get(word, len(wordToIndex))
            word_id = wordToIndex[word]
            counts[tag] = counts.get(tag, {})
            counts[tag][word_id] = counts[tag].get(word_id, 0) + 1
    data = numpy.zeros(shape=(len(counts), len(wordToIndex)))
    for tag, words in counts.iteritems():
        for word, count in words.iteritems():
            data[tag, word] = count
    word_totals = data.sum(0)
    indexToWord = {v : k for k, v in wordToIndex.iteritems()}
    with meta_open(target[0].rstr(), "w") as ofd:
        for word_counts in data:
            indices = list(reversed(word_counts.argsort()))
            ofd.write(" ".join(["%s:%d" % (indexToWord[i], word_counts[i]) for i in indices[0:10] if word_counts[i] > 0]) + "\n")
        ofd.write("")
    return None

def torque_key(action, env, target, source):
    return 1

def TOOLS_ADD(env):
    env["PYCFG_PATH"] = "/usr/local/py-cfg"
    env.Append(BUILDERS = {
            "MorphologyCFG" : Builder(action=morphology_cfg),
            "TaggingCFG" : Builder(action=tagging_cfg),
            "JointCFG" : Builder(action=joint_cfg),
            "GoldTagsJointCFG" : Builder(action=gold_tags_joint_cfg),
            "GoldSegmentationsJointCFG" : Builder(action=gold_segmentations_joint_cfg),
            "RunPYCFG" : Builder(generator=run_pycfg),
            "RunPYCFGTorque" : Builder(action=SCons.Action.Action(run_pycfg_torque, batch_key=torque_key)),
            "CollatePYCFGOutput" : Builder(action=collate_pycfg_output),
            })
