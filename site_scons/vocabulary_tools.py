from SCons.Builder import Builder
from SCons.Action import Action
from SCons.Subst import scons_subst
from SCons.Util import is_List
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import cPickle as pickle
import math
import xml.etree.ElementTree as et
import gzip
import subprocess
import shlex
import time
import shutil
import tempfile
import codecs
import locale
import bisect
from babel import ProbabilityList, Arpabo, Pronunciations, Vocabulary, FrequencyList
from common_tools import Probability, temp_file, meta_open
from torque_tools import run_command
import torque
from os.path import join as pjoin
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import tarfile

"""
file conventions: text.gz, vocabulary.gz, probabilities.gz, language_model.gz, pronunciations.gz, frequencies.gz
"""

def filter_by(target, source, env):
    """
    Keep words in first vocabulary file that also occur in second vocabulary file
    """
    with meta_open(source[0].rstr()) as ifdA, meta_open(source[1].rstr()) as ifdB:
        first = Vocabulary(ifdA)
        second = Vocabulary(ifdB)
    filtered = first.filter_by(second)
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(filtered.format())
    return None

def filter_by_emitter(target, source, env):
    if "." in os.path.basename(target[0].rstr()):
        return target, source
    else:
        return "%s_vocabulary.txt.gz" % (source[0].rstr()), source

def collect_text(target, source, env):
    pattern = re.compile(source[1].read())
    discard = re.compile(r"^(\[.*\]|\<.*\>|\(.*\)|\*.*\*)\s*$")
    keep = re.compile(r".*\w+.*", re.UNICODE)
    with tarfile.open(source[0].rstr()) as ifd, meta_open(target[0].rstr(), "w") as ofd:
        for name in ifd.getnames():
            if pattern.match(name):
                for line in ifd.extractfile(name):
                    if not discard.match(line):
                        words = [word for word in line.strip().split() if not discard.match(word) and keep.match(word)]
                    #words = [word for word in line.strip().split()]
                        if len(words) > 0:
                            ofd.write("%s\n" % (" ".join(words)))
    return None
    # words = set()
    # with meta_open(target[0].rstr(), "w") as ofd:
    #     for dname in source:
    #         for fname in glob(os.path.join(dname.rstr(), "*.txt")) + glob(os.path.join(dname.rstr(), "*.txt.gz")):
    #             with meta_open(fname) as ifd:
    #                 for line in ifd:
    #                     if not line.startswith("["):
    #                         toks = []
    #                         for x in line.lower().split():
    #                             if x == "<hes>":
    #                                 toks.append("<HES>")
    #                             elif not x.startswith("<"):
    #                                 toks.append(x)
    #                         for t in toks:
    #                             words.add(t)
    #                         if len(toks) > 0:
    #                             ofd.write("%s </s>\n" % (" ".join(toks)))
    # with meta_open(target[1].rstr(), "w") as ofd:
    #     ofd.write("# BOS: <s>\n# EOS: </s>\n# UNK: <UNK>\n<s>\n</s>\n<UNK>\n")
    #     ofd.write("\n".join(sorted(words)) + "\n")                                      
    # return None
def collect_text_emitter(target, source, env):
    if not tarfile.is_tarfile(source[0].rstr()):
        env.Exit()
    else:
        return target, source

def text_to_vocabulary(target, source, env):
    lower_case = len(source) == 1 or source[1].read()
    with meta_open(source[0].rstr()) as ifd:
        if lower_case:
            words = set(ifd.read().lower().split())
        else:
            words = set(ifd.read().split())
    vocab = Vocabulary.from_set([w for w in words if "_" not in w and not w.startswith("-") and not w.endswith("-")])
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(vocab.format())
    return None

def probability_list_to_vocabulary(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        probs = ProbabilityList(ifd)
    with meta_open(target[0].rstr(), "w") as ofd:
        vocab = Vocabulary.from_set(probs.get_words())
        ofd.write(vocab.format())
    return None

def TOOLS_ADD(env):
    env.Append(BUILDERS = {"FilterBy" : Builder(action=filter_by, emitter=filter_by_emitter),
                           "CollectText" : Builder(action=collect_text),
                           "TextToVocabulary" : Builder(action=text_to_vocabulary),
                           "ProbabilityListToVocabulary" : Builder(action=probability_list_to_vocabulary), #ensure_suffix=True, suffix="vocabulary.gz"),
                           }
               )
        
