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
from os.path import join as pjoin
import tarfile


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
    """Conventional way to add builders to an SCons environment."""
    env.Append(BUILDERS = {"FilterBy" : Builder(action=filter_by, emitter=filter_by_emitter),
                           "TextToVocabulary" : Builder(action=text_to_vocabulary),
                           "ProbabilityListToVocabulary" : Builder(action=probability_list_to_vocabulary), #ensure_suffix=True, suffix="vocabulary.gz"),
                           }
               )
