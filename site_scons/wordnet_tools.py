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
from babel import ProbabilityList
from subprocess import Popen, PIPE

def english_filter(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        words = ProbabilityList(ifd)
    #pid = Popen(env.subst("${FLOOKUP} ${TRMORPH}").split(), stdin=PIPE, stdout=PIPE)
    #stdout, stderr = pid.communicate("\n".join(words.keys()))
    #for l in stdout.split("\n"):
    #    toks = l.split()
    #    if len(toks) == 2 and toks[1] == "+?":
    #        del words[toks[0]]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(words.format())
    return None

#def turkish_filter_emitter(target, source, env):
#    if not target[0].rstr().endswith(".gz"):
#        new_target = "%s_trmorph.gz" % os.path.splitext(source[0].rstr())[0]
#    else:
#        new_target = target[0]
#    return new_target, source

def TOOLS_ADD(env):
    #env["FLOOKUP"] = "/home/tom/local/bin/flookup"
    #env["TRMORPH"] = "/home/tom/TRmorph/trmorph.fst"
    env.Append(BUILDERS = {
            "EnglishFilter" : Builder(action=english_filter), #, emitter=turkish_filter_emitter),
            })
