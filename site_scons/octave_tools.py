from SCons.Builder import Builder
from SCons.Script import FindFile
from SCons.Node.FS import Dir, File
from SCons.Node.Alias import Alias
from subprocess import Popen, PIPE
import re
from common_tools import meta_open, meta_basename, log_likelihood, generic_emitter
import logging
import os.path
import random
import cPickle as pickle
from functools import partial


def spectral_clustering(target, source, env, for_signature):
    return "${OCTAVE} -q --path ${LIN_DIR} --eval \"visualize('$SOURCE', 2, 1, '$TARGET', '%s');\"" % source[-1].read().get("kern", "gauss")


def eps_to_pdf(target, source, env, for_signature):
    return "epstopdf ${SOURCE} --outfile ${TARGET}"

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "SpectralClustering" : Builder(generator=spectral_clustering, emitter=partial(generic_emitter, name="spectral", ext="eps")),
        "EPSToPDF" : Builder(generator=eps_to_pdf, emitter=partial(generic_emitter, name="", ext="pdf")),
        })
