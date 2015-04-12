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
#import fst


fstcompile = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstcompose = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstprune = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstrmepsilon = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstprint = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstshortestpath = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstprintpaths = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")


#cmd = env.subst("fstcompose - ${SOURCES[0]} | fstprune -weight=${PRUNE} - | fstrmepsilon | fstprint -isymbols=${SOURCES[3]}  -osymbols=${SOURCES[1]} -  | perl ${CN_KWS_SCRIPTS}/collapse_fields.pl | fstcompile --acceptor -isymbols=${SOURCES[2]} -  | fstshortestpath --nshortest=10000 - | ${CN_KWS_SCRIPTS}/fstprintpaths ${SOURCES[2]} -  |  perl ${CN_KWS_SCRIPTS}/process.all.words.pl - KEY | sort -k 5 -gr |perl  ${CN_KWS_SCRIPTS}/clean_result.words.pl -", source=source, target=target)

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "FSTCompile" : Builder(action="${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}"),
        "FSTArcSort" : Builder(action="${OPENFST_BINARIES}/fstarcsort --sort_type=${SOURCES[1]} ${SOURCES[0]} ${TARGETS[0]}"),
    })
