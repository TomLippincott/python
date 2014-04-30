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
from common_tools import Probability, temp_file
from torque_tools import run_command
import torque
from os.path import join as pjoin
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy

def filter_by(target, source, env):
    """
    Keep words in first vocabulary file that also occur in second vocabulary file
    """
    return None

def filter_by_emitter(target, source, env):
    return target, source

def TOOLS_ADD(env):
    BUILDERS = {"FilterBy" : Builder(action=filter_by, emitter=filter_by_emitter),
                }
        
