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


def script_builder(command, piped=[]):
    return Builder()
