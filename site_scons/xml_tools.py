"""
"""

from SCons.Builder import Builder
from SCons.Script import FindFile
import cPickle
import re
#import libxml2
from tei import TeiDocument
from xml.etree import cElementTree as et
import xml.sax
import tei
from bible import get_book_order, book_names, Point, Citation
import gzip

def xsl(target, source, env, for_signature):
    """
    The sources are the stylesheet and the file it should be applied to.
    """
    if source[1].rstr().endswith("xml"):
        return "java -jar bin/saxon9.jar -s:${SOURCES[1]} -xsl:${SOURCES[0]} -o:$TARGET"
    else:
        return "gunzip -c ${SOURCES[1]} | java -jar bin/saxon9.jar -s:- -xsl:${SOURCES[0]} -o:$TARGET"

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "XSL" : Builder(generator=xsl),
        })
