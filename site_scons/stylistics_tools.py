from SCons.Builder import Builder
from SCons.Script import FindFile
from SCons.Node.FS import Dir, File
from SCons.Node.Alias import Alias
from subprocess import Popen, PIPE
import re
from common_tools import meta_open, meta_basename, log_likelihood, generic_emitter, unpack_numpy, unpoint
import logging
import os.path
import random
import cPickle as pickle
from functools import partial
try:
    import lxml.etree as et
except:
    import xml.etree.ElementTree as et
import sys
sys.path.append("../graphical_modeling")
#import graphmod
import logging


def run_lda(target, source, env):
    logging.info("loading docs...")
    args = source[-1].read()
    mat, labels, words = unpack_numpy(source[0].rstr(), dense=True)
    data = [[] for x in range(mat.shape[0])]
    for i, w in enumerate(words):
        lemma = w["_NAME"]
        for j in range(len(data)):
            data[j] += [lemma.encode("utf-8") for x in range(mat[j, i])]

    docs = graphmod.DocumentCollection([x for x in data if len(x) > 0])
    model = graphmod.LDA(args.get("topics", 100), args.get("alpha", .01), args.get("beta", .5))
    model.load_docs(docs)
    for i in range(1, args.get("iterations", 1000) + 1):
        logging.info("iteration %d/%d" % (i, args.get("iterations", 1000)))
        model.sample()
    
    assigns = model.get_assignments()
    topics = model.get_topic_word_counts()
    #print topics.shape, mat.shape
    pickle.dump((mat, labels, words, assigns, topics), meta_open(target[0].rstr(), "w"))
    return None


def add_definitions(target, source, env):
    mat, words, features = unpack_numpy(source[0].rstr())
    fd = meta_open(target[0].rstr(), "w")
    fd.write("<?xml-stylesheet href=\"presentation.xsl\" type=\"text/xsl\" ?><xml>")
    defs = {}
    for elem in [e for e in et.parse(source[1].rstr()).getiterator() if e.tag.endswith("div") and e.attrib.get("type", "") == "entry"]:
        defs[elem.attrib["n"]] = elem #[x.text for x in elem.getiterator() if x.tag.endswith("item")]
    logging.info("loaded %d dictionary entries", len(defs))

    for val, word in zip(mat, words):
        if word in defs:
            lemma = [x for x in defs[word].getiterator() if x.tag.endswith("w")][0].attrib["lemma"]
            fd.write("<w><lemma>%s</lemma><vals>" % lemma.encode("utf-8"))                    
            for h, v in zip(features, val):
                fd.write("<val>%s</val>\n" % h)
            fd.write("</vals><vals>")
            for h, v in zip(features, val):
                fd.write("<val>%f</val>\n" % v)
        
            fd.write("</vals><defs>")
            for d in [x for x in defs[word].getiterator() if x.tag.endswith("item")]:
                fd.write("<def>%s</def>\n" % d.text.encode("utf-8"))
            fd.write("</defs></w>\n")
        

    fd.write("</xml>")
    return None


def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "AddDefinitions" : Builder(action=add_definitions, emitter=partial(generic_emitter, name="strongs", ext="xml")),
        "RunLDA" : Builder(action=run_lda),
        })
