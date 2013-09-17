from SCons.Builder import Builder
from SCons.Script import FindFile
from SCons.Node.FS import Dir, File
from SCons.Node.Alias import Alias
from subprocess import Popen, PIPE
from weka import regex, arff
#import sqlite3
import re
import xml.etree.ElementTree as et
import csv
import weka
import feature_extractors
import numpy
import numpy.ma as ma
import logging
import os.path
import random
import cPickle as pickle
from functools import partial
import cPickle
import scipy
from common_tools import meta_open, generic_emitter
import cPickle as pickle
import tempfile
import os
import tarfile
from glob import glob
import gzip
import re


def ccg_parse(target, source, env):
    """
    Parse files, one sentence per line.
    """
    vals = []
    parser = candc.CCGParser(models="/home/tom/parsers/candc_svn/candc_models/models")
    for fname in source[0:-1]:
        for l in meta_open(fname.rstr()):
            label, line = l.split("\t")
            vals.append(parser.parse(line))
    pickle.dump(vals, meta_open(target[0].rstr(), "w"))
    return None


def rasp_parse(target, source, env):
    """
    Parse one file, one sentence per line.
    """
    pid = Popen(["/bin/sh", "/home/tom/parsers/rasp/scripts/rasp.sh"], cwd="/home/tom/parsers/rasp", stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = pid.communicate("\n".join([meta_open(f.rstr()).read() for f in source[0:-1]]))
    meta_open(target[0].rstr(), "w").write(out)
    return None


def ccg_train(target, source, env):
    """
    Train a parsing model using output from a previous run.
    """
    return None


def rasp_train(target, source, env):
    return None

def rasp_to_scfs(target, source, env):
    tmp = "data/patrules/temp.txt"
    open(tmp, "w").write(re.sub("gr-list: 1", "", meta_open(source[0].rstr()).read().replace("'", "_").replace('"', '_').replace("<", "_").replace(">", "_")))
    if os.path.exists("data/patrules/temp.out"):
        os.remove("data/patrules/temp.out")
    pid = Popen(["/bin/sh", "runLexiconBuilder.sh", "-i", "temp.txt", "-o", "temp.out"], cwd="data/patrules", stdout=PIPE, stderr=PIPE)    
    out, err = pid.communicate()
    meta_open(target[0].rstr(), "w").write(open("data/patrules/temp.out").read())
    os.remove("data/patrules/temp.out")
    os.remove(tmp)
    return None


def switch(m):
    n = "%s=\"%s\"%s" % (m.group(1), re.sub("\"|\>|\<", "_", m.group(2)), m.group(3))
    return n
rx1 = re.compile("([^\>])\s*\n\s*", re.S)
rx2 = re.compile("(\S+)\=\"(.*?)\"([\s\>\/])")
rx3 = re.compile("(\<instance.*?\<\/instance\>)", re.S)

def drop(m):
    try:
        if et.fromstring(m.group(1)):
            return m.group(1)
    except:
        return ""

def fix_xml(s):
    return re.sub("(\=\"[^\"]*)\&([^\"]*\")", r"\1_\2", s)

def extract_scfs(target, source, env):
    tmp = "temp/scfs.xml"
    newtext = gzip.open(source[0].rstr()).read()
    #newtext = re.sub(rx1, r"\1", gzip.open(source[0].rstr()).read())
    #newtext = re.sub(rx2, switch, newtext)
    #newtext = re.sub(rx3, drop, gzip.open(source[0].rstr()).read())
    open(tmp, "w").write(fix_xml(newtext))
    #et.fromstring(newtext.replace("&&", "__"))
    #et.parse(tmp)
    #sys.exit()
    base = os.path.splitext(os.path.splitext(os.path.basename(source[0].rstr()))[0])[0]
    tmpdir = "temp/%s" % base
    os.mkdir(tmpdir)
    pid = Popen(["/usr/groups/dict/subcat-2009/scripts/ext.pl", "-i", tmp, "-o", "%s/lex" % tmpdir, "-single", "-multislot", "2"])
    pid.communicate()
    fd = tarfile.open(target[0].rstr(), "w:gz")
    fd.add(tmpdir, arcname=base)
    fd.close()
    [os.remove(x) for x in glob("%s/*" % tmpdir)]
    os.rmdir(tmpdir)
    os.remove(tmp)
    return None

import StringIO
def combine_scfs(target, source, env):
    data = {}
    freqs = {}
    totals = {}
    for fname in source:
        ifd = tarfile.open(fname.rstr(), "r:gz")
        for mem in [x for x in ifd.getmembers() if x.isfile()]:
            name = os.path.basename(mem.name)            
            if name == "lex.freq":
                text = ifd.extractfile(mem).read()
                for m in re.finditer("(\S+) total=(\S+)\s*\n\s*\n(.*?)\n\s*(\n|$)", text, re.S):
                    verb = m.group(1)
                    if verb == "analyse":
                        verb = "analyze"
                    if verb == "recognise":
                        verb = "recognize"
                    totals[verb] = totals.get(verb, 0.0) + float(m.group(2))
                    if verb not in freqs:
                        freqs[verb] = {}
                    scf = None
                    for l in m.group(3).strip().split("\n"):
                        toks = l.split()
                        if len(toks) == 4:
                            scf = toks[0]
                            if scf not in freqs[verb]:
                                freqs[verb][scf] = {}
                        try:
                            freqs[verb][scf][toks[-1]] = freqs[verb][scf].get(toks[-1], 0.0) + float(toks[-2])
                        except:
                            pass
            else:
                data[name] = data.get(name, {})
                for verb, frame, freq, count in [x.split() for x in ifd.extractfile(mem)]:
                    if verb == "analyse":
                        verb = "analyze"
                    if verb == "recognise":
                        verb = "recognize"
                        
                    data[name][verb] = data[name].get(verb, {})
                    data[name][verb][frame] = data[name][verb].get(frame, 0) + float(count)
    ofd = tarfile.open(target[0].rstr(), "w:gz")
    bn = os.path.splitext(os.path.basename(target[0].rstr()))[0]
    os.mkdir("/local/scratch-2/tl318/temp/%s" % bn)
    fd = open("/local/scratch-2/tl318/temp/%s/lex.freq" % bn, "w")
    for verb, count in totals.iteritems():
        fd.write("%s total=%s\n\n" % (verb, count))
        for scf, vals in freqs[verb].iteritems():
            for i, (add, num) in enumerate(vals.iteritems()):
                if i == 0:
                    fd.write("%s\t%s\t%s\t%s\n" % (scf, num / float(count), num, add))
                else:
                    fd.write("\t%s\t%s\t%s\n" % (num / float(count), num, add))
        fd.write("\n")
    for mem, vals in data.iteritems():
        fd = open("/local/scratch-2/tl318/temp/%s/%s" % (bn, mem), "w")
        for verb, frames in vals.iteritems():
            for frame, count in frames.iteritems():                
                fd.write("%s %s %s %s\n" % (verb, frame, count / totals[verb], count))
        fd.close()
    ofd.add("/local/scratch-2/tl318/temp/%s" % (bn), arcname=bn)        
    ofd.close()
    [os.remove(x) for x in glob("/local/scratch-2/tl318/temp/%s/*" % bn)]
    os.rmdir("/local/scratch-2/tl318/temp/%s" % bn)
    return None


def filter_scfs(target, source, env):
    args = source[-1].read()
    lex = tempfile.NamedTemporaryFile(dir=env["TEMP_DIR"], delete=False, prefix="lex")
    data = {}
    if source[0].rstr().endswith(".tbz2"):
        tf = tarfile.open(source[0].rstr(), "r:bz2")
        lex.write(tf.extractfile([x for x in tf.getnames() if x.endswith("count.scf")][0]).read())
    else:
        lex.write(meta_open(source[0].rstr()).read())
    lex.close()
    tfile = tempfile.NamedTemporaryFile(dir=env["TEMP_DIR"], delete=False, prefix="lex")
    os.remove(tfile.name)
    pid = Popen(["%s/scripts/process-scf-counts.pl" % env["SUBCAT_2009"], lex.name, tfile.name] + args.get("EXTRA", []))
    pid.communicate()
    #print lex.name
    #print ["/home/tom/subcat-2009/scripts/process-scf-counts.pl", lex.name, target[0].rstr()] + args.get("EXTRA", [])
    os.remove(lex.name)
    meta_open(target[0].rstr(), "w").write(open(tfile.name).read())
    os.remove(tfile.name)
    return None

def evaluate_scfs(target, source, env):
    args = source[-1].read()
    raw = tempfile.NamedTemporaryFile(dir=env["TEMP_DIR"], delete=False, prefix="tlraw")
    traw = tempfile.NamedTemporaryFile(dir=env["TEMP_DIR"], delete=False, prefix="tltraw")
    tout = tempfile.NamedTemporaryFile(dir=env["TEMP_DIR"], delete=False, prefix="tlout")
    lexicon = tempfile.NamedTemporaryFile(dir=env["TEMP_DIR"], delete=False, prefix="tllex")
    gold = tempfile.NamedTemporaryFile(dir=env["TEMP_DIR"], delete=False, prefix="tlgold")
    if source[1].rstr().endswith(".tgz"):
        tf = tarfile.open(source[1].rstr(), "r:gz")
        traw.write(tf.extractfile([x for x in tf.getnames() if x.endswith("count.scf")][0]).read())
    else:
        traw.write(meta_open(source[1].rstr()).read())
    traw.close()
    raw_v = set([l.split()[0] for l in meta_open(traw.name)])
    lex_v = set([l.split()[0] for l in meta_open(source[0].rstr())])
    gold_v = set([l.split()[0] for l in meta_open(source[2].rstr()) if len(l.split()) == 1])
    if "FILTER_BY" in args:
        #filt_v = set([l.split()[0] for l in meta_open(args["FILTER_BY"]) if len(l.split()) == 1])
        filt_v = set([l.strip() for l in meta_open(args["FILTER_BY"])])
        verbs = set.intersection(raw_v, lex_v, gold_v, filt_v)
    else:
        verbs = set.intersection(raw_v, lex_v, gold_v)
    rx = re.compile("^(%s)\s+" % ("|".join(verbs)))
    for l in [l for l in meta_open(traw.name) if rx.match(l)]:
        verb, scf, freq, count = l.strip().split()
        scf = scf.split("_")[0].lstrip("0")
        raw.write("%s %s %s %s\n" % (verb, scf, freq, count))
    for l in [l for l in meta_open(source[0].rstr()) if rx.match(l)]:
        verb, scf, freq, count = l.strip().split()
        scf = scf.split("_")[0].lstrip("0")
        lexicon.write("%s %s %s %s\n" % (verb, scf, freq, count))
    gold.write(meta_open(source[2].rstr()).read())
    gold.close()
    raw.close()
    lexicon.close()
    pid = Popen(["%s/scripts/eval-scf-counts.pl" % env["SUBCAT_2009"], lexicon.name, tout.name, "-raw", raw.name, "-gold", gold.name])
    pid.communicate()
    text = open(tout.name).read()
    xml = et.TreeBuilder()
    xml.start("xml", {})
    for k, v in [x for x in args.iteritems() if not x[0].startswith("_")]:
        xml.start(k, {})
        xml.data(str(v))
        xml.end(k)
    xml.start("text", {})
    xml.data(text)
    xml.end("text")
    xml.end("xml")
    meta_open(target[0].rstr(), "w").write(et.tostring(xml.close()))
    os.remove(lexicon.name)
    os.remove(gold.name)
    os.remove(raw.name)
    os.remove(traw.name)
    os.remove(tout.name)
    return None


def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        'CCGParse' : Builder(action=ccg_parse, emitter=partial(generic_emitter, targets=[("", "txt.gz")])),
        'RASPParse' : Builder(action=rasp_parse, emitter=partial(generic_emitter, targets=[("", "txt")])),
        'RASPSCF' : Builder(action=rasp_to_scfs, emitter=partial(generic_emitter, targets=[("", "xml.gz")])),
        'ExtSCF' : Builder(action=extract_scfs, emitter=partial(generic_emitter, targets=[("", "tgz")])),
        'CombineSCFs' : Builder(action="python bin/combine_counts.py -o $TARGET ${SOURCES}"),
        'FilterSCFs' : Builder(action=filter_scfs, emitter=partial(generic_emitter, targets=[("", "txt")])),
        'EvaluateSCFs' : Builder(action=evaluate_scfs, emitter=partial(generic_emitter, targets=[("", "txt")])),
        'SCFThresh' : Builder(action="${SUBCAT_2009}/scripts/scf-thresh.pl ${SOURCES[0]} ${TARGET} -gold ${SOURCES[1]} -comlex %(S)s/comlex_paradigms -anlt %(S)s/single_paradigms" % {"S" : "data/paradigms"}),
        'SummarizeEval' : Builder(action="python bin/examine_results.py -i ${SOURCE} -o ${TARGET.dir}"),
        'GoldStandard' : Builder(action="python bin/create_gold_standard.py -i ${SOURCE} -o ${TARGET} ${FLAGS}"),
        'ExtractFeatures' : Builder(action="python bin/extract_features.py -i ${SOURCE} -o ${TARGET} ${FLAGS}"),
        'CoarsenFile' : Builder(action="python bin/make_inclusive_biolex.py -i ${SOURCE} -o ${TARGET} ${FLAGS}"),
        'BestMap' : Builder(action="python bin/make_best_biolexicon.py -i ${SOURCE} -o ${TARGET} ${FLAGS}"),
        'PrepareDists' : Builder(action="python bin/prepare_distros.py ${SOURCES} -o ${TARGET} ${FLAGS}"),
        'WekaToLexicon' : Builder(action="python bin/weka_to_lexicon.py ${SOURCE} -o ${TARGET} ${FLAGS}", emitter=partial(generic_emitter, targets=[("", "txt")])),
        'PennToConll' : Builder(action="${JYTHON} -J-Xmx${MEMORY}m bin/penn_to_conll.py -i ${SOURCE} -o ${TARGET} -s ${PARSER_PATH}/stanford/"),
        'StanfordTrain' : Builder(action="${JYTHON} -J-Xmx${MEMORY}m bin/stanford_train.py -i ${SOURCES[0]} -o ${TARGET} -s ${PARSER_PATH}/stanford/ ${SOURCES[1].read()} &> /dev/null"),
        'StanfordTest' : Builder(action="${JYTHON} -J-Xmx${MEMORY}m bin/stanford_run.py -m ${SOURCES[0]} -i ${SOURCES[1]} ${SOURCES[2].read()} -o ${TARGET} -s ${PARSER_PATH}/stanford/"),
        'StanfordEval' : Builder(action="${JYTHON} -J-Xmx${MEMORY}m bin/stanford_eval.py ${SOURCES[0:-1]} -o ${TARGET} -s ${PARSER_PATH}/stanford/ ${SOURCES[-1].read()}"),
        'MaltTrain' : Builder(action="java -Xmx${MEMORY}m -jar ${PARSER_PATH}/malt/malt.jar -m learn -c ${TARGET.rstr().split('.')[0]} -i ${SOURCE}"),
        })
