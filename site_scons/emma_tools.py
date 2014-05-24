from SCons.Builder import Builder
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
from common_tools import meta_open, DataSet, Probability
import cPickle as pickle
import math
import xml.etree.ElementTree as et
from subprocess import Popen, PIPE

def emma_score(env, target, gold, predictions):
    gold_emma = env.DatasetToEMMA("%s-emma" % gold[0], gold)
    pred_emma = env.DatasetToEMMA("%s-emma" % predictions, predictions)
    return env.RunEMMA(target, [gold_emma, pred_emma])
    # with meta_open(source[0].rstr()) as gold_ifd, meta_open(source[1].rstr()) as pred_ifd:
    #     gold = DataSet.from_stream(gold_ifd)
    #     pred = DataSet.from_stream(pred_ifd)
    # with meta_open(target[0].rstr(), "w") as ofd:
    #     ofd.write("EMMA\n")
    #     ofd.write("%.3f\n" % 1.0)
    #return [gold_emma, pred_emma]

def add_morphology(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)
    morphology = {}
    with meta_open(source[1].rstr()) as ifd:        
        for l in ifd:
            word, analyses = l.split("\t")
            morphology[word] = set()
            for analysis in analyses.split(", "):
                morphology[word].add(tuple([morph.split(":")[0] for morph in analysis.split() if not morph.startswith("~")]))
    #print [[(data.indexToWord[w], data.indexToTag.get(t, None), morphology.get(data.indexToWord[w], [])) for w, t, aa in s] for s in data.sentences][0:10]
    new_data = DataSet.from_sentences([[(data.indexToWord[w], data.indexToTag.get(t, None), morphology.get(data.indexToWord[w], [])) for w, t, aa in s] for s in data.sentences])
    with meta_open(target[0].rstr(), "w") as ofd:
        new_data.write(ofd)
    return None

def dataset_to_emma(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)
    with meta_open(target[0].rstr(), "w") as ofd:
        for word, analyses in sorted(data.get_analyses().iteritems()):
            x = "%s\t%s\n" % (word, ", ".join([" ".join(["%s:NULL" % m for m in a]) for a in analyses]))
            ofd.write(x.encode("utf-8"))
    return None

def run_emma(target, source, env):
    cmd = env.subst("python ${EMMA} -g ${SOURCES[0]} -p ${SOURCES[1]}", source=source, target=target)
    pid = Popen(cmd.split(), stdout=PIPE)
    out, err = pid.communicate()
    prec, rec, fscore = [float(x.strip().split()[-1]) for x in out.strip().split("\n")[-3:]]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\t".join(["MorphP", "MorphR", "MorphF"]) + "\n")
        ofd.write("\t".join(["%.3f" % x for x in [prec, rec, fscore]]) + "\n")
    return None


def TOOLS_ADD(env):
    env["EMMA"] = "bin/EMMA2.py"
    env.Append(BUILDERS = {
            "RunEMMA" : Builder(action=run_emma),
            "DatasetToEMMA" : Builder(action=dataset_to_emma),
            "AddMorphology" : Builder(action=add_morphology),
            })
    env.AddMethod(emma_score, "EMMAScore")
