from SCons.Builder import Builder
from SCons.Util import is_List
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
from common_tools import meta_open, DataSet, Probability, temp_file
import cPickle as pickle
import math
import xml.etree.ElementTree as et
from subprocess import Popen, PIPE
from tempfile import mkstemp
from EMMA2 import main_class, morphassignment, assigneval, tools
from scons_tools import make_generic_emitter
import csv

def emma_score(env, target, gold, predictions):
    if is_List(gold):
        gold = gold[0]
    if is_List(predictions):
        predictions = predictions[0]
    gold_emma = env.DatasetToEMMA("%s-emma" % gold, gold)
    pred_emma = env.DatasetToEMMA("%s-emma" % predictions, predictions)
    return env.RunEMMA(target, [gold_emma, pred_emma])
    # with meta_open(source[0].rstr()) as gold_ifd, meta_open(source[1].rstr()) as pred_ifd:
    #     gold = DataSet.from_stream(gold_ifd)
    #     pred = DataSet.from_stream(pred_ifd)
    # with meta_open(target[0].rstr(), "w") as ofd:
    #     ofd.write("EMMA\n")
    #     ofd.write("%.3f\n" % 1.0)
    #return [gold_emma, pred_emma]

def get_without_case(word, morph):
    retvals = []
    for analysis in morph.get(word.lower(), []):
        morphs = []
        total = 0
        for m in analysis:
            nm = word[total:total + len(m)]
            if nm != "":
                morphs.append(nm)
                total += len(nm)
        retvals.append(morphs)
    return retvals

def add_morphology(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[-1]
    morphology = {}
    with meta_open(source[1].rstr()) as ifd:        
        for l in ifd:
            word, analyses = l.split("\t")
            morphology[word] = set()
            for analysis in analyses.split(", "):
                morphology[word].add(tuple([morph.split(":")[0] for morph in analysis.split() if not morph.startswith("~")]))

    #print [[(data.indexToWord[w], data.indexToTag.get(t, None), morphology.get(data.indexToWord[w], [])) for w, t, aa in s] for s in data.sentences][0:10]
    new_data = DataSet.from_sentences([[(data.indexToWord[w], data.indexToTag.get(t, None), get_without_case(data.indexToWord[w], morphology)) for w, t, aa in s] for s in data.sentences])
    with meta_open(target[0].rstr(), "w") as ofd:
        new_data.write(ofd)
    return None
    
def dataset_to_emma(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[-1]
    with meta_open(target[0].rstr(), "w") as ofd:
        for word, analyses in sorted(data.get_analyses().iteritems()):
            #if not re.match(r".*\W.*", word):
            #    continue
            word = word.lower()
            if len(analyses) == 0:
                x = "%s\t%s:NULL\n" % (word, word) #", ".join([" ".join(["%s:NULL" % m for m in a]) for a in analyses]))
            else:
                x = "%s\t%s\n" % (word, ", ".join([" ".join(["%s:NULL" % m.lower() for m in a]) for a in analyses]))            
            ofd.write(x.encode("utf-8"))
    return None

def _run_emma(target, source, env):
    with temp_file() as gold, temp_file() as guess, meta_open(source[0].rstr()) as _guess, meta_open(source[1].rstr()) as _gold:
        guesses = [x for x in _guess]
        words = [x.split()[0] for x in guesses]
        keep = set([x for x in guesses if re.match(r"^\w+$", x) and not re.match(r".*\d.*", x)])
        with meta_open(gold, "w") as gold_fd:
            gold_fd.write("\n".join([x for x in _gold if x.split()[0] in keep]))
        with meta_open(guess, "w") as guess_fd:
            guess_fd.write("\n".join([x for x in guesses if x.split()[0] in keep]))            
        cmd = env.subst("python ${EMMA} -g %s -p %s -L ${LPSOLVE_PATH}" % (guess, gold), source=source, target=target)
        pid = Popen(cmd.split(), stdout=PIPE)
        #out, err = pid.communicate()
        #prec, rec, fscore = [float(x.strip().split()[-1]) for x in out.strip().split("\n")[-3:]]
    with meta_open(target[0].rstr(), "w") as ofd:
        pass
        #ofd.write("\t".join(["MorphP", "MorphR", "MorphF"]) + "\n")
        #ofd.write("\t".join(["%.3f" % x for x in [prec, rec, fscore]]) + "\n")
    return None

def prepare_datasets_for_emma(target, source, env):
    args = source[-1].read()


    try:
        with meta_open(source[0].rstr()) as ifdA:
            dataA = DataSet.from_stream(ifdA)[0]
            analysesA = {w : ", ".join([" ".join(["%s:NULL" % m for m in a]) for a in aa]) for w, aa in dataA.get_analyses().iteritems()}
    except:
        with meta_open(source[0].rstr()) as ifdA:
            analysesA = {w : r for w, r in [l.strip().split("\t") for l in ifdA]}


    try:
        with meta_open(source[1].rstr()) as ifdB:
            dataB = DataSet.from_stream(ifdB)[0]
            analysesB = {w : ", ".join([" ".join(["%s:NULL" % m for m in a]) for a in aa]) for w, aa in dataB.get_analyses().iteritems()}
    except:
        with meta_open(source[1].rstr()) as ifdB:
            analysesB = {w : r for w, r in [l.strip().split("\t") for l in ifdB]}            

            
    wordsA = set(analysesA.keys())
    wordsB = set(analysesB.keys()) 
    common_words = wordsA.intersection(wordsB)
    with meta_open(target[0].rstr(), "w") as ofdA, meta_open(target[1].rstr(), "w") as ofdB:
        for word in common_words:
            ofdA.write(("%s\t%s\n" % (word, analysesA[word])).encode("utf8"))
            ofdB.write(("%s\t%s\n" % (word, analysesB[word])).encode("utf8"))
    return None

def run_emma(target, source, env, for_signature):
    return "python ${EMMA} -g ${SOURCES[1]} -p ${SOURCES[0]} -L ${LPSOLVE_PATH} > ${TARGET}"

def collate_results(target, source, env):
    results = []
    for f in source:
        properties = {k : v for k, v in [x.split("=") for x in os.path.splitext(os.path.basename(f.rstr()))[0].split("-")]}
        with meta_open(f.rstr()) as ifd:
            precision, recall, fmeasure = re.match(r".*precision\s*:\s*(\d\.\d+).*recall\s*:\s*(\d\.\d+).*fmeasure\s*:\s*(\d\.\d+).*", ifd.read(), re.S|re.M).groups()
            results.append((properties, (precision, recall, fmeasure)))
            #properties["precision"] = precision
            #properties["recall"] = recall
            #properties["fmeasure"] = fmeasure
    fields = sorted(set(sum([x[0].keys() for x in results], []))) + ["Precision", "Recall", "F-Measure"]
    with meta_open(target[0].rstr(), "w") as ofd:
        #ofd.write("\t".join(fields) + "\n")
        c = csv.DictWriter(ofd, fields, delimiter="\t")
        c.writerow({k : k for k in fields})
        for x, (p, r, f) in results:
            x["Precision"] = p
            x["Recall"] = r
            x["F-Measure"] = f
            c.writerow(x)
    return None

def TOOLS_ADD(env):
    env["EMMA"] = "bin/EMMA2.py"
    env.Append(BUILDERS = {
        "PrepareDatasetsForEMMA" : Builder(action=prepare_datasets_for_emma, emitter=make_generic_emitter(["work/emma_input/${SPEC}-guesses.txt",
                                                                                                           "work/emma_input/${SPEC}-gold.txt"])),
        "RunEMMA" : Builder(generator=run_emma, emitter=make_generic_emitter(["work/emma_output/${SPEC}.txt"])),
        "DatasetToEMMA" : Builder(action=dataset_to_emma),
        "AddMorphology" : Builder(action=add_morphology),
        "CollateResults" : Builder(action=collate_results),
    })
    env.AddMethod(emma_score, "EMMAScore")
