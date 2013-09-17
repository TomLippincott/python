from SCons.Builder import Builder
from SCons.Script import FindFile
from SCons.Node.FS import Dir, File
from SCons.Node.Alias import Alias
from subprocess import Popen, PIPE
#import sqlite3
import re
import xml.etree.ElementTree as et
import csv
import numpy
import logging
import os.path
import os
import pickle


def mallet_line(target, source, env):
    fd = meta_open(target[0].rstr(), "w")
    for div in [x for x in et.parse(meta_open(source[0].rstr())).getiterator() if x.get("type", False) == source[1].read()]:
        text = " ".join(["".join([y.text for y in x.getiterator() if y.text]) for x in div.getiterator("fs") if x.get("type", None) == "token"])
        label = div.get("n").replace(" ", "_")
        fd.write("\t".join([label, label, text.replace("\n", " ").encode("utf-8")]) + "\n")
    fd.close()
    return None


def mallet_train(target, source, env, for_signature):
    return "${MALLET_PATH}/bin/mallet train-topics --input ${SOURCES[0]} --num-topics ${SOURCES[1]} --output-topic-keys ${TARGETS[0]} --output-doc-topics ${TARGETS[1]} --optimize-interval 0 --beta .02 --num-top-words 10 --show-topics-interval 1100 --num-iterations 100"


def mallet_train_emitter(target, source, env):
    return target, source
    base = meta_basename(source[0].rstr())
    path = os.path.dirname(source[0].rstr())
    return ["%s/%s.%s" % (path, base, x) for x in ["topic-key", "doc-key"]], source


def hdp_train(target, source, env, for_signature):
    args = source[-1].read()
    return "${HDP_PATH}/hdp --sample_hyper %(sample_hyper)s --split_merge %(split_merge)s --algorithm train --data ${SOURCES[0]} --directory %(directory)s --max_iter %(max_iter)s --save_lag %(save_lag)s" % (args)


def hdp_train_emitter(target, source, env):
    args = source[1].read()
    args["save_lag"] = args.get("save_lag", 100)
    args["max_iter"] = args.get("max_iter", 1000)
    for x in ["sample_hyper", "split_merge"]:
        args[x] = args.get(x, "no")
    args["directory"] = target[0].rstr()
    new_targets = []
    names = [".bin", "-topics.dat", "-word-assignments.dat"]
    for i in range(args["max_iter"] / args["save_lag"]):
        num = i * args["save_lag"]
        for name in names:
            new_targets.append(os.path.join(target[0].rstr(), "%05d%s" % (num, name)))
    new_targets += [os.path.join(target[0].rstr(), "mode%s" % x) for x in names] + [os.path.join(target[0].rstr(), "state.log")]
    return new_targets, source


def hdp_data(target, source, env):
    fd = open(target[0].rstr(), "w")
    atoks = {}
    for line in [x for x in open(source[0].rstr())]:
        toks = line.lower().strip().split() #re.sub(r"[^a-z\s]", "", line.lower()).strip().split()
        datum = {}
        for tok in toks[1:]:
            if tok not in atoks:
                atoks[tok] = len(atoks)
            datum[tok] = datum.get(tok, 0) + 1
        if len(datum) == 0:
            raise Exception("empty document!")
        fd.write("%s %s\n" % (len(datum), " ".join(["%s:%s" % (atoks[k], v) for k, v in datum.iteritems()])))
    ratoks = dict([(v, k) for k, v in atoks.iteritems()])
    open(target[1].rstr(), "w").write("\n".join([ratoks[i] for i in range(len(atoks))]))
    return None


def hdp_collate(target, source, env):
    assignments = [[int(y) for y in x.split()] for x in open(source[0].rstr()) if not x.startswith("d")]
    num_topics = len(set([x[2] for x in assignments]))
    num_docs = len(set([x[0] for x in assignments]))
    num_words = len(set([x[1] for x in assignments]))
    lookup = dict([(i, x.strip()) for i, x in enumerate(open(source[1].rstr()))])
    senders = [x.split()[0] for x in open(source[2].rstr())]
    txw = numpy.zeros(shape=(num_topics, num_words))
    dxt = numpy.zeros(shape=(num_docs, num_topics))
    for d, w, t, table in assignments:
        txw[t, w] += 1
        dxt[d, t] += 1
    dxt = numpy.transpose(dxt.T / dxt.sum(1))
    txw = numpy.transpose(txw.T / txw.sum(1))
    pickle.dump(dxt, open(target[0].rstr(), "w"))
    #dxt_fd = open(target[0].rstr(), "w")
    #for doc, (sender, props) in enumerate(zip(senders, dxt)):
    #    dxt_fd.write("%d %s %s\n" % (doc, sender, " ".join(["%d %f" % (i, v) for i, v in enumerate(props)])))
    pickle.dump(txw, open(target[1].rstr(), "w"))
    #txw_fd = open(target[1].rstr(), "w")
    #for doc, (sender, props) in enumerate(zip(senders, dxt)):
    #    dxt_fd.write("%d %s %s\n" % (doc, sender, " ".join(["%d %f" % (i, v) for i, v in enumerate(props)])))
    fd = open(target[2].rstr(), "w")
    for i, topic in enumerate(txw):
        fd.write("%s\n" % (" ".join([str(lookup[x]) for x in reversed(topic.argsort())][0:10])))
        #fd.write("%d 1 %s\n" % (i, " ".join([str(lookup[x]) for x in reversed(topic.argsort())][0:10])))
    fd.close()
    return None


def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        'MalletLine' : Builder(action=mallet_line),
        'MalletData' : Builder(action="${MALLET_PATH}/bin/mallet import-file --input $SOURCE --output $TARGET --keep-sequence"), # --token-regex \"[\\p{L}\\p{N}_]+\""),
        'MalletTrain' : Builder(generator=mallet_train, emitter=mallet_train_emitter),
        'HDPTrain' : Builder(generator=hdp_train, emitter=hdp_train_emitter),
        'HDPData' : Builder(action=hdp_data),
        'HDPCollate' : Builder(action=hdp_collate),
        })
