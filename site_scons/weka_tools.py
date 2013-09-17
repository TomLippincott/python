from SCons.Builder import Builder
from SCons.Script import FindFile
from SCons.Node.FS import Dir, File
from SCons.Node.Alias import Alias
from subprocess import Popen, PIPE
from weka import regex, arff
import sqlite3
import re
import xml.etree.ElementTree as et
import csv
import weka
import feature_extractors
from common_tools import meta_open, meta_basename, log_likelihood, generic_emitter
import numpy
import logging
import os.path
import random
import cPickle
from functools import partial


def verify(target, env):
    cmd = env.subst("${WEKA_CMD} weka.core.converters.ArffLoader")
    for f in target:
        if "Exception" in  Popen(cmd.split() + [f.rstr()], stdout=PIPE, stderr=PIPE, stdin=PIPE).communicate()[1]:
            return "ERROR processing %s!" % f.rstr()
    return None


def filter_counts(target, source, env):
    fd = weka.Arff(filename=source[0].rstr())
    feats = dict([(x, 0.0) for x, y in fd.attributes.iteritems() if not isinstance(y, set)])
    for d in fd.data:
        for k, v in d.iteritems():
            if isinstance(v, float):
                feats[k] += v
    remove = []
    for k, v in feats.iteritems():
        if v < source[1].read()["threshold"]:
            remove.append(k)
    for f in remove:
        s = f.split("_=_")[0]
        for d in fd.data:
            d["%s_+_" % s] = d.get("%s_+_" % s, 0.0) - d.get(f, 0)
        del fd.attributes[f]            
    fd.save(meta_open(target[0].rstr(), "w"))
    return None


def remove_features(target, source, env):
    fd = weka.Arff(filename=source[0].rstr())
    remove = []
    
    for f in fd.attributes.keys():
        if any([re.match(rx, f) for rx in source[1].read().get("REMOVE", [])]):
            remove.append(f)
    for f in remove:
        del fd.attributes[f]
    fd.save(meta_open(target[0].rstr(), "w"))
    return None


def filter_distros(target, source, env):
    fd = weka.Arff(filename=source[0].rstr())
    filt_fd = weka.Arff(filename=source[2].rstr())
    valid_feats = [x.split("_=_")[-1] for x in filt_fd.attributes.iteritems() if not isinstance(v, set)]
    fnames = [k for k, v in fd.attributes.iteritems() if not isinstance(v, set)]
    for f in fnames:
        if f not in valid_feats:
            del fd.attributes[f]
    fd.save(meta_open(target[0].rstr(), "w"))
    return None


def arff_file(target, source, env):
    """
    Build an arff file using feature extractors in env["FEATURE_EXTRACTORS"],
    where each file represents an instance.  The file name is used as
    a string attribute.  If the source is an Alias, use its children.  If the
    source is a Directory, use all files beneath it.
    """
    texts = []
    for f in source:
        if isinstance(f, File):
            for d, l in env["SPLIT"](f.rstr()):
                l.update(env["MAPPING"])
                texts += [(d, l)]
        elif isinstance(f, Dir):
            for fname in f.glob("*"):
                add_file(fname)
        elif isinstance(f, Alias):
            for fname in f.children():
                add_file(fname.rstr())
    fd = weka.Arff()
    for text, labels in texts:
        feats = labels
        for extractor in env["FEATURE_EXTRACTORS"]:
            new_feats = extractor(text)
            feats.update(new_feats)
        fd.add_datum(feats, force=True)
    fd.save(meta_open(target[0].rstr(), "w"), sparse=True)
    if env.get("VERIFY", True):
        return verify(target, env)
    else:
        return None


def subsample_arff_file(target, source, env):
    args = source[-1].read()
    sentences = [m.group(1) + m.group(2) for m in re.finditer("\n\s*\n(.*?)(^<c>.*?)$", meta_open(source[0].rstr()).read(), re.M | re.S)]
    assignments = [0 for i in range(int((1.0 - args["FRACTION"]) * len(sentences) / args["WINDOW"]))] + \
        [1 for i in range(int(args["FRACTION"] * len(sentences) / args["WINDOW"]))]
    #assignments = [0 for i in range(len(sentences) / args["WINDOW"] - 1)] + [1 for i in range(2)]
    fd = weka.Arff()
    for j in range(1, args["COUNT"] + 1):
        logging.error("processing random sample #%d", j)
        text = ""
        random.shuffle(assignments)
        for i, m in enumerate(assignments):
            if m == 1:
                text += "\n\n".join(sentences[i * args["WINDOW"] : min((i + 1) * args["WINDOW"], len(sentences))]).strip() + "\n\n"
        feats = {"LABEL_sample" : "sample_%d" % j}
        for extractor in env["FEATURE_EXTRACTORS"]:
            #print text
            new_feats = extractor(text)
            feats.update(new_feats)
        fd.add_datum(feats, force=True)
    fd.save(meta_open(target[0].rstr(), "w"))
    return None


def combine_arffs(target, source, env):
    new = weka.LazyArff()
    new.cname = env.get("CLASS", None)
    for fname in source[1:]:
        new.add_file(fname.rstr())
        logging.debug("processed %s", fname.rstr())
        logging.debug("%d attributes", len(new.attributes))
    logging.debug("saving to %s", target[0].rstr())
    new.save(meta_open(target[0].rstr(), "w"))
    return None
    #if env.get("VERIFY", False):
    #    return verify(target, env)
    #else:
    #    return None


def sum_arff(target, source, env):
    args = source[-1].read()
    new = weka.Arff(filename=source[0].rstr())
    new.sum_over(args["OVER"])
    new.save(meta_open(target[0].rstr(), "w"), sparse=True)
    if env.get("VERIFY", True):
        return verify(target, env)
    else:
        return None


def normalize_arff(target, source, env):
    new = weka.Arff(filename=source[0].rstr())
    new.normalize()
    new.save(meta_open(target[0].rstr(), "w"), sparse=True)
    if env.get("VERIFY", True):
        return verify(target, env)
    else:
        return None


def replace_missing(target, source, env):
    new = weka.Arff(filename=source[0].rstr())
    new.replace_missing()
    new.save(meta_open(target[0].rstr(), "w"), sparse=True)
    if env.get("VERIFY", True):
        return verify(target, env)
    else:
        return None

def split_arff(target, source, env):
    header = []
    fd = meta_open(source[0].rstr())
    for l in fd:
        header.append(l)
        if re.match(r"^@data.*$", l, re.I):
            break
    header = "".join(header)
    data = [l for l in fd]
    count = source[-1].read()
    per = len(data) / count
    for i, oname in zip(range(count), target):
        ofd = meta_open(oname.rstr(), "w")
        ofd.write(header + "\n")
        ofd.write("".join([l for l in data[i * per : (i + 1) * per]]))
        ofd.close()
    return None


def wordnet_filter(target, source, env):
    fd = weka.Arff(filename=source[0].rstr())
    newfd = weka.Arff()
    for d in fd.data:
        new_datum = {}
        for k, v in d.iteritems():
            m = re.match(source[2].read(), k)
            if m:
                lemma = wordnet.morphy(m.group("word").lower(), source[1].read())
                if lemma:
                    key = k.replace(m.group("word"), lemma)
                    new_datum[key] = new_datum.get(key, 0) + v
            else:
                new_datum[k] = v
        newfd.add_datum(new_datum)
    newfd.save(meta_open(target[0].rstr(), "w"), sparse=True)
    if env.get("VERIFY", True):
        return verify(target, env)
    else:
        return None

def remove_instances(target, source, env):
    fd = weka.LazyArff(fname=source[0].rstr(), cname=source[1].read().get("_CLASS", None))    
    filt = dict([(k, v) for k, v in source[1].read().iteritems() if not k.startswith("_")])
    fd.save(meta_open(target[0].rstr(), "w"), filt=filt)
    if env.get("VERIFY", False):
        return verify(target, env)
    else:
        return None


def apply_filter(target, source, env, for_signature):
    return "${WEKA_CMD} weka.filters.${SOURCES[1]} -i ${SOURCES[0]} ${SOURCES[2].read()} | gzip 1> ${TARGETS[0]}"


def filter_features(target, source, env, for_signature):
    rx = source[1].read()
    inv = ""
    if len(source) == 3 and source[2].read():
        inv = "-V"
    return "${WEKA_CMD} weka.filters.unsupervised.attribute.RemoveByName -i ${SOURCES[0]} -E %s %s | gzip 1> ${TARGETS[0]}" % (rx, inv)

def train_classifier(target, source, env, for_signature):
    return "${WEKA_CMD} weka.classifiers.${TYPE} ${ARGS} -t ${SOURCES[0]} -d ${TARGETS[0]} -i -k -o -v 1> ${TARGETS[1]}"


def apply_classifier(target, source, env, for_signature):
    return "${WEKA_CMD} weka.filters.supervised.attribute.AddClassification -c last -classification -distribution -i ${SOURCES[0]} -serialized ${SOURCES[1]} ${SOURCES[2]}|gzip 1> ${TARGETS[0]}"
    #return "${WEKA_CMD} weka.classifiers.${TYPE} -T ${SOURCES[0]} -v -o -i -l ${SOURCES[1]} ${SOURCES[2]} 1> ${TARGETS[0]}"


def train_clusterer(target, source, env, for_signature):
    #return "${WEKA_CMD} weka.clusterers.${SOURCES[1]} -t ${SOURCES[0]} -d ${TARGETS[0]} ${SOURCES[2].read()} 1> /dev/null"
    return "${WEKA_CMD} weka.clusterers.${SOURCES[1]} -t ${SOURCES[0]} -d ${TARGETS[0]} ${SOURCES[2].read()} 1> ${TARGETS[1]}"


def apply_clusterer(target, source, env, for_signature):
    return "${WEKA_CMD} weka.clusterers.${SOURCES[2]} -T ${SOURCES[1]} -l ${SOURCES[0]} -p last 1> ${TARGETS[0]}"


def map_classifier(target, source, env, for_signature):
    return "${WEKA_CMD} weka.classifiers.misc.InputMappedClassifier -L ${SOURCES[0]} -t ${SOURCES[1]} -T ${SOURCES[2]} -d ${TARGETS[0]} 1> ${TARGETS[1]}"


def compute_pairwise(target, source, env):
    fd = weka.Arff(filename=source[0].rstr())
    feats = fd.numeric()
    vals = {}
    comps = {}
    sigs = {}
    if "WEIGHTS" in env:
        wmat, wlabels, wfeatures = cPickle.load(meta_open(env["WEIGHTS"].rstr()))        
    for d in [x for x in fd.data if x[env.get("LABEL", "LABEL")] in env.get("FILTER", [])]:
        vals[d[env.get("LABEL", "LABEL")]] = d
    for v in set([frozenset([x, y]) for x in vals.keys() for y in vals.keys()]):
        #print v
        if len(v) == 1:
            a = b = list(v)[0]
        else:
            a, b = list(v)
        if a == b:
            res = env["EQUAL"]
        elif "WEIGHTS" in env and env.get("WTYPE", None) != "pairwise":
            indexA = wlabels.index(a)
            indexB = wlabels.index(b)
            weightsA = [x for x, y in zip(wmat[indexA], wmat[indexB])] #if x > 0.0 and y > 0.0]
            weightsB = [y for x, y in zip(wmat[indexA], wmat[indexB])] #if x > 0.0 and y > 0.0]
            weights = numpy.asarray([weightsA, weightsB]).mean(0)
            weights = weights / sum(weights)
            weights = dict([(x.split("_=_")[-1], y) for x, y in zip(wfeatures, weights)]) # if y > 0.0])
            #print weights
            res = env["FUNCTION"](vals[a], vals[b], distweights=weights)
        elif env.get("WTYPE", None) == "pairwise":
            index = wlabels.index(frozenset([a, b]))
            #indexB = wlabels.index(b)
            weights = [x for x in wmat[index]] #if x > 0.0 and y > 0.0]
            #weightsB = [y for x, y in zip(wmat[indexA], wmat[indexB])] #if x > 0.0 and y > 0.0]
            #weights = numpy.asarray([weightsA, weightsB]).mean(0)
            weights = weights / sum(weights)
            weights = dict([(x.split("_=_")[-1], y) for x, y in zip(wfeatures, weights)]) # if y > 0.0])
            #print weights
            res = env["FUNCTION"](vals[a], vals[b], distweights=weights)
        else:
            res = env["FUNCTION"](vals[a], vals[b])
        comps[v] = res
    newfd = weka.Arff()
    for k in vals.keys():
        features = {"LABEL" : k}
        for o in vals.keys():
            features["%s" % o] = comps[frozenset([k, o])]
        newfd.add_datum(features)
    newfd.save(meta_open(target[0].rstr(), "w"))
    return None


def compute_log_likelihood(target, source, env):
    fd = weka.Arff(filename=source[0].rstr())
    newfd = weka.Arff()
    freqs = {}
    totals = {}
    labels = set()
    items = set()    
    for d in fd.data:
        label = d[source[1].read()]
        labels.add(label)
        for k, v in [x for x in d.iteritems() if "_=_" in x[0]]:
            item = k.split("_=_")[1]
            if re.match("^\s*$", item):
                continue
            items.add(item)
            if item not in freqs:
                freqs[item] = {}
            freqs[item][label] = v
            totals[label] = totals.get(label, 0) + v
    cPickle.dump([freqs, totals], meta_open(target[0].rstr(), "w"))
    return None


def make_compatible(target, source, env):
    filt = weka.Arff(filename=source[1].rstr())
    print 1
    old = weka.Arff(filename=source[0].rstr())
    for k, v in filt.attributes.iteritems():
        if k not in old.attributes:
            old.attributes[k] = v
    for k in old.attributes.keys():
        if k not in filt.attributes.keys():
            del old.attributes[k]
    print 2
    #old.attributes = filt.attributes
    old.save(meta_open(target[0].rstr(), "w"), sparse=True)
    return None


def model_emitter(target, source, env):
    original = os.path.splitext(meta_basename(source[0].rstr()))[0]
    options = "_".join([x.read() for x in source[1:]])
    target[0] = "%s/%s_%s.model" % (env["DIR"], original, options)
    target.append("%s/%s_%s.out" % (env["DIR"], original, options))
    return target, source


def arff_emitter(target, source, env):
    if not any([target[0].rstr().endswith(x) for x in [".arff", ".arff.gz"]]):
        target[0] = "${BASE}/%s${UNIQUE}.arff" % (meta_basename(source[0].rstr()))
    env.Alias("arff", target)
    return target, source


def filter_emitter(target, source, env):
    target[0] = source[0].rstr().replace(".arff", "_%s_%s.arff" % (source[1].read().split(".")[-1], source[2].read()))
    return target, source

def apply_classifier_emitter(target, source, env):
    dname = os.path.splitext(os.path.basename(str(source[0])))[0]
    mname = os.path.splitext(os.path.basename(str(source[1])))[0]
    target = [os.path.join(env["DIR"], "%s_%s.arff.gz" % (mname, dname))]
    return target, source


def mapping_emitter(target, source, env):
    original = source[0].rstr()
    target[0] = source[0].rstr() + ".mapping"
    env.Alias("mappings", target)
    return target, source


def wordnet_filter_emitter(target, source, env):
    target[0] = source[0].rstr().replace(".arff", "_filtered.arff")
    return target, source


def subsample_emitter(target, source, env):
    args = source[-1].read()
    new_targets = []
    for i in range(1, args["COUNT"] + 1):
        new_targets.append(os.path.join(env["DIR"], "%s_%d.arff.gz" % (meta_basename(source[0].rstr()), i)))
    return new_targets, source


def split_arff_emitter(target, source, env):
    count = source[-1].read()
    new_targets = []
    for i in range(count):
        new_targets.append(target[0].rstr().replace(".arff", "%d.arff" % (i + 1)))
    return new_targets, source


def TOOLS_ADD(env):
    #env.Append(WEKAPATH=FindFile('weka.jar', ['/usr/share/weka/lib', '/usr/share/java']))
    #env.Append(WEKACLASSES=Popen(['jar', 'tf', env['WEKA_PATH'].rstr()], stdout=PIPE).stdout.read().replace('/', '.'))
    env.Append(BUILDERS = {
        'ArffFile' : Builder(action=arff_file, emitter=partial(generic_emitter, name="", ext="arff.gz")),        
        'SubsampleArffFile' : Builder(action=subsample_arff_file, emitter=partial(generic_emitter, name="subsamples", ext="arff.gz")),
        'ApplyFilter' : Builder(generator=apply_filter),
        'MapClassifier' : Builder(generator=map_classifier),
        #'TrainClassifier' : Builder(generator=train_classifier, emitter=model_emitter),
        'TrainClassifier' : Builder(generator=train_classifier, emitter=model_emitter),
        'TrainClusterer' : Builder(generator=train_clusterer, emitter=model_emitter),
        'ApplyClassifier' : Builder(generator=apply_classifier, emitter=apply_classifier_emitter),
        'ApplyThreshold' : Builder(action="python -m weka.arff ${SOURCES[0]} -o ${TARGET} -t ${SOURCES[1]} -a filter"),
        #'ApplyClassifier' : Builder(action=apply_classifier),
        'ApplyClusterer' : Builder(generator=apply_clusterer, emitter=mapping_emitter),
        'CSVtoARFF' : Builder(action = "${WEKA_CMD} weka.core.converters.CSVLoader $SOURCE > $TARGET"),
        'ARFFtoXRFF' : Builder(action = "${WEKA_CMD} weka.core.converters.XRFFSaver -i $SOURCE -compress -o $TARGET"),
        'Dedupe' : Builder(action = "/bin/sh bin/dedup_arff.sh $SOURCE $TARGET"),
        'CombineArffs' : Builder(action=combine_arffs),
        #'CombineArffs' : Builder(action="python -m weka.arff -o $TARGET ${SOURCES} ${OPTS}"),
        'FilterFeatures' : Builder(generator=filter_features),
        'SplitArff' : Builder(action=split_arff, emitter=split_arff_emitter),
        'NormalizeArff' : Builder(action=normalize_arff, emitter=partial(generic_emitter, name="normalized", ext="arff.gz")),
        'SumArff' : Builder(action=sum_arff, emitter=partial(generic_emitter, name="summed", ext="arff.gz")),
        'WordnetFilter' : Builder(action=wordnet_filter, emitter=wordnet_filter_emitter),
        'ReplaceMissing' : Builder(action=replace_missing),
        'RemoveInstances' : Builder(action=remove_instances),
        'RemoveFeatures' : Builder(action=remove_features),
        #'ComputePairwise' : Builder(action=compute_pairwise, emitter=partial(generic_emitter, name="pairwise", ext="arff.gz")),
        #'FilterCounts' : Builder(action=filter_counts, emitter=partial(generic_emitter, name="filtered", ext="arff.gz", remove="_filtered_threshold=\d+")),
        'FilterDistros' : Builder(action=filter_distros, emitter=partial(generic_emitter, name="filtered", ext="arff.gz")),
        'MakeCompatible' : Builder(action=make_compatible),
        })
