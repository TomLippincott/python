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
from common_tools import meta_open, meta_basename, log_likelihood as ct_log_likelihood, generic_emitter, unpack_numpy, pack_numpy, jensen_shannon_divergence
import numpy
import numpy.ma as ma
import logging
import os.path
from scipy import sparse, linalg, stats
import random
import cPickle as pickle
from functools import partial
import rpy2.robjects.numpy2ri
from scipy.cluster.hierarchy import linkage, dendrogram
import cPickle
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot, colors as mp_colors
from matplotlib.patches import Circle
#from scipy.sparse.linalg.eigen.arpack import svd
from scipy.linalg import svd
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.cluster.hierarchy import fclusterdata
from scipy.stats import entropy
from scipy.spatial.distance import pdist
import scipy
import rpy2.robjects.numpy2ri
from rpy2.robjects.conversion import ri2py



def weight_log_likelihood(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    wmat, wlabels, wfeatures = unpack_numpy(source[1].rstr())
    newmat = mat * wmat
    #pickle.dump((newmat, labels, features), meta_open(target[0].rstr(), "w"))
    return None


def log_likelihood(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    newmat = numpy.empty(mat.shape)
    obs_totals = mat.sum(1)
    feat_totals = mat.sum(0)
    total = feat_totals.sum()
    for col, feature in enumerate(features):
        for row, label in enumerate(labels):
            ll = ct_log_likelihood([
                (mat[row][col], obs_totals[row]),
                (feat_totals[col] - mat[row][col], total - obs_totals[row])
                ])
            newmat[row][col] = ll
    #pickle.dump((newmat, labels, features), meta_open(target[0].rstr(), "w"))
    return None


def pairwise_log_likelihood(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    combos = list(set([frozenset([x, y]) for x in labels for y in labels if x != y]))
    newmat = numpy.empty((len(combos), len(features)))
    obs_totals = mat.sum(1)
    for row, combo in enumerate(combos):
        for col, feature in enumerate(features):
            rowA = labels.index(list(combo)[0])
            rowB = labels.index(list(combo)[1])
            if mat[rowA][col] == 0 and mat[rowB][col] == 0:
                ll = 0.0
            else:
                ll = ct_log_likelihood([
                    (mat[rowA][col], obs_totals[rowA]),
                    (mat[rowB][col], obs_totals[rowB])
                    ])
            newmat[row][col] = ll
    #pickle.dump((newmat, combos, [x for x in features]), meta_open(target[0].rstr(), "w"))
    return None


def filter_counts(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    mat = mat.tocsc()
    totals = mat.sum(0)
    if "THRESHOLD" in args:
        keep = [i for i, x in enumerate(totals.T) if x >= args["THRESHOLD"]]
    elif "NUMBER" in args:
        keep = [j for y, j in sorted([(x, i) for i, x in enumerate(totals.T)], reverse=True)[0:args["NUMBER"]]]
        pass
    logging.info("keeping %d/%d features", len(keep), len(totals.T))
    mat = mat.tocsc()
    newmat = mat[:, keep]
    newfeatures = [features[i] for i in keep]
    pack_numpy(target[0].rstr(), data=newmat.tocoo(), labels=labels, features=newfeatures)
    return None


def filter_same_features(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    refmat, reflabels, reffeatures = unpack_numpy(source[1].rstr())
    mat = mat.tocsc()
    keep = [i for i, x in enumerate(features) if x in reffeatures]
    logging.info("keeping %d/%d features", len(keep), len(features))
    newmat = sparse.lil_matrix((len(labels), len(keep)))
    for i, col in enumerate([mat.getcol(i) for i in keep]):
        newmat[:, i] = col
    newfeatures = [features[i] for i in keep]
    pack_numpy(target[0].rstr(), data=newmat.tocoo(), labels=labels, features=newfeatures)
    return None


def merge_features(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    mat = mat.todense()
    merge = numpy.load(source[1].rstr())
    if "cluster" in merge:
        newmat = numpy.empty(shape=(len(labels), merge["centers"].shape[0] + 1))
        feature_map = dict([(k.get("_NAME"), v) for k, v in zip(merge["labels"], merge["cluster"])])
        clusters = [[] for i in range(merge["centers"].shape[0] + 1)]
        for i, f in enumerate(features):
            clusters[feature_map.get(f.get("_NAME"), -1)].append(i)
        for i in range(len(clusters)):
            newmat[:, i] = mat[:, clusters[i]].sum(1)
        newfeatures = [str(i) for i in range(len(clusters))]
    else:
        newmat = sparse.lil_matrix(numpy.empty(shape=(len(labels), len(merge))))
        features = [f.get("_NAME") for f in features]
        newfeatures = ["VCLUST_=_%s" % str(i) for i in range(len(merge))]
        for i, cluster in enumerate(merge):
            newmat[:, i] = mat[:, [features.index(f) for f in cluster if f in features]].sum(1)
    pack_numpy(target[0].rstr(), data=sparse.coo_matrix(newmat), labels=labels, features=newfeatures)
    return None


def feature_union(target, source, env):
    args = source[-1].read()
    dmat, dlabels, dfeatures = unpack_numpy(source[0].rstr())
    fmat, flabels, ffeatures = unpack_numpy(source[1].rstr())
    temp_dfeatures = [(re.match("^[A-Z]+-(.*)_(\=|\+)_.*$", x).group(1), i) for i, x in enumerate(dfeatures)]
    ffeatures = [x.split("_=_")[-1] for x in ffeatures]
    keep = [i for x, i in temp_dfeatures if x in ffeatures]
    newfeatures = sorted([(dfeatures[i], i) for i in keep])
    pack_numpy(target[0].rstr(), data=dmat[:, [i for n, i in newfeatures]], labels=dlabels, features=[n for n, i in newfeatures])
    return None


def compute_pairwise(target, source, env):
    args = source[-1].read()
    if "EQUAL" in env:
        equal = dict([(x, cPickle.load(open(y[0].rstr()))[0]) for x, y in env["EQUAL"].iteritems()])
    old = numpy.seterr(divide='ignore', invalid='ignore')
    mat, labels, features = unpack_numpy(source[0].rstr(), dense=True)
    # feature_names = [x.get("_NAME") for x in features]
    # label_names = [x.get("_NAME") for x in labels]
    # if env.get("DISTROS", True):
    #     distros = {}
    #     for i, feat in enumerate(feature_names):
    #         dist = feat.split("_=_")[0]
    #         if dist not in distros:
    #             distros[dist] = []
    #         distros[dist].append(i)
    #     logging.info("comparing %d distributions", len(distros))
    # else:
    #     distros = {"ALL" : range(len(features))}
    # for k in distros.keys():
    #     distros[k] = mat[:, distros[k]]
    # pairwise_distro_vals = numpy.empty(shape=(len(labels), len(labels), len(distros)))
    # print pairwise_distro_vals.shape
    # distro_order = []
    # for i, (name, distro) in enumerate(distros.iteritems()):
    #     distro_order.append(name.split("-")[-1])
    #     if i % 1000 == 0:        
    #         logging.info("processed distro #%d/%d", i, len(distros))
    # if "WEIGHTS" in args:
    #     ldata, llabels, lfeatures = unpack_numpy(args["WEIGHTS"])
    #     lfeatures = [x.get("_NAME").split("_=_")[-1] for x in lfeatures]
    #     indices = [i for i, x in enumerate(lfeatures) if x in distro_order]
    #     zeros = [i for i, x in enumerate(distro_order) if x not in lfeatures and i <= len(indices) ]
    #     ldata = ldata[:, indices]
    #     ldata = numpy.insert(ldata, zeros + [len(indices) for i in range(len(distro_order) - len(zeros) - len(indices))], 0.0, axis=1)
    #     weights = numpy.empty(shape=pairwise_distro_vals.shape)
    #     for r in range(weights.shape[0]):
    #         for c in range(weights.shape[1]):
    #             weights[r, c, :] = ldata[[r, c], :].sum(0)
    #     res = numpy.average(pairwise_distro_vals, axis=2, weights=weights)
    # else:
    #     res = numpy.average(pairwise_distro_vals, axis=2)
    # newmat = pairwise_distro_vals.sum(2) / pairwise_distro_vals.shape[2]
    # newmat = newmat[[i for i, x in enumerate(labels) if x["_NAME"] not in args.get("FILTER", [])], :][:, [i for i, x in enumerate(labels) if x["_NAME"] not in args.get("FILTER", [])]]
    labels = [x for x in labels if x["_NAME"] not in args.get("FILTER", [])]
    data = env["FUNCTION"](mat)
    pack_numpy(target[0].rstr(), data=data, labels=labels, features=labels)
    numpy.seterr(divide="warn", invalid="warn")
    return None


def normalize(target, source, env):
    old = numpy.seterr(divide='ignore', invalid='ignore')
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    feature_names = [x.get("_NAME") for x in features]
    feature_distros = [x.split("_=_")[0] for x in feature_names]
    distro_names = set(feature_distros)
    distros = dict([(i, []) for i in distro_names])
    for i, d in enumerate(feature_distros):
        distros[d].append(i)
    logging.info("found %d distribution(s)", len(distros))
    if len(distros) == 0:
        distros = {"ALL" : range(mat.shape[1])}
    C, R, V = [], [], []
    mat = mat.tolil()
    newmat = sparse.lil_matrix(numpy.zeros(shape=mat.shape))
    totals = numpy.empty(shape=(1, len(features)))
    for ri, row in [(i, mat.getrow(i).todense()) for i in range(mat.shape[0])]:
        logging.info("processing object %d/%d", ri, len(labels))
        for di, (name, indices) in enumerate(distros.iteritems()):
            totals[0, indices] = row[:, indices].sum()
        newmat[ri, :] = row / totals
    pack_numpy(target[0].rstr(), data=newmat.tocoo(), labels=labels, features=features)
    numpy.seterr(divide="warn", invalid="warn")
    return None


def numpy_file(target, source, env):
    args = source[-1].read()
    rows = []
    for text, label in sum([env["ARGS"]["SPLIT"](fd.rstr()) for fd in source[0:-1]], []):
        feats = {}
        for extractor in [env["ARGS"]["EXTRACT"]]:
            feats.update(extractor(text))
        rows.append((label.items(), feats))

    features = sorted(set(sum([x[1].keys() for x in rows], [])))
    labels = []
    if args.get("SPARSE", False):
        logging.info("creating sparse file")
        R, C, V = [], [], []
        for row, (name, vals) in enumerate(rows):
            labels.append(name) #"_".join([name[x] for x in source[-1].read()["LABEL"]]))
            new_vals = [(row, col, vals[feature]) for col, feature in enumerate(features) if feature in vals]
            R += [x[0] for x in new_vals]
            C += [x[1] for x in new_vals]
            V += [x[2] for x in new_vals]
        mat = sparse.coo_matrix((V, (R, C)), shape=(len(labels), len(features)))
    else:
        mat = numpy.zeros((len(rows), len(features)))
        for row, (name, vals) in enumerate(rows):
            labels.append(name) #"_".join([name[x] for x in source[-1].read()["LABEL"]]))
            new_vals = [(row, col, vals[feature]) for col, feature in enumerate(features) if feature in vals]
            for row, col, val in new_vals:
                mat[row, col] = val
    logging.info("created %dx%d data file", len(labels), len(features))
    pack_numpy(target[0].rstr(), data=mat, labels=labels, features=features)
    return None


def subsample_numpy_file(target, source, env):
    args = source[-1].read()
    labels = ["sample %d" % x for x in range(1, args["COUNT"] + 1)]
    items = []
    features = [] #set()
    item = {}
    for i, m in enumerate(re.finditer("\n\s*\n(.*?)(^<c>.*?)$", meta_open(source[0].rstr()).read(), re.M | re.S)):
        s = m.group(1) + m.group(2)
        for extractor in env["FEATURE_EXTRACTORS"]:
            feats = extractor(s)
            features += feats.keys() #features.union(feats.keys())
            for k, v in feats.iteritems():
                item[k] = item.get(k, 0.0) + v
        if i % args["WINDOW"] == 0 and i != 0:
            logging.info("made item #%d", len(items))
            items.append(item)
            item = {}
        if i % 100000 == 0 and i != 0:
            features = list(set(features))
    features = dict([(y, i) for i, y in enumerate(sorted(set(features)))])
    assignments = [0 for i in range(len(items) - args.get("ITEMS", 1))] + \
                  [1 for i in range(args.get("ITEMS", 1))]
    vals = []
    for r in range(args["COUNT"]):
        logging.info("building random sample #%d", (r + 1))
        random.shuffle(assignments)
        ritems = [items[i] for i, x in enumerate(assignments) if x == 1]
        #curfeats = set(sum([items[i].keys() for i, y in enumerate(assignments) if y == 1], []))
        curfeats = set(sum([i.keys() for i in ritems], []))
        logging.info("%d features", len(curfeats))
        #sample = dict([(f, sum([items[i].get(f, 0) for i, y in enumerate(assignments) if y == 1])) for f in curfeats])
        sample = dict([(f, sum([i.get(f, 0) for i in ritems])) for f in curfeats])
        vals += [(sample[f], r, features[f]) for f in curfeats]
    V, R, C = zip(*vals)
    mat = sparse.coo_matrix((V, (R, C)), shape=(len(labels), len(features)))
    logging.info("created %dx%d data file", len(labels), len(features))
    pack_numpy(target[0].rstr(), data=mat, labels=labels, features=sorted(features.keys()))
    return None


def sum_over_labels(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    labels = [tuple(x.iteritems()) for x in labels]
    all_labels = sorted(set(labels))
    label_mapping = dict([(b, a) for a, b in enumerate(all_labels)])
    total = len(sparse.find(mat)[0])
    if args.get("SINGLE", None):
        newlabels = [args.get("SINGLE", "?")]
        newmat = mat.sum(0)
    else:
        R, C, V = [], [], []
        for row, col, val in zip(*sparse.find(mat)):
            R.append(label_mapping[labels[row]])
            C.append(col)
            V.append(val)
    newmat = sparse.coo_matrix((V, (R, C)), shape=(len(all_labels), len(features)))
    pack_numpy(target[0].rstr(), data=newmat, labels=[dict(x) for x in all_labels], features=features)
    return None


def filter_features(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    for targ, filt in zip(target, args["FILTERS"]):
        indices = [i for i, x in enumerate(features) if x.startswith(filt)]
        newfeatures = [x for x in features if x.startswith(filt)]
        pickle.dump((mat[:, indices], labels, newfeatures), meta_open(targ.rstr(), "w"))
    return None


def remove_zero_features(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr(), dense=True)
    mat = numpy.asarray(mat)
    indices = [i for i, x in enumerate(mat.sum(0)) if x > 0]
    mat = mat[:, indices]
    features = [features[i] for i in indices]
    pack_numpy(target[0].rstr(), data=mat, labels=labels, features=features)
    return None


def lda_file(target, source, env):
    args = source[-1].read()
    tei = et.parse(meta_open(source[0].rstr()))
    docs = [(names, ["".join([y.text for y in x.getiterator() if y.text]) for x in doc.getiterator("fs") if x.get("type", None) == "token"]) for doc, names in env["ARGS"]["SPLIT"](tei)]
    vocabulary = sorted(set(sum([y for x, y in docs], [])))
    documents = []
    for names, doc in docs:
        documents.append([[vocabulary.index(x) for x in doc],
                          [doc.count(x) for x in doc]])
    pickle.dump((documents, vocabulary, [x[0] for x in docs]), meta_open(target[0].rstr(), "w"))
    return None


def compute_svd(target, source, env):
    args = source[-1].read()
    mat, docs, words = unpack_numpy(source[0].rstr(), dense=True)
    #mat = sparse.lil_matrix(mat)
    #mat = mat.asfptype()
    L, S, R = svd(mat, args.get("DIMENSIONS", 300))
    pack_numpy(target[0].rstr(), data=L, labels=docs, features=[{"SVD" : str(i)} for i in range(1, L.shape[1] + 1)])
    pack_numpy(target[1].rstr(), data=R.T, labels=words, features=[{"SVD" : str(i)} for i in range(1, R.shape[0] + 1)])
    return None


def project_svd(target, source, env):
    args = source[-1].read()
    data, labels, features = unpack_numpy(source[0].rstr(), dense=False)
    proj, plabels, newfeatures = unpack_numpy(source[1].rstr(), dense=False)
    newdata = data * proj
    pack_numpy(target[0].rstr(), data=newdata, labels=labels, features=newfeatures)
    return None


def compute_log_likelihoods(target, source, env):
    args = source[-1].read()
    old = numpy.seterr(divide='ignore', invalid='ignore')
    mat, labels, features = unpack_numpy(source[0].rstr(), dense=False)
    mat = numpy.asarray(mat.todense())
    totals = mat.sum(1)
    logging.info("created totals matrix of shape %s", totals.shape)
    newmat = numpy.empty(shape=mat.shape)
    for i in range(mat.shape[1]):
        newmat[:, i] = ct_log_likelihood(mat[:, i], totals)
    pack_numpy(target[0].rstr(), data=newmat, labels=labels, features=features)
    numpy.seterr(divide="warn", invalid="warn")
    return None


def compute_lsa(target, source, env):
    args = source[-1].read()
    logging.info("starting...")
    mat, labels, features = unpack_numpy(source[0].rstr())
    dimensions = args.get("DIMENSIONS", mat.shape[1])
    matrix = mat[:, range(dimensions)]
    items = labels
    K = args.get("CLUSTERS", len(items) / 5)
    if args.get("METHOD", None) == "kmeans":
        logging.info("extracting %d clusters from %d singular values in term matrix of order %s", K, dimensions, matrix.shape)
        whitened = whiten(matrix)
        book = numpy.array((whitened[0], whitened[2]))
        codebook, distortion = kmeans(whitened, K)
        pickle.dump((items, codebook, distortion, whitened), meta_open(target[0].rstr(), "w"))
    elif args.get("METHOD", None) == "hierarchical":
        pass
    return None


def plot_lsa(target, source, env):
    args = source[-1].read()
    words, codebook, distortion, whitened = cPickle.load(meta_open(source[0].rstr()))
    words = [x.get("_NAME") for x in words]
    assignments = zip(vq(whitened, codebook)[0], words)
    clusters = dict([(x, [y[1] for y in assignments if y[0] == x]) for x in set([y[0] for y in assignments])])
    fd = meta_open(target[0].rstr(), "w")
    for k, v in clusters.iteritems():
        fd.write("%s\n\t%s\n\n" % (k, "\n\t".join(v).encode("utf-8")))
    return None


def combine(target, source, env):
    files = [unpack_numpy(x.rstr(), dense=False) for x in source]
    all_labels = []
    all_features = [dict(y) for y in sorted(set(sum([[tuple(i.iteritems()) for i in x[2]] for x in files], [])))]
    feature_mapping = dict([(tuple(b.iteritems()), a) for a, b in enumerate(all_features)])
    R, C, V = [], [], []
    label_offset = 0
    for mat, labels, features in files:
        all_labels += [tuple(x.iteritems()) for x in labels]
        features_tuples = [tuple(x.iteritems()) for x in features]
        for row, col, val in zip(*sparse.find(mat)):
            R.append(row + label_offset)
            C.append(feature_mapping[features_tuples[col]])
            V.append(val)
        label_offset += mat.shape[0]
    newmat = sparse.coo_matrix((V, (R, C)), shape=(len(all_labels), len(all_features)))
    pack_numpy(target[0].rstr(), data=newmat, labels=[dict(x) for x in all_labels], features=all_features)
    return None


def split_numpy(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    indices = {}
    for pat, fname in zip(args["PATTERNS"], target):
        indices = [i for i, x in enumerate(features) if x.get("_NAME", "_NAME").startswith("%s_" % pat) or x.get("_NAME", "_NAME").startswith("%s-" % pat)]
        pack_numpy(fname.rstr(), data=sparse.coo_matrix(mat.todense()[:, indices]), labels=labels, features=[features[i] for i in indices])
    return None


def compare_distros(target, source, env):
    old = numpy.seterr(divide='ignore', invalid='ignore')
    mat, labels, features = unpack_numpy(source[0].rstr())
    labels = [x.get("LABEL_filename") for x in labels]
    features = [x.get("_NAME") for x in features]
    over = set([x.split("_=_")[-1] for x in features])
    mat = mat.todense()
    allverbs = [re.match("^[A-Z]+-(.*)_._.*$", x).group(1) for x in features]
    verbs = dict([(x, {"features" : [], "data" : {}}) for x in set(allverbs)])
    for i, v in enumerate(allverbs):
        ov = features[i].split("_=_")[-1]
        verbs[v]["data"][ov] = mat[:, i].T.tolist()[0]
    logging.info("comparing %s distributions", len(verbs))
    results = []
    distros = []
    verblist = []
    for verb, vals in [x for x in verbs.iteritems() if len(x[1]) > 1]: 
        distros.append(numpy.asarray([vals["data"].get(x, [0 for i in range(len(labels))]) for x in over]).T)
        #try:
        #    numpy.asarray([vals["data"].get(x, [0 for i in range(len(labels))]) for x in over])
        #except:
        #    print [vals["data"].get(x, [0 for i in range(len(labels))]) for x in over]
        #    sys.exit()
        jsd = jensen_shannon_divergence(distros[-1], counts=True)
        results.append(jsd)
        verblist.append(verb)
    distros = numpy.asarray(distros)
    logging.info("built distro matrix of shape %s", distros.shape)
    numpy.seterr(divide="warn", invalid="warn")
    numpy.savez(target[0].rstr(),
                results=results,
                distros=distros, #[sparse.lil_matrix(distros[:, :, i]) for i in range(distros.shape[-1])],
                verbs=verblist,
                labels=labels,
                features=list(over))
    return None


def update_numpy_file(target, source, env):
    data, labels, features = unpack_numpy(source[0].rstr())
    data = sparse.coo_matrix(data)
    pack_numpy(target[0].rstr(), data=data, labels=labels, features=features)
    return None


def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        'WeightLogLikelihood' : Builder(action=weight_log_likelihood, emitter=partial(generic_emitter, targets=[("weightloglikelihood", "npz")])),
        'LogLikelihood' : Builder(action=log_likelihood, emitter=partial(generic_emitter, targets=[("loglikelihood", "npz")])),
        'Normalize' : Builder(action=normalize, emitter=partial(generic_emitter, targets=[("normalized", "npz")])),
        'PWLogLikelihood' : Builder(action=pairwise_log_likelihood, emitter=partial(generic_emitter, targets=[("pwloglikelihood", "npz")])),
        'FilterCounts' : Builder(action=filter_counts, emitter=partial(generic_emitter, targets=[("filtered", "npz")])),
        'ComputePairwise' : Builder(action=compute_pairwise, emitter=partial(generic_emitter, targets=[("pairwise", "npz")])),
        'FeatureUnion' : Builder(action = feature_union, emitter=partial(generic_emitter, targets=[("unionfilter", "npz")])),
        'NumpyFile' : Builder(action = numpy_file, emitter = partial(generic_emitter, targets=[("", "npz")])),
        'Update' : Builder(action = update_numpy_file, emitter = partial(generic_emitter, targets=[("", "npz")])),
        'Combine' : Builder(action = combine),
        'Split' : Builder(action = split_numpy),
        'CompareDistros' : Builder(action = compare_distros, emitter=partial(generic_emitter, targets=[("comparison", "npz")])),
        'FilterFeatures' : Builder(action = filter_features),
        'SubsampleNumpyFile' : Builder(action = subsample_numpy_file, emitter = partial(generic_emitter, targets=[("numpy", "npz")])),
        'SumOverLabels' : Builder(action = sum_over_labels, emitter = partial(generic_emitter, targets=[("summed", "npz")])),
        'LDAFile' : Builder(action = lda_file, emitter = partial(generic_emitter, targets=[("lda_documents", "npz")])),
        'SVD' : Builder(action = compute_svd, emitter = partial(generic_emitter, targets=[("rows", "npz"), ("columns", "npz")])),
        'ProjectSVD' : Builder(action = project_svd, emitter = partial(generic_emitter, targets=[("projected", "npz")])),
        'LSA' : Builder(action = compute_lsa, emitter = partial(generic_emitter, targets=[("lsa", "npz")])),
        'LogLikelihoods' : Builder(action=compute_log_likelihoods, emitter=partial(generic_emitter, targets=[("loglikelihoods", "npz")])),
        'PlotLSA' : Builder(action = plot_lsa, emitter = partial(generic_emitter, targets=[("plot", "txt.gz")])),
        'MergeFeatures' : Builder(action = merge_features, emitter = partial(generic_emitter, targets=[("merged", "npz")])),
        'SameFeatures' : Builder(action = filter_same_features, emitter = partial(generic_emitter, targets=[("", "npz")])),
        'RemoveZeros' : Builder(action = remove_zero_features, emitter = partial(generic_emitter, targets=[("", "npz")])),
        })
