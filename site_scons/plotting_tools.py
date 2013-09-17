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
from matplotlib import rcdefaults, pyplot, colors as mp_colors
from matplotlib.patches import Circle
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.cluster.hierarchy import fclusterdata
from scipy.stats import entropy
from scipy.spatial.distance import pdist
import scipy
import rpy2.robjects.numpy2ri
from rpy2.robjects.conversion import ri2py
from rpy2.robjects.packages import importr


r_clv = importr("clv")
r_clustersim = importr("clusterSim")


colors = "bgrcmykwbgrcmykw"
shapes = ['+', 'o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', '8']


def plot_dendrogram(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    roworder = [i for i, f in enumerate(sorted(labels)) if "LABEL_FILTER" not in env or f in env["LABEL_FILTER"]]
    colorder = [i for i, f in enumerate(sorted(features)) if "FEATURE_FILTER" not in env or f in env["FEATURE_FILTER"]]
    mat = mat[roworder][:, colorder][0:50]
    if args.get("DIRECT", False):
        Y = squareform(mat, checks=False)
    else:
        Y = pdist(mat, "cosine")
    Z = abs(linkage(Y, source[-1].read().get("LINKAGE", 'average')))
    dendrogram(Z, labels=labels, orientation="right", color_threshold=-1, leaf_font_size="xx-small")
    pyplot.savefig(target[0].rstr(), dpi=100, bbox_inches="tight")
    pyplot.cla()
    pyplot.clf()
    return None


def plot_scatter(target, source, env):
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    X = mat[:, 0]
    Y = mat[:, 1]
    L = labels
    shapes = ['+', 'o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', '8']
    clustering = {}
    if env.get("CLUSTERS", None):
        for m in re.finditer("^(\d+) (\d+) \(\'?(.*?)\'?\)$", meta_open(env["CLUSTERS"].rstr()).read(), re.M):
            clustering[m.group(3)] = int(m.group(2))
    for x, y, label in zip(X, Y, L):
        try:
            point = pyplot.scatter([x], [y], s=10, c="black", marker=shapes[clustering.get(label, 0)])
        except:
            print x
            sys.exit()
        a = pyplot.annotate(label.title(), (x, y), xytext=(15, 0), textcoords="offset points", fontsize=10, arrowprops=dict(arrowstyle="-"))
    pyplot.xticks([], [])
    pyplot.yticks([], [])
    comps = {}
    xtext = features[0]
    pyplot.xlabel(xtext.rstrip().rstrip(","))
    ytext = features[1]
    pyplot.ylabel(ytext)
    pyplot.savefig(target[0].rstr(), dpi=100, bbox_inches="tight")
    pyplot.cla()
    return None


def plot_clustering_agreement(target, source, env):
    args = source[-1].read()
    counts = sorted([(int(x.rstr().split("/")[-2]),
                      os.path.splitext(x.rstr().split("/")[-1])[0],
                      ri2py(numpy.load(x.rstr())).rx2("cluster")) for x in source[0:-1]])
    features = {}
    gaps = []
    for feat in set([x[1] for x in counts]):
        features[feat] = [x[2] for x in counts if x[1] == feat]
    print features.keys()
    data = numpy.empty(shape=(len(features), len(features.values()[0]), len(features)))
    print data.shape
    for r, featA in enumerate(features.keys()):
        for c in range(data.shape[1]):
            for l, featB in enumerate(features.keys()):
                vals = r_clv.std_ext(features[featA][c], features[featB][c])
                data[r, c, l] = getattr(r_clv, "clv_%s" % args["TYPE"])(vals)[0]
    pyplot.figure(figsize=(15 * 2, 7 * len(features) + 1))
    for i, (nameA, feat) in enumerate(zip(features.keys(), data)):
        pyplot.subplot(len(data) / 2 + 1, 2, i)
        for j, (nameB, vals) in enumerate(zip(features.keys(), feat.T)):
            if nameB != nameA:
                if j > 6:
                    pyplot.plot(vals, label=nameB, ls="--")
                else:
                    pyplot.plot(vals, label=nameB)
        pyplot.legend(prop={"size" : 7}, numpoints=2)
        pyplot.title(nameA)
        pyplot.xticks(range(0, data.shape[1], 2), [i + 2 for i in range(0, data.shape[1], 2)])
        pyplot.xlabel("Clusters")
        pyplot.ylabel(args["TYPE"])
    pyplot.savefig(target[0].rstr(), bbox_inches="tight")
    pyplot.cla()
    return None


def plot_clusterings(target, source, env):
    """
    Plot items with clustering from first file, using 2-d coordinates from second file.
    The functions GET_COORDS and GET_CLUSTS specify operations to turn each file object
    into a mapping from item name to coordinate tuple or cluster number, respectively.
    """
    pyplot.rcdefaults()
    pyplot.figure(figsize=(10, 10))
    args = source[-1].read()
    # rcdefaults()
    clusts = dict(ri2py(eval("lambda x : %s" % args.get("GET_CLUSTERS", "x"))(numpy.load(source[0].rstr()))).rx2("cluster").iteritems())
    coords = eval("lambda x : %s" % args.get("GET_COORDS", "x"))(numpy.load(source[1].rstr()))
    labels = coords.keys()
    #if args.get("NORMALIZE", False):
    #    for i in [0, 1]:
    #        ttcoords[:, i] = (ttcoords[:, i] - ttcoords[:, i].min()) / numpy.max(ttcoords[:, i] - ttcoords[:, i].min())
    

    [pyplot.scatter(coords[l][0], coords[l][1], label=l, s=64, marker=shapes[clusts[l]], color=colors[clusts[l]]) for i, l in enumerate(labels)]
    ymin, ymax = pyplot.ylim()
    inc = (ymax - ymin) / 40.0

    [pyplot.text(coords[l][0], coords[l][1] + inc, l, fontsize=12, ha="center") for i, l in enumerate(labels)]
    pyplot.xticks([], [])
    pyplot.yticks([], [])
    pyplot.xlabel("First principal component")
    pyplot.ylabel("Second principal component")
    pyplot.savefig(target[0].rstr(), bbox_inches="tight")
    pyplot.cla()
    return None



def plot_spread(target, source, env):
    args = source[-1].read()
    raw_data = []
    labels = []
    for name, f in zip(args["SUBJECTS"], source):
        mat, l, features = unpack_numpy(f.rstr())
        raw_data.append(sum([[col for c, col in enumerate(row) if c != r] for r, row in enumerate(mat)], []))
        labels.append(name)
    raw_data = numpy.asarray(raw_data)
    #print data.min(), data.max()
    number = 50.0
    rmin, rmax = raw_data.min(), raw_data.max()
    inc = (rmax - rmin) / number
    data = numpy.empty(shape=(len(labels), number))
    for r, label in enumerate(labels):
        data[r, :] = [len([i for i in raw_data[r] if rmin + c * inc < i < rmin + (c + 1) * inc]) for c in range(int(number))]
    #print data.shape
    #pyplot.figure(figsize=(7, 7 * 6))
    for i, ls in enumerate(["-", "--", "-.", ":"]):
        #pyplot.subplot(6, 1, i)
        for name, datum in [x for x in zip(args["SUBJECTS"], data)][i * len(labels) / 4 : (i + 1) * len(labels) / 4]:
            pyplot.plot(datum, label=name, ls=ls)
    inc = (rmax - rmin) / 4.0
    pyplot.xticks([x * (50 / 4) for x in range(5)], ["%.5f" % (rmin + inc * x) for x in range(5)])
    pyplot.xlabel("Jensen-Shannon Divergence")
    pyplot.ylabel("Number of intra-subdomain random sample pairs")
    pyplot.legend(prop={"size" : 6}, ncol=3)
    pyplot.savefig(target[0].rstr(), dpi=100, bbox_inches="tight")

    pyplot.cla()
    return None


def plot_heatmap(target, source, env):
    """
    For an Arff file with N instances of M numeric features, plot an NxM matrix of cells
    where a cell's intensity is determined by the feature's value in that instance.
    """
    args = source[-1].read()
    mat, labels, features = unpack_numpy(source[0].rstr())
    labels = [x.get(args.get("LABEL_NAME", "_NAME")) for x in labels]
    features = [x.get(args.get("FEATURE_NAME", "_NAME")) for x in features]
    samples = {}
    for sub, fname in args.get("sampled", {}).iteritems():
        smat, slabels, sfeatures = unpack_numpy(fname)
        samples[sub] = sum([[smat[x, y] for x in range(smat.shape[0]) if x != y] for y in range(smat.shape[1])], [])
    roworder = [i for f, i in sorted([(f, i) for i, f in enumerate(labels)]) if "LABEL_FILTER" not in env or f in env["LABEL_FILTER"]]
    colorder = [i for f, i in sorted([(f, i) for i, f in enumerate(features)]) if "FEATURE_FILTER" not in env or f in env["FEATURE_FILTER"]]
    fig = pyplot.figure(figsize=(20, 20))
    C = []
    for row, datum in enumerate([mat[i] for i in roworder]):
        temp = []
        for col, val in enumerate([datum[i] for i in colorder]):
            if row == col and "sampled" in args:
                sub = sorted(samples)[row]
                temp.append(env.get("TRANSFORM", lambda x : x)(sum(samples[sub]) / float(len(samples[sub]))))
            elif row == col and "reflect" in args:
                temp.append(0.0)
            elif row - col > 0:
                temp.append(env.get("TRANSFORM", lambda x : x)(val))
            elif "sampled" in args:
                lab1 = sorted(labels)[row]
                lab2 = sorted(labels)[col]
                sub1 = float(len([i for i in samples[lab1] if i < val])) / len(samples[lab1])
                sub2 = float(len([i for i in samples[lab2] if i < val])) / len(samples[lab2])
                temp.append((sub1 + sub2) / 2.0)
            elif "reflect" in args:
                temp.append(env.get("TRANSFORM", lambda x : x)(val))
            else:
                temp.append(1.0)
        C.append(temp)
    C = numpy.asarray(C)
    pyplot.gray()
    top = numpy.triu(C, 1)
    bottom = numpy.tril(C)
    rbottom = numpy.tril(C)
    if "sampled" in args:
        bottom = bottom / bottom.max()
    C_scaled = top + bottom
    pyplot.pcolor(C_scaled, vmin=0.0, vmax=1.0)

#    pyplot.title("Lighter indicates higher JSD/significance\nDiagonal shows homogeny from random samples\n\n\nJensen-Shannon Divergence (not comparable across features)\nValues range from %f (black) to %f (white)" % (min([rbottom[x[0], x[1]] for x in zip(*rbottom.nonzero())]), rbottom.max()))
    if "sampled" in args:
        pyplot.text(C.shape[1] + .1, C.shape[0] / 2, "Significance score (comparable across features), values range from 0%% (black) to 100%% (white)", va="center", rotation=270, fontsize=25)
    A = numpy.asarray(C)
    fs = 250 / len(colorder)
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            if x < y or "sampled" in args or "reflect" in args:
                if A[y, x] < .5:
                    c = "white"
                else:
                    c = "black"
                pyplot.text(x + .05, y + .25, "%.1e" % (A[y, x]), color=c, fontsize=fs)
    if "sampled" not in args and "reflect" not in args:
        pyplot.text(A.shape[0] / 1.5, A.shape[1] / 2.5, "N/A", fontsize=50)
    fs = 600 / len(colorder)
    pyplot.xticks([x + .7 for x in range(len(colorder))], sorted([labels[x].title() for x in colorder]), rotation=45, ha="right", fontsize=fs)
    pyplot.yticks([x + .5 for x in range(len(roworder))], sorted([labels[x].title() for x in roworder]), fontsize=fs)
    pyplot.title("Jensen-Shannon Divergence", fontsize=25)
    #pyplot.title(args.get("title", ""), fontsize=20)
    #pyplot.text(40, 19, "Significance score, ranges from 0%% (black) to 100%% (white)", rotation=270, ha="center", va="center")
    pyplot.axes().autoscale_view()
    pyplot.savefig(target[0].rstr(), dpi=100, bbox_inches="tight")
    pyplot.cla()
    pyplot.clf()
    return None



def plot_distros(target, source, env):
    """
    Plots a matrix of distributions (rows) over features (columns)

    N per figure
    """
    N = 5
    args = source[-1].read()
    distros, labels, features = unpack_numpy(source[0].rstr(), dense=True)
    keep = [i for i, x in enumerate(distros) if x.sum() > 10]
    distros = distros[keep, :]
    labels = [labels[i] for i in keep]

    distros = numpy.transpose(distros.T / distros.sum(1))
    #print distros[0]
    features = [x["_NAME"] for x in features]
    labels = [x["_NAME"] for x in labels]
    num_figs = 1 + int(len(labels) / N)
    pyplot.figure(figsize=(7, 7 * num_figs))
    allvals = distros.sum(0) / distros.sum()
    order = [x[1] for x in sorted([(y, i) for i, y in enumerate(allvals) if y > 0], reverse=True)]
    for i in range(num_figs):
        start = N * i
        end = min(len(labels), N * (i + 1))
        pyplot.subplot(num_figs, 1, i + 1)
        if args.get("GRAYSCALE"):
            pyplot.bar(left=[i for i in range(len(order))], height=[allvals[i] for i in order], label="Average", width=1.0 / num_figs)
            for j, (name, vals) in enumerate([x for x in zip(labels[start : end], distros[start : end])]): # if x[1].sum() > 0]:
                total = vals.sum()
                vals = vals / vals.sum()
                pyplot.bar(left=[i + j * (1.0 / num_figs)  for i in range(len(order))], height=[vals[i] for i in order], label="%s" % name, width=1.0 / num_figs)
        else:
            pyplot.plot([allvals[i] for i in order], label="Average", lw=2)
            for name, vals in [x for x in zip(labels[start : end], distros[start : end])]: # if x[1].sum() > 0]:
                total = vals.sum()
                vals = vals / vals.sum()
                pyplot.plot([vals[i] for i in order], label="%s" % (name))
                pyplot.legend(prop={"size" : 8})
                pyplot.xticks(range(len(order)), [features[i].strip() for i in order], rotation=45, fontsize=6, ha="right")
                pyplot.gca().grid(color='lightgrey', linestyle='-', linewidth=1)
        pyplot.legend(prop={"size" : 8})
        pyplot.xticks(range(len(order)), [features[i].strip() for i in order], rotation=45, fontsize=6, ha="right")
        pyplot.gca().grid(color='lightgrey', linestyle='-', linewidth=1)
    pyplot.savefig(target[0].rstr(), bbox_inches="tight")
    
    pyplot.cla()
    return None



def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        'Dendrogram' : Builder(action = plot_dendrogram, emitter=partial(generic_emitter, targets=[("dendrogram", "png")])),
        'HeatMap' : Builder(action = plot_heatmap, emitter=partial(generic_emitter, targets=[("heatmap", "png")])),
        'Scatter' : Builder(action = plot_scatter, emitter=partial(generic_emitter, targets=[("scatter", "png")])),
        'PlotDistros' : Builder(action = plot_distros, emitter=partial(generic_emitter, targets=[("", "png")])),
        'PlotSpread' : Builder(action = plot_spread, emitter = partial(generic_emitter, targets=[("plot", "png")])),
        'PlotClusterings' : Builder(action = plot_clusterings, emitter = partial(generic_emitter, targets=[("pca", "png")])),
        'PlotClusteringAgreement' : Builder(action = plot_clustering_agreement, emitter = partial(generic_emitter, targets=[("", "png")])),
        })
