from SCons.Builder import Builder
from common_tools import meta_open, generic_emitter, unpack_numpy
import logging
import cPickle as pickle
from functools import partial
from rpy2.robjects import r, FloatVector, Formula
from rpy2.robjects.conversion import ri2py
from rpy2.rlike.container import TaggedList
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import numpy
from scipy import sparse
import sys


temp_out = sys.stdout
sys.stdout = open("/dev/null", "w")

modules = {}
for x in ["stats", "grDevices", "base", "kernlab", "lda", "ggplot2", "clusterSim", "cluster", "mclust", "clustvarsel", "clue"]:
    try:
        modules[x] = importr(x)
    except:
        print "failed to load R module: %s" % (x)

sys.stdout = temp_out


def npz_to_df(fname, rzero=True, transform=None, labelname="_NAME", featurename="_NAME", keep_labels=None, keep_features=None, transpose=False):
    mat, labels, features = unpack_numpy(fname)
    if sparse.issparse(mat):
        mat = numpy.asarray(mat.todense())
    if transform:
        mat = transform(mat)
    if keep_labels:
        indices = [i for i, x in enumerate(labels) if x.get(labelname) in keep_labels]
        mat = mat[indices, :]
        labels = [labels[i] for i in indices]
    if keep_features:
        indices = [i for i, x in enumerate(features) if x[featurename] in keep_features]
        mat = mat[indices, :]
        features = [features[i] for i in indices]
    if transpose:
        mat = r["as.data.frame"](mat.T)
        mat.colnames = [x.get(labelname, i) for i, x in enumerate(labels)]
        mat.rownames = [x.get(featurename, i) for i, x in enumerate(features)]
    else:
        mat = r["as.data.frame"](mat)
        mat.rownames = [x.get(labelname, i) for i, x in enumerate(labels)]
        mat.colnames = [x.get(featurename, i) for i, x in enumerate(features)]
    return mat


def df_to_npz(fname, result):
    items = dict([(x, result.rx2(x)) for x in result.names])
    numpy.savez(fname, **items)


def apply_r(target, source, env):
    """
    Calls FUNCTION in MODULE on data frame loaded from first file, with KW arguments    
    """
    args = source[-1].read()
    df = npz_to_df(source[0].rstr(),
                   keep_labels=args.get("LABELS", None),
                   keep_features=args.get("FEATURES", None),
                   transpose=args.get("TRANSPOSE", False))
    df = eval("lambda x : %s" % args.get("GET_DF", "x"))(df)
    if args.get("DF", False):
        r_args = args["KW"]
        r_args[args["DF"]] = df
        result = getattr(modules[args["MODULE"]], args["FUNCTION"])(Formula("~."), **r_args)
    else:
        result = getattr(modules[args["MODULE"]], args["FUNCTION"])(df, **args["KW"])
    pickle.dump(result, meta_open(target[0].rstr(), "w"))
    return None


def ksvm(target, source, env):
    """
    Calls FUNCTION in MODULE on data frame loaded from first file, with KW arguments    
    """
    args = source[-1].read()
    #df = npz_to_df(source[0].rstr(),
    #               keep_labels=args.get("LABELS", None),
    #               keep_features=args.get("FEATURES", None),
    #               transpose=args.get("TRANSPOSE", False))
    #df = eval("lambda x : %s" % args.get("GET_DF", "x"))(df)
    #print df
    #sys.exit()

    #if args.get("DF", False):
    #    r_args = args["KW"]
    #    r_args[args["DF"]] = df
    #    result = getattr(modules[args["MODULE"]], args["FUNCTION"])(Formula("~."), **r_args)
    #else:
    #    result = getattr(modules[args["MODULE"]], args["FUNCTION"])(df, **args["KW"])
    #pickle.dump(result, meta_open(target[0].rstr(), "w"))
    #return None


def plot_cluster(target, source, env):
    """
    """
    args = source[-1].read()
    mat = npz_to_df(source[0].rstr(), dense=True)    
    clus = modules["cluster"].pam(mat, 10)
    modules["grDevices"].png(target[0].rstr())
    modules["cluster"].clusplot(mat, clus.rx2("clustering"))
    modules["grDevices"].dev_off()    
    return None


def plot_heat(target, source, env):
    """
    """
    mat = npz_to_df(source[0].rstr())
    modules["grDevices"].png(target[0].rstr())
    colors = []
    m = max(numpy.array(mat).T.flat)
    colors = modules["grDevices"].grey([x / m for x in numpy.array(mat).T.flat])
    modules["stats"].heatmap(x=r["as.matrix"](mat), Colv=NA_bool, margins=FloatVector([2, 10]),  labCol=NA_bool, col=colors)
    modules["grDevices"].dev_off()
    return None


def plot_dendrogram(target, source, env):
    """
    """
    args = source[-1].read()
    mat = npz_to_df(source[0].rstr())
    modules["grDevices"].bitmap(target[0].rstr(), width=50 * mat.nrow, height=50 * mat.nrow, units="px", type="png256")
    #modules["grDevices"].png(target[0].rstr(),  width=50 * mat.nrow, height=50 * mat.nrow, units="px")
    colors = []
    if args.get("DIRECT", False):
        dm = r["as.dist"](mat)
    else:
        dm = r["dist"](mat)
    hc = r["hclust"](dm, method="average")
    d = r["as.dendrogram"](hc)
    r.par(mar=FloatVector([5, 5, 0, 25]))
    r.par(**{"ps" : ((60 * mat.nrow) / mat.nrow) / 2.0, "lwd" : 3})
    r["plot"](d, horiz=True, xlab="Cosine distance")
    modules["grDevices"].dev_off()
    return None


def plot_scatter(target, source, env):
    """
    """
    mat = npz_to_df(source[0].rstr())
    module["grDevices"].png(target[0].rstr())
    r.plot(mat)
    module["grDevices"].dev_off()
    return None


def TOOLS_ADD(env, prefix="R"):
    builders = {
        'Apply' : Builder(action=apply_r, emitter=partial(generic_emitter, targets=[("", "npz")])),
        'kSVM' : Builder(action=ksvm, emitter=partial(generic_emitter, targets=[("", "npz")])),
        'PlotHeatmap' : Builder(action=plot_heat, emitter=partial(generic_emitter, targets=[("", "png")])),
        'PlotScatter' : Builder(action=plot_scatter, emitter=partial(generic_emitter, targets=[("", "png")])),
        'PlotDendrogram' : Builder(action=plot_dendrogram, emitter=partial(generic_emitter, targets=[("", "bmp")])),
        'PlotCluster' : Builder(action=plot_cluster, emitter=partial(generic_emitter, targets=[("", "png")])),
        }    
    env.Append(BUILDERS=dict([("%s%s" % (prefix, k), v) for k, v in builders.iteritems()]))
