from SCons.Builder import Builder
from SCons.Script import FindFile
from SCons.Node.FS import Dir, File
from SCons.Node.Alias import Alias
from subprocess import Popen, PIPE
from weka import regex, arff
import weka
from common_tools import meta_open, meta_basename, log_likelihood, generic_emitter, unpack_numpy
import numpy
import logging
import cPickle as pickle
from functools import partial
from scipy import sparse


def arff_to_numpy(target, source, env):
    fd = weka.Arff(filename=source[0].rstr())
    numeric_attrs = [x for x, y in fd.attributes.iteritems() if not isinstance(y, set)]
    label_field = source[-1].read().get("LABEL", "LABEL")
    if env.get("SPARSE", True):
        mat = sparse.lil_matrix((len(fd.data), len(numeric_attrs)))
        labels = []
        for row, datum in enumerate(fd.data):
            labels.append(datum[label_field])
            for col, feature in enumerate(numeric_attrs):
                if feature in datum and datum[feature] != 0:                
                    mat[row, col] = datum[feature]
    else:
        mat = []
        for feature in numeric_attrs:
            mat.append([x.get(feature, 0.0) for x in fd.data])
        mat = numpy.asarray(mat).T
        labels = [x[label_field] for x in fd.data]
    pickle.dump((mat, labels, numeric_attrs), meta_open(target[0].rstr(), "w"))
    return None


def numpy_to_arff(target, source, env):
    mat, labels, features = unpack_numpy(source[0].rstr())
    features = [x.get("_NAME") for x in features]
    fd = weka.Arff()
    for label, values in zip(labels, mat):
        datum = dict(zip(features, values))
        datum.update(label)
        fd.add_datum(datum)
    fd.save(meta_open(target[0].rstr(), "w"))
    return None


def numpy_to_octave(target, source, env):
    mat, rows, cols = pickle.load(meta_open(source[0].rstr()))
    if hasattr(mat, "todense"):
        mat = numpy.asarray(mat.todense())
    dfd = open(target[0].rstr(), "w")
    open(target[1].rstr(), "w").write("\n".join(rows))
    for row, data in enumerate(mat, 1):
        for col, val in enumerate(data, 1):
            dfd.write("%d %d %f\n" % (row, col, val))
    return None


def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        'ArffToNumpy' : Builder(action=arff_to_numpy, emitter=partial(generic_emitter, targets=[("", "pkl.gz")])),
        'NumpyToArff' : Builder(action=numpy_to_arff, emitter=partial(generic_emitter, targets=[("", "arff.gz")])),
        'NumpyToOctave' : Builder(action=numpy_to_octave, emitter=partial(generic_emitter, name="", ext="", extra_targets=[("", "labels")])),
        })
