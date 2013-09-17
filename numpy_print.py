from common_tools import meta_open
import cPickle
import numpy
import pprint
import scipy
from scipy import sparse
import optparse
import re
import csv
from common_tools import unpack_numpy
import logging

parser = optparse.OptionParser()
parser.add_option("-r", dest="rows", action="append", default=[])
parser.add_option("-c", dest="cols", action="append", default=[])
parser.add_option("-o", dest="output", action="append", default=[])
parser.add_option("-i", dest="input", action="append", default=[])
parser.add_option("-t", dest="type", type="choice", choices=["sum_rows", "sum_cols", "print", "free"], default="print")
options, args = parser.parse_args()

logging.basicConfig(level = 0, format="%(asctime)s - %(message)s", filemode="w")

for fname in options.input:
    if options.type == "free":
        fd = numpy.load(fname)
        for k in fd.keys():
            print k, type(fd[k]), fd[k].shape
        continue

    mat, labels, features = unpack_numpy(fname, dense=True, oldstyle=True)
    labels = [x["_NAME"] for x in labels]
    features = [x["_NAME"] for x in features]
    
    row_indices = range(min(10, len(labels)))
    col_indices = range(min(10, len(features)))
    if options.rows:
        row_indices = [i for i, x in enumerate(labels) if any([re.match(rx, x) for rx in options.rows])]
    if options.cols:
        col_indices = [i for i, x in enumerate(features) if any([re.match(rx, x) for rx in options.cols])]

    print "\t".join([labels[i] for i in row_indices])
    print "\t".join([features[i] for i in col_indices])

    if options.type == "sum_cols":
        print mat[row_indices, :][:, col_indices].sum(1)
    elif options.type == "sum_rows":
        print mat[row_indices, :][:, col_indices].sum(0)
    else:
        print mat[row_indices, :][:, col_indices]
    
