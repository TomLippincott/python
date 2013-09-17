import os.path
import os
from glob import glob
import hashlib
import argparse
import shutil
import sys
import re

parser = argparse.ArgumentParser()
parser.add_argument("input", nargs="+")
parser.add_argument("-o", "--output", dest="output")
parser.add_argument("-a", "--action", dest="action", choices=["print", "save"], default="print")
options = parser.parse_args()

image_rx = re.compile(r".*\.(jpg|png|jpeg)", re.I)
def visit(path, image_files=[]):
    if os.path.isfile(path) and image_rx.match(path):
        return [path]
    elif os.path.isdir(path):
        return sum([visit(os.path.join(path, x)) for x in os.listdir(path)], [])
    else:
        return []

files = sum([visit(x) for x in options.input], [])

data = {}
for fname in files:
    m = hashlib.md5()
    m.update(open(fname).read())
    d = m.hexdigest()
    data[d] = data.get(d, []) + [fname]

for k, v in data.iteritems():
    if options.action == "print" and len(v) > 1:
        print k, v
    #elif args.action == "save":
    #    dest = os.path.join(args.output, os.path.basename(v[0]))
    #    shutil.copy2(v[0], dest)
