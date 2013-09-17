import os.path
import os
from glob import glob
import hashlib
import argparse
import shutil
import sys
from mimetypes import guess_type, add_type
import logging
from PIL import Image
import pickle
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("input", nargs="+")
parser.add_argument("-o", "--output", dest="output")
parser.add_argument("-d", "--dir", dest="dir")
parser.add_argument("-r", "--resume", dest="resume")
parser.add_argument("-a", "--action", dest="action", choices=["print", "save", "print_non_images"], default="print")
parser.add_argument("-s", "--stop", dest="stop", type=int, default=-1, help="0=make list, >0=how many to process")
options = parser.parse_args()

add_type("image/xcf", ".xcf")

if options.resume:
    files = pickle.load(open(options.resume))
else:
    files = dict([(x, {"type" : guess_type(x)[0]}) for x in sum([sum([[os.path.join(x[0], y) for y in x[2]] for x in os.walk(basepath)], []) for basepath in options.input], [])])

#if options.action == "print":
#    print len([(k, v) for k, v in files.iteritems() if "hash" in v]), len(files)

if options.stop == 0:
    if options.output:
        pickle.dump(files, open(options.output, "w"))
    sys.exit()

counter = 0
for fname in files.keys():
    vals = files[fname]
    if not vals.get("type") or not vals.get("type", "").startswith("image") or "hash" in vals:
        continue
    try:
        s = Image.open(fname).tostring()
        m = hashlib.md5()
        m.update(s)
        d = m.hexdigest()
    except:
        d = ""
    files[fname]["hash"] = d
    counter += 1
    if options.stop != -1 and counter > options.stop:
        break

if options.output:
    pickle.dump(files, open(options.output, "w"))

uniques = {}
if options.dir:
    for k, v in files.iteritems():
        if "hash" in v:
            h = v["hash"]
            uniques[h] = uniques.get(h, []) + [k]
    fsets = uniques.values()
    new_names = {}
    #primaries = [None for x in fsets]
    #taken = set()
    for i in range(len(fsets)):
        names = set([os.path.splitext(os.path.basename(x))[0] for x in fsets[i]])
        if len(names) == 1:
            name = list(names)[0]
        else:
            ntild = [x for x in names if "~" not in x]
            if len(ntild) > 0:
                name = ntild[0]
            else:
                name = list(names)[0]
        if name in new_names:
            app = 2
            while "%s_%d" % (name, app) in new_names:
                app += 1
            new_names["%s_%d" % (name, app)] = fsets[i]
        else:
            new_names[name] = fsets[i]



        #print " ".join(v) #[os.path.basename(x) for x in v])
    for k, v in new_names.iteritems():
        ext = os.path.splitext(v[0])[1].lower()
        if ext == ".jpeg":
            ext = ".jpg"
        target = os.path.join(options.dir, "%s%s" % (k, ext))
        source = v[0]
        shutil.copy(source, target)

    

        #print len(uniques), len(sum(uniques.values(), []))
#print uniques[""]

#print "\n".join([k for k, v in files.iteritems() if v.get("type") and v.get("type", "").startswith("video")])
