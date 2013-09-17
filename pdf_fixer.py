import tempfile
import subprocess
import argparse
import sys
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input")
parser.add_argument("-t", "--title", dest="title")
parser.add_argument("-a", "--author", dest="author")
options = parser.parse_args()

pdf_fid, pdf = tempfile.mkstemp()
header_fid, header = tempfile.mkstemp()

open(header, "w").write("""
InfoKey: Title
InfoValue: %s
InfoKey: Author
InfoValue: %s
""" % (options.title, options.author))

shutil.copy2(options.input, pdf)
subprocess.call(["pdftk", pdf, "update_info_utf8", header, "output", options.input])

os.remove(pdf)
os.remove(header)
