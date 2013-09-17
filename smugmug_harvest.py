import argparse
import urllib
import re
import os.path
import os
import sys
import xml.etree.ElementTree as et

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--url", dest="url", default="http://client.alwynphoto.com/LAURA-AND-TOM/")
parser.add_argument("-o", "--output", dest="output", default=".")
options = parser.parse_args()

outname = os.path.join(options.output, "main.xml")
if not os.path.exists(outname):
    open(outname, "w").write(urllib.urlopen(options.url).read())
main_text = open(outname).read()

albums = {}
for x in re.finditer("%s/([^\/]*)/([^\/\"]*)" % (options.url.rstrip("/")), main_text):
    albums[x.group(1)] = x.group(2)

to_download = set()
for name, key in albums.iteritems():
    feed = "http://client.alwynphoto.com/hack/feed.mg"
    data = urllib.urlencode({"Type" : "gallery", "Data" : key, "format" : "rss200"})
    outname = os.path.join(options.output, "%s.xml" % (name))
    if not os.path.exists(outname):
        open(outname, "w").write(urllib.urlopen(feed, data=data).read())
    outdir = os.path.join(options.output, "%s" % (name))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    feed_text = open(outname).read()
    feed_xml = et.parse(open(outname))
    for x in feed_xml.getiterator("{http://search.yahoo.com/mrss/}content"):        
        to_download.add((x.attrib["url"], outdir))

#print len(to_download), len([x for x in to_download if x.endswith("jpg")])
for url, path in to_download:
    outname = os.path.join(path, os.path.basename(url))
    if not os.path.exists(outname):
        open(outname, "w").write(urllib.urlopen(url).read())
        print outname
