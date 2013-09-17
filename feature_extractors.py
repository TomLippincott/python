"""
Converting an XML file (e.g. TEI) to an Arff file can be done by
specifying the documents, labels and features using the XPath
syntax (http://www.w3.org/TR/xpath).

Each document will be given a frequency for each unique value of
the given feature.

If the file input.xml contains:

<File>
  <div type="source" n="Genesis">
    <div type="name">
      <c>E</c>
      <c>v</c>
      <c>e</c>
    </div>
    <div type="name">
      <c>A</c>
      <c>d</c>
      <c>a</c>
      <c>m</c>
    </div>
  </div>
  <div type="source" n="Moby Dick">
    <div type="name">
      <c>S</c>
      <c>t</c>
      <c>u</c>
      <c>b</c>
      <c>b</c>
    </div>
    <div type="name">
      <c>A</c>
      <c>h</c>
      <c>a</c>
      <c>b</c>
    </div>
  </div>
</File>

then running:

python -m feature_extractors -i input.xml -o output.arff -l "//div[@type='source']/@n" -d "../div[@type='name']" -f "c"

produces this ARFF file:

@RELATION UNNAMED
@ATTRIBUTE 'A' NUMERIC
@ATTRIBUTE 'E' NUMERIC
@ATTRIBUTE 'S' NUMERIC
@ATTRIBUTE 'a' NUMERIC
@ATTRIBUTE 'b' NUMERIC
@ATTRIBUTE 'd' NUMERIC
@ATTRIBUTE 'e' NUMERIC
@ATTRIBUTE 'h' NUMERIC
@ATTRIBUTE 'm' NUMERIC
@ATTRIBUTE 't' NUMERIC
@ATTRIBUTE 'u' NUMERIC
@ATTRIBUTE 'v' NUMERIC
@ATTRIBUTE 'LABEL n' {'Moby Dick','Genesis'}
@DATA
'0','0.333333333333','0','0','0','0','0.333333333333','0','0','0','0','0.333333333333','Genesis'
'0.25','0','0','0.25','0','0.25','0','0','0.25','0','0','0','Genesis'
'0','0','0.2','0','0.4','0','0','0','0','0.2','0.2','0','Moby Dick'
'0.25','0','0','0.25','0.25','0','0','0.25','0','0','0','0','Moby Dick'
"""

from unicodedata import name
import cPickle
import xml.sax
import xml.etree.ElementTree as et
import gzip
from common_tools import meta_open
import re
import numpy
import logging

# HELPERS

def __unpoint(token):
    return u"".join([x for x in token.decode('utf-8') if "LETTER" in name(x, "UNKNOWN")])


# EXTRACTORS
#
# The text is either XML or plain text.  
# Plain text we tokenize ourselves, XML we process accordingly.
#

def morphology(text, morph):
    """
    Return the frequency counts in text for each morphological class described in morph.
    """
    mapping = {}
    counts = {}
    tree = et.parse(meta_open(morph))
    try:
        tokens = [x.text for x in text.getiterator("f") if x.attrib.get("name", "") == "morph"]
    except:
        tokens = text.split()
    for i, cat in enumerate(tree.getiterator("category")):
        for item in cat.getiterator("item"):
            mapping[item.text] = i
    for token in tokens:
        mc = "morphclass.%s" % mapping.get(token, None)
        counts[mc] = counts.get(mc, 0) + 1
    return counts


def bag_of_words(text, clean=False, unpoint=False):
    if clean:
        toks = ["".join([m.text.lower() for m in x.getiterator() if m.text]) 
                for x in text.getiterator("fs") if x.get("type", None) == "token"]
    elif unpoint:
        toks = [__unpoint("".join([m.text for m in x.getiterator() if m.text]).encode("utf-8"))
                for x in text.getiterator("fs") if x.get("type", None) == "token"]
    else:
        toks = ["".join([m.text for m in x.getiterator() if m.text]) for x in text.getiterator("fs") if x.get("type", None) == "token"]
    return dict([(x, toks.count(x)) for x in set(toks)])


def strongs(text):
    toks = [x.getiterator("numeric")[0].attrib["value"] for x in text.getiterator("f") if x.attrib.get("name", None) == "strongs"]
    return dict([(x, toks.count(x)) for x in set(toks)])


def unpointed_bag_of_words(text):
    toks = [__unpoint(x) for x in text.split()]
    return dict([(x, toks.count(x) / float(len(toks))) for x in set(toks)])


def stem_frequencies(text, stem_affix_file):
    rule_sets = cPickle.load(open(stem_affix_file, 'rb'))
    toks = [rule_sets.get(__unpoint(x), "unknown") for x in text.split()]
    return dict([(x, toks.count(x) / float(len(toks))) for x in set(toks)])


def ngrams(tokens, n=1):
    items = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return dict([(x, items.count(x) / float(len(items))) for x in set(items)])


def match(xpath, name, attr):
    xname = xpath.split('[')[0]
    xattr = xpath.split('[')[-1].strip(']').split(' and ')
    return xname == name and all([attr.get(a, None) == b.strip("'") for a, b in [x.strip('@').split('=') for x in xattr]])


#from nltk.corpus import wordnet
def scf_output(text):
    features = {}
    totals = {}
    for m in re.finditer("<class classnum=\"(\S+)\" target=\"(\S+)\"", text):
        scf = m.group(1)
        verb = m.group(2).lower()
        if not wordnet.morphy(verb, "v"):
            continue
        key = "SCF_=_%s" % scf
        features[key] = features.get(key, 0) + 1
        totals["SCF"] = totals.get("SCF", 0) + 1
        key = "VSCF-%s_=_%s" % (verb, scf)
        features[key] = features.get(key, 0) + 1
        totals["VSCF-%s" % verb] = totals.get("VSCF-%s" % verb, 0) + 1
    keys = features.keys()
    for k in keys:
        n, e = k.split("_=_")
        features["%s_+_" % n] = totals[n]        
    return features


def ccg_output(text, clean=True):
    """
    From C&C output, count tokens, lemmas, parts-of-speech, chunk, other,
    category, 
    sometimes, a pipe appears in the token/lemma
    """
    totals = {}
    features = {}
    alllengths = {}
    text = "\n\n" + text
    mapping = {"N" : "NOUNS", "V" : "VERBS", "J" : "ADJECTIVES", "R" : "ADVERBS"}
    for parse in re.finditer("\n\s*\n(.*?)(^<c>.*?)$", text, re.M | re.S):
        #logging.info("start sentence")        
        line = parse.group(2)
        grs = parse.group(1).strip().split("\n")
        alllengths["SENTENCE"] = alllengths.get("SENTENCE", []) + [float(len(line.split()))]
        gr_info = {}
        for gr in grs:
            # this takes too long
            vals = gr.strip()[1:-1].split()
            if len(vals) == 0:
                continue
            grname = vals[0]
            args = [x for x in vals if re.match("^\w+_\d+", x)]
            args = [int(x.split("_")[-1]) for x in args]
            for i in args:
                gr_info[i] = gr_info.get(i, []) + [vals[0]]
            if len(args) == 2:
                key = "%s" % grname
                if key in alllengths:
                    alllengths[key].append(abs(args[0] - args[1]))
                else:
                    alllengths[key] = [abs(args[0] - args[1])]
                #alllengths[key] = alllengths.get(key, []) + [1] #[float(abs(args[0] - args[1]))]
            key = "GR_=_%s" % grname
            features[key] = features.get(key, 0) + 1
            totals["GR"] = totals.get("GR", 0) + 1

        counter = 0
        #logging.info("start tokens")        
        for item in line.split()[1:]:
            try:
                toks = item.split("|")
                embed = (len(toks) - 4) / 2
                word = dict(zip(["TOKEN", "LEMMA", "POS", "CHUNK", "OTHER", "CATEGORY"], 
                                ["".join(toks[0:embed]), "".join(toks[embed:2*embed])] + toks[2*embed:]))
            except:
                continue
            if len(word) != 6:
                print toks, item, line
                sys.exit()
            if clean:
                word["LEMMA"] = re.sub("\W", "", word["LEMMA"].lower(), re.UNICODE).replace("_", "")
            key = "%s_=_%s" % (mapping.get(word["POS"][0], "OTHER"), word["LEMMA"])
            features[key] = features.get(key, 0) + 1
            totals["%s" % mapping.get(word["POS"][0], "OTHER")] = totals.get("%s" % mapping.get(word["POS"][0], "OTHER"), 0) + 1
            for k, v in word.iteritems():
                if k in ["TOKEN", "LEMMA"]:
                    continue
                key = "%s_=_%s" % (k, v)
                features[key] = features.get(key, 0) + 1
                totals[k] = totals.get(k, 0) + 1
            word["LEMMA"] = word["LEMMA"].lower()
            if word["POS"].startswith("V"):
                word["LEMMA"] = word["LEMMA"].lower()
                key = "VERBPOS-%s_=_%s" % (word["LEMMA"], word["POS"])
                features[key] = features.get(key, 0) + 1
                totals["VERBPOS-%s" % word["LEMMA"]] = totals.get("VERBPOS-%s" % word["LEMMA"], 0) + 1
                voice = "active"
                if "[pss]" in word["CATEGORY"]:
                    voice = "passive"
                key = "VERBVOICE-%s_=_%s" % (word["LEMMA"], voice)
                features[key] = features.get(key, 0) + 1
                totals["VERBVOICE-%s" % word["LEMMA"]] = totals.get("VERBVOICE-%s" % word["LEMMA"], 0) + 1
                for gr in gr_info.get(counter, []):
                    key = "VERBGR-%s_=_%s" % (word["LEMMA"], gr)
                    features[key] = features.get(key, 0) + 1
                    totals["VERBGR-%s" % word["LEMMA"]] = totals.get("VERBGR-%s" % word["LEMMA"], 0) + 1
            counter += 1
    keys = features.keys()
    #for k in keys:
    #    n, e = k.split("_=_")
    #    features["%s_+_" % n] = totals[n]
    for k, v in alllengths.iteritems():
        features["LENGTH-STDDEV-%s_=_" % k] = numpy.std(v)
        #features["LENGTH-STDDEV-%s_+_" % k] = 1.0
        features["LENGTH-AVERAGE-%s_=_" % k] = numpy.average(v)
        #features["LENGTH-AVERAGE-%s_+_" % k] = 1.0
    return features


def ccg_lemmas(text, clean=True):
    """
    From C&C output, count tokens, lemmas, parts-of-speech, chunk, other,
    category, 
    sometimes, a pipe appears in the token/lemma
    """
    totals = {}
    features = {}
    alllengths = {}
    for parse in re.finditer("\n\s*\n(.*?)(^<c>.*?)$", text, re.M | re.S):
        logging.info("start sentence")        
        line = parse.group(2)
        counter = 0
        logging.info("start tokens")        
        for item in line.split()[1:]:
            try:
                toks = item.split("|")
                embed = (len(toks) - 4) / 2
                word = dict(zip(["TOKEN", "LEMMA", "POS", "CHUNK", "OTHER", "CATEGORY"], 
                                ["".join(toks[0:embed]), "".join(toks[embed:2*embed])] + toks[2*embed:]))
            except:
                continue
            word["LEMMA"] = re.sub("\W", "", word["LEMMA"].lower(), re.UNICODE).replace("_", "")
            key = "LEMMA_=_%s" % (word["LEMMA"])
            features[key] = features.get(key, 0) + 1
            totals["LEMMA"] = totals.get("LEMMA", 0) + 1
    return features


def scf_from_ccg(text, clean=True):
    features = {}
    for m in re.finditer("class classnum=\"(.*?)\" target=\"(.*?)\"", text):
        scf = m.group(1)
        verb = m.group(2)
        if clean:
            verb = re.sub("\W", "", verb.lower(), re.UNICODE).replace("_", "")
        features["SCF_=_%s" % scf] = features.get("SCF_=_%s" % scf, 0) + 1
        features["VERBSCF-%s_=_%s" % (verb, scf)] = features.get("VERBSCF-%s_=_%s" % (verb, scf), 0) + 1
    return features


class FeatureExtractor(xml.sax.handler.ContentHandler):
    def __init__(self, label, document, features, fd):
        self.fd = fd
        self.arff = Arff()
        self.label = label
        self.document = document
        self.features = features
        self.cur_label = None
        self.cur_doc = None
        self.cur_feat = []
        self.docstack = 0
        self.labstack = 0
        self.featstack = 0

        
    def startElement(self, name, attrs):
        if name == self.label.split('[')[0]:
            if self.cur_label != None:
                self.labstack += 1
            elif match(self.label, name, attrs):
                self.labstack = 1
                self.cur_label = attrs['n']
        
        if name == self.document.split('[')[0]:
            if self.cur_doc != None:
                self.docstack += 1
            elif match(self.document, name, attrs):
                self.docstack = 1
                self.cur_doc = {'label' : self.cur_label}

        if name == 'f':
            self.cur_feat.append(attrs.get('name', '?'))

        if name == 'fs':
            self.cur_feat.append(attrs.get('type', '?'))

        if name == 'symbol' and self.cur_doc != None:
            key = "%s=%s" % ("_".join(self.cur_feat), attrs['value'])
            self.cur_doc[key.lstrip('?_')] = self.cur_doc.get('value', 0) + 1
                
    def endElement(self, name):
        if name == self.label.split('[')[0]:
            if self.cur_label != None:
                self.labstack -= 1
            if self.labstack == 0 and self.cur_label != None:
                self.cur_label = None

        if name == self.document.split('[')[0]:
            if self.cur_doc != None:
                self.docstack -= 1
            if self.docstack == 0 and self.cur_doc != None:
                self.arff.add_datum(self.cur_doc)
                self.cur_doc = None

        if name in ['f', 'fs']:
            self.cur_feat = self.cur_feat[0:-1]
        
    ## def characters(self, content):
    ##     if self.write == True:
    ##         self.fd.write("%s" % content.strip())
    ##     elif len(content.strip()) > 0:
    ##         self.fd.write(self.handler(content))

    def endDocument(self):
        self.arff.save(self.fd)



if __name__ == "__main__":

    from weka import Arff
    import optparse
    import libxml2
    import sys
    from lxml import etree
    import os

    parser = optparse.OptionParser()
    parser.add_option('-i', '--input', dest='input')
    parser.add_option('-o', '--output', dest='output')
    parser.add_option('-f', '--features', dest='features', default=[], action='append')
    parser.add_option('-l', '--label', dest='label', help='xpath')
    parser.add_option('-d', '--document', dest='document', help='xpath')
    options, remainder = parser.parse_args()

    if options.output:
        fd = open(options.output, 'w')
    else:
        fd = sys.stdout

    xml.sax.parse(options.input, FeatureExtractor(options.label, options.document, options.features, open(options.output, 'w')))
    sys.exit()

    a = Arff()

    #doc = libxml2.parseFile(options.input)
    doc = etree.parse(open(options.input))
    doc.xinclude()

    for labeled in doc.xpath(options.label):
        label = labeled
        if isinstance(labeled, etree._ElementStringResult):
            labeled = labeled.getparent()
        #labeled.xinclude()
        for text in labeled.xpath(options.document):
            if isinstance(text, etree._ElementStringResult):
                text = text.getparent()
            #print type(text)
            #text.xincludeProcessTree()
            attribs = {"LABEL" : label}
            #print text
            for feature in options.features:
                seq = [str(x) for x in text.xpath(feature)]
                #print seq
                for i in set(seq):
                    attribs[i] = seq.count(i) / float(len(seq))
            a.add_datum(attribs)

    a.save(fd)
