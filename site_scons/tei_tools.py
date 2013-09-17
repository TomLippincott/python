"""
"""

from SCons.Builder import Builder
from SCons.Script import FindFile
import cPickle
import re
#import libxml2
from tei import TeiDocument
from xml.etree import cElementTree as et
import xml.sax
import tei
from bible import get_book_order, book_names, Point, Citation
import gzip
import glob
import os.path
from common_tools import meta_open, generic_emitter
from functools import partial
import logging


def st_to_tei(target, source, env):
    data = {}
    for l in gzip.open(source[0].rstr()):
        try:
            book, chapter, verse, text = unicode(l, 'utf-8').strip().strip('~:').split('|')
        except:
            continue
        book = (('type', 'book'), ('_n', get_book_order(book, env["DOCUMENT"])), ('n', book_names.get(book.lower(), book)))
        chapter = (('type', 'chapter'), ('n', int(chapter)))
        verse = (('type', 'verse'), ('n', int(verse)))
        if not book in data:
            data[book] = {}
        if not chapter in data[book]:
            data[book][chapter] = {}
        data[book][chapter][verse] = re.sub('<.*?>', '', text)
    teifile = tei.TeiDocument(header = env["HEADER"])
    teifile.from_dict(data)
    fd = meta_open(target[0].rstr(), "w")
    fd.write("""<?xml version='1.0' encoding='utf-8'?>\n""")
    teifile.write(fd)
    return None


def add_hypothesis(target, source, env):
    hyp = et.parse(source[1].rstr())
    newtree = et.ElementTree(hyp.getroot())

    # make a list of all books relevant to this hypothesis
    books = set(sum([[x.attrib.get('from', '').split(':')[0], x.attrib.get('to', '').split(':')[0]] for x in hyp.getiterator('span')], []))

    # get the set of labels
    labels = {}
    for i in hyp.getiterator("category"):
        key = [k for k in i.keys() if k.endswith("id")][0]
        labels[i.get(key)] = list(i.getiterator("catDesc"))[0].text
    logging.info("%d labels", len(labels))

    main = et.parse(meta_open(source[0].rstr()))

    # make a list of its words
    words = []
    for b in [x for x in main.getiterator('div') if x.attrib['type'] == 'book']:
        if b.attrib['n'] in books:
            for c in [x for x in b.getiterator('div') if x.attrib['type'] == 'chapter']:
                for v in [x for x in c.getiterator('div') if x.attrib['type'] == 'verse']:
                    counter = 1
                    for fs in [x for x in v.getiterator('fs') if x.attrib.get('type', None) == "token"]:
                        words.append((fs, Point("%s:%s:%s.%d" % (b.attrib['n'].strip(), c.attrib['n'].strip(), v.attrib['n'].strip(), counter))))
                        counter += 1
            list(newtree.getiterator("body"))[0].append(b)
    logging.info("%d words", len(words))
    
    # process each labeling
    for ex in hyp.getiterator('span'):
        label = ex.attrib['type']
        if 'to' in ex.attrib:
            span = Citation("%s-%s" % (ex.attrib['from'], ex.attrib['to']))
        else:
            span = Citation(ex.attrib['from'])
        for fs, p in words:
            if span.contains(p):
                fs.attrib['hyp'] = label
            #fs.attrib['definition'] = dictionary.get(fs.attrib.get("lemmaRef", '?'), '?')

    # bubble up hypotheses
    for verse in [x for x in newtree.getiterator("div") if x.attrib.get("type", "") == "verse"]:
        l = set([x.attrib.get("hyp", "") for x in verse.getiterator("fs")])
        if len(l) == 1:
            verse.attrib["hyp"] = list(l)[0]
            [x.attrib.pop("hyp", "") for x in verse.getiterator("fs")]

    for chapter in [x for x in newtree.getiterator("div") if x.attrib.get("type", "") == "chapter"]:
        l = set([x.attrib.get("hyp", "") for x in chapter.getiterator("div") if x.attrib.get("type", "") == "verse"])
        if len(l) == 1:
            chapter.attrib["hyp"] = list(l)[0]
            [x.attrib.pop("hyp", "") for x in chapter.getiterator("div") if x.attrib.get("type", "") == "verse"]

    for book in [x for x in newtree.getiterator("div") if x.attrib.get("type", "") == "book"]:
        l = set([x.attrib.get("hyp", "") for x in book.getiterator("div") if x.attrib.get("type", "") == "chapter"])
        if len(l) == 1:
            book.attrib["hyp"] = list(l)[0]
            [x.attrib.pop("hyp", "") for x in book.getiterator("div") if x.attrib.get("type", "") == "chapter"]

    # replace hypothesis attributes with hypothesis div elements
    for e in [x for x in newtree.getiterator("div") if x.attrib.get("hyp", "") != ""]:
        children = e.getchildren()
        attribs = dict(e.items())
        del attribs["hyp"]
        hyp = e.attrib.get("hyp", "")
        e.clear()
        e.attrib = attribs
        div = et.Element("div", {"type" : "hypothesis", "n" : hyp})
        e.append(div)
        for c in children:
            div.append(c)

    for e in [x for x in newtree.getiterator("div") if x.attrib.get("type", "") == "verse"]:
        toks = e.getchildren()
        sublabels = set([x.attrib.get("hyp", "") for x in toks])
        if sublabels != set([""]):
            attribs = dict(e.items())
            e.clear()
            e.attrib = attribs
            div = None
            for t in toks:
                if div == None:
                    div = et.Element("div", {"type" : "hypothesis", "n" : t.attrib["hyp"]})
                elif div.attrib.get("hyp", "1") != t.attrib.get("hyp", "2"):
                    e.append(div)            
                    div = et.Element("div", {"type" : "hypothesis", "n" : t.attrib.get("hyp", "?")})
                try:
                    del t.attrib["hyp"]
                except:
                    pass
                div.append(t)

    for s in newtree.getiterator("string"):
        try:
            s.text = unpoint(s.text)
        except:
            pass

    # write the result
    fd = gzip.open(target[0].rstr(), 'w')
    fd.write("""<?xml version="1.0" encoding="UTF-8"?>\n""")
    newtree.write(fd)
    return None


def tei_to_html(target, source, env):
#    tree = et.parse(meta_open(source[0].rstr()))
#    divs = [x.attrib["type"] for x in tree.getiterator("div")]
#    ordered_divs = sorted(set(divs), lambda x, y : cmp(divs.index(x), divs.index(y)))
#    for level in ordered_divs:
#        structure = dict([() for x in ])
    return None


def xml_emitter(target, source, env):
    target[0] = "work/texts/${DOCUMENT}.xml.gz"
    env.Alias("xml", target)
    return target, source


def html_emitter(target, source, env):
    base = os.path.basename(source[0].rstr()).split(".")[0]
    target[0] = "work/html/%s/" % base
    env.Alias("html", target)
    return target, source


def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "STtoTei" : Builder(action=st_to_tei, emitter=partial(generic_emitter, targets=[("", "xml")])),
        "AddHypothesis" : Builder(action=add_hypothesis),
        "TEItoHTML" : Builder(action=tei_to_html, emitter=html_emitter),
                           })
