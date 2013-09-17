from matplotlib import pyplot
from SCons.Builder import Builder
from SCons.Script import FindFile, Progress
from subprocess import Popen, PIPE
import weka
import sqlite3
import re
import os
import gzip
import random
import glob
from subprocess import PIPE, Popen
import lxml.etree as et
import numpy
import cPickle
import csv
import numpy
import tempfile
import sys
import xmlrpclib
import logging
try:
    from candc import CCGParser
except:
    pass
#from SOAPpy import SOAPProxy, WSDL
from xml.sax import SAXParseException
from common_tools import meta_open, kullback_leibler_divergence, jensen_shannon_divergence, meta_basename, unpack_numpy
def esc_weka(token):
    for a, b in [("\t", " "), (",", " "), ("\n", " "), ("'", " "), (",", "_"), ("\"", "_")]:
        token = token.replace(a, b)
    return token



def genia_arff(target, source, env):
    mapping = env["MAPPING"]
    all_subs = set(sum([list(x) for x in mapping.values()], []))
    fd = weka.Arff()
    for f in source[0].children():
        journal = f.rstr().split("/")[-2]
        feats = dict([(x, str(x in mapping.get(journal, []))) for x in all_subs])
        feats["JOURNAL"] = journal
        for l in gzip.open(f.rstr()):
            for t, v in zip(["TOKEN", "LEMMA", "POS", "CHUNK", "NE"], l.strip().split()):
                if t != env["FEAT"]:
                    continue
                feat = "%s_%s" % (t, v)
                feats[feat] = feats.get(feat, 0) + 1
        fd.add_datum(feats, force=True)
    fd.save(gzip.open(target[0].rstr(), "w"), sparse=True)
    return None


def candc_arff(target, source, env):
    fd = weka.Arff()
    mapping = env["MAPPING"]
    all_subs = set(sum([list(x) for x in mapping.values()], []))
    for f in source[0].children():
        journal = f.rstr().split("/")[-2]
        feats = dict([(x, str(x in mapping.get(journal, []))) for x in all_subs])
        feats["JOURNAL"] = journal
        for s in re.split("\n\s*\n", gzip.open(f.rstr()).read()):            
            for l in s.strip().split("\n"):
                if l.startswith("<c>"):
                    for w in [dict(zip(["TOKEN", "LEMMA", "POS", "CHUNK", "NE", "CATEGORY"], w.split("|"))) for w in l[3:].split()]:
                        for k, v in w.iteritems():
                            if k != env["FEAT"]:
                                continue
                            feats["%s_%s" % (k, v)] = feats.get("%s_%s" % (k, v), 0) + 1
                #elif re.match("^\(\)\s*$", l):                
        fd.add_datum(feats, force=True)
    fd.save(gzip.open(target[0].rstr(), "w"), sparse=True)
    return None

def subject_overview(target, source, env):
    journals = {}
    for j in source[0].read():
        journal = os.path.basename(j)
        words = 0
        try:
            for s in re.finditer("^(<c>.*?)$", meta_open("%s/%s.parsed.txt.gz" % (env["PUBMED_CCG_PATH"], re.sub("[^a-zA-Z0-9_]", "", journal.lower().replace(" ", "_")))).read(), re.M):
                words += len(s.group(1).split())
        except:
            print "error processing %s" % journal
            continue
        subjects = set(sum([list(y) for x, y in env["MAPPING"].iteritems() if journal == x], []))
        journals[journal] = (subjects, words)
    cPickle.dump(journals, meta_open(target[0].rstr(), "w"))
    return None

def other_arff(target, source, env):
    fd = weka.Arff()
    mapping = env["MAPPING"]
    all_subs = set(sum([list(x) for x in mapping.values()], []))
    sentences = []
    for f in source[0].children():
        journal = f.rstr().split("/")[-2]
        feats = dict([(x, str(x in mapping.get(journal, []))) for x in all_subs])
        feats["JOURNAL"] = journal
        for s in re.split("\n\s*\n", gzip.open(f.rstr()).read()):            
            for l in s.strip().split("\n"):
                if l.startswith("<c>"):
                    words = [w for w in l[3:].split()]
                    feats["WORD_COUNT"] = feats.get("WORD_COUNT", 0) + len(words)
                    sentences.append(len(words))
        feats["AVG_WPS"] = sum(sentences) / float(len(sentences))
        feats["STDDEV_WPS"] = numpy.std(sentences)
        fd.add_datum(feats, force=True)
    fd.save(gzip.open(target[0].rstr(), "w"), sparse=True)
    return None

good_tags = ["p"]
def xml_to_text(target, source, env):
    xml = et.parse(meta_open(source[0].rstr()))
    processed = ""
    for s in xml.getiterator("body"):        
        processed += "  ".join([x.text for x in s.getiterator() if x.text and x.tag in good_tags])
    for s in xml.getiterator("abstract"):        
        processed += "  ".join([x.text for x in s.getiterator() if x.text and x.tag in good_tags])
    gzip.open(target[0].rstr(), "w").write(processed.encode("utf-8"))
    return None


def text_to_sentences(target, source, env):
    text = gzip.open(source[0].rstr()).read()
    p = Popen(["java", "-Xmx2048M", "-jar", "data/sptoolkit.jar", "-f", "2"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate(text)
    gzip.open(target[0].rstr(), "w").write(out)
    return None


def document_stats(target, source, env):
    types = env["TYPES"] #["VERBS", "NOUNS", "ADJECTIVES", "ADVERBS", "TOKEN_POS", "POS", "CATEGORY"]
    stats = dict([(t, {}) for t in types])
    for l in meta_open(source[0].rstr(), "r"):
        if l.startswith("<c>"):
            for word in [dict(zip(["token", "lemma", "pos", "chunk", "ne", "category"], w.split("|"))) for w in l[3:].split()]:
                if word["pos"].startswith("V"):
                    stats["VERBS"][word["lemma"]] = stats["VERBS"].get(word["lemma"], 0) + 1
                elif word["pos"].startswith("N"):
                    stats["NOUNS"][word["lemma"]] = stats["NOUNS"].get(word["lemma"], 0) + 1
                elif word["pos"].startswith("J"):
                    stats["ADJECTIVES"][word["lemma"]] = stats["ADJECTIVES"].get(word["lemma"], 0) + 1
                elif word["pos"].startswith("R"):
                    stats["ADVERBS"][word["lemma"]] = stats["ADVERBS"].get(word["lemma"], 0) + 1
                tps = "%s_%s" % (word["token"], word["pos"])
                stats["TOKEN_POS"][tps] = stats["TOKEN_POS"].get(tps, 0) + 1
                stats["POS"][word["pos"]] = stats["POS"].get(word["pos"], 0) + 1
                stats["CATEGORY"][word["category"]] = stats["CATEGORY"].get(word["category"], 0) + 1
    for targ, t in zip(target, types):
        fd = csv.writer(gzip.open(targ.rstr(), "w"), delimiter="\t")
        fd.writerow(["VALUE", "COUNT"])
        for k, v in sorted(stats[t].iteritems(), lambda x, y : cmp(y[1], x[1])):
            fd.writerow([k, v])
    return None


def aggregate_stats(target, source, env):
    stats = {}
    if env.get("ALIAS", False):
        sources = source[0].children()
    else:
        sources = source
    
    for f in sources:
        for line in csv.reader(gzip.open(f.rstr()), delimiter="\t"):
            if line[0] != "VALUE":
                stats[line[0]] = stats.get(line[0], 0) + int(line[1])
    fd = csv.writer(gzip.open(target[0].rstr(), "w"), delimiter="\t")
    fd.writerow(["VALUE", "COUNT"])
    for k, v in sorted(stats.iteritems(), lambda x, y : cmp(y[1], x[1])):
        fd.writerow([k, v])
    return None

def compute_pairwise(target, source, env):
    def f2s(s):
        return re.sub("\W", "_", os.path.basename(s).split("_")[0])
    dists = {}
    stat = {}
    setA = source[0].children()
    setB = source[0].children()
    words = {}

    for f in source[0].children():
        stat[f.rstr()] = {}
        for line in csv.reader(gzip.open(f.rstr()), delimiter="\t"):
            if line[0] != "VALUE":
                stat[f.rstr()][line[0]] = int(line[1])
    total = len(setA) * len(setB)
    for a in setA:
        for b in setB:       
            key = frozenset([f2s(a.rstr()), f2s(b.rstr())])            
            if key in dists:
                continue
            dists[key] = {}
            allwords = set(stat[a.rstr()]).union(set(stat[b.rstr()]))
            if env["TOP_WORDS"]:
                total = dict([(w, stat[a.rstr()].get(w, 0) + stat[b.rstr()].get(w, 0)) for w in allwords])
                try:
                    words = [k for k, v in sorted(total.iteritems(), lambda x, y : cmp(y[1], x[1]))][0 : min(len(total), env["TOP_WORDS"])]
                except:
                    words = None
            else:
                words = allwords
            stat1 = [stat[a.rstr()].get(w, 0) + 1 for w in words]
            stat2 = [stat[b.rstr()].get(w, 0) + 1 for w in words]
            logging.info("%d, %s to %s", len(dists), f2s(a.rstr()), f2s(b.rstr()))
            for name, func in env["FUNCTIONS"]:
                if not words:
                    dists[key][name] = -1
                elif len(key) == 1:
                    dists[key][name] = func([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
                else:
                    dists[key][name] = func(stat1, stat2)
        del stat[a.rstr()]

    for f, n in zip(target, [x for x, y in env["FUNCTIONS"]]):
        fd = csv.writer(meta_open(f.rstr(), "w"), delimiter="\t")
        fd.writerow(["SUBJECT"] + [f2s(x.rstr()) for x in setB])    
        for a in setA:
            fd.writerow([f2s(a.rstr())] + [dists[frozenset([f2s(x.rstr()), f2s(a.rstr())])][n] for x in setB])
    return None

def freqs_to_tab(target, source, env):
    items = {}
    data = {}
    tempdata = {}
    totals = {}
    avs = {}
    for f in source[0].children():
        subject = os.path.basename(f.rstr()).split("_")[0]
        tempdata[subject] = {}
        for r in csv.reader(meta_open(f.rstr()), delimiter="\t"):
            if r[1] == "COUNT":
                continue
            tempdata[subject][r[0]] = r[1]
            totals[subject] = totals.get(subject, 0) + int(r[1])
            items[r[0]] = items.get(r[0], 0) + int(r[1])
    for s, wordcounts in tempdata.iteritems():
        data[s] = {}
        for w, c in wordcounts.iteritems():
            data[s][w] = float(tempdata[s].get(w, 0)) / float(totals[s])
            avs[w] = avs.get(w, []) + [data[s][w]]
    #items = sorted([x for x, y in items.iteritems() if y > 20000 and not re.match("^.*\W.*$", x)])
    items = sorted([x for x, y in items.iteritems()], lambda x, y : cmp(sum(avs[y]), sum(avs[x])))[0:min(2000, len(items))]
    fd = csv.writer(meta_open(target[0].rstr(), "w"), delimiter="\t")
    fd.writerow(["SUBJECT"] + ["'%s'" % esc_weka(x) for x in items])
    for k, v in data.iteritems():
        fd.writerow(["'%s'" % esc_weka(k)] + [v.get(x, 0) for x in items])
    return None

def plot_js(target, source, env):
    data = []
    fr, to = [env["FROM"], env["TO"]]
    for f in source[0].children():
        fd = csv.reader(meta_open(f.rstr(), "r"), delimiter="\t")
        header = fd.next()
        for l in fd:
            if l[0] == fr:
                data.append(float(l[header.index(to)]))

    pyplot.plot(data)
    #pyplot.axes().set_ybound(0, 1)

    pyplot.savefig(target[0].rstr(), dpi=100)
    pyplot.cla()
    return None



def summarize_stats(target, source, env):
    variables = {}
    observations = []
    arff = weka.Arff()
    for f in source[0].children():
        obs = {}
        m = re.match("^PROCESSED (\d+) FILES: \n\t(.*?)\n(\d+) WORDS.*?(\d+) SENTENCES\n\nVERB FREQUENCIES\n(.*)$", 
                     gzip.open(f.rstr()).read().strip(), re.S)
        for a, b, c in [x.split("\t") for x in m.group(5).split("\n")]:
            variables[a] = variables.get(a, 0) + int(b)
            obs[a] = int(b)
        arff.add_datum(obs)
    arff.save(gzip.open(target[0].rstr(), "w"))
    return None

#import xlwt
def stats_to_spreadsheet(target, source, env):
    wb = xlwt.Workbook()
    for f in source[0].children():
        sheet = wb.add_sheet(os.path.basename(f.rstr()).split(".")[0])
        for r, row in enumerate(csv.reader(meta_open(f.rstr(), "r"), delimiter="\t")):
            for c, val in enumerate(row):
                if c == 0 or r == 0:
                    val = os.path.basename(val).split(".")[0].split("_")[0]
                sheet.write(r, c, val)
    wb.save(target[0].rstr())
    return None

def genia(target, source, env):
    geniatagger = xmlrpclib.ServerProxy(env["GENIA_URL"])
    text = re.sub("\S{40,}", "XXXXX", gzip.open(source[0].rstr()).read())
    fd = gzip.open(target[0].rstr(), "w")
    tagged = geniatagger.genia(text)
    fd.write(tagged.encode("utf-8"))
    return None


def rasp(target, source, env):    
    sents = gzip.open(source[0].rstr()).read()
    p = Popen([os.path.join(env["RASP_PATH"], "scripts", "rasp.sh")], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate(sents)
    gzip.open(target[0].rstr(), "w").write(out)
    return None


def rasp_to_scf(target, source, env):
    text = re.sub("gr-list: (\d+)\s*?\n", "", gzip.open(source[0].rstr()).read())
    sents = re.split("\n\s*\n", text)
    fd = gzip.open(target[0].rstr(), "w")
    for sent in [x for x in sents if len(x.split("\n")) < 20]:
        p = Popen([os.path.join(env["ALISP_PATH"], "alisp"), "-I", "bin/hierarchy.dxl"], stdin=PIPE,stdout=PIPE, stderr=PIPE)
        out, err = p.communicate(sent + "\n\n")        
        fd.write(out + sent.split("\n")[0].strip() + "***\n\n")
    return None


def old_candc(target, source, env):
    sents = gzip.open(source[0].rstr()).read()
    proxy = SOAPProxy(env["CANDC_URL"], "urn:candc-ccg", throw_faults=0)
    fd = gzip.open(target[0].rstr(), "w")
    try:
        parsed = proxy.parse_string(sents.decode("utf-8"))
        fd.write(parsed.encode("utf-8"))
    except SAXParseException:
        for l in gzip.open(source[0].rstr()):
            try:
                parsed = proxy.parse_string(l.decode("utf-8"))
                fd.write(parsed.encode("utf-8"))
            except SAXParseException:
                continue
    return None

#import candc
#import re
#parser = candc.CCGParser(models="~/models")
def candc(target, source, env):
    sents = [x for x in gzip.open(source[0].rstr()).read().split("\n") if not re.match("^\s*$", x)]
    fd = gzip.open(target[0].rstr(), "w")
    for s in sents:
        if len(s.split()) < 100:
            out = parser.parse(s)            
            fd.write(str(out) + "\n")
    fd.close()
    return None




def convert_pubmed_to_tei(target, source, env):
    stylesheet = source[0].rstr()
    fd = gzip.open(target[0].rstr(), 'w')
    fd.write(
        """<TEI><teiHeader type="source"><fileDesc><titleStmt><title>Biomed Corpus</title><author>Various</author>
        </titleStmt><publicationStmt><date>2009</date></publicationStmt><sourceDesc><p/></sourceDesc></fileDesc>
        <profileDesc><langUsage><language ident="en" usage="100">English</language></langUsage></profileDesc>
        </teiHeader><text><front/><body>""")
    for fname in random.sample(glob.glob("%s/*/*nxml" % env['DIR']), env["ARTICLES"]):
        text = et.tostring(et.parse(open(fname)).getroot())
        proc = Popen(["java",
                                 "-Xmx%s" % env["MAX_MEM"],
                                 "-jar",
                                 "bin/saxon9.jar",
                                 "-xsl:%s" % stylesheet,
                                 "-s:-",
                                 ], stdin=PIPE, stdout=PIPE)
        proc.stdin.write(re.sub("""<!DOCTYPE.*?>""", "", text, re.S))
        proc.stdin.close()
        fd.write(proc.stdout.read())
    fd.write("""</body><back/></text></TEI>""")
    fd.close()
    return None

def create_tagger(target, source, env):
    from nltk.corpus import brown
    import nltk
    sents = brown.tagged_sents()[0:100000]
    t1 = nltk.DefaultTagger("??")
    t2 = nltk.UnigramTagger(sents, backoff=t1)
    t3 = nltk.BigramTagger(sents, backoff=t2)
    t4 = nltk.TrigramTagger(sents, backoff=t3)
    cPickle.dump(t4, open(target[0].rstr(), 'wb'))
    return None


def create_instance_tab(target, source, env):
    args = env.Dictionary()
    files = glob.glob("%s/*/*nxml" % source[0].rstr())
    files = random.sample(files, min(env.get("COUNT", len(files)), len(files)))
    articles = []
    all_words = set()
    pats = dict([("N", "n"), ("J", "a"), ("R", "r")] + [(x, "v") for x in ["B", "V", "H", "D"]])
    all_cats = set()
    tagger = cPickle.load(open(source[1].rstr()))
    mapping = cPickle.load(open(source[2].rstr()))
    for fname, i in zip(files, range(len(files))):
        #if i % 500 == 0:
        #    print i
        tree = et.parse(fname)
        try:
            cat = esc_weka(mapping[[x for x in tree.getiterator("journal-id") if x.get("journal-id-type", None) == "nlm-ta"][0].text])
        except:
            continue
        all_cats.add(cat)
        article = {}            
        for text in [x.text for x in tree.getiterator("p") if x.text]:
            for w, t in [x for x in tagger.tag(text.split())]:
                w = esc_weka(w.strip(".:;'\"!,(){}[]").encode('utf-8'))
                stemmed = False
                if t[0] in pats:
                    stemmed = wordnet.morphy(w, pats[t[0]])
                if not stemmed and not args.get("INCLUDE_UNSTEMMED", False):
                    continue
                elif not stemmed and args.get("INCLUDE_UNSTEMMED", False):
                    w = "_".join([t, w])
                else:
                    w = "_".join([t, stemmed])
                try:
                    all_words.add(w.encode("utf-8"))
                    #all_words.add("POS_%s" % t)
                    article[w] = article.get(w, 0) + 1
                    #article["POS_%s" % t] = article.get("POS_%s" % t, 0) + 1
                except:
                    print "couldn't add %s" % w
        articles.append((cat, article, esc_weka(os.path.basename(fname))))


    print "%d articles, %d categories, %d lemmas" % (len(articles), len(all_cats), len(all_words))
    fd = csv.writer(gzip.open(target[0].rstr(), "w"), delimiter="\t")
    all_cats = list(all_cats)
    #fd.writerow(["CATEGORY", "FILE"] + [x.encode('utf-8') for x in all_words])
    fd.writerow(["CATEGORY"] + [x.encode('utf-8') for x in all_words])

    for cat, counts, fname in articles:
        #fd.writerow([w.decode('utf-8')] + [str(vals.get(x, 0) + 1) for x in all_words])
        #total_words = sum(vals.itervalues())
        #fd.writerow([cat, fname] + [str(counts.get(x, 0)) for x in all_words])
        fd.writerow([cat] + [str(counts.get(x, 0)) for x in all_words])
    return None

def create_verbframe_tab(target, source, env):
    args = env.Dictionary()
    files = glob.glob("%s/*/*nxml" % source[0].rstr())
    files = random.sample(files, min(env.get("COUNT", len(files)), len(files)))
    articles = []
    all_words = {}
    all_features = {}
    pats = dict([("N", "n"), ("J", "a"), ("R", "r")] + [(x, "v") for x in ["B", "V", "H", "D"]])
    all_cats = set()
    tagger = cPickle.load(open(source[1].rstr(), 'rb'))
    mapping = cPickle.load(open(source[2].rstr()))
    for fname, i in zip(files, range(len(files))):
        #if i % 500 == 0:
        #    print i
        tree = et.parse(fname)
        try:
            cat = esc_weka(mapping[[x for x in tree.getiterator("journal-id") if x.get("journal-id-type", None) == "nlm-ta"][0].text])
        except:
            continue
        all_cats.add(cat)
        article = {}            
        for text in [x.text for x in tree.getiterator("p") if x.text]:
            tagged = [x for x in tagger.tag(text.split())]
            for w, i in zip(tagged, range(len(tagged))):
                if w[1].startswith("V"):
                    stem = wordnet.morphy(w[0], "v")
                    if not stem:
                        continue
                    else:
                        stem = esc_weka(stem)
                    f = "_".join([z[1] for z in [(stem, stem)] + tagged[max([0, i - args["WINDOW"]]):i] + [("VERB", "VERB")] + tagged[i + 1:min([len(tagged), i + 1 + args["WINDOW"]])]])
                    f = esc_weka(f.strip(".:;'\"!,(){}[]").encode('utf-8'))
                    if "?" in f:
                        continue
                    try:
                        all_words[stem] = all_words.get(stem, 0) + 1
                        all_features[f.encode("utf-8")] = stem
                                                                     #set([cat])).union(set([cat]))
                        article[f] = article.get(f, 0) + 1
                    except:
                        print "couldn't add %s" % f
        articles.append((cat, article, esc_weka(os.path.basename(fname))))



    fd = csv.writer(gzip.open(target[0].rstr(), "w"), delimiter="\t")
    all_cats = list(all_cats)
    kept_words = [p[0] for p in sorted(all_words.iteritems(), lambda x, y : cmp(y[1], x[1]))][0:args["KEEP"]]
    #print kept_words
    kept_features = [p[0] for p in all_features.iteritems() if p[1] in kept_words]
    fd.writerow(["CATEGORY"] + [x.encode('utf-8') for x in kept_features])
    print "%d articles, %d categories, keeping %d/%d features" % (len(articles), len(all_cats), len(kept_features), len(all_features))
    for cat, counts, fname in articles:
        #fd.writerow([w.decode('utf-8')] + [str(vals.get(x, 0) + 1) for x in all_words])
        total_words = sum(counts.itervalues())
        if total_words != 0:
            fd.writerow([cat] + [str(counts.get(x, 0)) for x in kept_features])
    return None


def create_genia_tab(target, source, env):
    args = env.Dictionary()
    mapping = {}
    data = {}
    features = set()
    for i, fname in zip(range(len(source[1:])), sorted([x.rstr() for x in source[1:]])):
        journal = os.path.basename(fname)[0:-3]
        print i, journal
        data[journal] = {"JOURNAL" : journal}        
        for tok, lem, pos, ch, ne in [x.split() for x in gzip.open(fname) if len(x.split()) == 5]:
            lem = "LEM_%s" % lem
            pos = "POS_%s" % pos
            ne = "NE_%s" % ne
            for f in [lem, pos, ne]:
                features.add(f)
                data[journal][f] = data[journal].get(f, 0) + 1
    fd = csv.writer(gzip.open(target[0].rstr(), "w"))
    fd.writerow(["JOURNAL", "SUBJECT"] + [x for x in sorted(features)])
    for k, v in data.iteritems():
        fd.writerow([k] + [v.get(x, 0) for x in sorted(features)])
    return None


def create_verbframe_tab_ccg(target, source, env):
    args = env.Dictionary()
    files = glob.glob("%s/*.txt.gz" % source[0].rstr())
    files = random.sample(files, min(env.get("COUNT", len(files)), len(files)))
    mapping = {}
    data = {}
    journal_data = {}
    subdomain_data = {}
    all_lemmas = {}
    all_features = set()
    for k, v in cPickle.load(open(source[1].rstr())).iteritems():
        mapping[k.lower().replace(" ", "_")] = v
    counter = 0
    for fname in files:
        journal = os.path.basename(fname)[0:-14]        
        
        for subdomain in mapping.get(journal, ["NONE"]):

            temp_subdomain = "NONE"
            if subdomain:
                subdomain = esc_weka(subdomain)
                subdomain_data[subdomain] = subdomain_data.get(subdomain, {})
                temp_subdomain = subdomain
            #if not subdomain in subdomain_data:
            #    subdomain_data[subdomain] = {}       
            journal_data[journal] = journal_data.get(journal, {"SUBDOMAIN" : temp_subdomain})
            for line in [x for x in gzip.open(fname) if x.startswith("<c>")]:
                counter += 1
                if counter % 50000 == 0:
                    print counter
                words = [dict(zip(["token", "lemma", "pos", "chunk", "ne", "category"], w.split("|"))) for w in line.split()]
                for w in [x for x in words if x.get("pos", "A").startswith("V")]:
                    verb = re.sub("\W", "_", w["lemma"])
                    verbcat = "_".join(["VERB(%s)" % verb, "CCG(%s)" % w["category"]])
                    if subdomain:
                        subdomain_data[subdomain][verbcat] = subdomain_data[subdomain].get(verbcat, 0) + 1
                    journal_data[journal][verbcat] = journal_data[journal].get(verbcat, 0) + 1
                    all_features.add(verbcat)
                    all_lemmas[verb] = all_lemmas.get(verb, 0) + 1

    kept_lemmas = [p[0] for p in sorted(all_lemmas.iteritems(), lambda x, y: cmp(y[1], x[1]))][0:args["KEEP"]]
    kept_features = [x for x in all_features if re.match("^VERB\((.*)\)_.*$", x).group(1) in kept_lemmas]
    journal_fd = csv.writer(gzip.open(target[1].rstr(), "w"), delimiter="\t")
    journal_fd.writerow(["JOURNAL", "CATEGORY"] + [x for x in kept_features])
    subdomain_fd = csv.writer(gzip.open(target[0].rstr(), "w"), delimiter="\t")
    subdomain_fd.writerow(["CATEGORY"] + [x for x in kept_features])

    print "%d categories, keeping %d verbs, %d of %d CCG categories" % (len(subdomain_data), len(kept_lemmas), len(kept_features), len(all_features))
    for k, v in subdomain_data.iteritems():
        subdomain_fd.writerow([k] + [v.get(x, 0) + 1 for x in kept_features])
    for k, v in journal_data.iteritems():
        journal_fd.writerow([k, v.get("SUBDOMAIN", "NONE")] + [v.get(x, 0) + 1 for x in kept_features])
    return None


def scf_sents(target, source, env):
    fd = meta_open(target[0].rstr(), "w")
    regexes = [re.compile("[^V]*%s\S+V.*" % x) for x in source[0].read()]
    for i, f in enumerate(env["FILES"]):
        logging.info("%d %s", i, f)
        for i, s in enumerate(re.split("\n\s*\n", meta_open(f).read())):
            if any([x.match(s) for x in regexes]):
                fd.write("%s %d\n" % (f, i) + s.strip() + "\n\n\n")
    return None

def scf_task(target, source, env):
    """
    Takes C&C output and a list of verbs and creates a manual annotation task for potential SCF frames.
    """
    rasp = ""
    examples = {}
    tin = tempfile.mkstemp(prefix="rasp")
    tout = tempfile.mkstemp(prefix="scf")
    items = dict([(x, []) for x in env["SCF_VERBS"]])
    for s in random.sample(list(set(re.split("\n\s*\n", meta_open(source[0].rstr()).read()))), 150000):
        try:
            f, n = re.match("^(.*?)\n.*$", s, re.S).group(1).split()
        except:
            continue
        s = re.match("^.*?\n(.*)$", s, re.S).group(1)
        for k in env["SCF_VERBS"]:
            if re.match(".*%s.*" % k, s):
                items[k].append({"sentence" : s.strip().replace("gr-list: 1", ""), "file" : f, "number" : n})
    for k, v in items.iteritems():
        logging.info("%s %d", k, len(v))


    for k, v in items.iteritems():
        logging.info("VERB: %s", k)
        for i in random.sample(v, min(env["SENTENCES_PER_VERB"], len(v))):
            logging.info("\t%d", len(examples.get(k, [])))
            open(tin[1], "w").write("%s\n\n\n" % i["sentence"])        
            p = Popen(["/bin/sh", "%s/runLexiconBuilder.sh" % env["PATRULES_PATH"], 
                       "-i", tin[1], "-o", tout[1]], cwd=env["PATRULES_PATH"], stderr=PIPE)
            p.communicate()
            try:
                tree = et.parse(tout[1])
            except:
                continue
            for e in [x for x in tree.getiterator("entry") if any([re.match(y, x.get("target", None)) for y in env["SCF_VERBS"]])]:
                examples[k] = examples.get(k, []) + [{"sentence" : [(re.match("^%s.*$" % k, x.lower()), x.encode("utf-8")) for x in list(e.getiterator("sentence"))[0].text.split()], "subcats" : list(e.getiterator("instance"))[0].get("classnum", "").split("V"), "sfile" : i["file"], "snum" : i["number"]}]
    tb = et.TreeBuilder()
    tb.start("xml", {})
    n = 1
    for verb, items in examples.iteritems():
        tb.start("verb", {})
        tb.start("lemma", {}), tb.data(verb), tb.end("lemma")
        for i in random.sample(items, min(len(items), env.get("SENTENCES_PER_VERB", len(items)))):
            tb.start("item", {"n" : str(n)})
            tb.start("sentence", {"id" : "%s_%s" % (os.path.basename(i["sfile"]), i["snum"])})
            for t, w in i["sentence"]:
                if t:
                    tb.start("token", {"target" : "true"})
                else:
                    tb.start("token", {"target" : "false"})
                p = Popen(["%s/morph/morphg.x86_64_linux" % env["RASP_PATH"], "-c"], cwd="%s/morph" % env["RASP_PATH"], stderr=PIPE, stdin=PIPE, stdout=PIPE)
                out, err = p.communicate(re.sub("\:\d*_", "_", w))
                tb.data(out.decode("utf-8"))
                #tb.data(w.decode("utf-8"))
                tb.end("token")
            tb.end("sentence")
            tb.start("subcats", {})
            for sc in i["subcats"]:
                tb.start("subcat", {})
                tb.data(sc)
                tb.end("subcat")
            tb.end("subcats")
            tb.end("item")
            n += 1
        tb.end("verb")
    tb.end("xml")
    open(target[0].rstr(), "w").write(et.tostring(tb.close()))
    return None


def get_subsamples(target, source, env):
    args = source[-1].read()
    sentences = [m.group(1) + m.group(2) for m in re.finditer("\n\s*\n(.*?)(^<c>.*?)$", meta_open(source[0].rstr()).read(), re.M | re.S)]
    assignments = [0 for i in range(len(sentences) / (2 * args["WINDOW"]))] + [1 for i in range(len(sentences) / (2 * args["WINDOW"]))]
    for t in target:
        fd = meta_open(t.rstr(), "w")
        random.shuffle(assignments)
        for i, m in enumerate(assignments):
            if m == 1:
                fd.write("\n\n".join(sentences[i * args["WINDOW"] : min((i + 1) * args["WINDOW"], len(sentences))]).strip() + "\n\n")
    return None


def TOOLS_ADD(env):
    env.Append(BUILDERS = {#'PubmedToTEI' : Builder(action = convert_pubmed_to_tei),
                           #'PubmedToXrff' : Builder(action = pubmed_to_xrff),
                           #'CreateCounts' : Builder(action = create_counts),
                           #'ClusterItems' : Builder(action = cluster_items),
                           #'CatCsv' : cat_csv,
        #'CountFilter' : Builder(action = filter_word_counts),
        'Genia' : Builder(action = genia),
        'Rasp' : Builder(action = rasp),
        'CandC' : Builder(action = old_candc),
                           #'Tagger' : Builder(action = create_tagger),
                           #'InstanceTab' : Builder(action = create_instance_tab),
                           #'VerbframeTab' : Builder(action = create_verbframe_tab),
                           #'VerbframeTabCCG' : Builder(action = create_verbframe_tab_ccg),
                           #'SplitVerbframes' : Builder(action = split_verbframes, emitter = split_verbframes_emitter),
                           #'GeniaTab' : Builder(action = create_genia_tab),
        'XmlToText' : Builder(action = xml_to_text),
        'TextToSentences' : Builder(action = text_to_sentences),
                           #'RaspToSCF' : Builder(action = rasp_to_scf),
        'DocumentStats' : Builder(action = document_stats),
        'AggregateStats' : Builder(action = aggregate_stats),
        'SummarizeStats' : Builder(action = summarize_stats),
        'SubjectOverview' : Builder(action = subject_overview),
        'SCFSents' : Builder(action = scf_sents),
        'SCFTask' : Builder(action = scf_task),
        'GeniaArff' : Builder(action = genia_arff),
        'OtherArff' : Builder(action = other_arff),
        'CandCArff' : Builder(action = candc_arff),
        'ComputePairwise' : Builder(action = compute_pairwise),
        'StatsToSpreadsheet' : Builder(action = stats_to_spreadsheet),
        'PlotJS' : Builder(action = plot_js),
        'FreqsToTab' : Builder(action = freqs_to_tab),
        'GetAbstracts' : Builder(action = "python bin/get_abstracts.py -s ${SOURCES[0]} -S ${SOURCES[1]} -o ${TARGETS[0]} -d data/pubmed"),
        'Subsample' : Builder(action=get_subsamples),
        })


    
