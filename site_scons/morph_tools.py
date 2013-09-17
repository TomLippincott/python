"""
Create stand-off annotation with morphological information.

A morphological tool provides a function that takes a token
and returns a dictionary of attributes describing it.

Or, a list of probability-dictionary pairs.
"""
from SCons.Node.Python import Value
from SCons.Builder import Builder
from SCons.Script import FindFile
from proteus import schone, goldsmith, gaussier, Proteus
import cPickle
import re
#from custom.latin import inflection, words_inflection
#import libxml2
from tei import TeiDocument
from xml.etree import cElementTree as et
import xml.sax
import gzip
from common_tools import meta_open, meta_basename

class MorphTransform(xml.sax.handler.ContentHandler):
    def __init__(self, fname, handler):
        self.handler = handler
        self.fd = open(fname, 'w')
        self.write = True

    def startElement(self, name, attrs):
        if name == "w":
            self.write = False
        elif self.write == True:            
            self.fd.write("<%s %s>\n" % (name, " ".join(["%s=\"%s\"" % (x, attrs[x]) for x in attrs.getNames()])))

    def endElement(self, name):
        if name == "w":
            self.write = True
        elif self.write == True:
            self.fd.write("</%s>\n" % name)

    def characters(self, content):
        if self.write == True:
            self.fd.write("%s" % content.strip())
        elif len(content.strip()) > 0:
            self.fd.write(self.handler(content))

    def endDocument(self):
        self.fd.close()


def collatinus_morph(target, source, env):
    def handler(w):
        choices = inflection(w)
        if len(choices) == 0 and ('j' in w or 'J' in w):
            choices = inflection(w.replace('j', 'i').replace('J', 'I'))
        if len(choices) == 0 and ('v' in w or 'V' in w):
            choices = inflection(w.replace('v', 'u').replace('V', 'U'))
        e = et.Element("fs")
        #e.text = ''
        #e.tag = "fs"
        #s_f = et.SubElement(e, 'f', {'name' : 'surface'})
        #et.SubElement(s_f, 'string', {}).text = surf
        if len(choices) == 1:
            f_morph = et.SubElement(e, 'f', {'name' : 'morph'})
            fs_coll = et.SubElement(f_morph, 'fs', {'type' : 'collatinus'})
            for k, v in choices[0].iteritems():
                f = et.SubElement(fs_coll, 'f', {'name' : k})
                sym = et.SubElement(f, 'symbol', {'value' : v})
        elif len(choices) > 1:
            f_morph = et.SubElement(e, 'f', {'name' : 'morph'})               
            valt = et.SubElement(f_morph, 'vAlt', {})
            for choice in choices:
                fs_coll = et.SubElement(valt, 'fs', {'type' : 'collatinus'})
                for k, v in choice.iteritems():
                    f = et.SubElement(fs_coll, 'f', {'name' : k})
                    sym = et.SubElement(f, 'symbol', {'value' : v})
        else:
            return ''
            #all_values[surf.lower()] = all_values.get(surf.lower(), 0) + 1
        return et.tostring(e)
    
    xml.sax.parse(source[0].rstr(), MorphTransform(fname=target[0].rstr(), handler=handler))
    return None


    all_values = {}
    doc = et.parse(source[0].rstr())
    for p in doc.getiterator("p"):
        for e in p.getiterator("w"):
            if not e.get("type", None) == "token":
                p.remove(e)
        for e in p.getiterator("w"):
            if e.get("type", None) == "token":
                choices = inflection(e.text)
                if len(choices) == 0 and ('j' in e.text or 'J' in e.text):
                    choices = inflection(e.text.replace('j', 'i').replace('J', 'I'))
                if len(choices) == 0 and ('v' in e.text or 'V' in e.text):
                    choices = inflection(e.text.replace('v', 'u').replace('V', 'U'))
                #all_values += choices
                surf = e.text
                e.text = ''
                e.tag = "fs"
                #s_f = et.SubElement(e, 'f', {'name' : 'surface'})
                #et.SubElement(s_f, 'string', {}).text = surf
                if len(choices) == 1:
                    f_morph = et.SubElement(e, 'f', {'name' : 'morph'})
                    fs_coll = et.SubElement(f_morph, 'fs', {'type' : 'collatinus'})
                    for k, v in choices[0].iteritems():
                        f = et.SubElement(fs_coll, 'f', {'name' : k})
                        sym = et.SubElement(f, 'symbol', {'value' : v})
                elif len(choices) > 1:
                    f_morph = et.SubElement(e, 'f', {'name' : 'morph'})               
                    valt = et.SubElement(f_morph, 'vAlt', {})
                    for choice in choices:
                        fs_coll = et.SubElement(valt, 'fs', {'type' : 'collatinus'})
                        for k, v in choice.iteritems():
                            f = et.SubElement(fs_coll, 'f', {'name' : k})
                            sym = et.SubElement(f, 'symbol', {'value' : v})
                else:
                    all_values[surf.lower()] = all_values.get(surf.lower(), 0) + 1

    #cPickle.dump(all_values, open("test.pkl", 'wb'))
    doc.write(target[0].rstr())
    return None


def words_morph(target, source, env):
    all_values = {}
    doc = et.parse(source[0].rstr())
    for p in doc.getiterator("p"):
        for e in p.getiterator("w"):
            if not e.get("type", None) == "token":
                p.remove(e)
        for e in p.getiterator("w"):
            if e.get("type", None) == "token":
                choices = words_inflection(e.text)
                surf = e.text
                e.text = ''
                e.tag = "fs"
                et.SubElement(e, 'f', {'name' : 'surface', 'value' : surf})
                if len(choices) == 1:
                    f_morph = et.SubElement(e, 'f', {'name' : 'morph'})
                    fs_coll = et.SubElement(f_morph, 'fs', {'name' : 'collatinus'})
                    for k, v in choices[0].iteritems():
                        et.SubElement(fs_coll, 'f', {'name' : k, 'value' : v})
                elif len(choices) > 1:
                    valt = et.SubElement(e, 'vAlt', {})
                    for choice in choices:
                        f_morph = et.SubElement(valt, 'f', {'name' : 'morph'})
                        fs_coll = et.SubElement(f_morph, 'fs', {'name' : 'collatinus'})
                        for k, v in choice.iteritems():
                            et.SubElement(fs_coll, 'f', {'name' : k, 'value' : v})
                else:
                    all_values[surf.lower()] = all_values.get(surf.lower(), 0) + 1

    cPickle.dump(all_values, open("test.pkl", 'wb'))
    doc.write(target[0].rstr())
    return None


def schone_morph(target, source, env):
    docs = [x.split("\t")[-1].split() for x in source[0].get_contents().split("\n")]
    env_dict = env.Dictionary()
    top_words, m = schone.term_term_matrix(docs[0:env_dict.get('DOC_COUNT', 1000)],
                                           env_dict.get('WORD_COUNT', 1000), env_dict.get('WINDOW', 50))
    caffixes = schone.candidate_affixes(top_words)
    rule_sets = schone.rule_sets(caffixes)
    mapping = {}
    counter = 0
    for terms, rule in rule_sets:
        counter += 1
        for word in ["%s%s" % (stem, suffix) for stem in terms for suffix in rule]:
            mapping[word] = counter
    cPickle.dump(mapping, open(target[0].rstr(), 'wb'))
    return None


def gaussier_morph(target, source, env):
    if isinstance(source[0], Value):
        words = source[0].read()
    else:
        words = set([x.text for x in et.parse(meta_open(source[0].rstr())).getiterator("f") if x.attrib["name"] == "morph" and x.text and len(x.text) > 4])
    
    pairs = gaussier.suffix_pairs(words, min_psimilarity=int(env["MIN_PSIM"]), min_occurrence=int(env["MIN_OCCURRENCES"]))
    print len(pairs)
    #if len(pairs) > 0:
    #    dv, words = gaussier.similarity_matrix(pairs)
        #print pairs
    #try:
    rel_fams = dict([(i, [r + s for s in x[0] for r in x[1]]) for i, x in enumerate(pairs.iteritems())])
    #gaussier.relational_families(dv, words)
    #        print rel_fams
    p = Proteus(fams=rel_fams,
                title="""Gaussier Morphology, min. p-similarity=%s, unique words=%s,
                min. occurrences=%s, clustering=(method=%s, threshold=%s)""" % (env["MIN_PSIM"],
                                                                                env["WORD_COUNT"],
                                                                                env["MIN_OCCURRENCES"],
                                                                                env["CLUSTERING_METHOD"],
                                                                                env["CLUSTERING_THRESHOLD"]))
        
    fd = meta_open(target[0].rstr(), 'w')
    fd.write(
        """<?xml version="1.0" encoding="utf-8"?>
        <?xml-stylesheet href="morphology.xsl" type="text/xsl"?>"""
        )
    p.write(fd)
    #except:
    #    pass
    return None



def goldsmith_morph(target, source, env):
    tokens = set([re.sub('\W', '', x.lower()) for x in source[0].get_contents().split()])
    env_dict = env.Dictionary()
    sfs = goldsmith.successor_freqs(tokens)
    initial_stems_affixes = [goldsmith.stem_affix_break(x, sfs) for x in tokens]
    sigs = goldsmith.signatures([x for x in initial_stems_affixes if len(x) > 1])
    filtered_sigs = goldsmith.filter_signatures(sigs)
    cPickle.dump(filtered_sigs, open(target[0].rstr(), 'wb'))
    return None


def gaussier_emitter(target, source, env):
    target[0] = "${BASE}/%s.xml" % meta_basename(source[0].rstr())
    return target, source

def TOOLS_ADD(env):
    env.Append(BUILDERS = {'Gaussier' : Builder(action = gaussier_morph, emitter=gaussier_emitter),
                           'Schone' :Builder(action = schone_morph),
                           'Goldsmith' : Builder(action = goldsmith_morph),
                           'Collatinus' : Builder(action = collatinus_morph),
                           'Words' : Builder(action = words_morph),
                           })
