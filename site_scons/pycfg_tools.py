from SCons.Script import Builder
from SCons.Subst import scons_subst
from SCons.Action import CommandAction, CommandGeneratorAction
import re
from glob import glob
import logging
import os.path
import os
import codecs
from os.path import join as pjoin
from common_tools import meta_open, DataSet, v_measure, regular_word, list_to_tuples
from itertools import product
from subprocess import Popen, PIPE
from scons_tools import make_generic_emitter, make_command_builder
from nltk.parse.viterbi import ViterbiParser
from nltk.grammar import PCFG, ProbabilisticProduction, Nonterminal


def format_rules(rules):
    """Utility function to convert a list of rules into a string that can be written as an input grammar file for py-cfg.

    Inputs: list of tuples representing py-cfg rules
    Outputs: py-cfg-formatted grammar
    """
    lines = []
    for rule in rules:
        targets = " ".join(rule[-1])
        if len(rule) == 2:
            lines.append("%s --> %s" % (rule[0], targets))
        elif len(rule) == 3:
            lines.append("%s %s --> %s" % (rule[0], rule[1], targets))
        elif len(rule) == 4:
            lines.append("%s %s %s --> %s" % (rule[0], rule[1], rule[2], targets))
    return "\n".join(lines).encode("utf-8")


def normalize_pycfg_output(target, source, env):
    """Turns py-cfg parsed output into something more human-readable.

    Sources: segmented py-cfg output file
    Targets: reformatted segmentations file
    """
    analyses = {}
    with meta_open(source[0].rstr()) as ifd:
        text = re.split(r"\n\s*?\n", ifd.read().strip())[-1]
        for line in text.split("\n"):
            toks = tuple(["".join([unichr(int(y.group(1).replace(" ", ""), base=16)) for y in re.finditer("\(\S+ ([^\)\(]+)\)", x)]) for x in re.split(r"\(Prefix|\(Stem|\(Suffix", line)[1:]])
            toks = tuple([t for t in toks if len(t) > 0])
            word = "".join(toks)
            if len(toks) > 1:
                toks = tuple(["%s+" % (toks[0])] + ["+%s+" % (t) for t in toks[1:-1]] + ["+%s" % (toks[-1])])
            if not re.match(r"^\s*$", word) and "_" not in word:
                analyses[word] = analyses.get(word, []) + [toks]
    with meta_open(target[0].rstr(), "w") as ofd:
        for w, aa in sorted(analyses.iteritems()):
            ofd.write("%s\n" % (" ".join(aa[0])))
    return None


def character_productions(target, source, env):
    """Creates the portion of a grammar that handles the terminal character productions.

    Covers every non-whitespace character in the input file, except for the specified non-acoustic graphemes.
    Basically, just creates rules like "Char --> 'a'", "Char --> 'b'", etc.

    Sources: word list 1, word list 2, ...
    Targets: character grammar fragment
    """
    nag = [unichr(int(x, base=16)) for x in env.get("NON_ACOUSTIC_GRAPHEMES")]
    nag = ["%.4x" % (ord(x)) for x in nag]
    characters = set()
    for f in source:
        with meta_open(f.rstr()) as ifd:
            for l in ifd:
                for c in l:
                    if not re.match(r"\s", c):
                        characters.add(c)
    characters = ["%.4x" % (ord(x)) for x in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        for c in characters:
            ofd.write("0 1 Char --> %s\n" % (c))
    return None


def compose_grammars(target, source, env):
    """Combine grammar fragments into a single grammar.

    Right now this is just concatenation without even validation, but the door is open
    to much fancier techniques.

    Sources: fragment file 1, fragment file 2 ...
    Targets: composed grammar file
    """
    with meta_open(target[0].rstr(), "w") as ofd:
        for s in source:
            with meta_open(s.rstr()) as ifd:
                for l in ifd:
                    if re.match(r"^\s*$", l) or l.strip().startswith("#"):
                        continue
                    elif "|" in l:
                        lhs, rhs = re.match(r"^(.*) --> (.*)$", l.strip(), re.UNICODE).groups()
                        for r in rhs.split("|"):
                            x = ("%s --> %s\n" % (lhs, " ".join([c for c in r.strip()]))).encode("utf-8")
                            ofd.write(x)
                    else:
                        ofd.write(l.strip() + "\n")
    return None


def morphology_data(target, source, env):
    """Converts a list of words into the py-cfg data format.

    The input to py-cfg morphological models is each word as a sequence of space-separated graphemes,
    with special start ("^^^") and end ("$$$") symbols, one word per line.  Graphemes with no
    acoustic realizations (according to IBM) are ignored.

    Sources: word list file
    Targets: py-cfg data file
    """
    nag = [unichr(int(x, base=16)) for x in env.get("NON_ACOUSTIC_GRAPHEMES")]
    nag = ["%.4x" % (ord(x)) for x in nag]
    words = set()
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            tok = line.strip().split()[0]
            for word in tok.split("-"):
                if "_" not in word and "<" not in word and len(word) > 0:
                    words.add(word)
    with meta_open(target[0].rstr(), "w") as ofd:
        text = "\n".join([" ".join(["^^^"] + ["%.4x" % (ord(c)) for c in w] + ["$$$"]) for w in words])
        ofd.write(text)
    return None


def pycfg_generator(target, source, env, for_signature):
    """Runs an appropriate command-line invocation of py-cfg based on the sources and variables.

    Sources: py-cfg input grammar, py-cfg input data
    Targets: parsed data file, trained py-cfg model, debugging trace file
    """
    if source[1].rstr().endswith("gz"):
        cat = "zcat"
    else:
        cat = "cat"
    if len(source) == 2:
        return "%s ${SOURCES[1]}|${PYCFG_PATH}/py-cfg ${SOURCES[0]} -w 0.1 -N ${NUM_SAMPLES} -d 100 -E -n ${NUM_ITERATIONS} -e 1 -f 1 -g 10 -h 0.1 -T ${ANNEAL_INITIAL} -t ${ANNEAL_FINAL} -m ${ANNEAL_ITERATIONS} -A ${TARGETS[0]} -G ${TARGETS[1]} -F ${TARGETS[2]} 2> /dev/null" % cat
    elif len(source) == 3:
        return "%s ${SOURCES[1]}|${PYCFG_PATH}/py-cfg ${SOURCES[0]} -w 0.1 -N ${NUM_SAMPLES} -d 100 -E -n ${NUM_ITERATIONS} -e 1 -f 1 -g 10 -h 0.1 -T ${ANNEAL_INITIAL} -t ${ANNEAL_FINAL} -m ${ANNEAL_ITERATIONS} -A ${TARGETS[0]} -G ${TARGETS[1]} -F ${TARGETS[2]} -u ${SOURCES[2]} -U 'tee > ${TARGETS[3]}' -x ${NUM_ITERATIONS} 2> /dev/null" % cat
    elif len(source) == 4:
        return "%s ${SOURCES[1]}|${PYCFG_PATH}/py-cfg ${SOURCES[0]} -w 0.1 -N ${NUM_SAMPLES} -d 100 -E -n ${NUM_ITERATIONS} -e 1 -f 1 -g 10 -h 0.1 -T ${ANNEAL_INITIAL} -t ${ANNEAL_FINAL} -m ${ANNEAL_ITERATIONS} -A ${TARGETS[0]} -G ${TARGETS[1]} -F ${TARGETS[2]} -u ${SOURCES[2]} -U 'tee > ${TARGETS[3]}' -v ${SOURCES[3]} -V 'tee > ${TARGETS[4]}' -x ${NUM_ITERATIONS} 2> /dev/null" % cat


def apply_adaptor_grammar(target, source, env):
    """Apply an existing adaptor grammar model to new data.

    One of py-cfg's outputs is essentially a PCFG: this builder formats this a bit, then
    loads it as an NLTK PCFG, which is then applied to the provided word list to get new
    segmentations.  Note: the NLTK implementation is very slow, you may want to look into
    using one of Mark Johnson's other code bases, "cky.tbz", which is very fast and accepts
    a similar format to the py-cfg output.

    Sources: py-cfg grammar file, word list
    Targets: segmented word list
    """
    rules = {}
    nonterminals = set()
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            m = re.match(r"^(\S+)\s(\S+) --> (.*)$", line)
            if m:
                count = float(m.group(1))
                lhs = m.group(2)
                nonterminals.add(lhs)
                rhs = tuple(m.group(3).strip().split())
                rules[lhs] = rules.get(lhs, {})
                rules[lhs][rhs] = count
            else:
                m = re.match(r"^\((\S+)#\d+ (.*)$", line)
                lhs = m.group(1)
                rhs = tuple(re.sub(r"\(\S+", "", m.group(2)).replace(")", "").strip().split())
                rules[lhs] = rules.get(lhs, {})
                rules[lhs][rhs] = rules[lhs].get(rhs, 0) + 1
    productions = []
    for lhs, rhss in rules.iteritems():
        total = sum(rhss.values())
        for rhs, c in rhss.iteritems():
            mrhs = []
            for x in rhs:
                if x in nonterminals:
                    mrhs.append(Nonterminal(x))
                else:
                    mrhs.append(x)
            productions.append(ProbabilisticProduction(Nonterminal(lhs), mrhs, prob=(float(c) / total)))
    pcfg = PCFG(Nonterminal("Word"), productions)
    parser = ViterbiParser(pcfg)
    with meta_open(source[1].rstr()) as ifd:
        items = [l.strip().split() for l in ifd]
    with meta_open(target[0].rstr(), "w") as ofd:
        parsed = parser.parse_sents(items)
        for tree in [x.next() for x in parsed]:
            toks = [z for z in ["".join([unichr(int(y, base=16)) for y in x.leaves() if y not in ["^^^", "$$$"]]) for x in tree] if z != ""]
            if len(toks) == 1:
                ofd.write("%s\n" % (toks[0]))
            else:
                ofd.write(" ".join(["%s+" % toks[0]] + ["+%s+" % x for x in toks[1:-1]] + ["+%s" % toks[-1]]) + "\n")
    return None


def TOOLS_ADD(env):
    """Conventional way to add the four builders and the RunASR method to an SCons environment."""
    env.Append(BUILDERS = {
        "MorphologyData" : Builder(action=morphology_data),
        "ComposeGrammars" : Builder(action=compose_grammars),
        "CharacterProductions" : Builder(action=character_productions),
        "RunPYCFG" : Builder(generator=pycfg_generator),
        "NormalizePYCFGOutput" : Builder(action=normalize_pycfg_output),
        "ApplyAdaptorGrammar" : Builder(action=apply_adaptor_grammar),
    })
