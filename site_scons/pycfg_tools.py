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
from common_tools import meta_open, DataSet, v_measure, regular_word
from itertools import product
from subprocess import Popen, PIPE
from scons_tools import make_generic_emitter, make_command_builder
from nltk.parse.viterbi import ViterbiParser
from nltk.grammar import PCFG, ProbabilisticProduction, Nonterminal

def format_rules(rules):
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

def ngram_cfg(target, source, env):
    args = source[-1].read()
    n = args.get("n", 2)
    nulls = ["$$$" for i in range(n)]
    with meta_open(source[0].rstr()) as ifd:
        dataset = DataSet.from_stream(ifd)[0]
    if args.get("token_based", False):
        words = sum([[dataset.indexToWord[wid] for wid, tid, aids in s] for s in dataset.sentences], [])
    else:
        words = dataset.indexToWord.values()
    if args.get("lowercase", False):
        words = [nulls + [c for c in w.lower()] + nulls for w in words]
    else:
        words = [nulls + [c for c in w] + nulls for w in words]
    chars = set(sum(words, []))
    rules = [
        (1, "Word", ["$$$_$$$"]),
    ]
    for first in chars:
        for second in chars:
            #rules.append(("" % (first, second)
            pass
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(format_rules(rules))
    with meta_open(target[1].rstr(), "w") as ofd:
        ofd.write("\n".join([" ".join(w) for w in words]).encode("utf-8"))
    return None

def morphology_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = [word for word in sum([[data.indexToWord[w] for w, t, aa in s] for s in data.sentences], []) if regular_word(word)]
        if args.get("LOWER_CASE_MORPHOLOGY", False):
            words = [w.lower() for w in words]
        if args.get("TYPE_BASED", True):
            words = set(words)
        characters = set(sum([[c for c in w] for w in words], []))
    num_syntactic_classes = args.get("num_syntactic_classes", 1)
    rules = []
    rules += [
        (0, 1, "Word", ["Prefix", "Stem", "Suffix"]),
        ("Stem", ["Chars"]),
        ("Prefix", ["^^^"]),
        (0, 1, "Suffix", ["$$$"]),            
    ]

    if args["IS_AGGLUTINATIVE"]:
        internal = "Morphs"
        rules += [
            (0, 1, "Morphs", ["Morph"]),
            (0, 1, "Morphs", ["Morph", "Morphs"]),
            ("Morph", ["Chars"]),
        ]
    else:
        internal = "Chars"

    
    if args["HAS_PREFIXES"]:
        rules.append(("Prefix", ["^^^", internal]))

    if args["HAS_SUFFIXES"]:
        rules.append((0, 1, "Suffix", [internal, "$$$"]))

    rules += [
        ("Stem", ["Chars"]),
        (0, 1, "Chars", ["Char"]),
        (0, 1, "Chars", ["Char", "Chars"]),
    ]
    rules += [(0, 1, "Char", [character]) for character in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(format_rules(rules))
    with meta_open(target[1].rstr(), "w") as ofd:
        ofd.write("\n".join(["^^^ %s $$$" % (" ".join([c for c in w])) for w in words]).encode("utf-8"))
    return None

def tagging_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()        
        if args["LOWER_CASE_TAGGING"]:
            words = list(set([w.lower() for w in words]))
    num_tags = args.get("num_tags", 10)
    markov = args.get("markov", 2)    
    histories = product(range(num_tags), repeat=markov)
    rules = []
    for tag in range(num_tags):
        rules.append((0, 1, "Sentence", ["Tag%d" % tag]))
        for word in words:
            rules.append((0, 1, "Tag%d" % tag, [word]))
            for next_tag in range(num_tags):
                rules.append((0, 1, "Tag%d" % tag, [word, "Tag%d" % next_tag]))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(format_rules(rules))
    with meta_open(target[1].rstr(), "w") as ofd:
        sentences = [" ".join([data.indexToWord[w] for w, t, m in s]) for s in data.sentences if len(s) < env["MAXIMUM_SENTENCE_LENGTH"]]
        ofd.write("\n".join([s for s in sentences]).encode("utf-8").strip())
    return None

def joint_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        if args["LOWER_CASE_MORPHOLOGY"]:
            pass
        if args["LOWER_CASE_TAGGING"]:
            pass
        characters = set(sum([[c for c in w] for w in words], []))
    num_tags = args.get("num_tags", 10)
    markov = args.get("markov", 2)    
    histories = product(range(num_tags), repeat=markov)
    rules = []

    for tag in range(num_tags):
        rules += [
            (0, 1, "Sentence", ["Tag%d" % tag]),
            ("Prefix%d" % tag, ("^^^", "Chars")),
            ("Prefix%d" % tag, ["^^^"]),
            ("Stem%d" % tag, ["Chars"]),
            ("Suffix%d" % tag, ("Chars", "$$$")),
            ("Suffix%d" % tag, ["$$$"]),
            ]
        #for word in words:
        rules += [
            (0, 1, "Tag%d" % tag, ["Word%d" % (tag)]),
            (0, 1, "Word%d" % (tag), ["Prefix%d" % tag, "Stem%d" % tag, "Suffix%d" % tag]),
            #(0, 1, "Tag%d" % tag, ["%s%d" % (word, tag)]),
            #(0, 1, "%s%d" % (word, tag), ["Prefix%d" % tag, "Stem%d" % tag, "Suffix%d" % tag]),
        ]
        for next_tag in range(num_tags):
            rules += [
                (0, 1, "Tag%d" % tag, ["Word%d" % (tag), "Tag%d" % next_tag]),
                #(0, 1, "Tag%d" % tag, ["%s%d" % (word, tag), "Tag%d" % next_tag]),
            ]
    rules += [(0, 1, "Chars", ["Char"]),
              (0, 1, "Chars", ["Char", "Chars"]),
              ]
    rules += [(0, 1, "Char", [character]) for character in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(format_rules(rules))
    with meta_open(target[1].rstr(), "w") as ofd:
        lines = [" ".join([" ".join(["^^^"] + [c for c in data.indexToWord[w]] + ["$$$"]) for w, t, m in s]) for s in data.sentences if len(s) < env["MAXIMUM_SENTENCE_LENGTH"]]
        ofd.write("\n".join([l for l in lines]).encode("utf-8").strip())
    return None

def gold_segmentations_joint_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        characters = set(sum([[c for c in w] for w in words], []))
        sentences = [[(data.indexToWord[w], [data.indexToAnalysis[x] for x in xs]) for w, t, xs in s] for s in data.sentences if len(s) < env["MAXIMUM_SENTENCE_LENGTH"]]

    segmentations = {}
    for w, xs in sum(sentences, []):
        if len(xs) == 0 or len(xs[0]) == 1:
            segmentations[w] = ("", w, "")
        elif len(xs[0]) > 2:
            segmentations[w] = (xs[0][0], "".join(xs[0][1:-1]), xs[0][-1])
        else:
            if len(xs[0][0]) > len(xs[0][1]):
                segmentations[w] = ("", xs[0][0], xs[0][1])
            else:
                segmentations[w] = (xs[0][0], xs[0][1], "")

    sentences = [[(w, segmentations[w]) for w, xs in s] for s in sentences]
    num_tags = args.get("num_tags", 10)

    rules = []
    for tag in range(num_tags):
        rules += [
            (0, 1, "Sentence", ["Tag%d" % tag]),
            ("Prefix%d" % tag, ("^^^", "Chars")),
            ("Prefix%d" % tag, ["^^^"]),
            ("Stem%d" % tag, ("@@@", "Chars", "@@@")),
            ("Suffix%d" % tag, ("Chars", "$$$")),
            ("Suffix%d" % tag, ["$$$"]),
            (0, 1, "Tag%d" % tag, ["Word%d" % (tag)]),
            ("Word%d" % (tag), ["Prefix%d" % tag, "Stem%d" % tag, "Suffix%d" % tag]),
        ]
        for next_tag in range(num_tags):
            rules += [
                (0, 1, "Tag%d" % tag, ["Word%d" % (tag), "Tag%d" % next_tag]),
            ]

    rules += [
        (0, 1, "Chars", ("Chars", "Char")),
        (0, 1, "Chars", ["Char"]),
        ]
    rules += [(0, 1, "Char", [character]) for character in characters]

    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(format_rules(rules))
        #ofd.write("\n".join(["%s --> %s" % (k, " ".join(v)) for k, v in rules]))
    with meta_open(target[1].rstr(), "w") as ofd:
        #ofd.write("\n".join([" ".join([" ".join(["^^^"] + [c for c in pre] + ["@@@"] + [c for c in stm] + ["@@@"] + [c for c in suf] + ["$$$"]) for w, (pre, stm, suf) in s]) for s in sentences]))
        lines = [" ".join([" ".join(["^^^"] + [c for c in pre] + ["@@@"] + [c for c in stm] + ["@@@"] + [c for c in suf] + ["$$$"]) for w, (pre, stm, suf) in s]) for s in sentences]
        ofd.write("\n".join([l for l in lines]).encode("utf-8").strip())

    return None

def gold_tags_joint_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        characters = set(sum([[c for c in w] for w in words], []))
    #sentences = [[(data.indexToWord[w], data.indexToTag[t]) for w, t, xx in s] for s in data.sentences]
    sentences = [[(data.indexToWord[w], t) for w, t, xx in s] for s in data.sentences if len(s) < env["MAXIMUM_SENTENCE_LENGTH"]]
    tags = set([t for w, t in sum(sentences, [])])
    rules = [(0, 1, "Sentence", ["Tag%s" % tag]) for tag in tags]
    
    for tag in tags:
        rules += [
            #(0, 1, "Sentence", ["Tag%s" % tag]),
            ("Prefix%s" % tag, ["^^^%s" % tag, "Chars"]),
            ("Prefix%s" % tag, ["^^^%s" % tag]),
            ("Stem%s" % tag, ["Chars"]),
            ("Suffix%s" % tag, ["Chars", "$$$%s" % tag]),
            ("Suffix%s" % tag, ["$$$%s" % tag]),
            # ("Prefix%s" % tag, ("Prefix%s" % tag, "Chars")),
            # ("Prefix%s" % tag, ["^^^%s" % tag]),
            # ("Stem%s" % tag, ("Stem%s" % tag, "Chars")),
            # ("Stem%s" % tag, ["Chars"]),
            # ("Suffix%s" % tag, ("Chars", "Suffix%s" % tag)),
            # ("Suffix%s" % tag, ["$$$%s" % tag]),
            ]
        #for word in words:
        rules += [
            (0, 1, "Tag%s" % tag, ["Word%s" % (tag)]),
            #("Tag%s" % tag, ["%s%s" % (word, tag)]),
            (1, 1, "Word%s" % (tag), ["Prefix%s" % tag, "Stem%s" % tag, "Suffix%s" % tag]),
            #("%s%s" % (word, tag), ["Prefix%s" % tag, "Stem%s" % tag, "Suffix%s" % tag]),
        ]
        for next_tag in tags:
            rules += [
                (0, 1, "Tag%s" % tag, ["Word%s" % (tag), "Tag%s" % next_tag]),
            ]
    rules += [
        (0, 1, "Chars", ("Chars", "Char")),
        (0, 1, "Chars", ["Char"]),
        ]
    rules += [(0, 1, "Char", [character]) for character in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(format_rules(rules))
        #ofd.write("\n".join(["%s --> %s" % (k, " ".join(v)) for k, v in rules]))
    with meta_open(target[1].rstr(), "w") as ofd:
        #ofd.write("\n".join([" ".join([" ".join(["^^^%s" % t] + [c for c in w] + ["$$$%s" % t]) for w, t in s]) for s in sentences]))
        lines = [" ".join([" ".join(["^^^%s" % t] + [c for c in w] + ["$$$%s" % t]) for w, t in s]) for s in sentences]
        ofd.write("\n".join([l for l in lines]).encode("utf-8").strip())
    return None

#cmd = "cat ${SOURCES[1].abspath}|${PYCFG_PATH}/py-cfg ${SOURCES[0].abspath} -w 0.1 -N 1 -d 100 -E -n ${NUM_BURNINS} -e 1 -f 1 -g 10 -h 0.1 -T ${ANNEAL_INITIAL} -t ${ANNEAL_FINAL} -m ${ANNEAL_ITERATIONS} -A ${TARGETS[0]} -x 5 -G ${TARGETS[1]} -F ${TARGETS[2]}"

#def run_pycfg(target, source, env, for_signature):
#    if source[1].rstr().endswith("z"):
#        return env.subst("z%s" % cmd, target=target, source=source)    
#    else:
#        return env.subst(cmd, target=target, source=source)

def list_to_tuples(xs, n=2):
    return [[xs[i * n + j] for j in range(n)] for i in range(len(xs) / n)]

# def run_pycfg_torque(target, source, env):
#     """
#     Pairs of inputs and outputs
#     """
#     jobs = []
#     interval = 30
#     for (cfg, data), (out, grammar, log) in zip(list_to_tuples(source), list_to_tuples(target, 3)):
#         job = torque.Job("py-cfg",
#                          commands=[env.subst(cmd, target=[out, grammar, log], source=[cfg, data])],
#                          #path=args["path"],
#                          stdout_path=os.path.abspath("torque"),
#                          stderr_path=os.path.abspath("torque"),
#                          array=0,
#                          other=["#PBS -W group_list=yeticcls"],
#                          resources={
#                              "cput" : "40:00:00",
#                              "walltime" : "40:00:00",
#                          },
#                          )
#         job.submit(commit=True)
#         jobs.append(job)
#     running = torque.get_jobs(True)
#     while any([job.job_id in [x[0] for x in running] for job in jobs]):
#         logging.info("sleeping...")
#         time.sleep(interval)
#         running = torque.get_jobs(True)
#     return None

def collate_tagging_output(target, source, env):
    sentences = []
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            sentence = []
            for m in re.finditer(r"Tag(\d+)(#\d+)? (\S+)", line):
                tag, customers, word = m.groups()
                word = word.rstrip(")")
                tag = int(tag)
                sentence.append((word, tag))
            sentences.append(sentence)
    with meta_open(target[0].rstr(), "w") as ofd:
        for sentence in sentences:
            ofd.write(" ".join(["%s/%d/%s" % (word, tag, word) for word, tag in sentence]) + "\n")
    return None

def morphology_output_to_emma(target, source, env):
    analyses = {}
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            toks = tuple(["".join([y.group(1).replace(" ", "") for y in re.finditer("\(\S+ ([^\)\(]+)\)", x)]) for x in re.split(r"\(Prefix|\(Stem|\(Suffix", line)[1:]])
            #toks = tuple(["".join([y.group(1) for y in re.finditer("\(\S+ ([^\)\(]+)\)", x)]) for x in re.split(r"\(Prefix|\(Stem|\(Suffix", line)[1:]])
            word = "".join(toks)
            if not re.match(r"^\s*$", word):
                analyses[word] = analyses.get(word, []) + [toks]
    with meta_open(target[0].rstr(), "w") as ofd:
        for w, aa in sorted(analyses.iteritems()):
            if re.match(r"^\w+$", w) and not re.match(r"^.*\d.*$", w):
                ofd.write("%s\t%s\n" % (w, ", ".join([" ".join(["%s:NULL" % m for m in a if m != ""]) for a in set(aa)])))
    return None

def normalize_pycfg_output(target, source, env):
    analyses = {}
    with meta_open(source[0].rstr()) as ifd:
        text = re.split(r"\n\s*?\n", ifd.read().strip())[-1]
        for line in text.split("\n"):
            toks = tuple(["".join([unichr(int(y.group(1).replace(" ", ""), base=16)) for y in re.finditer("\(\S+ ([^\)\(]+)\)", x)]) for x in re.split(r"\(Prefix|\(Stem|\(Suffix", line)[1:]])
            toks = tuple([t for t in toks if len(t) > 0])
            #toks = tuple(["".join([y.group(1) for y in re.finditer("\(\S+ ([^\)\(]+)\)", x)]) for x in re.split(r"\(Prefix|\(Stem|\(Suffix", line)[1:]])
            word = "".join(toks)
            if len(toks) > 1:
                toks = tuple(["%s+" % (toks[0])] + ["+%s+" % (t) for t in toks[1:-1]] + ["+%s" % (toks[-1])])
            if not re.match(r"^\s*$", word) and "_" not in word:
                analyses[word] = analyses.get(word, []) + [toks]
    with meta_open(target[0].rstr(), "w") as ofd:
        for w, aa in sorted(analyses.iteritems()):
            #if re.match(r"^\w+$", w, re.UNICODE) and not re.match(r"^.*\d.*$", w):
            #ofd.write("%s\t%s\n" % (w, ", ".join([" ".join(["%s" % m for m in a if m != ""]) for a in set(aa)])))
            ofd.write("%s\n" % (" ".join(aa[0])))
    return None

def collate_joint_output(target, source, env):
    data = []
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            if not re.match(r"^\s*$", line):
                locations = []
                for l in re.split(r"\(Tag", line)[1:]:
                    tag, prefix_str, stem_str, suffix_str = re.match(r"^(\d+).*\(Prefix(.*)\(Stem(.*)\(Suffix(.*)$", l).groups()
                    prefix = "".join([x.groups()[0] for x in re.finditer(r"\(Char (\S)\)", prefix_str)])
                    stem = "".join([x.groups()[0] for x in re.finditer(r"\(Char (\S)\)", stem_str)])
                    suffix = "".join([x.groups()[0] for x in re.finditer(r"\(Char (\S)\)", suffix_str)])
                    word = prefix + stem + suffix
                    locations.append((int(tag), prefix, stem, suffix, word))
                data.append(locations)
    with meta_open(target[0].rstr(), "w") as ofd:
        for sentence in data:
            ofd.write(" ".join(["%s/%d/%s+%s+%s" % (word, tag, prefix, stem, suffix) for tag, prefix, stem, suffix, word in sentence]) + "\n")
    return None



def tagging_output_to_dataset(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        tags = [[m.group(1) for m in re.finditer(r"\(Tag(\S+)", l)] for l in ifd if not re.match(r"^\s*$", l)]
    with meta_open(source[1].rstr()) as ifd:
        dataset = DataSet.from_stream(ifd)[0]
    new_sentences = [[(dataset.indexToWord[w], dataset.indexToTag[t], [dataset.indexToAnalysis[a] for a in aa]) for w, t, aa in s] for s in dataset.sentences if len(s) < 40]
    assert(len(tags) == len(new_sentences))
    assert(all([len(x) == len(y) for x, y in zip(tags, new_sentences)]))    
    new_dataset = DataSet.from_sentences([[(w, t, aa) for t, (w, ot, aa) in zip(tt, s)] for tt, s in zip(tags, new_sentences)])
    gold_dataset = DataSet.from_sentences(new_sentences)
    with meta_open(target[0].rstr(), "w") as ofd:
        gold_dataset.write(ofd)
    with meta_open(target[1].rstr(), "w") as ofd:
        new_dataset.write(ofd)
    return None

def evaluate_tagging(env, *args, **kw):
    target, (output, gold) = args
    #filtered_gold, proposal = env.TaggingOutputToDataset(["%s-dataset_gold.xml.gz" % target, "%s-dataset.xml.gz" % target], [output, gold])
    #env.EvaluateTaggingVM(target, [proposal, filtered_gold])
    return None
    
def evaluate_morphology(env, *args, **kw):
    parses, training, gold_morphology = args
    emma_output = "work/emma/%s" % os.path.basename(parses.rstr())
    eval_output = "work/evaluation/%s" % os.path.basename(parses.rstr())
    emma = env.MorphologyOutputToEMMA(emma_output, parses)
    morphology_results = env.RunEMMA(eval_output, [emma, gold_morphology])
    return None

def evaluate_joint(env, *args, **kw):
    target, (source, train) = args
    gold_morphology = env.subst("data/${LANGUAGE}_morphology.txt")
    if os.path.exists(gold_morphology):
        emma = env.MorphologyOutputToEMMA("%s-emma" % target, source)
        morphology_results = env.RunEMMA(target, [emma, gold_morphology])
    return None


def evaluate_many_morphology(target, source, env):
    
    gold_morphology = source[-1]
    results = []
    with meta_open(source[0].rstr()) as ifd:
        for text in ifd.read().strip().split("\n\n"):
            analyses = {}
            for m in re.finditer(r"\(Prefix(.*?)\(Stem(.*?)\(Suffix(.*?)(\(Word|$)", text, re.S|re.M):
                prefix_str, stem_str, suffix_str, rem = m.groups()
                prefix = "".join([x.groups()[0] for x in re.finditer(r"\(Char (\S)\)", prefix_str)])
                stem = "".join([x.groups()[0] for x in re.finditer(r"\(Char (\S)\)", stem_str)])
                suffix = "".join([x.groups()[0] for x in re.finditer(r"\(Char (\S)\)", suffix_str)])
                word = prefix + stem + suffix
                if not re.match(r"^\s*$", word):
                    analyses[word] = analyses.get(word, []) + [(prefix, stem, suffix)]
            with meta_open("temp.txt", "w") as ofd:
                for w, aa in sorted(analyses.iteritems()):
                    ofd.write("%s\t%s\n" % (w, ", ".join([" ".join(["%s:NULL" % m for m in a if m != ""]) for a in set(aa)])))
            cmd = env.subst("python ${EMMA} -p temp.txt -g ${SOURCES[1]}", source=source, target=target)
            print cmd
            pid = Popen(cmd.split(), stdout=PIPE)
            out, err = pid.communicate()
            prec, rec, fscore = [float(x.strip().split()[-1]) for x in out.strip().split("\n")[-3:]]
            


            results.append("%f\t%f\t%f" % (prec, rec, fscore))
            print results[-1]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\t".join(["MorphP", "MorphR", "MorphF"]) + "\n")
        ofd.write("\n".join(results))

    return None


def extract_affixes(target, source, env):
    prefixes, suffixes = {}, {}
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        
    for a in data.indexToAnalysis.values():
        if len(a) > 1:
            a = map(lambda x : x.lower(), a)
            lengths = map(len, a)
            mlen = max(lengths)
            mind = lengths.index(mlen)
            pre = a[0:mind]
            suf = a[mind + 1:]                        
            if len(pre) > 0:
                pre = re.sub(r"[^\w]", "", pre[0])
                prefixes[pre] = prefixes.get(pre, 0) + 1
            if len(suf) > 0:
                suf = re.sub(r"[^\w]", "", suf[-1])
                suffixes[suf] = suffixes.get(suf, 0) + 1
    for fname, affixes in zip(target, [prefixes, suffixes]):
        with meta_open(fname.rstr(), "w") as ofd:
            ofd.write("\n".join(["%d\t%s" % (c, a) for a, c in sorted(affixes.iteritems(), lambda x, y : cmp(y[1], x[1]))]))
    return None


def character_productions(target, source, env):
    nag = [unichr(int(x, base=16)) for x in env.get("NON_ACOUSTIC_GRAPHEMES")]
    nag = ["%.4x" % (ord(x)) for x in nag]
    characters = set()
    for f in source:
        #try:
        #    with meta_open(f.rstr()) as ifd:
        #        data = DataSet.from_stream(ifd)[0]
        #        words = [word for word in sum([[data.indexToWord[w] for w, t, aa in s] for s in data.sentences], [])] # if regular_word(word)]
                #words = [w.lower() for w in words]
        #        words = set(words)
        #        characters = set(sum([[c for c in w] for w in words], []))
        #except:
        with meta_open(f.rstr()) as ifd:
            for l in ifd:
                for c in l:
                    if not re.match(r"\s", c):
                        characters.add(c)
                        #for n in nag:
                        #    characters.add("%s%s" % (n, c))
                        #    characters.add("%s%s$" % (c, n))
    #print "W" in characters
    characters = ["%.4x" % (ord(x)) for x in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        for c in characters:
            #ofd.write("0 1 Char --> %s\n" % (c.encode("utf-8")))
            ofd.write("0 1 Char --> %s\n" % (c))
    return None

def compose_grammars(target, source, env):
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
    # lc = False #source[1].read()
    # try:
    #     with meta_open(source[0].rstr()) as ifd:
    #         data = DataSet.from_stream(ifd)[0]
    #         words = set([word.lower() for word in sum([[data.indexToWord[w] for w, t, aa in s] for s in data.sentences], []) if regular_word(word)])
    # except:
    #     with codecs.open(source[0].rstr(), "r", "utf-8") as ifd: #meta_open(source[0].rstr()) as ifd:
    #         if lc:
    #             words = set(sum([l.lower().split() for l in ifd], []))
    #         else:                
    #             words = set(sum([l.split() for l in ifd], []))
    nag = [unichr(int(x, base=16)) for x in env.get("NON_ACOUSTIC_GRAPHEMES")]
    nag = ["%.4x" % (ord(x)) for x in nag]
    words = set()
    with meta_open(source[0].rstr()) as ifd:
        for line in ifd:
            toks = line.strip().split()
            for tok in toks:
                for word in tok.split("-"):
                    if "_" not in word and "<" not in word and len(word) > 0:
                        words.add(word)
    with meta_open(target[0].rstr(), "w") as ofd:
        text = "\n".join([" ".join(["^^^"] + ["%.4x" % (ord(c)) for c in w] + ["$$$"]) for w in words])
        ofd.write(text)
    return None

def pycfg_generator(target, source, env, for_signature):
    if source[1].rstr().endswith("gz"):
        cat = "zcat"
    else:
        cat = "cat"
    # -w: default rule prob or dirichlet prior, -d debug level, -E estimate rule prob, -e/-f beta prior on pya, -g/-h gamma prior on pyb
    if len(source) == 2:
        return "%s ${SOURCES[1]}|${PYCFG_PATH}/py-cfg ${SOURCES[0]} -w 0.1 -N ${NUM_SAMPLES} -d 100 -E -n ${NUM_ITERATIONS} -e 1 -f 1 -g 10 -h 0.1 -T ${ANNEAL_INITIAL} -t ${ANNEAL_FINAL} -m ${ANNEAL_ITERATIONS} -A ${TARGETS[0]} -G ${TARGETS[1]} -F ${TARGETS[2]} 2> /dev/null" % cat
    elif len(source) == 3:
        return "%s ${SOURCES[1]}|${PYCFG_PATH}/py-cfg ${SOURCES[0]} -w 0.1 -N ${NUM_SAMPLES} -d 100 -E -n ${NUM_ITERATIONS} -e 1 -f 1 -g 10 -h 0.1 -T ${ANNEAL_INITIAL} -t ${ANNEAL_FINAL} -m ${ANNEAL_ITERATIONS} -A ${TARGETS[0]} -G ${TARGETS[1]} -F ${TARGETS[2]} -u ${SOURCES[2]} -U 'tee > ${TARGETS[3]}' -x ${NUM_ITERATIONS} 2> /dev/null" % cat
    elif len(source) == 4:
        return "%s ${SOURCES[1]}|${PYCFG_PATH}/py-cfg ${SOURCES[0]} -w 0.1 -N ${NUM_SAMPLES} -d 100 -E -n ${NUM_ITERATIONS} -e 1 -f 1 -g 10 -h 0.1 -T ${ANNEAL_INITIAL} -t ${ANNEAL_FINAL} -m ${ANNEAL_ITERATIONS} -A ${TARGETS[0]} -G ${TARGETS[1]} -F ${TARGETS[2]} -u ${SOURCES[2]} -U 'tee > ${TARGETS[3]}' -v ${SOURCES[3]} -V 'tee > ${TARGETS[4]}' -x ${NUM_ITERATIONS} 2> /dev/null" % cat
    
def pycfg_emitter(target, source, env):
    if len(target) == 1:
        new_target = [target[0]] + ["work/pycfg_output/%s_%s.txt" % (os.path.basename(target[0].rstr()), x) for x in ["grammar", "trace"]]
    else:
        new_target = target
    return new_target, source

def apply_adaptor_grammar(target, source, env):
    # PCFG, Production, ViterbiParser
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
        #for word in items:
            #try:
                #tree = parser.parse(word).next()
            toks = [z for z in ["".join([unichr(int(y, base=16)) for y in x.leaves() if y not in ["^^^", "$$$"]]) for x in tree] if z != ""]
            if len(toks) == 1:
                ofd.write("%s\n" % (toks[0]))
            else:
                ofd.write(" ".join(["%s+" % toks[0]] + ["+%s+" % x for x in toks[1:-1]] + ["+%s" % toks[-1]]) + "\n")
            #except:
            #    ofd.write("%s\n" % (word))
    return None


def TOOLS_ADD(env):
    #env["PYCFG_PATH"] = "/usr/local/py-cfg"
    env.Append(BUILDERS = {
        "MorphologyData" : Builder(action=morphology_data),
        "ComposeGrammars" : Builder(action=compose_grammars),
        "CharacterProductions" : Builder(action=character_productions),
        "ExtractAffixes" : Builder(action=extract_affixes),
        "CreateMorphologyModel" : make_command_builder("${CABAL}/bin/create_model ${MODEL} -i ${SOURCES[0]} -m ${TARGETS[0]} -d ${TARGETS[1]} ${HAS_PREFIXES and '--hasprefixes' or ''} ${HAS_SUFFIXES and '--hassuffixes' or ''} -k ${KEEP} --prefixfile ${SOURCES[1]} --suffixfile ${SOURCES[2]}", 
                                             ["Model", "Data"], 
                                             ["LANGUAGE", "MODEL", "HAS_PREFIXES", "HAS_SUFFIXES", "KEEP"], 
                                             "work/models"),


        "CreateNgramModel" : make_command_builder("${CABAL}/bin/create_model ${MODEL} -i ${SOURCES[0]} -m ${TARGETS[0]} -d ${TARGETS[1]} -n ${N}", 
                                                  ["Model", "Data"], 
                                                  ["LANGUAGE", "N"], 
                                                  "work/models"),


        "NGram" : Builder(action=ngram_cfg),
        "MorphologyCFG" : Builder(action=morphology_cfg,
                                  emitter=make_generic_emitter(["work/ag_models/${SPEC}.txt", "work/ag_data/${SPEC}.txt"])),
        "TaggingCFG" : Builder(action=tagging_cfg),
        "JointCFG" : Builder(action=joint_cfg),
        "GoldTagsJointCFG" : Builder(action=gold_tags_joint_cfg),
        "GoldSegmentationsJointCFG" : Builder(action=gold_segmentations_joint_cfg),
        "CollateTaggingOutput" : Builder(action=collate_tagging_output),
        "MorphologyOutputToEMMA" : Builder(action=morphology_output_to_emma), #emitter=make_generic_emitter(["work/ag_emma_format/${SPEC}.txt"])),
        "CollateJointOutput" : Builder(action=collate_joint_output),
        "TaggingOutputToDataset" : Builder(action=tagging_output_to_dataset),
        "EvaluateManyMorphology" : Builder(action=evaluate_many_morphology),
        "NormalizePYCFGOutput" : Builder(action=normalize_pycfg_output),
        "ApplyAdaptorGrammar" : Builder(action=apply_adaptor_grammar),
    })
    pycfg_action = CommandGeneratorAction(pycfg_generator, {})
    pycfg_builder = Builder(action=pycfg_action, emitter=pycfg_emitter)
    env.Append(BUILDERS = {"RunPYCFG" : pycfg_builder})
    env.AddMethod(evaluate_tagging, "EvaluateTagging")
    env.AddMethod(evaluate_tagging, "EvaluateGoldSegmentationsJoint")
    env.AddMethod(evaluate_morphology, "EvaluateMorphology")
    env.AddMethod(evaluate_morphology, "EvaluateGoldTagsJoint")
    env.AddMethod(evaluate_joint, "EvaluateJoint")

#  -N nanal-its    -- print analyses during last nanal-its iterations
#  -C              -- print compact trees omitting uncached categories
#  -D              -- delay grammar initialization until all sentences are parsed
#  -E              -- estimate rule prob (theta) using Dirichlet prior
#  -H              -- skip Hastings correction of tree probabilities
#  -I              -- parse sentences in order (default is random order)
#  -P              -- use a predictive Earley parse to filter useless categories
#  -R nr           -- resample PY cache strings during first nr iterations (-1 = forever)
#  -r rand-init    -- initializer for random number generator (integer)
#  -a a            -- default PY a parameter
#  -b b            -- default PY b parameter
#  -e pya-beta-a   -- if positive, parameter of Beta prior on pya; if negative, number of iterations to anneal pya
#  -f pya-beta-b   -- if positive, parameter of Beta prior on pya
#  -g pyb-gamma-s  -- if non-zero, parameter of Gamma prior on pyb
#  -h pyb-gamma-c  -- parameter of Gamma prior on pyb
#  -w weight       -- default value of theta (or Dirichlet prior) in generator
#  -s train_frac   -- train only on train_frac percentage of training sentences (ignore remainder)
#  -S              -- randomise training fraction of sentences (default: training fraction is at front)
#  -Z ztemp        -- temperature used just before stopping
#  -z zits         -- perform zits iterations at temperature ztemp at end of run
