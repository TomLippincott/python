from SCons.Builder import Builder
from SCons.Script import *
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import cPickle as pickle
import math
import sys
import gzip
from os.path import join as pjoin
from os import listdir
import tarfile
import operator
from random import randint
from subprocess import Popen, PIPE
from common_tools import meta_open, DataSet, v_measure
from itertools import product
#import numpy
#from torque_tools import submit_job
import torque

def format_rules(rules):
    lines = []
    for rule in rules:
        targets = " ".join(rule[-1])
        if len(rule) == 2:
            lines.append("%s --> %s" % (rule[0], targets))
        elif len(rule) == 4:
            lines.append("%s %s %s --> %s" % (rule[0], rule[1], rule[2], targets))
    return "\n".join(lines).encode("utf-8")

def morphology_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        characters = set(sum([[c for c in w] for w in words], []))
    num_syntactic_classes = args.get("num_syntactic_classes", 1)
    rules = []
    #if num_syntactic_classes == 1:
    rules += [#(0, 1, "Word", ["Compound"]),                  
              #(0, 1, "Word", ["Prefix", "Stem"]),
              #(0, 1, "Word", ["Stem", "Suffix"]),
              (0, 1, "Word", ["Prefix", "Stem", "Suffix"]),
              ("Prefix", ["^^^", "Chars"]),
              ("Prefix", ["^^^"]),
              ("Stem", ["Chars"]),
              ("Suffix", ["Chars", "$$$"]),
              ("Suffix", ["$$$"]),
          ]
    # else:
    #     for syntactic_class in range(num_syntactic_classes):
    #         rules += [("Word", ["Word%d" % syntactic_class]),
    #                   ("Word%d" % syntactic_class, ["Stem"]),
    #                   #("Word%d" % syntactic_class, ["Prefix", "Stem"]),
    #                   #("Word%d" % syntactic_class, ["Stem", "Suffix"]),
    #                   ("Word%d" % syntactic_class, ["Prefix", "Stem", "Suffix"]),
    #                   ("Prefix", ["^^^", "Chars"]),
    #                   ("Stem", ["Chars"]),
    #                   ("Suffix", ["Chars", "$$$"]),
    #                   ]
    rules += [(0, 1, "Chars", ["Char"]),
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
        sentences = [" ".join([data.indexToWord[w] for w, t, m in s]) for s in data.sentences if len(s) < 20]
        ofd.write("\n".join([s for s in sentences]).encode("utf-8").strip())
    return None

def joint_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
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
        lines = [" ".join([" ".join(["^^^"] + [c for c in data.indexToWord[w]] + ["$$$"]) for w, t, m in s]) for s in data.sentences if len(s) < 20]
        ofd.write("\n".join([l for l in lines]).encode("utf-8").strip())
    return None

def gold_segmentations_joint_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        characters = set(sum([[c for c in w] for w in words], []))
        sentences = [[(data.indexToWord[w], [data.indexToAnalysis[x] for x in xs]) for w, t, xs in s] for s in data.sentences]

    segmentations = {}
    for w, xs in sum(sentences, []):
        if len(xs) == 0 or len(xs[0]) == 1:
            segmentations[w] = ("", w, "")
        elif len(xs[0]) > 2:
            segmentations[w] = (xs[0], "".join(xs[1:-1]), xs[-1])
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
            ("Sentence", ["Tag%d" % tag]),
            ("Prefix%d" % tag, ("Prefix%d" % tag, "Chars")),
            ("Prefix%d" % tag, ["^^^"]),
            ("Stem%d" % tag, ("@@@", "Chars", "@@@")),
            ("Stem%d" % tag, ["Chars"]),
            ("Suffix%d" % tag, ("Chars", "Suffix%d" % tag)),
            ("Suffix%d" % tag, ["$$$"]),
            ]
        for word in words:
            rules += [
                ("Tag%d" % tag, ["%s%d" % (word, tag)]),
                ("%s%d" % (word, tag), ["Prefix%d" % tag, "Stem%d" % tag, "Suffix%d" % tag]),
                ]
            for next_tag in range(num_tags):
                rules += [
                    ("Tag%d" % tag, ["%s%d" % (word, tag), "Tag%d" % next_tag]),
                    ]

    rules += [
        ("Chars", ("Chars", "Char")),
        ("Chars", ["Char"]),
        ]
    rules += [("Char", [character]) for character in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(format_rules(rules))
        #ofd.write("\n".join(["%s --> %s" % (k, " ".join(v)) for k, v in rules]))
    with meta_open(target[1].rstr(), "w") as ofd:
        ofd.write("\n".join([" ".join([" ".join(["^^^"] + [c for c in pre] + ["@@@"] + [c for c in stm] + ["@@@"] + [c for c in suf] + ["$$$"]) for w, (pre, stm, suf) in s]) for s in sentences]))

    return None

def gold_tags_joint_cfg(target, source, env):
    args = source[-1].read()
    with meta_open(source[0].rstr()) as ifd:
        data = DataSet.from_stream(ifd)[0]
        words = data.indexToWord.values()
        characters = set(sum([[c for c in w] for w in words], []))
    sentences = [[(data.indexToWord[w], data.indexToTag[t]) for w, t, xx in s] for s in data.sentences]
    tags = set([t for w, t in sum(sentences, [])])
    rules = []
    for tag in tags:
        rules += [
            ("Sentence", ["Tag%s" % tag]),
            ("Prefix%s" % tag, ("Prefix%s" % tag, "Chars")),
            ("Prefix%s" % tag, ["^^^%s" % tag]),
            ("Stem%s" % tag, ("Stem%s" % tag, "Chars")),
            ("Stem%s" % tag, ["Chars"]),
            ("Suffix%s" % tag, ("Chars", "Suffix%s" % tag)),
            ("Suffix%s" % tag, ["$$$%s" % tag]),
            ]
        for word in words:
            rules += [
                ("Tag%s" % tag, ["Word%s" % (word, tag)]),
                #("Tag%s" % tag, ["%s%s" % (word, tag)]),
                ("%s%s" % (word, tag), ["Prefix%s" % tag, "Stem%s" % tag, "Suffix%s" % tag]),
                #("%s%s" % (word, tag), ["Prefix%s" % tag, "Stem%s" % tag, "Suffix%s" % tag]),
                ]
            for next_tag in tags:
                rules += [
                    ("Tag%s" % tag, ["%s%s" % (word, tag), "Tag%s" % next_tag]),
                    ]
    rules += [
        ("Chars", ("Chars", "Char")),
        ("Chars", ["Char"]),
        ]
    rules += [("Char", [character]) for character in characters]
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(format_rules(rules))
        #ofd.write("\n".join(["%s --> %s" % (k, " ".join(v)) for k, v in rules]))
    with meta_open(target[1].rstr(), "w") as ofd:
        ofd.write("\n".join([" ".join([" ".join(["^^^%s" % t] + [c for c in w] + ["$$$%s" % t]) for w, t in s]) for s in sentences]))
    return None

cmd = "zcat ${SOURCES[1].abspath}|${PYCFG_PATH}/py-cfg ${SOURCES[0].abspath} -w 0.1 -A ${TARGETS[0].abspath} -N 1 -d 100 -E -n ${NUM_BURNINS} -e 1 -f 1 -g 10 -h 0.1 -T ${ANNEAL_INITIAL} -t ${ANNEAL_FINAL} -m ${ANNEAL_ITERATIONS} -G ${TARGETS[1]} > ${TARGETS[2].abspath}"

def run_pycfg(target, source, env, for_signature):
    return env.subst(cmd, target=target, source=source)

def list_to_tuples(xs):
    return [(xs[i * 2], xs[i * 2 + 1]) for i in range(len(xs) / 2)]

def run_pycfg_torque(target, source, env):
    """
    Pairs of inputs and outputs
    """
    jobs = []
    interval = 30
    for (cfg, data), (out, log) in zip(list_to_tuples(source), list_to_tuples(target)):
        job = torque.Job("py-cfg",
                         commands=[env.subst(cmd, target=[out, log], source=[cfg, data])],
                         #path=args["path"],
                         stdout_path=os.path.abspath("torque"),
                         stderr_path=os.path.abspath("torque"),
                         array=0,
                         other=["#PBS -W group_list=yeticcls"],
                         resources={
                             "cput" : "40:00:00",
                             "walltime" : "40:00:00",
                         },
                         )
        job.submit(commit=True)
        jobs.append(job)
    running = torque.get_jobs(True)
    while any([job.job_id in [x[0] for x in running] for job in jobs]):
        logging.info("sleeping...")
        time.sleep(interval)
        running = torque.get_jobs(True)
    return None

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
        for m in re.finditer(r"\(Prefix(.*?)\(Stem(.*?)\(Suffix(.*?)(\(Word|$)", ifd.read(), re.S|re.M):
            prefix_str, stem_str, suffix_str, rem = m.groups()
            prefix = "".join([x.groups()[0] for x in re.finditer(r"\(Char (\S)\)", prefix_str)])
            stem = "".join([x.groups()[0] for x in re.finditer(r"\(Char (\S)\)", stem_str)])
            suffix = "".join([x.groups()[0] for x in re.finditer(r"\(Char (\S)\)", suffix_str)])
            word = prefix + stem + suffix
            if not re.match(r"^\s*$", word):
                analyses[word] = analyses.get(word, []) + [(prefix, stem, suffix)]
    with meta_open(target[0].rstr(), "w") as ofd:
        for w, aa in analyses.iteritems():
            ofd.write("%s\t%s\n" % (w, ", ".join([" ".join(["%s:NULL" % m for m in a if m != ""]) for a in set(aa)])))
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

def torque_key(action, env, target, source):
    return 1

def evaluate_tagging(env, *args, **kw):
    target, (source, train) = args
    return None

def evaluate_morphology(env, *args, **kw):
    target, (source, train) = args
    gold_morphology = env.subst("data/${LANGUAGE}_morphology.txt")
    if os.path.exists(gold_morphology):
        emma = env.MorphologyOutputToEMMA("%s-emma" % target, source)    
        morphology_results = env.RunEMMA(target, [gold_morphology, emma])
    return None

def evaluate_joint(env, *args, **kw):
    target, (source, train) = args
    gold_morphology = env.subst("data/${LANGUAGE}_morphology.txt")
    if os.path.exists(gold_morphology):
        emma = env.MorphologyOutputToEMMA("%s-emma" % target, source)
        morphology_results = env.RunEMMA(target, [gold_morphology, emma])
    return None

def TOOLS_ADD(env):
    env["PYCFG_PATH"] = "/usr/local/py-cfg"
    env.Append(BUILDERS = {
        "MorphologyCFG" : Builder(action=morphology_cfg),
        "TaggingCFG" : Builder(action=tagging_cfg),
        "JointCFG" : Builder(action=joint_cfg),
        "GoldTagsJointCFG" : Builder(action=gold_tags_joint_cfg),
        "GoldSegmentationsJointCFG" : Builder(action=gold_segmentations_joint_cfg),
        "CollateTaggingOutput" : Builder(action=collate_tagging_output),
        "MorphologyOutputToEMMA" : Builder(action=morphology_output_to_emma),
        "CollateJointOutput" : Builder(action=collate_joint_output),
    })
    if env["HAS_TORQUE"] == True:
        runner = Builder(action=SCons.Action.Action(run_pycfg_torque, batch_key=torque_key))
    else:
        runner = Builder(generator=run_pycfg)
    env.Append(BUILDERS = {"RunPYCFG" : runner})
    env.AddMethod(evaluate_tagging, "EvaluateTagging")
    env.AddMethod(evaluate_morphology, "EvaluateMorphology")
    env.AddMethod(evaluate_joint, "EvaluateJoint")
    
