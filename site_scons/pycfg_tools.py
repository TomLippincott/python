from SCons.Script import Builder
import re
from glob import glob
import logging
import os.path
import os
from os.path import join as pjoin
from common_tools import meta_open, DataSet, v_measure
from itertools import product
import torque
from subprocess import Popen, PIPE

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
        words = [word for word in sum([[data.indexToWord[w] for w, t, aa in s] for s in data.sentences if len(s) < env["MAXIMUM_SENTENCE_LENGTH"]], []) if re.match(r"^[a-zA-Z]+$", word)]
        if args["LOWER_CASE_MORPHOLOGY"]:
            words = [w.lower() for w in words]
        characters = set(sum([[c for c in w] for w in words], []))
    num_syntactic_classes = args.get("num_syntactic_classes", 1)
    rules = []
    rules += [(1, 1, "Word", ["Prefix", "Stem", "Suffix"]),
              ("Prefix", ["^^^", "Chars"]),
              ("Prefix", ["^^^"]),
              ("Stem", ["Chars"]),
              ("Suffix", ["Chars", "$$$"]),
              ("Suffix", ["$$$"]),
          ]
    rules += [(1, 1, "Chars", ["Char"]),
              (1, 1, "Chars", ["Char", "Chars"]),
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
        ofd.write("\n".join([" ".join([" ".join(["^^^"] + [c for c in pre] + ["@@@"] + [c for c in stm] + ["@@@"] + [c for c in suf] + ["$$$"]) for w, (pre, stm, suf) in s]) for s in sentences]))

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
        ofd.write("\n".join([" ".join([" ".join(["^^^%s" % t] + [c for c in w] + ["$$$%s" % t]) for w, t in s]) for s in sentences]))
    return None

cmd = "zcat ${SOURCES[1].abspath}|${PYCFG_PATH}/py-cfg ${SOURCES[0].abspath} -w 0.1 -N 1 -d 100 -E -n ${NUM_BURNINS} -e 1 -f 1 -g 10 -h 0.1 -T ${ANNEAL_INITIAL} -t ${ANNEAL_FINAL} -m ${ANNEAL_ITERATIONS} -A ${TARGETS[0]} -x 5 -G ${TARGETS[1]} -F ${TARGETS[2]} > ${TARGETS[3].abspath}"

def run_pycfg(target, source, env, for_signature):
    return env.subst(cmd, target=target, source=source)

def list_to_tuples(xs, n=2):
    return [[xs[i * n + j] for j in range(n)] for i in range(len(xs) / n)]

def run_pycfg_torque(target, source, env):
    """
    Pairs of inputs and outputs
    """
    jobs = []
    interval = 30
    for (cfg, data), (out, grammar, log) in zip(list_to_tuples(source), list_to_tuples(target, 3)):
        job = torque.Job("py-cfg",
                         commands=[env.subst(cmd, target=[out, grammar, log], source=[cfg, data])],
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
        for w, aa in sorted(analyses.iteritems()):
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
    target, (source, train) = args
    gold_morphology = env.subst("data/${LANGUAGE}_morphology.txt")
    if os.path.exists(gold_morphology):
        emma = env.MorphologyOutputToEMMA("%s-emma" % target, source)    
        morphology_results = env.RunEMMA(target, [emma, gold_morphology])
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



def TOOLS_ADD(env):
    env["PYCFG_PATH"] = "/usr/local/py-cfg-quad"
    env.Append(BUILDERS = {
        "MorphologyCFG" : Builder(action=morphology_cfg),
        "TaggingCFG" : Builder(action=tagging_cfg),
        "JointCFG" : Builder(action=joint_cfg),
        "GoldTagsJointCFG" : Builder(action=gold_tags_joint_cfg),
        "GoldSegmentationsJointCFG" : Builder(action=gold_segmentations_joint_cfg),
        "CollateTaggingOutput" : Builder(action=collate_tagging_output),
        "MorphologyOutputToEMMA" : Builder(action=morphology_output_to_emma),
        "CollateJointOutput" : Builder(action=collate_joint_output),
        "TaggingOutputToDataset" : Builder(action=tagging_output_to_dataset),
        "EvaluateManyMorphology" : Builder(action=evaluate_many_morphology),
    })
    if env["HAS_TORQUE"] == True:
        runner = Builder(action=SCons.Action.Action(run_pycfg_torque, batch_key=torque_key))
    else:
        runner = Builder(generator=run_pycfg)
    env.Append(BUILDERS = {"RunPYCFG" : runner})
    env.AddMethod(evaluate_tagging, "EvaluateTagging")
    env.AddMethod(evaluate_tagging, "EvaluateGoldSegmentationsJoint")
    env.AddMethod(evaluate_morphology, "EvaluateMorphology")
    env.AddMethod(evaluate_morphology, "EvaluateGoldTagsJoint")
    env.AddMethod(evaluate_joint, "EvaluateJoint")
    
