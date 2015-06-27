from SCons.Builder import Builder
from SCons.Action import Action
from common_tools import meta_open, DataSet
import re
import unicodedata

def graphemic_pronunciations(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        items = [x.strip() for x in ifd]
    with meta_open(target[0].rstr(), "w") as ofd:
        #ofd.write("\n".join(["%s\t%s" % (w, " ".join(["u%.4x" % (ord(c)) for c in w if c not in ["+", "-", unichr(1100), unichr(1098), ]])) for w in items]))
        ofd.write("\n".join(["%s\t%s" % (w, " ".join(["u%.4x" % (ord(c)) for c in w if unicodedata.category(c)[0] == "L" and c not in [unichr(1100), unichr(1098)]])) for w in items]))
    return None

def segmented_pronunciations(target, source, env):
    sep = "+"
    data = {}
    segs = {}
    morphs = set()
    w2w = {}
    with meta_open(source[0].rstr()) as pron_fd, meta_open(source[1].rstr()) as seg_fd:
        for l in pron_fd:
            word, pron = re.match(r"^(\S+)\(\d+\) (.*)$", l.strip().replace(" [ wb ]", "")).groups()
            if env.get("LOWER_CASE"):
                word = word.lower()
            data[word] = data.get(word, []) + [pron]
            #w2s = {"".join(ms) : sep.join(ms) for ms in 
        #d = DataSet.from_stream(seg_fd)
        vals = [[y.strip(sep) for y in x.split()] for x in seg_fd]
        if env.get("LOWER_CASE"):
            vals = [[y.lower() for y in x] for x in vals]
        segs = {sep.join(x) : x for x in vals}
        w2s = {"".join(x) : sep.join(x) for x in vals} #d[0].indexToAnalysis.values()}
        for m in segs.values():
            if len(m) == 1:
                morphs.add(m[0])
            if len(m) >= 2:
                morphs.add("%s%s" % (m[0], sep))
                morphs.add("%s%s" % (sep, m[1]))
                for x in m[1:-1]:
                    morphs.add("%s%s%s" % (sep, x, sep))
    with meta_open(target[0].rstr(), "w") as seg_ofd, meta_open(target[1].rstr(), "w") as morph_ofd:
        morph_ofd.write("\n".join(morphs))
        seg_ofd.write("\n".join(sum([["%s %s" % (w2s.get(k, k), p) for p in v] for k, v in data.iteritems()], [])))
    return None

def train_g2p(target, source, env, for_signature):
    if len(source) == 1:
        return "PYTHONPATH=${G2P_PATH}:$${PYTHONPATH} ${G2P} -e utf-8 -t ${SOURCES[0]} -n ${TARGETS[0]}"
    elif len(source) == 2:
        return "PYTHONPATH=${G2P_PATH}:$${PYTHONPATH} ${G2P} -e utf-8 -m ${SOURCES[0]} --ramp-up -t ${SOURCES[1]} -n ${TARGETS[0]}"

def pronunciations_to_vocab_dict(target, source, env):
    graphemic = source[-1].read()
    prons = {}
    with meta_open(source[0].rstr()) as ifd:
        for l in ifd:
            try:
                morph, num, prob, phones = l.strip().split("\t")
            except:
                try:
                    morph, num, prob = l.strip().split("\t")
                except:
                    try:
                        morph, phones = l.strip().split("\t")
                        num = "1"
                    except:
                        morph = l.strip()
                        phones = "SIL"
                        num = "1"
            num = int(num) + 1
            prons["%s(%.2d)" % (morph, num)] = (morph, phones.split())
    with meta_open(target[0].rstr(), "w") as vocab_ofd, meta_open(target[1].rstr(), "w") as dict_ofd:
        wb = ["[", "wb", "]"]
        for w, (m, p) in prons.iteritems():
            if not graphemic:
                if len(p) == 1:
                    p = p + wb
                else:
                    p = [p[0]] + wb + p[1:] + wb
            dict_ofd.write("%s %s\n" % (w, " ".join(p)))
            vocab_ofd.write("%s %s\n" % (w, m))
        vocab_ofd.write("""<s>(01) <s>
</s>(01) </s>
~SIL(01) VOCAB_NIL_WORD 1.4771
~SIL(02) VOCAB_NIL_WORD 1.4771
~SIL(03) VOCAB_NIL_WORD 1.4771
""")
        if graphemic:
            dict_ofd.write("""<s>(01) SIL
</s>(01) SIL
~SIL(01) SIL
~SIL(02) NS
~SIL(03) VN
""")
        else:
            dict_ofd.write("""<s>(01) SIL [ wb ]
</s>(01) SIL [ wb ]
~SIL(01) SIL [ wb ]
~SIL(02) NS [ wb ]
~SIL(03) VN [ wb ]
""")
    return None

def TOOLS_ADD(env):
    env.Append(BUILDERS = {
        "SegmentedPronunciations" : Builder(action=segmented_pronunciations),
        "TrainG2P" : Builder(generator=train_g2p),
        "ApplyG2P" : Builder(action="PYTHONPATH=${G2P_PATH}:$${PYTHONPATH} ${G2P} -e utf-8 -m ${SOURCES[0]} -a ${SOURCES[1]} --variants-number=1 > ${TARGETS[0]}"),
        "PronunciationsToVocabDict" : Builder(action=pronunciations_to_vocab_dict),
        "GraphemicPronunciations" : Builder(action=graphemic_pronunciations),
    })
