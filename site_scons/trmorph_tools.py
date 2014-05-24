from SCons.Builder import Builder
from SCons.Action import Action
from subprocess import Popen, PIPE
from acceptor_tools import language_filter

def split_words(target, source, env, words):
    cmd = env.subst("${FLOOKUP} ${TURKISH_FST}")
    try:
        text = "\n".join(words).encode("utf-8")
    except:
        text = "\n".join(words)
    pid = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = pid.communicate(text)
    good, bad = set(), set()
    for l in stdout.split("\n"):
        toks = l.strip().split()
        if len(toks) > 0:            
            if toks[-1].endswith("?"):
                bad.add(toks[0])
            else:
                good.add(toks[0])
    return (good, bad)

def TOOLS_ADD(env):
    env["FOMA"] = "${LOCAL_PATH}/bin/foma"
    env["FLOOKUP"] = "${LOCAL_PATH}/bin/flookup"
    env["TURKISH_FST"] = "${EMNLP_TOOLS_PATH}/turkish/TRmorph/trmorph.fst"
    env.Append(BUILDERS = {
            "TurkishFilter" : Builder(action=Action(language_filter(split_words), "TurkishFilter($TARGETS, $SOURCES)")),
            })
