from SCons.Builder import Builder
from SCons.Action import Action
from subprocess import Popen, PIPE
from acceptor_tools import language_filter

def split_words(target, source, env, words):
    cmd = env.subst("perl -I ${ALMOR_PATH}/releases ${ALMOR_PATH}/babel-analyzer.pl ${ALMOR_PATH}/releases/almor-s31.db")
    pid = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = pid.communicate("\n".join(words))
    bad = set([x.strip() for x in stdout.split("\n")])
    good = set([x for x in words if x not in bad])
    return (good, bad)

def TOOLS_ADD(env):
    env["ALMOR_PATH"] = "${EMNLP_TOOLS_PATH}/arabic"
    env.Append(BUILDERS = {
            "ArabicFilter" : Builder(action=Action(language_filter(split_words), "ArabicFilter($TARGETS, $SOURCES)")),
            })
