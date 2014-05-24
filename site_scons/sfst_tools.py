from SCons.Builder import Builder
from SCons.Action import Action
from acceptor_tools import language_filter
import re
from subprocess import Popen, PIPE

def make_filter(language):
    def split_words(target, source, env, words):
        pid = Popen(env.subst("${SFST} ${%s_FST}" % language.upper(), target=target, source=source).split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        try:
            stdout, stderr = pid.communicate("\n".join(words).encode("utf-8"))
        except:
            stdout, stderr = pid.communicate("\n".join(words))
        #bad = set([m.group(1) for m in re.finditer(r"^no analysis for \"(.*)\"$", stdout.decode("utf-8"), re.M)])
        bad = set([m.group(1) for m in re.finditer(r"^no analysis for \"(.*)\"$", stdout, re.M)])
        for b in bad:
            if b not in words:
                print b, "<- huh?"
        good = set([x for x in words if x not in bad])
        return (good, bad)
    return Action(language_filter(split_words), "%sFilter($TARGETS, $SOURCES)" % language.title())

def TOOLS_ADD(env):
    env["SFST"] = "${LOCAL_PATH}/bin/fst-parse"
    env["ENGLISH_FST"] = "${EMNLP_TOOLS_PATH}/english/EMOR/emor.a"
    env["GERMAN_FST"] = "${EMNLP_TOOLS_PATH}/german/morphisto-02022011.a"    
    env.Append(BUILDERS = {
            "EnglishFilter" : Builder(action=make_filter("ENGLISH")),
            "GermanFilter" : Builder(action=make_filter("GERMAN")),
            })
