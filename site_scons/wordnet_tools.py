from SCons.Builder import Builder
from common_tools import meta_open
from subprocess import Popen

def english_filter(target, source, env):
    with meta_open(source[0].rstr()) as ifd:
        words = ProbabilityList.from_stream(ifd)
    new_words = ProbabilityList()
    for w, p in words.iteritems():        
        pid = Popen(env.subst("${WORDNET} %s -over" % w).split())        
        stdout, stderr = pid.communicate()
        if pid.returncode > 0:
            new_words[w] = p
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(new_words.format())
    return None

def english_filter_emitter(target, source, env):
    if not target[0].rstr().endswith(".gz"):
        new_target = "%s_wordnet-filtered.gz" % os.path.splitext(source[0].rstr())[0]
    else:
        new_target = target[0]
    return new_target, source

def TOOLS_ADD(env):
    env["WORDNET"] = "wn"
    env.Append(BUILDERS = {
            "EnglishFilter" : Builder(action=english_filter, emitter=english_filter_emitter),
            })
