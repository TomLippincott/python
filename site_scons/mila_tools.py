from SCons.Builder import Builder
from SCons.Action import Action
from common_tools import meta_open, temp_file, temp_dir
import xml.etree.ElementTree as et
from subprocess import Popen, PIPE
from acceptor_tools import language_filter
import os.path

def split_words(target, source, env, words):
    print len(words)
    good, bad = set(), set()
    r = True
    with temp_dir(remove=r) as raw, temp_dir(remove=r) as tokenized, temp_dir(remove=r) as analyzed:
        with meta_open(os.path.join(raw, "file.txt"), "w") as ofd:
            ofd.write(" ".join(words)) #.encode("utf-8"))
        cmd = env.subst("java -Xmx4024M -jar ${MILA_PATH}/tokenizer.jar %s %s" % (raw, tokenized))
        pid = Popen(cmd.split(), cwd=env.subst("${MILA_PATH}"), stdout=PIPE, stderr=PIPE)
        out, err = pid.communicate()
        #print out, err
        cmd = env.subst("java -Xmx4024M -jar ${MILA_PATH}/morphAnalyzer.jar false %s %s" % (tokenized, analyzed))
        pid = Popen(cmd.split(), cwd=env.subst("${MILA_PATH}"), stdout=PIPE, stderr=PIPE)
        out, err = pid.communicate()
        #print out, err
        with meta_open(os.path.join(analyzed, "file.xml")) as ifd:
            xml = et.parse(ifd)
            for token in xml.getiterator("token"):
                word = token.get("surface")
                unk = [x for x in token.getiterator("unknown")]
                if len(unk) == 0:
                    good.add(word)
                else:
                    bad.add(word)
    return (good, bad)

def TOOLS_ADD(env):
    env["MILA_PATH"] = "${EMNLP_TOOLS_PATH}/hebrew/HebMorphAnalyzer"
    env.Append(BUILDERS = {
            "HebrewFilter" : Builder(action=Action(language_filter(split_words), "HebrewFilter($TARGETS, $SOURCES)")),
            })
