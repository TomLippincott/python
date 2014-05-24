from common_tools import meta_open, DataSet
from subprocess import Popen, PIPE
import re

def language_filter(split_words):
    def actual_filter(target, source, env):
        with meta_open(source[0].rstr()) as ifd:
            if "xml" in source[0].rstr():
                words = DataSet.from_stream(ifd).indexToWord.values()
            else:
                words = sum([[x for x in re.split(r"\s+", l.strip()) if not re.match(r".*\d.*", x)] for l in ifd], [])
        #words = [w for w in words if re.match(r"^\w+$", w, re.UNICODE)]
        good, bad = split_words(target, source, env, words)
        try:
            with open(target[0].rstr(), "w") as good_ofd, open(target[1].rstr(), "w") as bad_ofd:
                good_ofd.write("\n".join(sorted(good)).encode("utf-8"))
                bad_ofd.write("\n".join(sorted(bad)).encode("utf-8"))
        except:
            with meta_open(target[0].rstr(), "w") as good_ofd, meta_open(target[1].rstr(), "w") as bad_ofd:
                good_ofd.write("\n".join(sorted(good)))
                bad_ofd.write("\n".join(sorted(bad)))
        return None
    return actual_filter
