from SCons.Builder import Builder
from SCons.Action import Action
import re
from glob import glob
from functools import partial
import logging
import os.path
from os.path import join as pjoin
import os
import cPickle as pickle
import math
import xml.etree.ElementTree as et
import gzip
import subprocess
import shlex
import time
import shutil
import tempfile
from common_tools import meta_open, temp_dir
from torque_tools import run_command, SimpleCommandThreadedBuilder, SimpleTorqueBuilder
from babel import ProbabilityList, Arpabo, Pronunciations, Vocabulary
import torque

def query_files(target, source, env):
    # OUTPUT:
    # iv oov map w2w <- 
    # INPUT: kw, iv, id
    # pad id to 4
    remove_vocab = source[-1].read()
    with meta_open(source[0].rstr()) as kw_fd, meta_open(source[1].rstr()) as iv_fd:
        keyword_xml = et.parse(kw_fd)
        keywords = set([(x.get("kwid"), x.find("kwtext").text.lower()) for x in keyword_xml.getiterator("kw")])
        #print list(keywords)[0][1].split()
        vocab = [x.decode("utf-8") for x in Pronunciations(iv_fd).get_words()]
        #print list(vocab)[0].split()
        #set([x.split()[1].strip().decode("utf-8") for x in iv_fd])
        if remove_vocab:
            remove_vocab = Vocabulary(meta_open(remove_vocab)).get_words()
        else:
            remove_vocab = []
        iv_keywords = sorted([(int(tag.split("-")[-1]), tag, term) for tag, term in keywords if all([y in vocab for y in term.split()]) and term not in remove_vocab])
        oov_keywords = sorted([(int(tag.split("-")[-1]), tag, term) for tag, term in keywords if any([y not in vocab for y in term.split()])])
        language_id = source[-2].read()
        with meta_open(target[0].rstr(), "w") as iv_ofd, meta_open(target[1].rstr(), "w") as oov_ofd, meta_open(target[2].rstr(), "w") as map_ofd, meta_open(target[3].rstr(), "w") as w2w_ofd, meta_open(target[4].rstr(), "w") as kw_ofd:
            iv_ofd.write("\n".join([x[2].encode("utf-8") for x in iv_keywords]))
            oov_ofd.write("\n".join([x[2].encode("utf-8") for x in oov_keywords]))
            map_ofd.write("\n".join(["%s %.5d %.5d" % x for x in 
                                     sorted([("iv", gi, li) for li, (gi, tag, term) in enumerate(iv_keywords, 1)] + 
                                            [("oov", gi, li) for li, (gi, tag, term) in enumerate(oov_keywords, 1)], lambda x, y : cmp(x[1], y[1]))]))
            w2w_ofd.write("\n".join([("0 0 %s %s 0" % (x.encode("utf-8"), x.encode("utf-8"))) for x in vocab if x != "VOCAB_NIL_WORD"] + ["0"]))
            for x in keyword_xml.getiterator("kw"):
                x.set("kwid", "KW%s-%s" % (language_id, x.get("kwid").split("-")[-1]))
            keyword_xml.write(kw_ofd) #.write(et.tostring(keyword_xml.))
    return None

def ecf_file(target, source, env):
    with open(source[0].rstr()) as fd:
        files = [(fname, int(chan), float(begin), float(end)) for fname, num, sph, chan, begin, end in [line.split() for line in fd]]
        data = {}
        for fname in set([x[0] for x in files]):
            relevant = [x for x in files if x[0] == fname]
            start = sorted(relevant, lambda x, y : cmp(x[2], y[2]))[0][2]
            end = sorted(relevant, lambda x, y : cmp(x[3], y[3]))[-1][3]
            channels = relevant[0][1]
            data[fname] = (channels, start, end)
        total = sum([x[2] - x[1] for x in data.values()])
        tb = et.TreeBuilder()
        tb.start("ecf", {"source_signal_duration" : "%f" % total, "language" : "", "version" : ""})
        for k, (channel, start, end) in data.iteritems():
            tb.start("excerpt", {"audio_filename" : k, "channel" : str(channel), "tbeg" : str(start), "dur" : str(end-start), "source_type" : "splitcts"})
            tb.end("excerpt")
        tb.end("ecf")        
        with open(target[0].rstr(), "w") as ofd:
            ofd.write(et.tostring(tb.close()))
    return None

def database_file(target, source, env):
    return None

def lattice_list(target, source, env):
    """
    Creates a file that's simply a list of the lattices in the given directory (absolute paths, one per line).
    """
    lattice_dir = source[1].read()
    if not os.path.exists(lattice_dir):
        return "No such directory: %s" % lattice_dir
    meta_open(target[0].rstr(), "w").write("\n".join([os.path.abspath(x) for x in glob(os.path.join(lattice_dir, "*"))]) + "\n")
    return None

def word_pronounce_sym_table(target, source, env):
    """
    convert dictionary format:
      
      WORD(NUM) WORD

    to numbered format:

      WORD(NUM) NUM

    with <EPSILON> as 0
    """
    ofd = meta_open(target[0].rstr(), "w")
    ofd.write("<EPSILON>\t0\n")
    for i, line in enumerate(meta_open(source[0].rstr())):
        ofd.write("%s\t%d\n" % (line.split()[0], i + 1))
    ofd.close()
    return None

def clean_pronounce_sym_table(target, source, env):
    """
    Deduplicates lexicon, removes "(NUM)" suffixes, and adds <query> entry.
    """
    ofd = meta_open(target[0].rstr(), "w")
    seen = set()
    for line in meta_open(source[0].rstr()):
        if line.startswith("<EPSILON>"):
            word = "<EPSILON>"
        else:
            word = re.match(r"^(.*)\(\d+\)\s+.*$", line).groups()[0]
        if word not in seen:
            ofd.write("%s\t%d\n" % (word, len(seen)))
            seen.add(word)
    ofd.write("<query>\t%d\n" % (len(seen)))
    return None

def munge_dbfile(target, source, env):
    """
    NEEDS WORK!
    Converts a database file.
    """
    with meta_open(source[1].rstr()) as fd:
        lattice_files = set([os.path.basename(x.strip()) for x in fd])
    ofd = meta_open(target[0].rstr(), "w")
    for line in meta_open(source[0].rstr()):
        toks = line.split()
        fname = "%s.fsm.gz" % ("#".join(toks[0:2]))
        if fname in lattice_files:
            ofd.write(" ".join(toks[0:4] + ["0.0", toks[5]]) + "\n")
    ofd.close()
    return None

def create_data_list(target, source, env):
    """
    NEEDS WORK!
    Creates the master list of lattice transformations.
    """
    args = source[-1].read()
    data = {}
    for line in meta_open(source[0].rstr()):
        toks = line.split()
        bn = os.path.basename(toks[2])
        data[toks[0]] = data.get(toks[0], {})
        data[toks[0]][toks[1]] = (bn, toks[4], toks[5])
    ofd = meta_open(target[0].rstr(), "w")
    for lattice_file in glob(os.path.join(args["LATTICE_DIR"], "*")):
        bn = os.path.basename(lattice_file)
        path = os.path.join(env["BASE_PATH"], "lattices")
        uttname, delim, uttnum = re.match(r"(.*)([^\w])(\d+)\.%s$" % (args["oldext"]), bn).groups()
        try:
            name, time, timeend = data[uttname][uttnum]
            newname = os.path.abspath(os.path.join(path, "%s%s%s.%s" % (uttname, delim, uttnum, args["ext"])))
            ofd.write("%s %s %s %s %s.osym %s\n" % (os.path.splitext(name)[0], time, timeend, newname, newname, os.path.abspath(lattice_file)))
        except:
            return "lattice file not found in database: %s (are you sure your database file matches your lattice directory?)" % bn
    ofd.close()
    return None

def split_list(target, source, env):
    """
    Splits the lines in a file evenly among the targets.
    """
    lines = [x for x in meta_open(source[0].rstr())]
    per_file = len(lines) / len(target)
    for i, fname in enumerate(target):
        start = int((i * float(len(lines))) / len(target))
        end = int(((i + 1) * float(len(lines))) / len(target))
        if i == len(target) - 1:
            meta_open(fname.rstr(), "w").write("".join(lines[start : ]))
        else:
            meta_open(fname.rstr(), "w").write("".join(lines[start : end]))
    return None

def word_to_phone_lattice_torque(target, source, env):
    args = source[-1].read()
    data_list, lattice_list, wordpron, dic = source[0:4]
    args["DICTIONARY"] = dic.abspath
    args["DATA_FILE"] = data_list.abspath
    args["FSMGZ_FORMAT"] = "true"
    args["CONFUSION_NETWORK"] = ""
    args["FSM_DIR"] = "temp"
    args["WORDPRONSYMTABLE"] = wordpron.abspath
    argstr = "-d %(DICTIONARY)s -D %(DATA_FILE)s -t %(FSMGZ_FORMAT)s -s %(WORDPRONSYMTABLE)s -S %(EPSILON_SYMBOLS)s %(CONFUSION_NETWORK)s -P %(PRUNE_THRESHOLD)d %(FSM_DIR)s" % args
    if not os.path.exists(os.path.dirname(target[0].rstr())):
        os.makedirs(os.path.dirname(target[0].rstr()))
    command = env.subst("cat ${SOURCES[1].abspath}|${WRD2PHLATTICE} %s" % (argstr), source=source)
    interval = args.get("interval", 10)
    args["path"] = args.get("path", target[0].get_dir())
    stdout = env.Dir(args.get("stdout", args["path"])).Dir("stdout").rstr()
    stderr = env.Dir(args.get("stderr", args["path"])).Dir("stderr").rstr()
    if not os.path.exists(stdout):
        os.makedirs(stdout)
    if not os.path.exists(stderr):
        os.makedirs(stderr)
    job = torque.Job(args.get("name", "scons-wrd2phlattice"),
                     commands=["mkdir -p temp", command],
                     path=args.get("path", target[0].get_dir()).rstr(),
                     stdout_path=stdout,
                     stderr_path=stderr,
                     other=args.get("other", ["#PBS -W group_list=yeticcls"]),
                     )
    job.submit(commit=True)
    while job.job_id in [x[0] for x in torque.get_jobs(True)]:
        logging.info("sleeping...")
        time.sleep(interval)
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(time.asctime() + "\n")
    return None


def word_to_phone_lattice(target, source, env):
    args = source[-1].read()
    data_list, lattice_list, wordpron, dic = source[0:4]
    args["DICTIONARY"] = dic.rstr()
    args["DATA_FILE"] = data_list.rstr()
    args["FSMGZ_FORMAT"] = "true"
    args["CONFUSION_NETWORK"] = ""
    args["FSM_DIR"] = "temp"
    args["WORDPRONSYMTABLE"] = wordpron.rstr()
    return ("${WRD2PHLATTICE} -d %(DICTIONARY)s -D %(DATA_FILE)s -t %(FSMGZ_FORMAT)s -s %(WORDPRONSYMTABLE)s -S %(EPSILON_SYMBOLS)s %(CONFUSION_NETWORK)s -P %(PRUNE_THRESHOLD)d %(FSM_DIR)s",
            args)




def get_file_list(target, source, env):
    """
    Extracts the third field from a database file (i.e. the name of the lattice file).
    """
    meta_open(target[0].rstr(), "w").write("\n".join([x.split()[3] for x in meta_open(source[0].rstr())]))
    return None

def build_index(target, source, env):
    """
    Creates an index of files listed in the input, using the IBM binary 'buildindex'.

Usage: /mnt/calculon-minor/lorelei_svn/KWS/bin64/buildindex [-opts] [lattice_list] [output_file]            
Options:                                                                 
-f file     filter fst (default : none)                                  
-p          push costs                                                   
-J int      job-batch (for parallel run)                                 
-N int      total number of jobs (for parallel run)                      
-v          (verbose) if specified all debug output is printed to stderr 
-?      help
    """
    command = env.subst("${BUILDINDEX} -p ${SOURCE} ${TARGET}", target=target, source=source)
    stdout, stderr, success = run_command(command, env={"LD_LIBRARY_PATH" : env.subst(env["LIBRARY_OVERLAY"])}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not success:
        return stderr
    return None

def build_pad_fst(target, source, env):
    """
printing usage
Usage: /mnt/calculon-minor/lorelei_svn/KWS/bin64/buildpadfst [symtable_file] [output_fst_file]
    """
    command = env.subst("${BUILDPADFST} ${SOURCE} ${TARGET}", target=target, source=source)
    stdout, stderr, success = run_command(command, env={"LD_LIBRARY_PATH" : env.subst(env["LIBRARY_OVERLAY"])}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not success:
        return stderr
    return None

def fst_compile(target, source, env):
    """
    Compile an FST using OpenFST's binary 'fstcompile'.
    """
    command = env.subst("${FSTCOMPILE} --isymbols=${SOURCES[0]} --osymbols=${SOURCES[0]} ${SOURCES[1]}", target=target, source=source)
    stdout, stderr, success = run_command(command, env={"LD_LIBRARY_PATH" : env.subst(env["LIBRARY_OVERLAY"])}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not success:
        return stderr
    meta_open(target[0].rstr(), "w").write(stdout)
    return None

def query_to_phone_fst(target, source, env):
    """
Usage: /mnt/calculon-minor/lorelei_svn/KWS/bin64/query2phonefst [-opts] [outputdir] [querylist]                                        
-d file         dictionary file                                                                 
-s file         use external phone table specified                                              
-O file         file containing prons for oovs, output of l2s system                            
-l file         file to output list of all fsts corresponding to queries                        
-I int          ignore (print empty fst-file) if query has less than <int> phones               
-t double       if specified, tag oovs with soft threshold indicated.                           
-u              if specified, ignore weight of alternative prons                                
-g              add gamma penalty for query length p = p^gamma (gamma=1/lenght-phone)           
-w              if specified, query is represented as one arc per word, not converted to phones 
-p p2pfile      p2pfile, to allow for fuzziness in query (default:no p2p)                       
-n nbest        if p2pfile, this limits number of paths retaind after composing query with p2p  
-?              info/options
    """
    args = source[-1].read()
    try:
        os.makedirs(args["OUTDIR"])
    except:
        pass
    command = env.subst("${QUERY2PHONEFST} -p ${SOURCES[0]} -s ${SOURCES[1]} -d ${SOURCES[2]} -l ${TARGETS[0]} -n %(n)d -I %(I)d %(OUTDIR)s ${SOURCES[3]}" % args, target=target, source=source)
    #command = env.subst("${BABEL_REPO}/KWS/bin64/query2phonefst -s ${SOURCES[1]} -d ${SOURCES[2]} -l ${TARGETS[0]} -I %(I)d %(OUTDIR)s ${SOURCES[3]}" % args, target=target, source=source)
    #print command
    stdout, stderr, success = run_command(command, env={"LD_LIBRARY_PATH" : env.subst(env["LIBRARY_OVERLAY"])}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not success:
        return stderr
    return None

def standard_search(target, source, env):
    """
Usage: /mnt/calculon-minor/lorelei_svn/KWS/bin64/stdsearch [-opts] [result_file] [query_file]                     
Options:                                                                        
-d file          data file [data.list] with lines in the following format :     
                 utt_name start_time fst_path (default: data.list)              
-f filt          filter fst (default: none)                                     
-i fst           index fst (default: index.fst)                                 
-n N             return N-best results (default: return all)                    
-p fst           pad fst (default: fspad.fst)                                   
-s symbols       arc symbols (default: word.list)                               
-t threshold     min score needed to decide YES,                                
                 (if specified it overrides term-spec-threshold (default))      
-T true/false    true (default)=queries are in text format, false=fst format    
                 txtformat: query list is a list of queries,                    
                 fstformat: query list is a list of full-paths to query fsts    
-J int           job-batch (for parallel run)                                   
-N int           total number of jobs (for parallel run)                        
-a string        title on results list (default: stdbn.tlist.xml)               
-b string        prefix on termid (default: TERM-0)                             
-m string        termid numerical formatting string (default: -1524500936)             
-O               if specified, don't optimize (default : optimize = true)       
-v               (verbose) if specified, print all debug outputs to stderr      
-?               info/options
    """
    data_list, isym, idx, pad, queryph = source[0:5]
    args = source[-1].read()
    if source[-2].stat().st_size == 0:
        with meta_open(target[0].rstr(), "w") as ofd:
            ofd.write("""<stdlist termlist_filename="std.xml" indexing_time="68.51" language="english" index_size="" system_id="" />\n""")
        return None
    command = env.subst("${STDSEARCH} -F ${TARGET} -i ${SOURCES[2]} -b KW%(LANGUAGE_ID)s- -s ${SOURCES[1]} -p ${SOURCES[3]} -d ${SOURCES[0]} -a %(TITLE)s -m %(PRECISION)s ${SOURCES[4]}" % args, target=target, source=source)
    stdout, stderr, success = run_command(command, env={"LD_LIBRARY_PATH" : env.subst(env["LIBRARY_OVERLAY"])}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not success:
        return stderr
    return None

def standard_search_torque(target, source, env):
    """
Usage: /mnt/calculon-minor/lorelei_svn/KWS/bin64/stdsearch [-opts] [result_file] [query_file]                     
Options:                                                                        
-d file          data file [data.list] with lines in the following format :     
                 utt_name start_time fst_path (default: data.list)              
-f filt          filter fst (default: none)                                     
-i fst           index fst (default: index.fst)                                 
-n N             return N-best results (default: return all)                    
-p fst           pad fst (default: fspad.fst)                                   
-s symbols       arc symbols (default: word.list)                               
-t threshold     min score needed to decide YES,                                
                 (if specified it overrides term-spec-threshold (default))      
-T true/false    true (default)=queries are in text format, false=fst format    
                 txtformat: query list is a list of queries,                    
                 fstformat: query list is a list of full-paths to query fsts    
-J int           job-batch (for parallel run)                                   
-N int           total number of jobs (for parallel run)                        
-a string        title on results list (default: stdbn.tlist.xml)               
-b string        prefix on termid (default: TERM-0)                             
-m string        termid numerical formatting string (default: -1524500936)             
-O               if specified, don't optimize (default : optimize = true)       
-v               (verbose) if specified, print all debug outputs to stderr      
-?               info/options
    """
    data_list, isym, idx, pad, queryph = source[0:5]
    args = source[-1].read()
    interval = args.get("interval", 30)
    if source[-2].stat().st_size == 0:
        with meta_open(target[0].rstr(), "w") as ofd:
            ofd.write("""<stdlist termlist_filename="std.xml" indexing_time="68.51" language="english" index_size="" system_id="" />\n""")
        return None
    command = env.subst("${STDSEARCH} -F ${TARGET} -i ${SOURCES[2]} -b KW%(LANGUAGE_ID)s- -s ${SOURCES[1]} -p ${SOURCES[3]} -d ${SOURCES[0]} -a %(TITLE)s -m %(PRECISION)s ${SOURCES[4]}" % args, target=target, source=source)
    job = torque.Job(args.get("name", "scons-stdsearch"),
                     commands=[command],
                     path=args.get("path", target[0].get_dir()).rstr(),
                     stdout_path=pjoin(target[0].get_dir().rstr(), "stdout"),
                     stderr_path=pjoin(target[0].get_dir().rstr(), "stderr"),
                     other=args.get("other", ["#PBS -W group_list=yeticcls"]),
                     )
    if env["HAS_TORQUE"]:
        print str(job)
        job.submit(commit=True)
        while job.job_id in [x[0] for x in torque.get_jobs(True)]:
            logging.info("sleeping...")
            time.sleep(interval)
    else:
        logging.info("no Torque server, but I would submit:\n%s" % (job))
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(time.asctime() + "\n")
    return None

def merge(target, source, env):
    """
    NEEDS WORK!
    CONVERT TO BUILDER!
    Combines the output of several searches
    input: XML files (<term>)
    output: 
    """
    args = source[-1].read()
    #stdout, stderr, success = run_command(env.subst("${BABEL_REPO}/KWS/scripts/printQueryTermList.prl -padlength=%(PADLENGTH)d ${SOURCES[0]}" % args, 
    #                                                target=target, source=source), env={"LD_LIBRARY_PATH" : env.subst("${LIBRARY_OVERLAY}")})
    stdout, stderr, success = run_command(env.subst("${PRINTQUERYTERMLISTPRL} -prefix=KW%(LANGUAGE_ID)s- -padlength=%(PADLENGTH)d ${SOURCES[0]}" % args, 
                                                    target=target, source=source), env={"LD_LIBRARY_PATH" : env.subst("${LIBRARY_OVERLAY}")})
    meta_open(target[0].rstr(), "w").write(stdout)
    meta_open(target[1].rstr(), "w").write("\n".join([x.rstr() for x in source[1:-1]]))
    if args["MODE"] == "merge-atwv":
        return "merge-atwv option not supported!"
    else:        
        merge_search_from_par_index = "${MERGESEARCHFROMPARINDEXPRL} -force-decision=\"YES\" ${TARGETS[0]} ${TARGETS[1]}"
        stdout, stderr, success = run_command(env.subst(merge_search_from_par_index, target=target, source=source), env={"LD_LIBRARY_PATH" : env.subst("${LIBRARY_OVERLAY}")})
        meta_open(target[2].rstr(), "w").write(stdout)
        meta_open(target[3].rstr(), "w").write("\n".join(stdout.split("\n")))
    return None
1
def merge_scores(target, source, env):
    """
    NEEDS WORK!
    CONVERT TO BUILDER!
    """
    stdout, stderr, success = run_command(env.subst("${MERGESCORESSUMPOSTNORMPL} ${SOURCES[0]}", target=target, source=source), env={"LD_LIBRARY_PATH" : env.subst("${LIBRARY_OVERLAY}")})
    if not success:
        return stderr
    meta_open(target[0].rstr(), "w").write(stdout)
    return None

def merge_iv_oov(target, source, env):
    """
    The in-vocabulary and out-of-vocabulary search terms have their own numberings, and must be mapped into the universal numbering for evaluation.

    This corresponds to the 'merge_iv_oov.pl' script from IBM.
    """
    iv_xml = et.parse(open(source[0].rstr()))
    oov_xml = et.parse(open(source[1].rstr()))
    term_map = {(s, int(sn)) : int(un) for s, un, sn in [x.strip().split() for x in open(source[2].rstr())]}
    tb = et.TreeBuilder()
    stdlist = tb.start("stdlist", {"termlist_filename" : os.path.basename(source[-1].rstr()), "indexing_time" : "", "language" : "", "index_size" : "", "system_id" : ""})
    for term_list in iv_xml.getiterator("detected_termlist"):
        p, n = term_list.get("termid").split("-")
        term_list.set("termid", "%s-%0.5d" % (p, term_map[("iv", int(n))]))
        stdlist.append(term_list)
    for term_list in oov_xml.getiterator("detected_termlist"):
        p, n = term_list.get("termid").split("-")
        term_list.set("termid", "%s-%0.5d" % (p, term_map[("oov", int(n))]))
        stdlist.append(term_list)
    tb.end("stdlist")
    open(target[0].rstr(), "w").write(et.tostring(tb.close()))
    return None

def normalize(target, source, env):
    """
    Takes the combined IV and OOV results
    Remove keywords not in kwlist
    NEEDS WORK!
    CONVERT TO BUILDER!
    """
    tmpfile_fid, tmpfile_name = tempfile.mkstemp()
    res_xml = et.parse(meta_open(source[0].rstr()))

    kw_ids = {(a, b.lstrip("0")) : "%s-%s" % (a, b) for a, b in [x.split("-") for x in set([x.get("kwid") for x in et.parse(meta_open(source[1].rstr())).getiterator("kw")])]}
    elems = [x for x in res_xml.getiterator("detected_termlist")] # if x.get("termid") not in kw_ids]
    for e in elems:
        a, b = e.get("termid").split("-")
        b = b.lstrip("0")
        if (a, b) not in kw_ids:
            res_xml.getroot().remove(e)
            #print kw_ids[(a, b)]
        else:
            #print kw_ids[(a, b)]
            e.set("termid", kw_ids[(a, b)])

    res_xml.write(tmpfile_name)
    stdout, stderr, success = run_command(env.subst("${PYTHON} ${F4DENORMALIZATIONPY} ${SOURCE} ${TARGET}", target=target, source=tmpfile_name))
    os.remove(tmpfile_name)
    if not success:
        print stderr
    return None

def normalize_sum_to_one(target, source, env):
    """
    NEEDS WORK!
    CONVERT TO BUILDER!
    """
    stdout, stderr, success = run_command(env.subst("java -cp ${JAVA_NORM} normalization.ApplySumToOneNormalization ${SOURCE} ${TARGET}", target=target, source=source))
    if not success:
        return stderr
    return None

def score(target, source, env):
    """
    NEEDS WORK!
    CONVERT TO BUILDER!
    """
    args = source[-1].read()

    with temp_dir("kws_work") as work_dir, temp_dir("kws_out") as out_dir:
        cmd = env.subst("${PERL} ${F4DE}/bin/BABEL13_Scorer -XmllintBypass -sys ${SOURCE} -dbDir ${INDUS_DB} -comp %s -res %s -exp %s" % (work_dir, out_dir, args.get("EXPID", "KWS13_IBM_babel106b-v0.2g_conv-dev_BaDev_KWS_FullLP_BaseLR_NTAR_p-test-STO_1")), source=source)
        #cmd = env.subst("${F4DE}/KWSEval/BABEL/Participants/BABEL_Scorer.pl -XmllintBypass -sys ${SOURCE} -dbDir ${INDUS_DB} -comp %s -res %s -exp %s" % (work_dir, out_dir, args.get("EXPID", "KWS13_IBM_babel106b-v0.2g_conv-dev_BaDev_KWS_FullLP_BaseLR_NTAR_p-test-STO_1")), source=source)
        stdout, stderr, success = run_command(cmd, env={"LD_LIBRARY_PATH" : env.subst("${LIBRARY_OVERLAY}"), 
                                                        "F4DE_BASE" : env.subst(env["F4DE"]),
                                                        "PERL5LIB" : env.subst("$PERL_LIBRARIES"),
                                                        "PATH" : ":".join([env.subst("${OVERLAY}/bin")] + os.environ["PATH"].split(":"))})
        if not success:
            return stderr + stdout
        else:
            shutil.rmtree(os.path.dirname(target[0].rstr()), ignore_errors=False)
            shutil.copytree(out_dir, os.path.dirname(target[0].rstr()))
    return None
    #tmpfile_fid, tmpfile_name = tempfile.mkstemp()
    

    #theargs = {}
    #theargs.update(args)
    #theargs.update({"KWS_LIST_FILE" : source[0].rstr(), "PREFIX" : tmpfile_name})
    #cmd = env.subst("${KWSEVALPL} -e %(ECF_FILE)s -r %(RTTM_FILE)s -s %(KWS_LIST_FILE)s -t ${SOURCES[1]} -o -b -f %(PREFIX)s" % theargs,
    #                source=source, target=target)                    
    #cmd = env.subst("${F4DE}/bin/BABEL13_Scorer -e %(ECF_FILE)s -r %(RTTM_FILE)s -s %(KWS_LIST_FILE)s -t ${SOURCES[1]} -o -b -f %(PREFIX)s" % theargs,
    #source=source, target=target)                    
    #stdout, stderr, success = run_command(cmd, env={"LD_LIBRARY_PATH" : env.subst("${LIBRARY_OVERLAY}"), "PERL5LIB" : env.subst("${OVERLAY}/lib/perl5/site_perl:${F4DE}/common/lib:${F4DE}/KWSEval/lib/"), "PATH" : "/usr/bin"})
    #if not success:
    #    return stderr + stdout
    #os.remove(tmpfile_name)
    #shutil.move("%s.sum.txt" % tmpfile_name, target[0].rstr())
    #shutil.move("%s.bsum.txt" % tmpfile_name, target[1].rstr())
    return None

def score_emitter(target, source, env):
#    new_targets = [os.path.join(target[0].rstr(), x) for x in ["Ensemble.AllOccur.results.txt"]]
#    print new_targets
    return target, source

def alter_iv_oov(target, source, env):
    """
    NEEDS WORK!
    If the vocabulary has been expanded, some OOV terms are now IV.
    """
    iv_q, oov_q, iv, term_map, kw_file, w2w_file = source
    with meta_open(iv_q.rstr()) as iv_q_fd, meta_open(oov_q.rstr()) as oov_q_fd, meta_open(iv.rstr()) as iv_fd, meta_open(term_map.rstr()) as term_map_fd, meta_open(kw_file.rstr()) as kw_file_fd, meta_open(w2w_file.rstr()) as w2w_fd:
        iv_queries = [x.strip() for x in iv_q_fd]
        oov_queries = [x.strip() for x in oov_q_fd]
        iv_words = [x.strip().split("(")[0] for x in iv_fd]
        oov_to_iv_indices = [i for i, q in enumerate(oov_queries) if all([x in iv_words for x in q.split()])]
        oov_to_oov_indices = enumerate([i for i, q in enumerate(oov_queries) if not all([x in iv_words for x in q.split()])])
        new_iv_queries = iv_queries + [oov_queries[i] for i in oov_to_iv_indices]
        new_oov_queries = [x for i, x in enumerate(oov_queries) if i not in oov_to_iv_indices]
        old_mapping = {(y[0], int(y[2])) : y[1] for y in [x.strip().split() for x in term_map_fd]}
        new_mapping = old_mapping.copy()
        for i, old_oov_num in enumerate(oov_to_iv_indices):
            x = old_mapping[("oov", old_oov_num + 1)]
            new_iv_num = len(iv_queries) + i + 1
            del new_mapping[("oov", old_oov_num + 1)]
            new_mapping[("iv", new_iv_num)] = x
        for new_oov, old_oov in oov_to_oov_indices:
            x = old_mapping[("oov", old_oov + 1)]
            del new_mapping[("oov", old_oov + 1)]
            new_mapping[("oov", new_oov + 1)] = x
        #old_xml = et.fromstring(kw_file_fd.read())
        new_w2w = [" ".join(y) for y in set([tuple(x.split()) for x in w2w_fd if len(x.split()) == 5] + [("0", "0", x, x, "0") for x in iv_words])]
        open(target[0].rstr(), "w").write("\n".join(new_iv_queries) + "\n")
        open(target[1].rstr(), "w").write("\n".join(new_oov_queries) + "\n")
        #open(target[2].rstr(), "w").write(open(term_map.rstr()).read())
        open(target[2].rstr(), "w").write("\n".join(["%s %s %0.5d" % (s, on, n) for (s, n), on in sorted(new_mapping.iteritems(), lambda x, y : cmp(x[1], y[1]))]))
        open(target[3].rstr(), "w").write(kw_file_fd.read())
        open(target[4].rstr(), "w").write("\n".join(new_w2w) + "\n0\n")
    return None

def collate_scores(target, source, env):
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write("\t".join(["Experiment", "PMiss", "PMissMTWV", "MTWV"]) + "\n")
        for fname in [x.rstr() for x in source]:
            exp = fname.split("/")[1]
            toks = [[x.strip() for x in l.split("|")] for l in open(fname)][-1]
            pmiss = toks[-10]
            pmiss2, mtwv = toks[-6:-4]
            ofd.write("\t".join([exp, pmiss, pmiss2, mtwv]) + "\n")
    return None


def TOOLS_ADD(env):
    BUILDERS = {'LatticeList' : Builder(action=lattice_list), 
                'ECFFile' : Builder(action=ecf_file),
                'WordPronounceSymTable' : Builder(action=word_pronounce_sym_table), 
                'CleanPronounceSymTable' : Builder(action=clean_pronounce_sym_table), 
                'MungeDatabase' : Builder(action=munge_dbfile), 
                'CreateDataList' : Builder(action=create_data_list),
                'SplitList' : Builder(action=split_list), 
                #'WordToPhoneLattice' : Builder(action=word_to_phone_lattice), 
                'GetFileList' : Builder(action=get_file_list), 
                'BuildIndex' : Builder(action=build_index), 
                'BuildPadFST' : Builder(action=build_pad_fst), 
                'FSTCompile' : Builder(action=fst_compile), 
                'QueryToPhoneFST' : Builder(action=query_to_phone_fst),
                #'StandardSearch' : Builder(action=standard_search), 
                'Merge' : Builder(action=merge),
                'MergeScores' : Builder(action=merge_scores),
                'MergeIVOOV' : Builder(action=merge_iv_oov),
                'Normalize' : Builder(action=normalize),
                'NormalizeSTO' : Builder(action=normalize_sum_to_one),
                'Score' : Builder(action=score, emitter=score_emitter),
                'AlterIVOOV' : Builder(action=alter_iv_oov),
                "QueryFiles" : Builder(action=query_files),
                "DatabaseFile" : Builder(action=database_file),
                "CollateScores" : Builder(action=collate_scores),

                "NewTermMap2QueryTerm" : Builder(action="${KWS_SCRIPTS}/termmap2queryterm.pl ${LANGUAGE_ID} ${KWS_RESOURCES} ${TARGET[0]} ${TARGET[1]} ${TARGET[2]}"),
                "NewCreateWP" : Builder(action="${KWS_SCRIPTS}/create_wp.pl ${IV_DICT} ${OOV_DICT} ${DATA_FST}"),
                "NewCreateP2P" : Builder(action="${KWS_SCRIPTS}/create_p2p.pl ${P2P} ${DATA_FST}}"),
                "NewCreateQueryDIRIV" : Builder(action="${KWS_SCRIPTS}/createQueryDIR_iv.pl ${LANGUAGE_ID} ${IV_QUERY} ${DATA_IV} ${DATA_FST}"),
                #"NewCreateExpandedQueryDIR" : Builder(action="${KWS_SCRIPTS}/create_expanded_queryDIR.pl"),
                #"NewNBestP2P" : Builder(action=""),
                
                }
    #if env.get("HAS_TORQUE", False):
    BUILDERS["WordToPhoneLattice"] = Builder(action=word_to_phone_lattice_torque)
    BUILDERS["StandardSearch"] = Builder(action=standard_search_torque)
    #else:
    #    BUILDERS["WordToPhoneLattice"] = SimpleCommandThreadedBuilder(
    #        action="${WRD2PHLATTICE} -d ${SOURCES[2]} -D ${SOURCES[0]} -t %(FSMGZ_FORMAT)s -s ${SOURCES[1]} -S %(EPSILON_SYMBOLS)s %(CONFUSION_NETWORK)s -P %(PRUNE_THRESHOLD)d ${TARGET}")
            #"WordToPhoneLattice",
            #env)
        #{},
        #    {},
        #    None
        #    )
        #Builder(action=word_to_phone_lattice)
    #    BUILDERS["StandardSearch"] = Builder(action=standard_search)
    env.Append(BUILDERS=BUILDERS)
    # args["DICTIONARY"] = dic.rstr()
    # args["DATA_FILE"] = data_list.rstr()
    # args["FSMGZ_FORMAT"] = "true"
    # args["CONFUSION_NETWORK"] = ""
    # args["FSM_DIR"] = "temp"
    # args["WORDPRONSYMTABLE"] = wordpron.rstr()
    # return ("${WRD2PHLATTICE} -d %(DICTIONARY)s -D %(DATA_FILE)s -t %(FSMGZ_FORMAT)s -s %(WORDPRONSYMTABLE)s -S %(EPSILON_SYMBOLS)s %(CONFUSION_NETWORK)s -P %(PRUNE_THRESHOLD)d %(FSM_DIR)s",
    #         args)
#create queryterm files (iv,oov)
#${KWS_SCRIPTS}/termmap2queryterm.pl ${LANGUAGE_ID} ${RESOURCE} ${IV_QUERY} ${OOV_QUERY} ${DATA_FST} 
#create words2phones.fst,  phones2words.fst  words.isyms  phones.isyms
#${KWS_SCRIPTS}/create_wp.pl ${DICT_IV} ${DICT_OOV} ${DATA_FST}
#create p2p fst
#${KWS_SCRIPTS}/create_p2p.pl ${P2P} ${DATA_FST}
#${KWS_SCRIPTS}/create_queryDIR_iv.pl ${LANGUAGE_ID} ${IV_QUERY} ${DATA_IV} ${DATA_FST}
#${KWS_SCRIPTS}/split_list.pl  ${FILE} ${JOB} ${JOBN} | ${KWS_SCRIPTS}/create_expanded_queryDIR.pl - ${OUTDIR} ${DATA_FST} P2P.fst  ${NBESTP2P} ${MINPHLENGTH}
#ls ${DIR}/*p2p2w.${NBESTP2P}.fst > $DIR/${NBESTP2P}.list
#${KWS_SCRIPTS}/combine_expanded_fsts.pl ${DIR}/${NBESTP2P}.list ${DIR}/all.${NBESTP2P}.fsm ${DATA_FST}
###

#${KWS_SCRIPTS}/split_list.pl ${LIST} ${JOB} ${JOBN} | ${KWS_SCRIPTS}/make_index.pl - ${INDEX}/index.part${JOB}.fsm ${PRINT_WORDS_THRESH} ${PRINT_EPS_THRESH}
#rc=$?
#if (( $rc )); then
#  echo "rc=",$rc
#  rm -f $INDEX/index.part${JOB}.fsm 
#  exit $rc
#fi
#${KWS_SCRIPTS}/compile_fst_index.pl $INDEX/index.part${JOB}.fsm ${DATA_FST} ${KWS_SCRIPTS}

###

