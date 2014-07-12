from SCons.Builder import Builder
from SCons.Action import Action
from SCons.Subst import scons_subst
import re
from glob import glob
from functools import partial
import logging
import os.path
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
import torque
import time
from common_tools import temp_file
from os.path import join as pjoin

#def SimpleCommandThreadedBuilder(**kw):
    
#    return Builder(action=None)

def SimpleTorqueBuilder(**kw):
    return Builder(action=action)

#def make_parallel_f2d_command(name, command, env, environment={}):
def SimpleCommandThreadedBuilder(to_split, **kw):
    def action(target, source, env):
        args = source[-1].read()
        #max_local_jobs = env["MAXIMUM_LOCAL_JOBS"]
        #num_scons_jobs = env.GetOption("num_jobs")
        how_many = env["LOCAL_JOBS_PER_SCONS_INSTANCE"] #max(1, max_local_jobs / num_scons_jobs)
        lines = [l for l in meta_open(source[0].rstr())]
        per = len(lines) / how_many
        temp_files = []
        for i in range(how_many):
            (fid, fname) = tempfile.mkstemp()
            temp_files.append(fname)
            start = per * i
            end = per * (i + 1)
            if i == how_many - 1:
                end = len(lines)
            meta_open(fname, "w").write("\n".join(lines[start:end]))
            print env.subst(command % args, target=target[0].get_dir(), source=[fname] + source[1:])
        for fname in temp_files:
            try:
                os.remove(fname)
            except:
                pass
        return None
    def emitter(target, source, env):
        new_targets = pjoin(target[0].rstr(), "timestamp.txt")
        return new_targets, source
    return Builder(action=Action(action)) #, cmdstr="%s(${TARGETS}, ${SOURCES})" % (name)), emitter=emitter)


def meta_open(file_name, mode="r"):
    """
    Convenience function for opening a file with gzip if it ends in "gz", uncompressed otherwise.
    """
    if os.path.splitext(file_name)[1] == ".gz":
        return gzip.open(file_name, mode)
    else:
        return open(file_name, mode)


def run_command(cmd, env={}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, data=None):
    """
    Simple convenience wrapper for running commands (not an actual Builder).
    """
    if isinstance(cmd, basestring):
        cmd = shlex.split(cmd)
    logging.debug("Running command: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, env=env, stdin=stdin, stdout=stdout, stderr=stderr)
    if data:
        out, err = process.communicate(data)
    else:
        out, err = process.communicate()
    return out, err, process.returncode == 0



def submit_job(target, source, env):
    if env["HAS_TORQUE"]:
        args = source[-1].read()
        stdout = args.get("stdout", os.path.join(args["path"], "stdout"))
        stderr = args.get("stderr", os.path.join(args["path"], "stderr"))
        if not os.path.exists(stdout):
            os.makedirs(stdout)
        if not os.path.exists(stderr):
            os.makedirs(stderr)
        interval = args.get("interval", 10)
        job = torque.Job(args.get("name", "scons"),
                         commands=[env.subst(x) for x in args["commands"]],
                         path=args["path"],
                         stdout_path=stdout,
                         stderr_path=stderr,
                         array=args.get("array", 0),
                         other=args.get("other", []))
        job.submit(commit=True)
        while job.job_id in [x[0] for x in torque.get_jobs(True)]:
            logging.info("sleeping...")
            time.sleep(interval)
    with meta_open(target[0].rstr(), "w") as ofd:
        ofd.write(time.asctime() + "\n")
    return None



def TOOLS_ADD(env):
    env.Append(BUILDERS = {"SubmitJob" : Builder(action=submit_job),
                           })
