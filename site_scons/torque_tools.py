from SCons.Builder import Builder
from SCons.Action import Action
from SCons.Subst import scons_subst
from SCons.Taskmaster import Task
from SCons.Executor import Executor
import SCons.dblite
import SCons.Warnings
import SCons.SConsign
import SCons.compat
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import math
import xml.etree.ElementTree as et
import gzip
import subprocess
import shlex
import time
import shutil
import tempfile
from os.path import join as pjoin
import pickle
from common_tools import temp_file, meta_open
import torque
import os

class TorqueTask(Task):
    pass


def TorqueCommandBuilder(**kw):
    def torque_builder(target, source, env):
        cmd = env.subst(kw["action"].genstring(target, source, env), target=target, source=source)
        job = torque.Job("scons",
                         commands=[cmd],
                         stdout_path="work/torque_output",
                         stderr_path="work/torque_output",
                         other=["#PBS -W group_list=yeticcls"])
        job.submit(commit=True, hold=False)
        interval = 120
        while job.job_id in [x[0] for x in torque.get_jobs(True)]:
            logging.info("sleeping...")
            time.sleep(interval)
        return None
    return Builder(action=torque_builder, emitter=kw["emitter"])


def torque_run(target, source, env):
    args = {}
    running = []
    resources = {
        "mem" : env["TORQUE_MEMORY"],
        "cput" : env["TORQUE_TIME"],
        "walltime" : env["TORQUE_TIME"],
        }
    for t in target:
        cmd = env.subst("scons -Q TORQUE_WORKER_NODE=True TORQUE_SUBMIT_NODE=False THREADED_SUBMIT_NODE=False THREADED_WORKER_NODE=False ${TARGET}", target=t, source=source)
        stdout = env.subst("${TARGET}.out", target=t, source=source)
        stderr = env.subst("${TARGET}.err", target=t, source=source)
        job = torque.Job(args.get("name", "scons"),
                         commands=["source /vega/ccls/users/tml2115/projects/bashrc.txt", cmd],
                         resources=resources,
                         path=args.get("path", os.getcwd()),
                         stdout_path=stdout,
                         stderr_path=stderr,
                         array=args.get("array", 0),
                         other=args.get("other", ["#PBS -W group_list=yeticcls"]))
        job.submit(commit=True)
        running.append(job.job_id)
    while len(running) > 0:
        logging.info("monitoring %d torque jobs..." % (len(running)))
        time.sleep(int(env["TORQUE_INTERVAL"]))
        active = [x[0] for x in torque.get_jobs(True)]
        running = [x for x in running if x in active]
    return None


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
    pass
    #env["TORQUE_SUBMIT_NODE"] = False
