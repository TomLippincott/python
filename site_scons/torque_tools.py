from SCons.Builder import Builder, BuilderBase
from SCons.Action import Action, CommandGeneratorAction, FunctionAction
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
from scons_tools import make_batch_builder
import functools


def torque_executor(env, commands):
    args = {}
    running = []
    resources = {
        "mem" : env["TORQUE_MEMORY"],
        "cput" : env["TORQUE_TIME"],
        "walltime" : env["TORQUE_TIME"],
    }    
    for cmd in commands:
        stdout = env.subst("${TORQUE_LOG}/")
        stderr = env.subst("${TORQUE_LOG}/")
        job = torque.Job(args.get("name", "scons"),
                         commands=[env.subst("source ${BASHRC_TXT}"), env.subst(cmd)],
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


make_torque_builder = functools.partial(make_batch_builder, torque_executor, name="torque")
