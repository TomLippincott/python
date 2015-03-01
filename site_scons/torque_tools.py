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
                         commands=["source /vega/ccls/users/tml2115/projects/bashrc.txt", env.subst(cmd)],
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


make_torque_builder = functools.partial(make_batch_builder, torque_executor)

# def make_torque_builder(builder, targets_per_job=1, sources_per_job=1):
#     def torque_print(target, source, env):
#         target_sets = [target[i * targets_per_job : (i + 1) * targets_per_job] for i in range(len(target) / targets_per_job)]
#         source_sets = [source[i * sources_per_job : (i + 1) * sources_per_job] for i in range(len(source) / sources_per_job)]
#         if isinstance(builder.action, CommandGeneratorAction):
#             inside = "\n\t".join([env.subst(builder.action.genstring(t, s, env), target=t, source=s) for t, s in zip(target_sets, source_sets)])
#         elif isinstance(builder.action, FunctionAction):
#             inside = "\n\t".join([env.subst(builder.action.genstring(t, s, env).replace("target", "${TARGETS}").replace("source", "${SOURCES}"), target=t, source=s)
#                                   for t, s in zip(target_sets, source_sets)])
#         return "torque(\n\t" + inside + "\n)"
#     def run_builder(target, source, env):
#         args = {}
#         running = []
#         resources = {
#             "mem" : env["TORQUE_MEMORY"],
#             "cput" : env["TORQUE_TIME"],
#             "walltime" : env["TORQUE_TIME"],
#         }
#         target_sets = [target[i * targets_per_job : (i + 1) * targets_per_job] for i in range(len(target) / targets_per_job)]
#         source_sets = [source[i * sources_per_job : (i + 1) * sources_per_job] for i in range(len(source) / sources_per_job)]
#         with meta_open(".sconsign.dblite", "r", None) as ifd, temp_file(dir="work") as ofd_name:
#             cmd = "scons -Q THREADED_SUBMIT_NODE=False THREADED_WORKER_NODE=True TORQUE_SUBMIT_NODE=False TORQUE_WORKER_NODE=False SCONSIGN_FILE=%s ${TARGET}" % (ofd_name)
#             with meta_open(ofd_name, "w", None) as ofd:
#                 ofd.write(ifd.read())
#                 for t, s in zip(target_sets, source_sets):                    
#                     stdout = env.subst("${TARGET}.out", target=t, source=source)
#                     stderr = env.subst("${TARGET}.err", target=t, source=source)
#                     job = torque.Job(args.get("name", "scons"),
#                                      commands=["source /vega/ccls/users/tml2115/projects/bashrc.txt", env.subst(cmd, target=t, source=s)],
#                                      resources=resources,
#                                      path=args.get("path", os.getcwd()),
#                                      stdout_path=stdout,
#                                      stderr_path=stderr,
#                                      array=args.get("array", 0),
#                                      other=args.get("other", ["#PBS -W group_list=yeticcls"]))
#                     job.submit(commit=True)
#                     running.append(job.job_id)
#                 while len(running) > 0:
#                     logging.info("monitoring %d torque jobs..." % (len(running)))
#                     time.sleep(int(env["TORQUE_INTERVAL"]))
#                     active = [x[0] for x in torque.get_jobs(True)]
#                     running = [x for x in running if x in active]
#         return None
#     return Builder(action=Action(run_builder, torque_print, batch_key=torque_batch_key))

# def TOOLS_ADD(env):
#     pass
