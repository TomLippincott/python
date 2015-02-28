from SCons.Builder import Builder, BuilderBase
from SCons.Action import Action, CommandGeneratorAction, FunctionAction
from SCons.Subst import scons_subst
import os.path
from subprocess import Popen, PIPE, STDOUT
from multiprocessing import Pool
import subprocess
import shlex
import tarfile
from common_tools import meta_open, temp_file
import re
import logging

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

def strip_file(f):
    return os.path.splitext(os.path.basename(f))[0]

def make_generic_emitter(targets):
    def emitter(target, source, env):
        for s in source:
            try:
                name = strip_file(s.rstr())
                s.__setattr__("basename", name)
            except:
                pass
        try:
            args = source[-1].read()
            spec = "-".join(["%s=%s" % (k, v) for k, v in sorted(args.iteritems())])
        except:
            spec = os.path.splitext(os.path.basename(source[0].rstr()))[0]
        env.Replace(SPEC=spec)
        return targets, source
    return emitter

def make_command_builder(command, targets, arg_names, directory):
    def generator(target, source, env, for_signature):
        args = source[-1].read()
        return scons_subst(command, env, target=target, source=source, lvars=args, gvars=env.gvars())
    def emitter(target, source, env):
        args = source[-1].read()
        spec = "-".join(["%s=%s" % (n, args.get(n, None)) for n in arg_names])
        new_target = [os.path.join(directory, "%s_%s.txt" % (t, spec)) for t in targets]
        return new_target, source
    return Builder(generator=generator, emitter=emitter)

class ThreadedBuilder(BuilderBase):
    def __init__(self, action):
        print action

    def __call__(self, env, target, source, chdir=None, **kw):
        print env.subst(target)
        pass

def threaded_run(target, source, env):
    cmd = env.subst("scons -Q THREADED_SUBMIT_NODE=False THREADED_WORKER_NODE=True TORQUE_SUBMIT_NODE=False TORQUE_WORKER_NODE=False ${TARGET}", target=target, source=source)
    pid = Popen(shlex.split(cmd))
    pid.communicate()
    return None

def call(x):
    subprocess.call(x.split())

def make_threaded_builder(builder, targets_per_job=1, sources_per_job=1):
    def threaded_print(target, source, env):
        target_sets = [target[i * targets_per_job : (i + 1) * targets_per_job] for i in range(len(target) / targets_per_job)]
        source_sets = [source[i * sources_per_job : (i + 1) * sources_per_job] for i in range(len(source) / sources_per_job)]
        if isinstance(builder.action, CommandGeneratorAction):
            inside = "\n\t".join([env.subst(builder.action.genstring(t, s, env), target=t, source=s) for t, s in zip(target_sets, source_sets)])
        elif isinstance(builder.action, FunctionAction):
            inside = "\n\t".join([env.subst(builder.action.genstring(t, s, env).replace("target", "${TARGETS}").replace("source", "${SOURCES}"), target=t, source=s)
                                  for t, s in zip(target_sets, source_sets)])
        return "threaded(\n\t" + inside + "\n)"
    def run_builder(target, source, env):
        target_sets = [target[i * targets_per_job : (i + 1) * targets_per_job] for i in range(len(target) / targets_per_job)]
        source_sets = [source[i * sources_per_job : (i + 1) * sources_per_job] for i in range(len(source) / sources_per_job)]
        cmd = "scons -Q THREADED_SUBMIT_NODE=False THREADED_WORKER_NODE=True TORQUE_SUBMIT_NODE=False TORQUE_WORKER_NODE=False ${TARGET}"
        with meta_open(".sconsign.dblite", "r", None) as ifd, temp_file() as ofd_name:
            with meta_open(ofd_name, "w", None) as ofd:
                ofd.write(ifd.read())
                p = Pool(3)
                for t, s in zip(target_sets, source_sets):                    
                    p.apply_async(call, (env.subst(cmd, target=t, source=s),))
                p.close()
                p.join()
        return None
    return Builder(action=Action(run_builder, threaded_print, batch_key=True))

def tar_member(target, source, env):
    pattern = env.subst(source[1].read())
    with meta_open(target[0].rstr(), "w") as ofd:
        with tarfile.open(source[0].rstr()) as tf:
            for name in [n for n in tf.getnames() if re.match(pattern, n)]:
                ofd.write(tf.extractfile(name).read())
    return None

def maybe(self, pattern):
    r = self.Glob(pattern)
    if len(r) == 0:
        return None
    else:
        return r[0].rstr()

def TOOLS_ADD(env):
    BUILDERS = {"TarMember" : Builder(action=tar_member),
                }
    env.Append(BUILDERS=BUILDERS)
    env.AddMethod(maybe)
