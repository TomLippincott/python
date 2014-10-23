import subprocess
import re
import sys
import random
import logging
import os.path
import os

class Job():
    def __init__(self, name="UNNAMED", resources={}, dependencies=[], commands=[], path="", array=0, stdout_path=None, stderr_path=None, other=[]):
        self.name = name
        self.resources = resources
        self.dependencies = dependencies
        self.commands = commands
        self.path = os.path.abspath(path)
        self.array = array
        self.resources["cput"] = resources.get("cput", "15:30:00")
        self.resources["walltime"] = resources.get("walltime", "15:30:00")
        self.resources["mem"] = "4000mb"
        self.commands.append("exit 0")
        self.stdout_path = os.path.abspath(stdout_path)
        self.stderr_path = os.path.abspath(stderr_path)
        self.other = other

    def __str__(self):
        lines = ["#PBS -N %s" % self.name] + self.other + ["#PBS -l %s=%s" % (k, v) for k, v in self.resources.iteritems()]
        if self.stdout_path:
            lines.append("#PBS -o %s" % os.path.join(self.stdout_path, "%s-${PBS_JOBID}.out" % (self.name)))
        if self.stderr_path:
            lines.append("#PBS -e %s" % os.path.join(self.stderr_path, "%s-${PBS_JOBID}.err" % (self.name)))
        if self.dependencies:
            arrays = [x.job_id for x in self.dependencies if x.array > 0]
            nonarrays = [x.job_id for x in self.dependencies if x.array == 0]
            if len(arrays) > 0 and len(nonarrays) > 0:
                depline = "%s,%s" % ("afterany:%s" % (":".join([str(x) for x in nonarrays])), "afteranyarray:%s" % (":".join(["%d[]" % (x) for x in arrays])))
            elif len(arrays) == 0 and len(nonarrays) > 0:
                depline = "afterany:%s" % (":".join([str(x) for x in nonarrays]))
            else:
                depline = "afteranyarray:%s" % (":".join(["%d[]" % (x) for x in arrays]))
            lines.append("#PBS -W depend=%s" % depline)
        if self.array > 0:
            lines.append("#PBS -t 0-%d" % (self.array - 1))
        if self.path:
            lines.append("cd %s" % self.path)
        lines += [c for c in self.commands]
        return "\n".join(lines) + "\n"

    def submit(self, commit=False, hold=False, propagate=True):
        cmd = ["qsub"]
        if propagate:
            cmd.append("-V")
        if hold:
            cmd.append("-h")
        if commit:
            logging.debug("Submitting the following job specification via \"%s\":\n%s" % (" ".join(cmd), self))
            out, err = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE).communicate(str(self))
            try:
                logging.info("Got job id: %s" % out)
                job_id = int(re.match(r"^(\d+).*", out).group(1))
                self.job_id = job_id
            except:
                print out, err
                sys.exit()
        else:
            logging.debug("Would submit the following job specification via \"%s\":\n%s" % (" ".join(cmd), self))
            self.job_id = random.randint(1, 10000)
            
def get_nodes(commit):
    if commit:
        out, err = subprocess.Popen(["qnodes"], stdout=subprocess.PIPE).communicate()
        return [x.strip() for x in out.split("\n") if re.match(r"^\S+.*$", x)]
    else:
        return ["dummy%d" % x for x in range(1, 5)]

def get_jobs(commit):
    if commit:
        out, err = subprocess.Popen(["qstat"], stdout=subprocess.PIPE).communicate()
        return [(int(y[0]), y[4]) for y in [x.groups() for x in re.finditer(r"^(\d+)\S*\s+(\S+)\s+(\S+)\s+(\S+)\s+(Q|R|E|H)\s+(\S+)\s*$", out, re.M)]]
    else:
        return []
