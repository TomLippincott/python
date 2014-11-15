from SCons.Builder import Builder
from SCons.Action import Action
from SCons.Subst import scons_subst
import os.path

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

