import os
import os.path
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
import cPickle as pickle
import numpy
import math
import lxml.etree as et
import xml.sax
import sys
import gzip
from os.path import join as pjoin
from os import listdir
import tarfile
from common_tools import meta_open

from SCons.Builder import Builder
from SCons.Script import *

import SCons.Action
import SCons.Builder
from SCons.Node.FS import _my_normcase
#from SCons.Tool.ScalaCommon import parse_scala_file
import SCons.Util

import os
import os.path
import re

# scala_parsing = 1

# default_scala_version = '1.4'

# if scala_parsing:
#     # Parse Scala files for class names.
#     #
#     # This is a really cool parser from Charles Crain
#     # that finds appropriate class names in Scala source.

#     # A regular expression that will find, in a scala file:
#     #     newlines;
#     #     double-backslashes;
#     #     a single-line comment "//";
#     #     single or double quotes preceeded by a backslash;
#     #     single quotes, double quotes, open or close braces, semi-colons,
#     #         periods, open or close parentheses;
#     #     floating-point numbers;
#     #     any alphanumeric token (keyword, class name, specifier);
#     #     any alphanumeric token surrounded by angle brackets (generics);
#     #     the multi-line comment begin and end tokens /* and */;
#     #     array declarations "[]".
#     _reToken = re.compile(r'(\n|\\\\|//|\\[\'"]|[\'"\{\}\;\.\(\)]|' +
#                           r'\d*\.\d*|[A-Za-z_][\w\$\.]*|<[A-Za-z_]\w+>|' +
#                           r'/\*|\*/|\[\])')

#     class OuterState(object):
#         """The initial state for parsing a Scala file for classes,
#         interfaces, and anonymous inner classes."""
#         def __init__(self, version=default_scala_version):

#             if not version in ('1.1', '1.2', '1.3','1.4', '1.5', '1.6', '1.7',
#                                '5', '6'):
#                 msg = "Scala version %s not supported" % version
#                 raise NotImplementedError(msg)

#             self.version = version
#             self.listClasses = []
#             self.listOutputs = []
#             self.stackBrackets = []
#             self.brackets = 0
#             self.nextAnon = 1
#             self.localClasses = []
#             self.stackAnonClassBrackets = []
#             self.anonStacksStack = [[0]]
#             self.package = None

#         def trace(self):
#             pass

#         def __getClassState(self):
#             try:
#                 return self.classState
#             except AttributeError:
#                 ret = ClassState(self)
#                 self.classState = ret
#                 return ret

#         def __getPackageState(self):
#             try:
#                 return self.packageState
#             except AttributeError:
#                 ret = PackageState(self)
#                 self.packageState = ret
#                 return ret

#         def __getAnonClassState(self):
#             try:
#                 return self.anonState
#             except AttributeError:
#                 self.outer_state = self
#                 ret = SkipState(1, AnonClassState(self))
#                 self.anonState = ret
#                 return ret

#         def __getSkipState(self):
#             try:
#                 return self.skipState
#             except AttributeError:
#                 ret = SkipState(1, self)
#                 self.skipState = ret
#                 return ret
        
#         def __getAnonStack(self):
#             return self.anonStacksStack[-1]

#         def openBracket(self):
#             self.brackets = self.brackets + 1

#         def closeBracket(self):
#             self.brackets = self.brackets - 1
#             if len(self.stackBrackets) and \
#                self.brackets == self.stackBrackets[-1]:
#                 self.listOutputs.append('$'.join(self.listClasses))
#                 self.localClasses.pop()
#                 self.listClasses.pop()
#                 self.anonStacksStack.pop()
#                 self.stackBrackets.pop()
#             if len(self.stackAnonClassBrackets) and \
#                self.brackets == self.stackAnonClassBrackets[-1]:
#                 self.__getAnonStack().pop()
#                 self.stackAnonClassBrackets.pop()

#         def parseToken(self, token):
#             if token[:2] == '//':
#                 return IgnoreState('\n', self)
#             elif token == '/*':
#                 return IgnoreState('*/', self)
#             elif token == '{':
#                 self.openBracket()
#             elif token == '}':
#                 self.closeBracket()
#             elif token in [ '"', "'" ]:
#                 return IgnoreState(token, self)
#             elif token == "new":
#                 # anonymous inner class
#                 if len(self.listClasses) > 0:
#                     return self.__getAnonClassState()
#                 return self.__getSkipState() # Skip the class name
#             elif token in ['class', 'interface', 'enum']:
#                 if len(self.listClasses) == 0:
#                     self.nextAnon = 1
#                 self.stackBrackets.append(self.brackets)
#                 return self.__getClassState()
#             elif token == 'package':
#                 return self.__getPackageState()
#             elif token == '.':
#                 # Skip the attribute, it might be named "class", in which
#                 # case we don't want to treat the following token as
#                 # an inner class name...
#                 return self.__getSkipState()
#             return self

#         def addAnonClass(self):
#             """Add an anonymous inner class"""
#             if self.version in ('1.1', '1.2', '1.3', '1.4'):
#                 clazz = self.listClasses[0]
#                 self.listOutputs.append('%s$%d' % (clazz, self.nextAnon))
#             elif self.version in ('1.5', '1.6', '1.7', '5', '6'):
#                 self.stackAnonClassBrackets.append(self.brackets)
#                 className = []
#                 className.extend(self.listClasses)
#                 self.__getAnonStack()[-1] = self.__getAnonStack()[-1] + 1
#                 for anon in self.__getAnonStack():
#                     className.append(str(anon))
#                 self.listOutputs.append('$'.join(className))

#             self.nextAnon = self.nextAnon + 1
#             self.__getAnonStack().append(0)

#         def setPackage(self, package):
#             self.package = package

#     class AnonClassState(object):
#         """A state that looks for anonymous inner classes."""
#         def __init__(self, old_state):
#             # outer_state is always an instance of OuterState
#             self.outer_state = old_state.outer_state
#             self.old_state = old_state
#             self.brace_level = 0
#         def parseToken(self, token):
#             # This is an anonymous class if and only if the next
#             # non-whitespace token is a bracket. Everything between
#             # braces should be parsed as normal scala code.
#             if token[:2] == '//':
#                 return IgnoreState('\n', self)
#             elif token == '/*':
#                 return IgnoreState('*/', self)
#             elif token == '\n':
#                 return self
#             elif token[0] == '<' and token[-1] == '>':
#                 return self
#             elif token == '(':
#                 self.brace_level = self.brace_level + 1
#                 return self
#             if self.brace_level > 0:
#                 if token == 'new':
#                     # look further for anonymous inner class
#                     return SkipState(1, AnonClassState(self))
#                 elif token in [ '"', "'" ]:
#                     return IgnoreState(token, self)
#                 elif token == ')':
#                     self.brace_level = self.brace_level - 1
#                 return self
#             if token == '{':
#                 self.outer_state.addAnonClass()
#             return self.old_state.parseToken(token)

#     class SkipState(object):
#         """A state that will skip a specified number of tokens before
#         reverting to the previous state."""
#         def __init__(self, tokens_to_skip, old_state):
#             self.tokens_to_skip = tokens_to_skip
#             self.old_state = old_state
#         def parseToken(self, token):
#             self.tokens_to_skip = self.tokens_to_skip - 1
#             if self.tokens_to_skip < 1:
#                 return self.old_state
#             return self

#     class ClassState(object):
#         """A state we go into when we hit a class or interface keyword."""
#         def __init__(self, outer_state):
#             # outer_state is always an instance of OuterState
#             self.outer_state = outer_state
#         def parseToken(self, token):
#             # the next non-whitespace token should be the name of the class
#             if token == '\n':
#                 return self
#             # If that's an inner class which is declared in a method, it
#             # requires an index prepended to the class-name, e.g.
#             # 'Foo$1Inner' (Tigris Issue 2087)
#             if self.outer_state.localClasses and \
#                 self.outer_state.stackBrackets[-1] > \
#                 self.outer_state.stackBrackets[-2]+1:
#                 locals = self.outer_state.localClasses[-1]
#                 try:
#                     idx = locals[token]
#                     locals[token] = locals[token]+1
#                 except KeyError:
#                     locals[token] = 1
#                 token = str(locals[token]) + token
#             self.outer_state.localClasses.append({})
#             self.outer_state.listClasses.append(token)
#             self.outer_state.anonStacksStack.append([0])
#             return self.outer_state

#     class IgnoreState(object):
#         """A state that will ignore all tokens until it gets to a
#         specified token."""
#         def __init__(self, ignore_until, old_state):
#             self.ignore_until = ignore_until
#             self.old_state = old_state
#         def parseToken(self, token):
#             if self.ignore_until == token:
#                 return self.old_state
#             return self

#     class PackageState(object):
#         """The state we enter when we encounter the package keyword.
#         We assume the next token will be the package name."""
#         def __init__(self, outer_state):
#             # outer_state is always an instance of OuterState
#             self.outer_state = outer_state
#         def parseToken(self, token):
#             self.outer_state.setPackage(token)
#             return self.outer_state

#     def parse_scala_file(fn, version=default_scala_version):
#         return parse_scala(open(fn, 'r').read(), version)

#     def parse_scala(contents, version=default_scala_version, trace=None):
#         """Parse a .scala file and return a double of package directory,
#         plus a list of .class files that compiling that .scala file will
#         produce"""
#         package = None
#         initial = OuterState(version)
#         currstate = initial
#         for token in _reToken.findall(contents):
#             # The regex produces a bunch of groups, but only one will
#             # have anything in it.
#             currstate = currstate.parseToken(token)
#             if trace: trace(token, currstate)
#         if initial.package:
#             package = initial.package.replace('.', os.sep)
#         return (package, initial.listOutputs)

# else:
#     # Don't actually parse Scala files for class names.
#     #
#     # We might make this a configurable option in the future if
#     # Scala-file parsing takes too long (although it shouldn't relative
#     # to how long the Scala compiler itself seems to take...).

#     def parse_scala_file(fn):
#         """ "Parse" a .scala file.

#         This actually just splits the file name, so the assumption here
#         is that the file name matches the public class name, and that
#         the path to the file is the same as the package name.
#         """
#         return os.path.split(file)

# # Local Variables:
# # tab-width:4
# # indent-tabs-mode:nil
# # End:
# # vim: set expandtab tabstop=4 shiftwidth=4:


# def classname(path):
#     """Turn a string (path name) into a Scala class name."""
#     return os.path.normpath(path).replace(os.sep, '.')

# def emit_scala_classes(target, source, env):
#     """Create and return lists of source scala files
#     and their corresponding target class files.
#     """
#     scala_suffix = env.get('SCALASUFFIX', '.scala')
#     class_suffix = env.get('SCALACLASSSUFFIX', '.class')

#     target[0].must_be_same(SCons.Node.FS.Dir)
#     classdir = target[0]

#     s = source[0].rentry().disambiguate()
#     if isinstance(s, SCons.Node.FS.File):
#         sourcedir = s.dir.rdir()
#     elif isinstance(s, SCons.Node.FS.Dir):
#         sourcedir = s.rdir()
#     else:
#         raise SCons.Errors.UserError("Scala source must be File or Dir, not '%s'" % s.__class__)

#     slist = []
#     js = _my_normcase(scala_suffix)
#     for entry in source:
#         entry = entry.rentry().disambiguate()
#         if isinstance(entry, SCons.Node.FS.File):
#             slist.append(entry)
#         elif isinstance(entry, SCons.Node.FS.Dir):
#             result = SCons.Util.OrderedDict()
#             dirnode = entry.rdir()
#             def find_scala_files(arg, dirpath, filenames):
#                 scala_files = sorted([n for n in filenames
#                                        if _my_normcase(n).endswith(js)])
#                 mydir = dirnode.Dir(dirpath)
#                 scala_paths = [mydir.File(f) for f in scala_files]
#                 for jp in scala_paths:
#                      arg[jp] = True
#             for dirpath, dirnames, filenames in os.walk(dirnode.get_abspath()):
#                find_scala_files(result, dirpath, filenames)
#             entry.walk(find_scala_files, result)

#             slist.extend(list(result.keys()))
#         else:
#             raise SCons.Errors.UserError("Scala source must be File or Dir, not '%s'" % entry.__class__)

#     version = env.get('SCALAVERSION', '1.4')
#     full_tlist = []
#     for f in slist:
#         tlist = []
#         source_file_based = True
#         pkg_dir = None
#         if not f.is_derived():
#             pkg_dir, classes = parse_scala_file(f.rfile().get_abspath(), version)
#             if classes:
#                 source_file_based = False
#                 if pkg_dir:
#                     d = target[0].Dir(pkg_dir)
#                     p = pkg_dir + os.sep
#                 else:
#                     d = target[0]
#                     p = ''
#                 for c in classes:
#                     t = d.File(c + class_suffix)
#                     t.attributes.scala_classdir = classdir
#                     t.attributes.scala_sourcedir = sourcedir
#                     t.attributes.scala_classname = classname(p + c)
#                     tlist.append(t)

#         if source_file_based:
#             base = f.name[:-len(scala_suffix)]
#             if pkg_dir:
#                 t = target[0].Dir(pkg_dir).File(base + class_suffix)
#             else:
#                 t = target[0].File(base + class_suffix)
#             t.attributes.scala_classdir = classdir
#             t.attributes.scala_sourcedir = f.dir
#             t.attributes.scala_classname = classname(base)
#             tlist.append(t)

#         for t in tlist:
#             t.set_specific_source([f])

#         full_tlist.extend(tlist)

#     return full_tlist, slist

# ScalaAction = SCons.Action.Action('$SCALACCOM', '$SCALACCOMSTR')

# ScalaBuilder = SCons.Builder.Builder(action = ScalaAction,
#                     emitter = emit_scala_classes,
#                     target_factory = SCons.Node.FS.Entry,
#                     source_factory = SCons.Node.FS.Entry)

# class pathopt(object):
#     """
#     Callable object for generating scalac-style path options from
#     a construction variable (e.g. -classpath, -sourcepath).
#     """
#     def __init__(self, opt, var, default=None):
#         self.opt = opt
#         self.var = var
#         self.default = default

#     def __call__(self, target, source, env, for_signature):
#         path = env[self.var]
#         if path and not SCons.Util.is_List(path):
#             path = [path]
#         if self.default:
#             default = env[self.default]
#             if default:
#                 if not SCons.Util.is_List(default):
#                     default = [default]
#                 path = path + default
#         if path:
#             return [self.opt, os.pathsep.join(map(str, path))]
#         else:
#             return []

# def Scala(env, target, source, *args, **kw):
#     """
#     A pseudo-Builder wrapper around the separate ScalaClass{File,Dir}
#     Builders.
#     """
#     if not SCons.Util.is_List(target):
#         target = [target]
#     if not SCons.Util.is_List(source):
#         source = [source]

#     # Pad the target list with repetitions of the last element in the
#     # list so we have a target for every source element.
#     target = target + ([target[-1]] * (len(source) - len(target)))

#     scala_suffix = env.subst('$SCALASUFFIX')
#     result = []

#     for t, s in zip(target, source):
#         if isinstance(s, SCons.Node.FS.Base):
#             if isinstance(s, SCons.Node.FS.File):
#                 b = env.ScalaClassFile
#             else:
#                 b = env.ScalaClassDir
#         else:
#             if os.path.isfile(s):
#                 b = env.ScalaClassFile
#             elif os.path.isdir(s):
#                 b = env.ScalaClassDir
#             elif s[-len(scala_suffix):] == scala_suffix:
#                 b = env.ScalaClassFile
#             else:
#                 b = env.ScalaClassDir
#         result.extend(b(t, s, *args, **kw))

#     return result



# def CreateJarBuilder(env):
#     try:
#         scala_jar = env['BUILDERS']['Jar']
#     except KeyError:
#         fs = SCons.Node.FS.get_default_fs()
#         jar_com = SCons.Action.Action('$JARCOM', '$JARCOMSTR')
#         scala_jar = SCons.Builder.Builder(action = jar_com,
#                                          suffix = '$JARSUFFIX',
#                                          src_suffix = '$SCALACLASSSUFIX',
#                                          src_builder = 'ScalaClassFile',
#                                          source_factory = fs.Entry)
#         env['BUILDERS']['Jar'] = scala_jar
#     return scala_jar

# def CreateScalaHBuilder(env):
#     try:
#         scala_scalah = env['BUILDERS']['ScalaH']
#     except KeyError:
#         fs = SCons.Node.FS.get_default_fs()
#         scala_scalah_com = SCons.Action.Action('$SCALAHCOM', '$SCALAHCOMSTR')
#         scala_scalah = SCons.Builder.Builder(action = scala_scalah_com,
#                                            src_suffix = '$SCALACLASSSUFFIX',
#                                            target_factory = fs.Entry,
#                                            source_factory = fs.File,
#                                            src_builder = 'ScalaClassFile')
#         env['BUILDERS']['ScalaH'] = scala_scalah
#     return scala_scalah

# def CreateScalaClassFileBuilder(env):
#     try:
#         scala_class_file = env['BUILDERS']['ScalaClassFile']
#     except KeyError:
#         fs = SCons.Node.FS.get_default_fs()
#         scalac_com = SCons.Action.Action('$SCALACCOM', '$SCALACCOMSTR')
#         scala_class_file = SCons.Builder.Builder(action = scalac_com,
#                                                 emitter = {},
#                                                 #suffix = '$SCALACLASSSUFFIX',
#                                                 src_suffix = '$SCALASUFFIX',
#                                                 src_builder = ['ScalaFile'],
#                                                 target_factory = fs.Entry,
#                                                 source_factory = fs.File)
#         env['BUILDERS']['ScalaClassFile'] = scala_class_file
#     return scala_class_file

# def CreateScalaClassDirBuilder(env):
#     try:
#         scala_class_dir = env['BUILDERS']['ScalaClassDir']
#     except KeyError:
#         fs = SCons.Node.FS.get_default_fs()
#         scalac_com = SCons.Action.Action('$SCALACCOM', '$SCALACCOMSTR')
#         scala_class_dir = SCons.Builder.Builder(action = scalac_com,
#                                                emitter = {},
#                                                target_factory = fs.Dir,
#                                                source_factory = fs.Dir)
#         env['BUILDERS']['ScalaClassDir'] = scala_class_dir
#     return scala_class_dir

# def CreateScalaFileBuilder(env):
#     try:
#         scala_file = env['BUILDERS']['ScalaFile']
#     except KeyError:
#         scala_file = SCons.Builder.Builder(action = {},
#                                           emitter = {},
#                                           suffix = {None:'$SCALASUFFIX'})
#         env['BUILDERS']['ScalaFile'] = scala_file
#         env['SCALASUFFIX'] = '.scala'
#     return scala_file


def recursive_find(path, pattern):
   entries = [os.path.join(path, e) for e in listdir(path)]
   return [e for e in entries if os.path.isfile(e) and re.match(pattern, os.path.basename(e))] + sum([recursive_find(e, pattern) for e in entries if os.path.isdir(e)], [])

def scala_compile_generator(target, source, env, for_signature):
   return "scalac -deprecation -cp work/classes -d work/classes ${CHANGED_SOURCES}"

def target_from_source(path, fname):
   pkg = re.match(r"^\s*package\s+(\S+)\s*$", open(fname).read(), re.M).group(1).replace(".", "/")
   return re.sub(r"^[^\/]*", path, fname.replace(".scala", ".class")).replace("bhmm", pkg)

#def scala_compile_emitter(target, source, env):
#   new_sources = [env.File(x) for x in recursive_find(source[0].rstr(), r".*\.scala$")]    
#   new_targets = [env.File(target_from_source(target[0].rstr(), x.rstr())) for x in new_sources]
#   return new_targets, new_sources

def scala(env, target, source):
   if not SCons.Util.is_List(target):
      target = [target]
   if not SCons.Util.is_List(source):
      source = [source]
   new_sources = [env.File(x) for x in recursive_find(source[0].rstr(), r".*\.scala$")]
   new_targets = [env.File(target_from_source(target[0].rstr(), x.rstr())) for x in new_sources]
   #for x in new_sources:
   #   print (x.rstr(), dir(x.get_ninfo()))

   #print type(new_sources[0])
   if not env["DEBUG"]:
      new_sources = env.StripLogging(["work/debug_src/%s" % (os.path.basename(s.rstr())) for s in new_sources], new_sources)
   env.ScalaCompile(new_targets, new_sources)

def strip_logging(target, source, env):
   for t, s in zip(target, source):
      with meta_open(s.rstr()) as ifd:
         lines = [l for l in ifd if "logger.fin" not in l and "assert" not in l]
         with meta_open(t.rstr(), "w") as ofd:
            ofd.write("".join(lines))
   return None

def TOOLS_ADD(env):
   fs = SCons.Node.FS.get_default_fs()
   env.Append(BUILDERS = {
         "ScalaCompile" : Builder(generator=scala_compile_generator, source_factory=fs.File),# emitter=scala_compile_emitter),
         "StripLogging" : Builder(action=strip_logging),
         })
   env.AddMethod(scala, "Scala")

# def TOOLS_ADD(env):
#     """Add Builders and construction variables for scalac to an Environment."""
#     scala_file = CreateScalaFileBuilder(env)
#     scala_class = CreateScalaClassFileBuilder(env)
#     scala_class_dir = CreateScalaClassDirBuilder(env)
#     scala_class.add_emitter(None, emit_scala_classes)
#     scala_class.add_emitter(env.subst('$SCALASUFFIX'), emit_scala_classes)
#     scala_class_dir.emitter = emit_scala_classes

#     env.AddMethod(Scala)

#     env['SCALAC']                    = 'scalac'
#     env['SCALACFLAGS']               = SCons.Util.CLVar('')
#     env['SCALABOOTCLASSPATH']        = []
#     env['SCALACLASSPATH']            = []
#     env['SCALASOURCEPATH']           = []
#     env['_scalapathopt']             = pathopt
#     env['_SCALABOOTCLASSPATH']       = '${_scalapathopt("-bootclasspath", "SCALABOOTCLASSPATH")} '
#     env['_SCALACLASSPATH']           = '${_scalapathopt("-classpath", "SCALACLASSPATH")} '
#     env['_SCALASOURCEPATH']          = '${_scalapathopt("-sourcepath", "SCALASOURCEPATH", "_SCALASOURCEPATHDEFAULT")} '
#     env['_SCALASOURCEPATHDEFAULT']   = '${TARGET.attributes.scala_sourcedir}'
#     env['_SCALACCOM']                = '$SCALAC $SCALACFLAGS $_SCALABOOTCLASSPATH $_SCALACLASSPATH -d ${TARGET.attributes.scala_classdir} $_SCALASOURCEPATH $SOURCES'
#     env['SCALACCOM']                 = "${TEMPFILE('$_SCALACCOM')}"
#     env['SCALACLASSSUFFIX']          = '.class'
#     env['SCALASUFFIX']               = '.scala'

# def exists(env):
#     return 1

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
