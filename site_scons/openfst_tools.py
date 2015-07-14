from SCons.Builder import Builder
from SCons.Script import *


fstcompile = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstcompose = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstprune = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstrmepsilon = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstprint = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstshortestpath = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")
fstprintpaths = Action("${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}")


def TOOLS_ADD(env):
    """Conventional way to add the builders to an SCons environment."""
    env.Append(BUILDERS = {
        "FSTCompile" : Builder(action="${OPENFST_BINARIES}/fstcompile --isymbols=${SOURCES[0]} --osymbols=${SOURCES[1]} ${SOURCES[2]} > ${TARGETS[0]}"),
        "FSTArcSort" : Builder(action="${OPENFST_BINARIES}/fstarcsort --sort_type=${SOURCES[1]} ${SOURCES[0]} ${TARGETS[0]}"),
    })
