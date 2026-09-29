# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# sif-exodus: the exodus finder run on a parameter file.

# The configuration `--template` prints lives in template.lua, as a Lua file
# the documentation includes as it is, and is compiled in as a byte array: one
# text for both, so they cannot drift apart. Editing it re-runs this, and the
# copy only changes -- and the program only rebuilds -- when the text does.
set(_template ${CMAKE_CURRENT_SOURCE_DIR}/template.lua)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_template})
file(READ ${_template} _hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/template.c.tmp
  "/* Generated from template.lua by CMakeLists.txt: edit that instead. */\n"
  "const char config_template[] = {${_bytes}0x00};\n")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/template.c.tmp
  ${CMAKE_CURRENT_BINARY_DIR}/template.c COPYONLY)

sif_add_program(sif-exodus
  main.c config.c pipeline.c ${CMAKE_CURRENT_BINARY_DIR}/template.c)
