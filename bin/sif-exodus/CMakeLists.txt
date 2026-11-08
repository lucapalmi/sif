# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# sif-exodus: the exodus finder run on a parameter file.

# The configurations `--template box` and `--template survey` print live in
# template_box.lua and template_survey.lua, as Lua files the documentation
# includes as they are, and are compiled in as byte arrays: one text for
# both, so they cannot drift apart. Editing one re-runs this, and the copy
# only changes -- and the program only rebuilds -- when the text does.
set(_templates "")
foreach(_kind box survey)
  set(_template ${CMAKE_CURRENT_SOURCE_DIR}/template_${_kind}.lua)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_template})
  file(READ ${_template} _hex HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
  string(APPEND _templates
    "const char config_template_${_kind}[] = {${_bytes}0x00};\n")
endforeach()
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/template.c.tmp
  "/* Generated from template_*.lua by CMakeLists.txt: edit those instead. */\n"
  "${_templates}")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/template.c.tmp
  ${CMAKE_CURRENT_BINARY_DIR}/template.c COPYONLY)

sif_add_program(sif-exodus
  main.c config.c pipeline.c ${CMAKE_CURRENT_BINARY_DIR}/template.c)
