# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The command-line programs: stand-alone executables built on sif, one
# directory each. Added by the top-level CMakeLists.txt when
# SIF_BUILD_PROGRAMS is on.
#
# Where a program ends up:
#
#   build tree  <build>/bin/, runnable from there.
#   install     ${CMAKE_INSTALL_BINDIR}, as usual.
#   pip         pysif/bin/ inside the package, next to the extension, where
#               the wheel repair (auditwheel, delocate) finds it and points
#               it at the libraries the wheel bundles. It is not on PATH
#               there: pip installs a launcher that is, one per program --
#               python/pysif/_cli.py and [project.scripts] in pyproject.toml.
#
# Adding a program: a directory with its sources and a CMakeLists.txt calling
# sif_add_program(), an add_subdirectory() below, and on the pip side a
# launcher in _cli.py and its line in [project.scripts].

function(sif_add_program name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE sif)
  target_compile_definitions(${name} PRIVATE
    SIF_VERSION_STRING="${PROJECT_VERSION}")
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  sif_record_rpath(${name})

  if(SKBUILD)
    install(TARGETS ${name} RUNTIME DESTINATION pysif/bin)
  else()
    install(TARGETS ${name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endfunction()

add_subdirectory(sif-exodus)
