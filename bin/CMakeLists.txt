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

# Lua, which the programs read their configuration files with (vendor/lua,
# see README.sif there). The core and the libraries a configuration may use,
# no more: package/require, io, debug and coroutines are neither compiled nor
# opened, and os is trimmed further at run time.
set(_lua_dir ${PROJECT_SOURCE_DIR}/vendor/lua)
add_library(sif_lua STATIC
  # core
  ${_lua_dir}/lapi.c ${_lua_dir}/lcode.c ${_lua_dir}/lctype.c
  ${_lua_dir}/ldebug.c ${_lua_dir}/ldo.c ${_lua_dir}/ldump.c
  ${_lua_dir}/lfunc.c ${_lua_dir}/lgc.c ${_lua_dir}/llex.c ${_lua_dir}/lmem.c
  ${_lua_dir}/lobject.c ${_lua_dir}/lopcodes.c ${_lua_dir}/lparser.c
  ${_lua_dir}/lstate.c ${_lua_dir}/lstring.c ${_lua_dir}/ltable.c
  ${_lua_dir}/ltm.c ${_lua_dir}/lundump.c ${_lua_dir}/lvm.c ${_lua_dir}/lzio.c
  # libraries
  ${_lua_dir}/lauxlib.c ${_lua_dir}/lbaselib.c ${_lua_dir}/lmathlib.c
  ${_lua_dir}/loslib.c ${_lua_dir}/lstrlib.c ${_lua_dir}/ltablib.c
  ${_lua_dir}/lutf8lib.c)
target_include_directories(sif_lua SYSTEM PUBLIC ${_lua_dir})
target_compile_definitions(sif_lua PRIVATE LUA_USE_POSIX)
# Lua's numbers are IEEE doubles and it relies on them being so -- NaN is the
# one value unequal to itself, math.huge is infinite -- which the project's
# -ffast-math would let the compiler assume away. Upstream's own flags, with
# its warnings silenced: vendored code is not ours to fix.
target_compile_options(sif_lua PRIVATE -fno-fast-math -w)
target_link_libraries(sif_lua PRIVATE m)

# The licence of each library a wheel carries goes into it; Lua is compiled
# into the programs, so its travels with them.
if(SKBUILD)
  install(FILES ${_lua_dir}/LICENSE
    DESTINATION "${SKBUILD_METADATA_DIR}/licenses/bundled" RENAME Lua.txt)
endif()

function(sif_add_program name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE sif sif_lua)
  target_compile_definitions(${name} PRIVATE
    SIF_VERSION_STRING="${PROJECT_VERSION}")
  # Whether the library can write HDF5, which the programs refuse to promise
  # otherwise. The library's own definition is private.
  if(SIF_WITH_HDF5)
    target_compile_definitions(${name} PRIVATE SIF_HAVE_HDF5)
  endif()
  if(SIF_WITH_FITS)
    target_compile_definitions(${name} PRIVATE SIF_HAVE_FITS)
  endif()
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
