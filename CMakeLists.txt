# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

cmake_minimum_required(VERSION 3.15)
project(sif VERSION 0.1.0 LANGUAGES C)

# Configuration
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set compile flags
#
# -ffast-math, less one of the promises it makes: -ffinite-math-only tells the
# compiler no value is ever NaN or infinite, which entitles it to delete every
# check that asks. sif validates its inputs with exactly those checks --
# `!(x >= 0 && x < box)` rejects a NaN coordinate because the comparison is
# false for it -- and GCC does delete them: a NaN coordinate then passes the
# chain mesh's validation and becomes a cell index. Keeping NaN semantics costs
# nothing measurable in the hot loops, which never meet one.
#
# A compile option rather than part of the flags below, which also reach the
# link line, where a compiler-only flag draws an "unused argument" warning per
# executable. Compile options follow CMAKE_C_FLAGS_RELEASE on the command line,
# so it lands after -ffast-math and takes the one promise back.
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math")
add_compile_options($<$<CONFIG:Release>:-fno-finite-math-only>)
set(CMAKE_C_FLAGS_DEBUG "-g -O2 -Wall -Wextra -fno-omit-frame-pointer")

# Tuning for the CPU of the machine doing the build. Right for a build that
# runs where it was compiled; wrong for one that runs elsewhere, where the
# first instruction the other CPU lacks kills the process with SIGILL. Turn it
# off for a binary meant to be shipped (a Python wheel), and on a cluster
# whose login nodes differ from its compute nodes -- there, name the target
# instead: -DSIF_NATIVE_ARCH=OFF -DCMAKE_C_FLAGS="-march=znver3".
option(SIF_NATIVE_ARCH "Optimize for the build machine's CPU (-march=native)" ON)
if(SIF_NATIVE_ARCH)
  string(APPEND CMAKE_C_FLAGS_RELEASE " -march=native")
  message(STATUS "CPU tuning: native (SIF_NATIVE_ARCH=ON)")
else()
  message(STATUS "CPU tuning: generic, plus whatever CMAKE_C_FLAGS names")
endif()

option(SIF_VECTORIZATION_REPORT "Enable compiler vectorization reports" OFF)

# Build the tests when sif is the project being built, but not when it is
# pulled in as a subdirectory, nor when pip is building the Python extension.
# SKBUILD is set by scikit-build-core, the build backend in pyproject.toml.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT SKBUILD)
  set(SIF_TESTS_DEFAULT ON)
else()
  set(SIF_TESTS_DEFAULT OFF)
endif()
option(SIF_BUILD_TESTS "Build the sif test suite" ${SIF_TESTS_DEFAULT})

# Debug-only invariant checks (bounds asserts in the hot accessors). Not tied
# to NDEBUG, which the release preset never defines.
option(SIF_DEBUG_CHECKS "Enable internal assertions" OFF)

# The Python extension, pysif. On whenever pip builds the project; a plain
# CMake build can ask for it too, to work on the bindings without pip.
if(SKBUILD)
  set(SIF_PYTHON_DEFAULT ON)
else()
  set(SIF_PYTHON_DEFAULT OFF)
endif()
option(SIF_BUILD_PYTHON "Build the Python extension (pysif)" ${SIF_PYTHON_DEFAULT})

# Offline data-generation programs. Not part of the library and not built by
# default: they exist to produce emulator training sets on a cluster.
option(SIF_BUILD_TOOLS "Build the offline tools in tools/" OFF)

# OpenMP
#
# 4.5 or newer. The measured histograms reduce into array sections
# (`reduction(+ : counts[ : n_bins])`), which is a 4.5 feature; GCC has had it
# since 6.1 and Clang since 9, so the floor costs nothing in practice. Stated
# here rather than discovered as a compile error in one file.
find_package(OpenMP REQUIRED)

if(OpenMP_C_FOUND AND OpenMP_C_VERSION VERSION_LESS 4.5)
  message(FATAL_ERROR
    "sif needs OpenMP 4.5 or newer for array-section reductions; "
    "the compiler reports ${OpenMP_C_VERSION}")
endif()

# fftw3

# this might need to be modified depending on the destination system
set(FFTW_HINTS $ENV{FFTW3_ROOT} $ENV{FFTW_HOME})

# FFTW's threading backend.
#
#   omp     - shares the OpenMP runtime with sif, so FFTW and the rest of the
#             library do not oversubscribe the machine. Correct ONLY when FFTW
#             was built against the same OpenMP runtime as this project.
#   threads - FFTW's pthreads backend. Always safe, at the cost of FFTW keeping
#             its own thread pool.
#
# Mixing two OpenMP runtimes in one process aborts at startup with
# "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
# initialized". That happens on a stock Homebrew FFTW built against a different
# compiler than the one used here; switch to `threads` in that case.
set(SIF_FFTW_THREADING "omp" CACHE STRING "FFTW threading backend (omp|threads)")
set_property(CACHE SIF_FFTW_THREADING PROPERTY STRINGS omp threads)

if(NOT SIF_FFTW_THREADING MATCHES "^(omp|threads)$")
  message(FATAL_ERROR "SIF_FFTW_THREADING must be 'omp' or 'threads'")
endif()

find_library(FFTW3_LIB  NAMES fftw3 HINTS ${FFTW_HINTS} PATH_SUFFIXES lib lib64 REQUIRED)
find_library(FFTW3F_LIB NAMES fftw3f HINTS ${FFTW_HINTS} PATH_SUFFIXES lib lib64 REQUIRED)
find_library(FFTW3_THREAD_LIB
  NAMES fftw3_${SIF_FFTW_THREADING} HINTS ${FFTW_HINTS} PATH_SUFFIXES lib lib64 REQUIRED)
find_library(FFTW3F_THREAD_LIB
  NAMES fftw3f_${SIF_FFTW_THREADING} HINTS ${FFTW_HINTS} PATH_SUFFIXES lib lib64 REQUIRED)
find_path(FFTW3_INCLUDE NAMES fftw3.h HINTS ${FFTW_HINTS} PATH_SUFFIXES include REQUIRED)

message(STATUS "FFTW threading backend: ${SIF_FFTW_THREADING}")

# HDF5
#
# Optional: the HDF5 readers and writers are an interchange format on top of
# the native .xfield and the ASCII paths, not something the library needs.
#
#   ON   - required; configuring fails without a usable HDF5.
#   OFF  - never looked for. Nothing in sif references HDF5, so a project that
#          embeds sif and has no use for it sets this and never sees it:
#            set(SIF_HDF5_SUPPORT OFF CACHE STRING "" FORCE)
#            add_subdirectory(sif)
#   AUTO - used when found, skipped otherwise. Convenient, and exactly as
#          dependent on the machine as it sounds: a cluster build where the
#          HDF5 module was not loaded comes out without it. The status line
#          below says which, and sif_init() logs it again at run time.
#
# Only a serial HDF5 is accepted. A parallel build needs MPI headers and the
# MPI compiler wrappers everywhere HDF5's own headers are seen, and sif is not
# an MPI code. Asked for it with ON, that is an error; under AUTO it is
# skipped as if absent.
set(SIF_HDF5_SUPPORT "AUTO" CACHE STRING "HDF5 I/O support (ON|OFF|AUTO)")
set_property(CACHE SIF_HDF5_SUPPORT PROPERTY STRINGS ON OFF AUTO)

string(TOUPPER "${SIF_HDF5_SUPPORT}" _sif_hdf5_mode)
if(NOT _sif_hdf5_mode MATCHES "^(ON|OFF|AUTO)$")
  message(FATAL_ERROR
    "SIF_HDF5_SUPPORT must be ON, OFF or AUTO, not '${SIF_HDF5_SUPPORT}'")
endif()

set(SIF_WITH_HDF5 OFF)
set(_sif_hdf5_why "disabled (SIF_HDF5_SUPPORT=OFF)")

if(NOT _sif_hdf5_mode STREQUAL "OFF")
  if(_sif_hdf5_mode STREQUAL "ON")
    find_package(HDF5 REQUIRED COMPONENTS C)
  else()
    find_package(HDF5 QUIET COMPONENTS C)
  endif()

  if(HDF5_FOUND)
    # Asked of HDF5's own configuration header rather than of the variables
    # find_package() leaves behind: which of those exist depends on whether
    # HDF5 was found through its CMake package or through the h5cc wrapper,
    # and on the CMake version. H5_HAVE_PARALLEL is the one answer every
    # HDF5 install gives the same way.
    include(CheckSymbolExists)
    set(CMAKE_REQUIRED_INCLUDES ${HDF5_C_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_QUIET ON)
    check_symbol_exists(H5_HAVE_PARALLEL "H5pubconf.h" SIF_HDF5_IS_PARALLEL)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_QUIET)

    if(SIF_HDF5_IS_PARALLEL)
      if(_sif_hdf5_mode STREQUAL "ON")
        message(FATAL_ERROR
          "the HDF5 found (${HDF5_VERSION}) is a parallel (MPI) build, which "
          "sif does not support. Point CMake at a serial HDF5 "
          "(-DHDF5_ROOT=...), or set SIF_HDF5_SUPPORT=OFF")
      endif()
      set(_sif_hdf5_why
        "skipped: the HDF5 found (${HDF5_VERSION}) is a parallel build")
    else()
      set(SIF_WITH_HDF5 ON)
      set(_sif_hdf5_why "HDF5 ${HDF5_VERSION}")
    endif()
  else()
    set(_sif_hdf5_why
      "not found (SIF_HDF5_SUPPORT=AUTO; set it to ON to require HDF5)")
  endif()
endif()

message(STATUS "HDF5 I/O: ${_sif_hdf5_why}")

# Readable from a project that adds sif as a subdirectory, which cannot see
# this directory's variables otherwise.
set(SIF_WITH_HDF5 ${SIF_WITH_HDF5} CACHE INTERNAL "sif was built with HDF5")

# Source files
set(SOURCES
  # data structures
    src/structures/grid.c
    src/structures/catalog.c
    src/structures/chain_mesh.c
    src/structures/tessellation.c
    src/structures/cell_linked_list.c
    src/structures/field.c
    src/structures/octree.c
    src/structures/bitmask.c
    src/structures/delta_distribution.c
    src/structures/delta_moments.c
    src/structures/size_function.c

  # finder
    src/finder/spherical_finder.c
    src/finder/exodus_finder.c
    src/finder/utils.c

  # measure
    src/measure/profiles.c
    src/measure/size_function.c
    src/measure/delta_common.c
    src/measure/delta_distribution.c
    src/measure/delta_moments.c

  # model
    src/model/delta_moments.c
    src/model/bbks.c
    src/model/size_function.c
    src/model/excursion_set.c
    src/model/ep_upcrossing.c
    src/model/ep_emu.c

  # io -- the HDF5 half is added below, one file or the other
    src/io/core_io.c
    src/io/catalog_io.c
    src/io/profiles_io.c
    src/io/field_io.c
    src/io/gadget_io.c
    src/io/grid_io.c

  # math
    src/math/fft.c
    src/math/nn.c

  # utils
    src/utils/timer.c
    src/utils/logger.c
    src/utils/str.c
    src/utils/align.c
    src/utils/array.c
    src/utils/crc32.c

  # library
    src/core/system.c
    src/core/settings.c

  # vendor
    vendor/predicates/predicates.c
)

# Disable fast math for exact arithmetic
set(PREDICATES_FLAGS "-fno-fast-math -ffp-contract=off")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  string(APPEND PREDICATES_FLAGS " -mno-fma")
endif()

set_source_files_properties(vendor/predicates/predicates.c
    PROPERTIES COMPILE_FLAGS "${PREDICATES_FLAGS}"
)


# The HDF5 entry points always exist; which file implements them decides
# whether they do the work or report that this build cannot.
if(SIF_WITH_HDF5)
  list(APPEND SOURCES src/io/hdf5.c src/io/gadget_hdf5.c)
else()
  list(APPEND SOURCES src/io/hdf5_off.c)
endif()
list(APPEND SOURCES src/io/hdf5_common.c)

# Create library
add_library(sif STATIC ${SOURCES})

# The project version, for the library to stamp into the files it writes.
# PRIVATE: it is the build's own idea of itself, not something a caller
# compiles against.
target_compile_definitions(sif PRIVATE SIF_VERSION_STRING="${PROJECT_VERSION}")

# Internal assertions. PUBLIC because the guarded macros live in headers that
# callers (and the tests) include, so both sides must agree.
if(SIF_DEBUG_CHECKS)
  target_compile_definitions(sif PUBLIC SIF_DEBUG_CHECKS)
endif()

# Include directories
target_include_directories(sif
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${FFTW3_INCLUDE}
    src
    vendor
)

# HDF5 is private to the sources that use it: no public sif header includes
# hdf5.h, so nothing compiled against sif needs HDF5's headers, and a build
# without it has no trace of it at all.
if(SIF_WITH_HDF5)
  target_include_directories(sif PRIVATE
    ${HDF5_C_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
  target_compile_definitions(sif PRIVATE
    SIF_HAVE_HDF5 ${HDF5_C_DEFINITIONS} ${HDF5_DEFINITIONS})
  target_link_libraries(sif PRIVATE ${HDF5_C_LIBRARIES})
endif()

# Linking
target_link_libraries(sif PRIVATE
    ${FFTW3_LIB}
    ${FFTW3F_LIB}
    ${FFTW3_THREAD_LIB}
    ${FFTW3F_THREAD_LIB}
    m
    $<$<BOOL:${OpenMP_C_FOUND}>:OpenMP::OpenMP_C>
)

if(SIF_VECTORIZATION_REPORT)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(sif PRIVATE -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)
  elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(sif PRIVATE -fopt-info-vec-optimized -fopt-info-vec-missed)
  endif()
endif()

# tests

if(SIF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SIF_BUILD_TOOLS)
  foreach(_tool ep_gen_case ep_validate)
    add_executable(${_tool} tools/${_tool}.c)
    target_link_libraries(${_tool} PRIVATE sif m)
    set_target_properties(${_tool} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)
  endforeach()
endif()

if(SIF_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# installation
#
# Not under pip: a wheel carries the extension alone, which has sif linked in,
# and has no place for a static library, headers and CMake files.

if(SKBUILD)
  return()
endif()

install(TARGETS sif
    EXPORT sifTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT sifTargets
    FILE sifTargets.cmake
    NAMESPACE sif::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sif
)

# A static library carries its private dependencies into whatever links it,
# and HDF5 can arrive as an imported target rather than a path -- so a sif
# built with HDF5 has to find it again on the consumer's side. Built without,
# the consumer never hears of it.
if(SIF_WITH_HDF5)
  set(_sif_config_hdf5 "find_dependency(HDF5 COMPONENTS C)\n")
else()
  set(_sif_config_hdf5 "")
endif()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/sifConfig.cmake
"include(CMakeFindDependencyMacro)\n"
"find_dependency(OpenMP)\n"
"${_sif_config_hdf5}"
"include(\"\${CMAKE_CURRENT_LIST_DIR}/sifTargets.cmake\")\n"
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sifConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sif
)
