cmake_minimum_required(VERSION 3.15)
project(sif VERSION 0.0.0 LANGUAGES C)

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
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math")
set(CMAKE_C_FLAGS_DEBUG "-g -O2 -Wall -Wextra -fno-omit-frame-pointer")

option(SIF_VECTORIZATION_REPORT "Enable compiler vectorization reports" OFF)

# Build the tests when sif is the project being built, but not when it is
# pulled in as a subdirectory (the Python extension does exactly that).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(SIF_TESTS_DEFAULT ON)
else()
  set(SIF_TESTS_DEFAULT OFF)
endif()
option(SIF_BUILD_TESTS "Build the sif test suite" ${SIF_TESTS_DEFAULT})

# Debug-only invariant checks (bounds asserts in the hot accessors). Not tied
# to NDEBUG, which the release preset never defines.
option(SIF_DEBUG_CHECKS "Enable internal assertions" OFF)

# OpenMP
find_package(OpenMP REQUIRED)

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

# Source files
set(SOURCES
  # data structures
    src/structures/grid.c
    src/structures/catalog.c
    src/structures/chain_mesh.c
    src/structures/cell_linked_list.c
    src/structures/field.c
    src/structures/octree.c
    src/structures/tessellation.c
    src/structures/bitmask.c
    src/structures/deltadistribution.c
    src/structures/deltamoments.c
    src/structures/sizefunction.c

  # finder
    src/finder/spherical_finder.c
    src/finder/rescaled_spherical_finder.c
    src/finder/utils.c

  # measure
    src/measure/profiles.c
    src/measure/sizefunction.c
    src/measure/delta_common.c
    src/measure/deltadistribution.c
    src/measure/deltamoments.c

  # model
    src/model/deltamoments.c
    src/model/bbks.c
    src/model/sizefunction.c
    src/model/excursionset.c

  # io
    src/io/core_io.c
    src/io/catalog_io.c
    src/io/field_io.c
    src/io/grid_io.c

  # math
    src/math/fft.c

  # utils
    src/utils/timer.c
    src/utils/logger.c
    src/utils/stringy.c
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


# Create library
add_library(sif STATIC ${SOURCES})

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

# installation

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

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/sifConfig.cmake
"include(CMakeFindDependencyMacro)\n"
"find_dependency(OpenMP)\n"
"include(\"\${CMAKE_CURRENT_LIST_DIR}/sifTargets.cmake\")\n"
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sifConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sif
)
