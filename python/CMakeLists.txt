# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

cmake_minimum_required(VERSION 3.15...3.27)
project(pysif VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)

# 1. Find Python and execute a quick script to locate NumPy headers
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include(), end='')"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE # Strips the newline so paths don't break
)

# 2. Build the Core C Library!
# We point to the parent directory where your main C CMakeLists.txt lives.
# CMake will build the 'sif' target from there and make it available here.
add_subdirectory(${CMAKE_SOURCE_DIR}/.. ${CMAKE_BINARY_DIR}/core_sif)

# 3. Define the Python wrapper sources
set(PYTHON_SOURCES
    ${CMAKE_SOURCE_DIR}/src/sif_library.c

    # Submodule Entry Points
    ${CMAKE_SOURCE_DIR}/src/measure_module.c
    ${CMAKE_SOURCE_DIR}/src/model_module.c
    ${CMAKE_SOURCE_DIR}/src/io_module.c
    ${CMAKE_SOURCE_DIR}/src/types_register.c
    ${CMAKE_SOURCE_DIR}/src/finders_module.c

    # Measure Submodule
    ${CMAKE_SOURCE_DIR}/src/measure/py_size_function.c
    ${CMAKE_SOURCE_DIR}/src/measure/py_profiles.c
    ${CMAKE_SOURCE_DIR}/src/measure/py_delta.c

    # IO Submodule
    ${CMAKE_SOURCE_DIR}/src/io/py_io.c

    # Structures Submodule
    ${CMAKE_SOURCE_DIR}/src/structures/py_catalog.c
    ${CMAKE_SOURCE_DIR}/src/structures/py_grid.c
    ${CMAKE_SOURCE_DIR}/src/structures/py_field.c
    ${CMAKE_SOURCE_DIR}/src/structures/py_chain_mesh.c
    ${CMAKE_SOURCE_DIR}/src/structures/py_cell_linked_list.c
    ${CMAKE_SOURCE_DIR}/src/structures/py_octree.c
    ${CMAKE_SOURCE_DIR}/src/structures/py_bitmask.c
    ${CMAKE_SOURCE_DIR}/src/structures/py_delta_distribution.c
    ${CMAKE_SOURCE_DIR}/src/structures/py_delta_moments.c
    ${CMAKE_SOURCE_DIR}/src/structures/py_size_function.c

    # Model Submodule
    ${CMAKE_SOURCE_DIR}/src/model/py_delta_moments.c
    ${CMAKE_SOURCE_DIR}/src/model/py_bbks.c
    ${CMAKE_SOURCE_DIR}/src/model/py_size_function.c
    ${CMAKE_SOURCE_DIR}/src/model/py_excursion_set.c

    # Finders Submodule
    ${CMAKE_SOURCE_DIR}/src/finders/py_finders.c
)

# 4. Build the Python Extension Module
# The target name 'pysif' must match your PyInit_pysif function in C.
Python3_add_library(pysif MODULE WITH_SOABI ${PYTHON_SOURCES})

# 5. Include Directories
target_include_directories(pysif
    PRIVATE
    ${NumPy_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/../include  # So wrapper files can see sif headers
    ${CMAKE_SOURCE_DIR}/src         # So submodules can easily #include "py_common.h"
)

# 6. Link the Python module against your core static C library ('sif')
# Because the 'sif' target privately/publicly links OpenMP and FFTW in the parent
# CMake, those dependencies automatically transit up to 'pysif' here!
target_link_libraries(pysif PRIVATE sif)

# 7. Handle the macOS + Conda dynamic linking problem
if (APPLE AND DEFINED ENV{CONDA_PREFIX})
  set_target_properties(pysif PROPERTIES
        INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

install(TARGETS pysif DESTINATION .)
