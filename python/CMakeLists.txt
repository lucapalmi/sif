# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The Python extension, pysif: the bindings in src/ compiled into one module
# with sif linked in statically. Added by the top-level CMakeLists.txt when
# SIF_BUILD_PYTHON is on, which it is whenever pip builds the project.

# 1. Find Python and execute a quick script to locate NumPy headers
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include(), end='')"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE # Strips the newline so paths don't break
)

# 2. Define the Python wrapper sources
set(PYTHON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sif_library.c

    # Submodule Entry Points
    ${CMAKE_CURRENT_SOURCE_DIR}/src/measure_module.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_module.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io_module.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/types_register.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/finders_module.c

    # Measure Submodule
    ${CMAKE_CURRENT_SOURCE_DIR}/src/measure/py_size_function.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/measure/py_profiles.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/measure/py_delta.c

    # IO Submodule
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/py_io.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/py_hdf5.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/py_gadget.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/py_fits.c

    # Structures Submodule
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_catalogue.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_grid.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_field.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_chain_mesh.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_tessellation.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_cell_linked_list.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_octree.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_bitmask.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_delta_distribution.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_delta_moments.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_size_function.c

    # Model Submodule
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/py_delta_moments.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/py_bbks.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/py_size_function.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/py_excursion_set.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/py_cosmology.c

    # Finders Submodule
    ${CMAKE_CURRENT_SOURCE_DIR}/src/finders/py_finders.c
)

# 3. Build the Python Extension Module
# Built as _pysif (OUTPUT_NAME, below), to match PyInit__pysif in C.
Python3_add_library(pysif MODULE WITH_SOABI ${PYTHON_SOURCES})

# 4. Include Directories (sif's own headers come with the target, below)
target_include_directories(pysif
    PRIVATE
    ${NumPy_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src  # So submodules can easily #include "py_common.h"
)

# 5. Link the Python module against the static C library ('sif')
# sif links OpenMP, FFTW and HDF5 privately; being static, they still reach
# pysif's link line through it.
target_link_libraries(pysif PRIVATE sif)

# The oldest NumPy the module runs with is set by the C API level it is
# compiled for, and NumPy's headers pick a default that rises with their own
# version. Stated here, so that it is the numpy>=1.23 in pyproject.toml
# whatever NumPy the build happens to use.
target_compile_definitions(pysif PRIVATE NPY_TARGET_VERSION=NPY_1_23_API_VERSION)

# The bindings word their warnings and errors by whether sif has HDF5; the
# library's own definition is private, so it is repeated here.
if(SIF_WITH_HDF5)
  target_compile_definitions(pysif PRIVATE SIF_HAVE_HDF5)
endif()

# The FITS bindings refuse up front in a build without cfitsio, rather than
# call a reader that can only refuse.
if(SIF_WITH_FITS)
  target_compile_definitions(pysif PRIVATE SIF_HAVE_FITS)
endif()

# 6. Where the libraries pysif links are found at run time: see
# SIF_RECORD_RPATH in the top-level CMakeLists.txt.
sif_record_rpath(pysif)

# 7. Handle the macOS + Conda dynamic linking problem
if (APPLE AND DEFINED ENV{CONDA_PREFIX})
  # Appended, so the directories recorded under SIF_RECORD_RPATH survive.
  set_property(TARGET pysif APPEND PROPERTY INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
  set_target_properties(pysif PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# pysif is a package: the extension is its compiled half, pysif._pysif, next
# to the Python files in python/pysif/. pip puts those files into the wheel
# itself (wheel.packages in pyproject.toml) and the module goes in beside
# them.
#
# A plain CMake build assembles the same package in the build tree, so that
# PYTHONPATH=<build>/python finds a working pysif without pip. The files are
# copied, not linked: edit them in python/pysif/, and rebuild.
set_target_properties(pysif PROPERTIES
  OUTPUT_NAME _pysif
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pysif)

file(GLOB _pysif_py CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/pysif/*.py)
foreach(_py IN LISTS _pysif_py)
  get_filename_component(_name "${_py}" NAME)
  configure_file("${_py}" "${CMAKE_CURRENT_BINARY_DIR}/pysif/${_name}" COPYONLY)
endforeach()

if(SKBUILD)
  install(TARGETS pysif DESTINATION pysif)
endif()

# The license texts of the libraries a distributed wheel carries inside it
# (FFTW, HDF5, the OpenMP runtime), collected by packaging/build-deps.sh and
# placed in the wheel's .dist-info next to sif's own. Set only by the wheel
# builds in pyproject.toml; a build from source bundles nothing.
set(SIF_BUNDLED_LICENSES "" CACHE PATH
  "Directory of third-party license texts to ship in the wheel")
if(SKBUILD AND SIF_BUNDLED_LICENSES)
  install(DIRECTORY "${SIF_BUNDLED_LICENSES}/"
    DESTINATION "${SKBUILD_METADATA_DIR}/licenses/bundled")
endif()
