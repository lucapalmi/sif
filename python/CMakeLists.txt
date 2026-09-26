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

    # Structures Submodule
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structures/py_catalog.c
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

    # Finders Submodule
    ${CMAKE_CURRENT_SOURCE_DIR}/src/finders/py_finders.c
)

# 3. Build the Python Extension Module
# The target name 'pysif' must match your PyInit_pysif function in C.
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

# 6. Where the libraries pysif links are found at run time.
#
# A build from source runs where it was built, against the FFTW, HDF5 and
# OpenMP it was linked with -- which on a cluster usually live in module
# directories the loader does not search. Without their paths recorded in the
# module, `import pysif` fails on libfftw3.so or libhdf5.so unless
# LD_LIBRARY_PATH is set in every job. So they are recorded: every directory a
# linked library came from, the compiler's OpenMP runtime included, less the
# system's own, which the loader searches anyway.
#
# A wheel for distribution must say OFF, as it does in pyproject.toml: it
# carries its own copies of the libraries, and must not point at directories
# on the machine that built it.
option(SIF_PYTHON_RPATH
  "Record the linked libraries' directories in pysif (off for distributed wheels)"
  ON)

if(SIF_PYTHON_RPATH)
  set_target_properties(pysif PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

  # The OpenMP runtime arrives as a compiler flag (-fopenmp) rather than as a
  # library path, so the line above does not see it; FindOpenMP knows where it
  # is. Not a problem with the system compiler, whose runtime is in a system
  # directory -- exactly the case of a gcc loaded from a module.
  set(_pysif_omp_dirs "")
  foreach(_lib IN LISTS OpenMP_C_LIB_NAMES)
    if(OpenMP_${_lib}_LIBRARY)
      get_filename_component(_dir "${OpenMP_${_lib}_LIBRARY}" DIRECTORY)
      list(APPEND _pysif_omp_dirs "${_dir}")
    endif()
  endforeach()
  if(_pysif_omp_dirs)
    list(REMOVE_DUPLICATES _pysif_omp_dirs)
    list(REMOVE_ITEM _pysif_omp_dirs ${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES})
    set_property(TARGET pysif APPEND PROPERTY INSTALL_RPATH ${_pysif_omp_dirs})
  endif()
endif()

# 7. Handle the macOS + Conda dynamic linking problem
if (APPLE AND DEFINED ENV{CONDA_PREFIX})
  # Appended, so the directories recorded under SIF_PYTHON_RPATH survive.
  set_property(TARGET pysif APPEND PROPERTY INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
  set_target_properties(pysif PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Into the root of the wheel, where `import pysif` finds it. A plain CMake
# build leaves the module in the build tree, to be used from there.
if(SKBUILD)
  install(TARGETS pysif DESTINATION .)
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
