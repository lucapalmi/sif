# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# sif test suite
#
# Each test is a self-contained executable that returns non-zero on failure,
# so ctest needs no extra harness. Run with:
#
#   cmake -S . -B build -DSIF_BUILD_TESTS=ON
#   cmake --build build
#   ctest --test-dir build --output-on-failure

# Tests call sif_init(), which creates and writes to $HOME/.sif. Point HOME at
# the build tree so a test run never touches the developer's home directory.
set(SIF_TEST_HOME "${CMAKE_CURRENT_BINARY_DIR}/test_home")
file(MAKE_DIRECTORY "${SIF_TEST_HOME}")

set(SIF_TESTS
  test_utils        # allocation, arrays, crc32, prng, sort
  test_structures   # bitmask, catalogue, cell linked list, chain mesh
  test_field_grid   # morton ordering, require_* helpers, CIC
  test_io           # .xfield/.xgrid round trips, checksums, ASCII and raw binary
  test_octree       # build integrity + nearest neighbour vs brute force
  test_fft          # transform round-trip, filters, buffer ownership
  test_profiles     # radial profiles vs a brute-force particle count
  test_tessellation # sampled Voronoi volumes, and the profiles they give
  test_vsf          # void size function binning, normalisation, merging
  test_delta_stats  # Fourier delta PDF, spectral moments, phase shuffle
  test_vsf_models   # SvdW / Vdn size functions and the sigma slope
  test_ep           # correlated walks, Cholesky, first-crossing multiplicity
  test_ep_emu       # the emulator against its Python reference, and invariants
  test_finders      # end-to-end: both finders on a synthetic field
  test_survey       # end-to-end: the survey finder on a synthetic footprint
  test_hdf5         # the HDF5 catalogue file, or its text fallback
  test_gadget       # GADGET snapshots, all formats, from tests/data/gadget
  test_cosmology    # comoving distances, sky coordinates, and their refusals
  test_fits         # FITS tables: columns, expressions, filters, subsamples
)

foreach(test_name IN LISTS SIF_TESTS)
  add_executable(${test_name} ${test_name}.c)

  target_link_libraries(${test_name} PRIVATE sif)

  # test_fft reaches into the internal fft wrapper, which is not a public
  # header; the rest only use the installed API. That header pulls in fftw3.h,
  # whose location is private to the sif target.
  target_include_directories(${test_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${FFTW3_INCLUDE}
  )

  if(OpenMP_C_FOUND)
    target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_C)
  endif()

  # What sif itself was built with, so a test can expect the HDF5 entry
  # points to work or to refuse. The library's own definition is private.
  if(SIF_WITH_HDF5)
    target_compile_definitions(${test_name} PRIVATE SIF_HAVE_HDF5)
  endif()

  # test_hdf5 also makes files sif would not -- someone else's HDF5, a newer
  # layout -- to check they are refused, and that takes HDF5 itself.
  if(SIF_WITH_HDF5 AND test_name STREQUAL "test_hdf5")
    target_include_directories(${test_name} PRIVATE
      ${HDF5_C_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
    target_link_libraries(${test_name} PRIVATE ${HDF5_C_LIBRARIES})
  endif()

  # test_fits writes its own fixture with cfitsio, and reads it back with
  # sif, when sif has cfitsio; without, it checks the reader refuses.
  if(SIF_WITH_FITS AND test_name STREQUAL "test_fits")
    target_compile_definitions(${test_name} PRIVATE SIF_HAVE_FITS)
    target_include_directories(${test_name} PRIVATE ${CFITSIO_INCLUDE})
    target_link_libraries(${test_name} PRIVATE ${CFITSIO_LIB})
  endif()

  # Fixture files that live in the source tree, read in place.
  target_compile_definitions(${test_name} PRIVATE
    SIF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

  add_test(NAME ${test_name} COMMAND ${test_name})

  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "HOME=${SIF_TEST_HOME};OMP_NUM_THREADS=4"
    TIMEOUT 600
  )
endforeach()

# The command-line programs, from the outside: a shell script per program,
# run against the executable the build made.
if(SIF_BUILD_PROGRAMS)
  add_test(NAME test_sif_exodus
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/programs/test_sif_exodus.sh
      $<TARGET_FILE:sif-exodus> ${CMAKE_CURRENT_SOURCE_DIR}/data)
  set_tests_properties(test_sif_exodus PROPERTIES
    ENVIRONMENT "HOME=${SIF_TEST_HOME};OMP_NUM_THREADS=4"
    TIMEOUT 600
  )
endif()
