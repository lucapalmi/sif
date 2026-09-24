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
  test_utils        # allocation, arrays, string scanning, crc32, prng, sort
  test_structures   # bitmask, catalog, cell linked list, chain mesh
  test_field_grid   # morton ordering, require_* helpers, CIC
  test_io           # .xfield/.xgrid round trips, checksums, ASCII catalogues
  test_octree       # build integrity + nearest neighbour vs brute force
  test_fft          # transform round-trip, filters, buffer ownership
  test_profiles     # radial profiles vs a brute-force particle count
  test_tessellation # sampled Voronoi volumes, and the profiles they give
  test_vsf          # void size function binning, normalization, merging
  test_delta_stats  # Fourier delta PDF, spectral moments, phase shuffle
  test_vsf_models   # SvdW / Vdn size functions and the sigma slope
  test_ep           # correlated walks, Cholesky, first-crossing multiplicity
  test_ep_emu       # the emulator against its Python reference, and invariants
  test_finders      # end-to-end: both finders on a synthetic field
  test_survey       # end-to-end: the survey finder on a synthetic footprint
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

  add_test(NAME ${test_name} COMMAND ${test_name})

  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "HOME=${SIF_TEST_HOME};OMP_NUM_THREADS=4"
    TIMEOUT 600
  )
endforeach()
