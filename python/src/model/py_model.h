/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_MODEL_PY_MODEL_H
#define SIF_PY_MODEL_PY_MODEL_H

#include "py_common.h"
#include "sif/model/cosmology.h"

/*
 * The cosmology wherever a function takes one, from its keyword arguments:
 * omega_de None (or NULL) makes it flat. Validated here, so that a bad value
 * raises a ValueError that names it rather than one pointing at the log.
 *
 * @return 0 with *out set, -1 with the exception set.
 */
int py_sif_cosmology_from(double omega_m, PyObject* omega_de, double omega_r,
  double w0, double wa, sif_cosmology_t* out);

/* The ValueError for a cosmology with no expansion history out to z; always
 * returns NULL. */
PyObject* py_sif_cosmology_range_error(double z);

PyObject* py_sif_comoving_distance(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_delta_moments_pk(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_bbks_g(PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_bbks_number_density_differential(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_bbks_number_density_cumulative(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_size_function_bbks(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_delta_sigma_slope_pk(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_spherical_map_nonlinear(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_spherical_map_linear(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_svdw_multiplicity_function(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_size_function_svdw(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_size_function_vdn(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_delta_covariance_pk(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_ep_barrier_smt(PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_ep_first_crossing_counts(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_ep_multiplicity_function(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_ep_multiplicity_function_emu(
  PyObject* self, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_MODEL_PY_MODEL_H */
