/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_MODEL_PY_MODEL_H
#define SIF_PY_MODEL_PY_MODEL_H

#include "py_common.h"

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
