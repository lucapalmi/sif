#ifndef __SIF_PY_MODEL_H__
#define __SIF_PY_MODEL_H__

#include "py_common.h"

PyObject* py_sif_delta_moments_pk(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_g_bbks(PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_differential_number_density_bbks(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_cumulative_number_density_bbks(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_size_function_bbks(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_sigma_slope_pk(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_delta_nonlinear(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_delta_linear(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_multiplicity_function_svdw(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_size_function_svdw(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_size_function_vdn(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_delta_covariance_pk(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_barrier_smt(PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_first_crossing_counts_ep(
  PyObject* self, PyObject* args, PyObject* kwds);

PyObject* py_sif_multiplicity_function_ep(
  PyObject* self, PyObject* args, PyObject* kwds);

#endif /* __SIF_PY_MODEL_H__ */
