/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "py_survey_volume.h"

#include "sif/measure/survey_volume.h"
#include "structures/py_field.h"

PyObject* py_sif_survey_volume(PyObject* self, PyObject* args, PyObject* kwds) {
  PyObject* field_obj;
  PyObject* observer_obj = Py_None;

  static char* kwlist[] = {"randoms", "observer", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kwds, "O!|O", kwlist, &sifFieldType, &field_obj, &observer_obj))
    return NULL;

  sif_real observer[3];
  const sif_real* obs = NULL;
  if (observer_obj != Py_None) {
    PyObject* seq =
      PySequence_Fast(observer_obj, "observer: expected 3 coordinates");
    if (!seq)
      return NULL;
    if (PySequence_Fast_GET_SIZE(seq) != 3) {
      PyErr_Format(PyExc_ValueError, "observer: %zd coordinates, expected 3",
        PySequence_Fast_GET_SIZE(seq));
      Py_DECREF(seq);
      return NULL;
    }
    for (int a = 0; a < 3; a++) {
      const double v = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, a));
      if (v == -1.0 && PyErr_Occurred()) {
        Py_DECREF(seq);
        return NULL;
      }
      observer[a] = (sif_real)v;
    }
    Py_DECREF(seq);
    obs = observer;
  }

  const sif_field_t* randoms = ((sifFieldObject*)field_obj)->field;
  sif_real volume = 0.0f;
  int status;
  sif_error_clear();
  Py_BEGIN_ALLOW_THREADS status = sif_survey_volume(randoms, obs, &volume);
  Py_END_ALLOW_THREADS

    if (status != SIF_OK) return py_sif_raise(status, "survey_volume");

  return PyFloat_FromDouble((double)volume);
}
