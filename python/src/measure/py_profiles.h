/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_MEASURE_PY_PROFILES_H
#define SIF_PY_MEASURE_PY_PROFILES_H

#include "py_common.h"
#include "sif/measure/profiles.h"

/*
 * @brief Unified Python container wrapping density and/or velocity structures
 */
typedef struct {
  PyObject_HEAD sif_density_profiles_t* dens;
  sif_velocity_profiles_t* vel;
} sifProfilesObject;

extern PyTypeObject sifProfilesType;

/* Functional API Entry Point */
PyObject* py_sif_profiles(PyObject* self, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_MEASURE_PY_PROFILES_H */