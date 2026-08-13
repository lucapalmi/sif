/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_STRUCTURES_PY_TESSELLATION_H
#define SIF_PY_STRUCTURES_PY_TESSELLATION_H

#include "py_common.h"
#include "sif/structures/tessellation.h"

typedef struct {
  PyObject_HEAD sif_tessellation_t* tess;
} sifTessellationObject;

extern PyTypeObject sifTessellationType;

#endif /* SIF_PY_STRUCTURES_PY_TESSELLATION_H */