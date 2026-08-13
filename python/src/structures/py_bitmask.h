/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_STRUCTURES_PY_BITMASK_H
#define SIF_PY_STRUCTURES_PY_BITMASK_H

#include "py_common.h"
#include "sif/structures/bitmask.h"

/*
 * @brief Python object wrapping the C sif_bitmask_t struct
 */
typedef struct {
  PyObject_HEAD sif_bitmask_t* bitmask;
} sifBitmaskObject;

/*
 * @brief Expose the Type Object for module registration and type-checking
 */
extern PyTypeObject sifBitmaskType;

#endif /* SIF_PY_STRUCTURES_PY_BITMASK_H */
