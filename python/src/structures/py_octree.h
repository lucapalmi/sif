/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_STRUCTURES_PY_OCTREE_H
#define SIF_PY_STRUCTURES_PY_OCTREE_H

#include "py_common.h"
#include "sif/structures/octree.h"

/*
 * @brief Python object wrapping the C sif_octree_t struct
 */
typedef struct {
  PyObject_HEAD sif_octree_t* tree;
} sifOctreeObject;

/*
 * @brief Expose the Type Object for module registration and type-checking
 */
extern PyTypeObject sifOctreeType;

#endif /* SIF_PY_STRUCTURES_PY_OCTREE_H */
