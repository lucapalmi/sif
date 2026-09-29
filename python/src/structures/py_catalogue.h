/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_STRUCTURES_PY_CATALOGUE_H
#define SIF_PY_STRUCTURES_PY_CATALOGUE_H

#include "py_common.h"
#include "sif/structures/catalogue.h"

/*
 * @brief Python object wrapping the C sif_catalogue_t struct
 */
typedef struct {
  PyObject_HEAD sif_catalogue_t* catalogue;
} sifCatalogueObject;

/*
 * @brief Expose the Type Object for module registration and type-checking
 */
extern PyTypeObject sifCatalogueType;

/* pysif.catalogue_from_numpy(): a new Catalogue from NumPy arrays. */
PyObject* pysif_catalogue_from_numpy(
  PyObject* module, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_STRUCTURES_PY_CATALOGUE_H */