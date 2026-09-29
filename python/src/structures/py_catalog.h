/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_PY_STRUCTURES_PY_CATALOG_H
#define SIF_PY_STRUCTURES_PY_CATALOG_H

#include "py_common.h"
#include "sif/structures/catalog.h"

/*
 * @brief Python object wrapping the C sif_catalog_t struct
 */
typedef struct {
  PyObject_HEAD sif_catalog_t* catalog;
} sifCatalogObject;

/*
 * @brief Expose the Type Object for module registration and type-checking
 */
extern PyTypeObject sifCatalogType;

/* pysif.catalog_from_numpy(): a new Catalog from NumPy arrays. */
PyObject* pysif_catalog_from_numpy(
  PyObject* module, PyObject* args, PyObject* kwds);

#endif /* SIF_PY_STRUCTURES_PY_CATALOG_H */