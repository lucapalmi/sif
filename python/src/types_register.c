/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Registration of the data structure types onto the package root.
 *
 * These used to live in a pysif.structures submodule. They are the types every
 * other submodule takes and returns, and a type is a noun rather than an
 * operation, so they belong beside the package rather than one level down:
 * pysif.Field, not pysif.structures.Field.
 */

#define Py_MODULE_HEAD_UNIFIED
#include "structures/py_bitmask.h"
#include "structures/py_catalog.h"
#include "structures/py_cell_linked_list.h"
#include "structures/py_chain_mesh.h"
#include "structures/py_delta_distribution.h"
#include "structures/py_delta_moments.h"
#include "structures/py_field.h"
#include "structures/py_grid.h"
#include "structures/py_octree.h"
#include "structures/py_size_function.h"

/* One table, so adding a type is one line and cannot be half-done: a type that
 * is readied but not added, or added but not readied, is a crash waiting for
 * the first user. */
static const struct {
  const char* name;
  PyTypeObject* type;
} SIF_PY_TYPES[] = {
  {"Bitmask", &sifBitmaskType},
  {"Catalog", &sifCatalogType},
  {"CellLinkedList", &sifCellLinkedListType},
  {"ChainMesh", &sifChainMeshType},
  {"DeltaDistribution", &sifDeltaDistributionType},
  {"DeltaMoments", &sifDeltaMomentsType},
  {"Field", &sifFieldType},
  {"Grid", &sifGridType},
  {"Octree", &sifOctreeType},
  {"SizeFunction", &sifSizeFunctionType},
};

int py_sif_register_types(PyObject* module) {
  const size_t n = sizeof(SIF_PY_TYPES) / sizeof(SIF_PY_TYPES[0]);

  for (size_t i = 0; i < n; i++) {
    if (PyType_Ready(SIF_PY_TYPES[i].type) < 0)
      return -1;

    Py_INCREF(SIF_PY_TYPES[i].type);
    if (PyModule_AddObject(
          module, SIF_PY_TYPES[i].name, (PyObject*)SIF_PY_TYPES[i].type) < 0) {
      Py_DECREF(SIF_PY_TYPES[i].type);
      return -1;
    }
  }

  return 0;
}
