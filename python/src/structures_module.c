#define Py_MODULE_HEAD_UNIFIED
#include "structures/py_catalog.h"
#include "structures/py_field.h"
#include "structures/py_grid.h"
#include "structures/py_chain_mesh.h"
#include "structures/py_cell_linked_list.h"
#include "structures/py_octree.h"
#include "structures/py_bitmask.h"
#include "structures/py_tessellation.h"
#include "structures/py_delta_distribution.h"
#include "structures/py_delta_moments.h"
#include "structures/py_size_function.h"

static PyMethodDef structures_methods[] = {
  {"tessellation_build", (PyCFunction)py_sif_tessellation_build,
   METH_VARARGS | METH_KEYWORDS,
   "Build a tessellation using 'random', 'voxel', or 'exact' methods."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef structures_module = {
  PyModuleDef_HEAD_INIT,
  .m_name = "pysif.structures",
  .m_doc = "SIF sub-module for core data structures and spatial memory management.",
  .m_size = -1,
  .m_methods = structures_methods /* Hooked up! */
};

/* Submodule exporter called from the parent module initialization routing */
PyObject* py_sif_init_structures(void) {
  /* Initialize Type Structures safely */
  if (PyType_Ready(&sifCatalogType) < 0) return NULL;
  if (PyType_Ready(&sifFieldType) < 0) return NULL;
  if (PyType_Ready(&sifGridType) < 0) return NULL;
  if (PyType_Ready(&sifChainMeshType) < 0) return NULL;
  if (PyType_Ready(&sifCellLinkedListType) < 0) return NULL;
  if (PyType_Ready(&sifOctreeType) < 0) return NULL;
  if (PyType_Ready(&sifBitmaskType) < 0) return NULL;
  if (PyType_Ready(&sifTessellationType) < 0) return NULL;
  if (PyType_Ready(&sifDeltaDistributionType) < 0) return NULL;
  if (PyType_Ready(&sifDeltaMomentsType) < 0) return NULL;
  if (PyType_Ready(&sifSizeFunctionType) < 0) return NULL;

  PyObject* m = PyModule_Create(&structures_module);
  if (!m) return NULL;

  /* Register types into the submodule context namespace */
  Py_INCREF(&sifCatalogType);
  PyModule_AddObject(m, "Catalog", (PyObject*)&sifCatalogType);

  Py_INCREF(&sifFieldType);
  PyModule_AddObject(m, "Field", (PyObject*)&sifFieldType);

  Py_INCREF(&sifGridType);
  PyModule_AddObject(m, "Grid", (PyObject*)&sifGridType);

  Py_INCREF(&sifChainMeshType);
  PyModule_AddObject(m, "ChainMesh", (PyObject*)&sifChainMeshType);

  Py_INCREF(&sifCellLinkedListType);
  PyModule_AddObject(m, "CellLinkedList", (PyObject*)&sifCellLinkedListType);

  Py_INCREF(&sifOctreeType);
  PyModule_AddObject(m, "Octree", (PyObject*)&sifOctreeType);

  Py_INCREF(&sifBitmaskType);
  PyModule_AddObject(m, "Bitmask", (PyObject*)&sifBitmaskType);

  Py_INCREF(&sifTessellationType);
  PyModule_AddObject(m, "Tessellation", (PyObject*)&sifTessellationType);

  Py_INCREF(&sifDeltaDistributionType);
  PyModule_AddObject(
    m, "DeltaDistribution", (PyObject*)&sifDeltaDistributionType);

  Py_INCREF(&sifDeltaMomentsType);
  PyModule_AddObject(m, "DeltaMoments", (PyObject*)&sifDeltaMomentsType);

  Py_INCREF(&sifSizeFunctionType);
  PyModule_AddObject(m, "SizeFunction", (PyObject*)&sifSizeFunctionType);

  return m;
}