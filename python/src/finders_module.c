#include "finders/py_finders.h"

static PyMethodDef finders_methods[] = {
  {"rescaled_spherical", (PyCFunction)py_sif_finder_rescaled_spherical, METH_VARARGS | METH_KEYWORDS,
   "Executes the high-performance rescaled spherical threshold void finder algorithm."},
   
  {"spherical", (PyCFunction)py_sif_finder_spherical, METH_VARARGS | METH_KEYWORDS,
   "Executes the pure geometric spherical void finder algorithm."},

  {"suggest_mesh_cells", (PyCFunction)py_sif_finder_suggest_mesh_cells,
   METH_VARARGS | METH_KEYWORDS,
   "suggest_mesh_cells(n_particles, box_length, max_radius=0.0) -> int\n\n"
   "Chain-mesh resolution to pass to ChainMesh, tuned for the rescaled\n"
   "finder. Resolution changes only speed and memory -- the catalog is\n"
   "identical at any resolution -- but the widest search sphere still has to\n"
   "fit inside the mesh, so max_radius (the largest smoothing radius of the\n"
   "run) puts a floor under the answer. Pass 0 to skip that constraint.\n"
   "Raises ValueError when the geometry admits no mesh at all."},

  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef finders_module = {
  PyModuleDef_HEAD_INIT,
  .m_name = "pysif.finders",
  .m_doc = "SIF sub-module for cosmic void identification engines.",
  .m_size = -1,
  .m_methods = finders_methods
};

PyObject* py_sif_init_finders(void) {
  return PyModule_Create(&finders_module);
}