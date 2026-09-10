/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "finders/py_finders.h"

static PyMethodDef finders_methods[] = {
  {"exodus", (PyCFunction)py_sif_finder_exodus, METH_VARARGS | METH_KEYWORDS,
    "exodus(grid, mesh, radii, threshold, overlap_fraction=0.0, "
    "consume_grid=False, search_factor=1.5)\n"
    "--\n\n"
    "Find voids, growing each to the radius its tracers support.\n\n"
    "Uses the radius ladder to locate candidates, then walks the enclosed\n"
    "density outward from each accepted centre until it crosses the\n"
    "threshold. The radii therefore come from the data rather than from the\n"
    "ladder, which is why this needs the particles as well as the grid.\n\n"
    "The mesh is borrowed and never freed, so it can be reused across runs\n"
    "at different thresholds. It only needs positions: weights, velocities\n"
    "and original indices are never read.\n\n"
    "Args:\n"
    "    grid: Density contrast field. Smoothed in place and restored\n"
    "        afterwards unless consume_grid is set.\n"
    "    mesh: ChainMesh over the same box.\n"
    "    radii: Smoothing radii to try.\n"
    "    threshold: Density contrast a cell must reach to seed a void;\n"
    "        negative, since voids are underdensities.\n"
    "    overlap_fraction: How much two voids may overlap, as a fraction of\n"
    "        the smaller one's radius. 0 forbids overlap entirely.\n"
    "    consume_grid: Skip restoring the grid, saving one inverse FFT.\n"
    "    search_factor: How far past a rung the rescaling looks for the\n"
    "        crossing, as a multiple of the rung. Default 1.5. This is the\n"
    "        finder's dominant cost and it goes as factor**3 - 1, so 1.5\n"
    "        does about a third the work of 2.0. Lower is not free: a rung\n"
    "        whose crossing lies past its reach defers to a larger rung, and\n"
    "        where no larger rung saw that void the catalogue changes.\n"
    "        Snapped to the nearest of 1.25, 1.5, 1.75, 2.0; pass 2.0 to\n"
    "        reproduce catalogues made before this argument existed.\n\n"
    "Returns:\n"
    "    Catalog: The voids found."},

  {"spherical", (PyCFunction)py_sif_finder_spherical,
    METH_VARARGS | METH_KEYWORDS,
    "spherical(grid, radii, threshold, overlap_fraction=0.0, "
    "consume_grid=False)\n"
    "--\n\n"
    "Find voids as fixed-radius spheres on a density grid.\n\n"
    "Walks the radii from largest to smallest, accepting a sphere wherever\n"
    "one does not overlap a void already accepted. Voids therefore come out\n"
    "with radii drawn from the ladder you passed in; use exodus() to get\n"
    "radii from the data.\n\n"
    "Args:\n"
    "    grid: Density contrast field. Smoothed in place and restored\n"
    "        afterwards unless consume_grid is set.\n"
    "    radii: Smoothing radii to try. Order does not matter.\n"
    "    threshold: Density contrast a cell must reach to seed a void.\n"
    "    overlap_fraction: How much two voids may overlap, as a fraction of\n"
    "        the smaller one's radius.\n"
    "    consume_grid: Skip restoring the grid.\n\n"
    "Returns:\n"
    "    Catalog: The voids found."},

  {"suggest_mesh_cells", (PyCFunction)py_sif_finder_suggest_mesh_cells,
    METH_VARARGS | METH_KEYWORDS,
    "suggest_mesh_cells(n_particles, box_length, max_radius=0.0)\n"
    "--\n\n"
    "Suggest a ChainMesh resolution for exodus().\n\n"
    "Resolution changes only speed and memory -- the catalogue is identical\n"
    "at any resolution -- so it is safe to tune and worth tuning.\n\n"
    "Args:\n"
    "    n_particles: Tracers the mesh will hold.\n"
    "    box_length: Physical side length, matching the grid's.\n"
    "    max_radius: Largest smoothing radius the run will use. The search\n"
    "        sphere it implies must fit inside the mesh, which puts a floor\n"
    "        under the resolution. Pass 0 to skip that constraint.\n\n"
    "Returns:\n"
    "    int: n_cells to pass to ChainMesh, or 0 if no mesh can hold a\n"
    "    search sphere that wide."},

  {NULL, NULL, 0, NULL}};

static struct PyModuleDef finders_module = {PyModuleDef_HEAD_INIT,
  .m_name = "pysif.finders",
  .m_doc = "Void finders.\n\n"
           "Both take a density-contrast grid and return a Catalog. They\n"
           "differ in where the radii come from: spherical() can only report\n"
           "radii from the ladder you pass in, while exodus() grows each\n"
           "void to the radius its tracers actually support, and so also\n"
           "needs the particles as a ChainMesh.",
  .m_size = -1, .m_methods = finders_methods};

PyObject* py_sif_init_finders(void) { return PyModule_Create(&finders_module); }