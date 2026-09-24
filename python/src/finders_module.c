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
    "at different thresholds. Velocities and original indices are never\n"
    "read. Weights are: on a mesh built from a weighted Field, a void's\n"
    "radius is where the enclosed weight reaches the threshold against the\n"
    "mean weight density -- the same criterion the grid applies when it\n"
    "was built from that Field. The weights must be finite and\n"
    "non-negative. A Tessellation mesh carries its density only in its\n"
    "weights, so it is always run this way.\n\n"
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

  {"exodus_survey", (PyCFunction)py_sif_finder_exodus_survey,
    METH_VARARGS | METH_KEYWORDS,
    "exodus_survey(data_grid, random_grid, data_mesh, random_mesh, radii, "
    "threshold, overlap_fraction=0.0, consume_grid=False, search_factor=1.5)\n"
    "--\n\n"
    "Find voids in a survey: any geometry, described by a random catalogue.\n\n"
    "The same finder as exodus(), with the box mean replaced by the randoms.\n"
    "A sphere's expected content is the random weight inside it times\n"
    "alpha = W_data / W_random, so its contrast is D(<r) / (alpha R(<r)) - 1:\n"
    "the footprint, its holes and the radial selection all come in through\n"
    "the randoms. They may follow the survey's n(z), and both sets may carry\n"
    "weights.\n\n"
    "Where the randoms are is the footprint: a grid cell holding at least\n"
    "one random is observed, any other is not. Voids are only centred in\n"
    "observed cells, and every void found is kept, with the fraction of its\n"
    "sphere and of the shell out to twice its radius that was observed --\n"
    "Catalog.footprint and Catalog.footprint_shell.\n\n"
    "Coordinates are comoving Cartesian, converted beforehand (astropy does\n"
    "it). Nothing wraps around, so the survey needs empty padding inside the\n"
    "box, about the largest search sphere on every side: survey_box() works\n"
    "out the box and the offset, Field.translate() moves data and randoms\n"
    "in, and Catalog.translate(-offset) moves the voids back out.\n\n"
    "Args:\n"
    "    data_grid: Grid of the data after assign_cic(). A density, NOT a\n"
    "        density contrast. Smoothed in place and restored afterwards\n"
    "        unless consume_grid is set.\n"
    "    random_grid: Grid of the randoms, same cells and box. Restored the\n"
    "        same way.\n"
    "    data_mesh: ChainMesh of the data over the same box.\n"
    "    random_mesh: ChainMesh of the randoms over the same box; its cells\n"
    "        need not match the data mesh's.\n"
    "    radii: Smoothing radii to try.\n"
    "    threshold: Density contrast a void is grown to, against the\n"
    "        randoms.\n"
    "    overlap_fraction: As in exodus().\n"
    "    consume_grid: Skip restoring both grids.\n"
    "    search_factor: As in exodus(). Pass the same one to survey_box().\n\n"
    "Returns:\n"
    "    Catalog: The voids found, with their footprint.\n\n"
    "Raises:\n"
    "    RuntimeError: If the finder refused its inputs -- most often a\n"
    "        survey too close to the box faces, which the log explains."},

  {"survey_box", (PyCFunction)py_sif_finder_survey_box,
    METH_VARARGS | METH_KEYWORDS,
    "survey_box(randoms, radii, n_cells, search_factor=1.5)\n"
    "--\n\n"
    "The box a survey has to be searched in, and the offset that moves it\n"
    "there.\n\n"
    "exodus_survey() needs empty padding around the survey, and how much\n"
    "depends on the radii, the search factor and the grid cell -- which\n"
    "depends on the box. This works it out from the randoms: a cubic box,\n"
    "just large enough, with the survey centred in it.\n\n"
    "    offset, box = pysif.finders.survey_box(randoms, radii, n_cells)\n"
    "    data.translate(offset); randoms.translate(offset)\n"
    "    # Grid(n_cells, box), ChainMesh(..., box, ...), exodus_survey(...)\n"
    "    catalog.translate(-offset)\n\n"
    "Args:\n"
    "    randoms: Field of the randoms, in your own Cartesian frame.\n"
    "    radii: The radii exodus_survey() will be run with.\n"
    "    n_cells: Cells per side of the grids you will build; at least 16.\n"
    "    search_factor: The one exodus_survey() will be run with.\n\n"
    "Returns:\n"
    "    tuple: (offset, box_length). offset is a length-3 array to add to\n"
    "    every position, data and randoms alike."},

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
           "spherical() and exodus() take a density-contrast grid of a\n"
           "periodic box and return a Catalog. They differ in where the radii\n"
           "come from: spherical() can only report radii from the ladder you\n"
           "pass in, while exodus() grows each void to the radius its tracers\n"
           "actually support, and so also needs the particles as a\n"
           "ChainMesh.\n\n"
           "exodus_survey() is exodus() for a survey, whose geometry comes\n"
           "from a random catalogue; survey_box() sizes the box to run it\n"
           "in.",
  .m_size = -1, .m_methods = finders_methods};

PyObject* py_sif_init_finders(void) { return PyModule_Create(&finders_module); }