/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#define Py_MODULE_HEAD_UNIFIED
#include "measure/py_delta.h"
#include "measure/py_profiles.h"
#include "measure/py_size_function.h"

static PyMethodDef measure_methods[] = {
  {"size_function_catalog", (PyCFunction)py_sif_size_function_catalog,
    METH_VARARGS | METH_KEYWORDS,
    "size_function_catalog(catalog, box_length, n_bins, bins='ln', "
    "r_min=0.0, r_max=0.0)\n"
    "--\n\n"
    "Bin a catalogue into a void size function.\n\n"
    "Divides the counts by the bin width and the box volume, so the result\n"
    "is a number density and catalogues from different volumes are\n"
    "comparable. The raw counts are kept, and the Poisson error follows\n"
    "from them.\n\n"
    "Args:\n"
    "    catalog: Catalogue to bin.\n"
    "    box_length: Physical side length, which sets the volume.\n"
    "    n_bins: Radial bins.\n"
    "    bins: 'ln' for bins uniform in ln R, 'linear' for uniform in R.\n"
    "    r_min, r_max: Radius bounds; 0 or less takes the catalogue's own\n"
    "        extremes.\n\n"
    "Returns:\n"
    "    SizeFunction: The binned size function."},

  {"size_function_combine", (PyCFunction)py_sif_size_function_combine,
    METH_VARARGS | METH_KEYWORDS,
    "size_function_combine(vsfs, master_bins, domains=None, method='mean')\n"
    "--\n\n"
    "Merge several size functions onto one radius grid.\n\n"
    "How measurements from boxes of different resolution are joined: each\n"
    "resolves a different range of radii well, and domains says where each\n"
    "should be trusted.\n\n"
    "Args:\n"
    "    vsfs: Sequence of SizeFunction objects. They must share a binning\n"
    "        convention -- merging an 'ln' with a 'linear' one combines two\n"
    "        different quantities.\n"
    "    master_bins: Radial bins in the result.\n"
    "    domains: Optional (r_min, r_max) per input; None uses each input's\n"
    "        full range.\n"
    "    method: 'mean', 'median' or 'stitch'.\n\n"
    "Returns:\n"
    "    SizeFunction: The merged size function."},

  {"profiles", (PyCFunction)py_sif_profiles, METH_VARARGS | METH_KEYWORDS,
    "profiles(catalog, mesh, n_bins, ext=5.0, compute_velocity=False, "
    "use_pbc=True, differential=False, weighted_velocity=False)\n"
    "--\n\n"
    "Stack radial density and velocity profiles around voids.\n\n"
    "Radii are scaled by each void's own radius, so profiles of\n"
    "different-sized voids stack directly. Densities are normalized to the\n"
    "box mean, so a profile approaches 1 far from the centre. On a mesh\n"
    "whose field carried weights the densities are weighted ones.\n\n"
    "Args:\n"
    "    catalog: Voids to profile.\n"
    "    mesh: ChainMesh of the tracers, which also supplies the box length\n"
    "        and the mean density -- so it has to hold the whole sample, not\n"
    "        a subset. Size it with profiles_suggest_mesh_cells(), or reuse\n"
    "        the mesh a finder was given: any resolution gives the same\n"
    "        profiles. Must carry velocities if compute_velocity.\n"
    "    n_bins: Radial bins per profile.\n"
    "    ext: Outer edge of the profile, in units of each void's radius.\n"
    "        Anything not positive selects the default of 5.\n"
    "    compute_velocity: Also stack the radial velocity profile.\n"
    "    use_pbc: Treat the box as periodic.\n"
    "    differential: Make each density bin the contrast of its own shell\n"
    "        rather than of everything enclosed within its outer edge. The\n"
    "        one worth measuring from a Tessellation, since a shell is all\n"
    "        boundary. Velocity bins are shell means either way.\n"
    "    weighted_velocity: Average each shell's radial velocity over the\n"
    "        tracer weights, sum(w v) / sum(w), rather than over the tracers.\n"
    "        Only differs on a weighted mesh. Off by default because of what\n"
    "        it means on a Tessellation mesh: the plain mean over samples is\n"
    "        the volume-weighted velocity, the weighted one is not.\n\n"
    "Returns:\n"
    "    Profiles: The stacked profiles."},

  {"profiles_suggest_mesh_cells",
    (PyCFunction)py_sif_profiles_suggest_mesh_cells,
    METH_VARARGS | METH_KEYWORDS,
    "profiles_suggest_mesh_cells(n_particles)\n"
    "--\n\n"
    "Suggest a ChainMesh resolution for profiles().\n\n"
    "Resolution changes only speed and memory -- the profiles are identical\n"
    "at any of them -- so a mesh built for something else is always a valid\n"
    "input, and one sized for finders.exodus() is as good as one sized\n"
    "here. It is still worth tuning: the minimum is broad, but sitting a\n"
    "long way off it costs a factor of a few.\n\n"
    "Args:\n"
    "    n_particles: Tracers the mesh will hold. The void size does not\n"
    "        enter: the balance is between cell overhead and tracers read\n"
    "        outside the sphere, and it lands in the same place whatever\n"
    "        the voids measure.\n\n"
    "Returns:\n"
    "    int: n_cells to pass to ChainMesh."},

  {"delta_distribution_grid", (PyCFunction)py_sif_delta_distribution_grid,
    METH_VARARGS | METH_KEYWORDS,
    "delta_distribution_grid(grid, radii, n_bins, delta_min, delta_max, "
    "shuffle='none', window='top_hat', seed=0, keep_cic_window=False)\n"
    "--\n\n"
    "Measure the one-point PDF of the smoothed density contrast.\n\n"
    "Pass shuffle='phases' or 'gaussian' to measure the same field with its\n"
    "Fourier phases randomized: that preserves the power spectrum and\n"
    "destroys everything above it, so the difference between the two PDFs\n"
    "is the non-Gaussian information the field carries.\n\n"
    "The same seed and grid size give the identical surrogate here and in\n"
    "delta_moments_grid(), which is what lets a PDF and a set of moments\n"
    "describe one field.\n\n"
    "Args:\n"
    "    grid: Density contrast field.\n"
    "    radii: Smoothing radii, each at least two cells and at most half\n"
    "        the box. Smaller radii resolve grid artefacts, not field, and\n"
    "        are rejected.\n"
    "    n_bins: Histogram bins per radius.\n"
    "    delta_min, delta_max: Histogram range. Samples outside it are\n"
    "        counted but not binned, so a row integrates to the fraction\n"
    "        that fell in range.\n"
    "    shuffle: 'none', 'phases' or 'gaussian'.\n"
    "    window: 'top_hat' or 'gaussian'.\n"
    "    seed: Seed for the phase shuffle.\n"
    "    keep_cic_window: Skip deconvolving the CIC assignment window. Set\n"
    "        this only for a grid that was not built by CIC.\n\n"
    "Returns:\n"
    "    DeltaDistribution: One PDF per radius."},

  {"delta_moments_grid", (PyCFunction)py_sif_delta_moments_grid,
    METH_VARARGS | METH_KEYWORDS,
    "delta_moments_grid(grid, radii, order, shuffle='none', "
    "window='top_hat', n_tracers=0, seed=0, keep_cic_window=False)\n"
    "--\n\n"
    "Measure the spectral moments of a gridded density field.\n\n"
    "Pass window='gaussian' when order >= 2 has to be meaningful: the\n"
    "top-hat sum is cut off by the grid, not by the field.\n\n"
    "Args:\n"
    "    grid: Density contrast field.\n"
    "    radii: Smoothing radii; see delta_distribution_grid().\n"
    "    order: Highest moment order. Each extra order is nearly free, but\n"
    "        pass 0 if only sigma_0 is wanted and one array is allocated.\n"
    "    shuffle: 'none', 'phases' or 'gaussian'.\n"
    "    window: 'top_hat' or 'gaussian'.\n"
    "    n_tracers: Tracer count behind the grid, used to subtract the shot\n"
    "        noise. 0 leaves it in.\n"
    "    seed: Seed for the phase shuffle; shared with\n"
    "        delta_distribution_grid().\n"
    "    keep_cic_window: Skip deconvolving the CIC assignment window.\n\n"
    "Returns:\n"
    "    DeltaMoments: sigma_0..sigma_order at every radius."},

  {NULL, NULL, 0, NULL}};

static struct PyModuleDef measure_module = {PyModuleDef_HEAD_INIT,
  .m_name = "pysif.measure",
  .m_doc = "Measurements taken from data.\n\n"
           "Size functions and profiles from a Catalog, and the statistics\n"
           "of the density field itself -- its one-point PDF and its spectral\n"
           "moments -- from a Grid. The counterparts that predict the same\n"
           "quantities from theory are in pysif.model, and deliberately\n"
           "return the same containers so the two can be compared.",
  .m_size = -1, .m_methods = measure_methods};

/* Submodule exporter called from the parent module initialization routing */
PyObject* py_sif_init_measure(void) {
  if (PyType_Ready(&sifProfilesType) < 0)
    return NULL;

  PyObject* m = PyModule_Create(&measure_module);
  if (!m)
    return NULL;

  Py_INCREF(&sifProfilesType);
  PyModule_AddObject(m, "Profiles", (PyObject*)&sifProfilesType);

  return m;
}
