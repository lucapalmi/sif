/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#define Py_MODULE_HEAD_UNIFIED
#include "measure/py_delta.h"
#include "measure/py_profiles.h"
#include "measure/py_size_function.h"
#include "measure/py_survey_volume.h"
#include "structures/py_delta_distribution.h"
#include "structures/py_delta_moments.h"

static PyMethodDef measure_methods[] = {
  {"size_function_catalogue", (PyCFunction)py_sif_size_function_catalogue,
    METH_VARARGS | METH_KEYWORDS,
    "size_function_catalogue(catalogue, box_length, n_bins, bins='ln', "
    "r_min=0.0, r_max=0.0)\n"
    "--\n\n"
    "Bin a catalogue into a void size function.\n\n"
    "Divides the counts by the bin width and the box volume, so the result\n"
    "is a number density and catalogues from different volumes are\n"
    "comparable. The raw counts are kept, and the Poisson error follows\n"
    "from them.\n\n"
    "Args:\n"
    "    catalogue: Catalogue to bin.\n"
    "    box_length: Physical side length, which sets the volume.\n"
    "    n_bins: Radial bins.\n"
    "    bins: 'ln' for bins uniform in ln R, 'linear' for uniform in R.\n"
    "    r_min, r_max: Radius bounds; 0 or less takes the catalogue's own\n"
    "        extremes.\n\n"
    "Returns:\n"
    "    SizeFunction: The binned size function."},

  {"size_function_combine", (PyCFunction)py_sif_size_function_combine,
    METH_VARARGS | METH_KEYWORDS,
    "size_function_combine(vsfs, master_bins, domains=None, method='mean', "
    "bin='ln')\n"
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
    "    method: 'mean', 'median' or 'stitch'.\n"
    "    bin: 'ln' or 'linear', the result's binning convention: the inputs'\n"
    "        own, as size_function_catalogue() made them.\n\n"
    "Returns:\n"
    "    SizeFunction: The merged size function."},

  {"survey_volume", (PyCFunction)py_sif_survey_volume,
    METH_VARARGS | METH_KEYWORDS,
    "survey_volume(randoms, observer=None)\n"
    "--\n\n"
    "The comoving volume of a survey's footprint, from its randoms.\n\n"
    "The footprint is taken to be separable -- one angular mask, seen out\n"
    "to the same range of distances in every direction -- which is how\n"
    "survey randoms are made. The volume is then\n"
    "Omega * (r_max**3 - r_min**3) / 3, with r_min and r_max the randoms'\n"
    "nearest and furthest distances and Omega the mask's solid angle,\n"
    "measured from the randoms' directions with the edges and holes\n"
    "resolved statistically. Their density on the sky need not be uniform:\n"
    "randoms drawn with an angular completeness are measured as they are.\n"
    "Unbiased against shapes of known volume to within about its scatter:\n"
    "about 0.5% at 10**4 randoms, 0.05-0.1% at 10**5 and 0.015% at 10**6.\n\n"
    "A catalogue whose radial range changes across the sky -- regions with\n"
    "different redshift cuts -- breaks the assumption; that is reported in\n"
    "a warning, and such regions are measured one call each. Weights are\n"
    "ignored: this is the geometric volume.\n\n"
    "Args:\n"
    "    randoms: Field of the randoms, in comoving Cartesian coordinates.\n"
    "    observer: Where the observer is, in the same frame. Default: the\n"
    "        origin, where Field's sky conversion puts it. Pass it if the\n"
    "        randoms have been moved since.\n\n"
    "Returns:\n"
    "    float: The volume, in the cube of the positions' unit.\n\n"
    "Raises:\n"
    "    ValueError: For sky coordinates, a position that is not finite,\n"
    "        or too few randoms to resolve the mask."},

  {"profiles", (PyCFunction)py_sif_profiles, METH_VARARGS | METH_KEYWORDS,
    "profiles(catalogue, mesh, n_bins, ext=5.0, compute_velocity=False, "
    "use_pbc=True, differential=False, weighted_velocity=False)\n"
    "--\n\n"
    "Stack radial density and velocity profiles around voids.\n\n"
    "Radii are scaled by each void's own radius, so profiles of\n"
    "different-sized voids stack directly. Densities are normalised to the\n"
    "box mean, so a profile approaches 1 far from the centre. On a mesh\n"
    "whose field carried weights the densities are weighted ones.\n\n"
    "Args:\n"
    "    catalogue: Voids to profile.\n"
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
    "shuffle='none', window='top_hat', seed=0, keep_cic_window=False, *, "
    "deconvolution='full')\n"
    "--\n\n"
    "Measure the one-point PDF of the smoothed density contrast.\n\n"
    "Pass shuffle='phases' or 'gaussian' to measure the same field with its\n"
    "Fourier phases randomised: that preserves the power spectrum and\n"
    "destroys everything above it, so the difference between the two PDFs\n"
    "is the non-Gaussian information the field carries.\n\n"
    "The same seed and grid size give the identical surrogate here and in\n"
    "delta_moments_grid(), which is what lets a PDF and a set of moments\n"
    "describe one field.\n\n"
    "Requires pysif.system_init().\n\n"
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
    "    keep_cic_window: Skip deconvolving the assignment window: the\n"
    "        older spelling of deconvolution='none'.\n"
    "    deconvolution: How the window of the scheme the grid was assigned\n"
    "        with is undone: 'full', 'aliasing' (allowing for the aliases,\n"
    "        and for interlacing) or 'none'; see pysif.finders.exodus().\n\n"
    "Returns:\n"
    "    DeltaDistribution: One PDF per radius.\n\n"
    "Raises:\n"
    "    RuntimeError: If pysif.system_init() has not been called, or the\n"
    "        computation failed (the sif log says why)."},

  {"delta_moments_grid", (PyCFunction)py_sif_delta_moments_grid,
    METH_VARARGS | METH_KEYWORDS,
    "delta_moments_grid(grid, radii, order, shuffle='none', "
    "window='top_hat', n_tracers=0, seed=0, keep_cic_window=False, *, "
    "deconvolution='full')\n"
    "--\n\n"
    "Measure the spectral moments of a gridded density field.\n\n"
    "Pass window='gaussian' when order >= 2 has to be meaningful: the\n"
    "top-hat sum is cut off by the grid, not by the field.\n\n"
    "Requires pysif.system_init().\n\n"
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
    "    keep_cic_window: As in delta_distribution_grid().\n"
    "    deconvolution: As in delta_distribution_grid().\n\n"
    "Returns:\n"
    "    DeltaMoments: sigma_0..sigma_order at every radius.\n\n"
    "Raises:\n"
    "    RuntimeError: If pysif.system_init() has not been called, or the\n"
    "        computation failed (the sif log says why)."},

  {NULL, NULL, 0, NULL}};

static struct PyModuleDef measure_module = {PyModuleDef_HEAD_INIT,
  .m_name = "pysif.measure",
  .m_doc = "Measurements taken from data.\n\n"
           "Size functions and profiles from a Catalogue, and the statistics\n"
           "of the density field itself -- its one-point PDF and its spectral\n"
           "moments -- from a Grid. The counterparts that predict the same\n"
           "quantities from theory are in pysif.model, and deliberately\n"
           "return the same containers so the two can be compared.",
  .m_size = -1, .m_methods = measure_methods};

/* The containers measure's functions return; pysif.model's
 * delta_moments_pk() returns a DeltaMoments too. */
static const struct {
  const char* name;
  PyTypeObject* type;
} MEASURE_TYPES[] = {
  {"DeltaDistribution", &sifDeltaDistributionType},
  {"DeltaMoments", &sifDeltaMomentsType},
  {"Profiles", &sifProfilesType},
};

/* Submodule exporter called from the parent module initialisation routing */
PyObject* py_sif_init_measure(void) {
  const size_t n = sizeof(MEASURE_TYPES) / sizeof(MEASURE_TYPES[0]);
  for (size_t i = 0; i < n; i++)
    if (PyType_Ready(MEASURE_TYPES[i].type) < 0)
      return NULL;

  PyObject* m = PyModule_Create(&measure_module);
  if (!m)
    return NULL;

  for (size_t i = 0; i < n; i++) {
    if (PyModule_AddObjectRef(
          m, MEASURE_TYPES[i].name, (PyObject*)MEASURE_TYPES[i].type) < 0) {
      Py_DECREF(m);
      return NULL;
    }
  }
  return m;
}
