/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#define Py_MODULE_HEAD_UNIFIED
#include "model/py_model.h"
#include "model/py_variance_kernel.h"

static PyMethodDef model_methods[] = {
  {"comoving_distance", (PyCFunction)py_sif_comoving_distance,
    METH_VARARGS | METH_KEYWORDS,
    "comoving_distance(z, omega_m, omega_de=None, omega_r=0.0, w0=-1.0, "
    "wa=0.0)\n"
    "--\n\n"
    "Line-of-sight comoving distance to redshift z, in Mpc/h.\n\n"
    "For a w0waCDM background: dark energy with w(a) = w0 + wa (1 - a),\n"
    "curvature 1 - omega_m - omega_de - omega_r. The distance a survey's\n"
    "tracers are placed at by Field.convert_sky_coordinates(), with the\n"
    "same arguments.\n\n"
    "Args:\n"
    "    z: A redshift, or an array of them; at least 0.\n"
    "    omega_m: Matter density today.\n"
    "    omega_de: Dark-energy density today; None, the default, makes the\n"
    "        model flat (1 - omega_m - omega_r).\n"
    "    omega_r: Radiation density today.\n"
    "    w0, wa: The dark-energy equation of state: -1 and 0 for a\n"
    "        cosmological constant.\n\n"
    "Returns:\n"
    "    float or ndarray: D_C in Mpc/h, shaped like z.\n\n"
    "Raises:\n"
    "    ValueError: For a negative or non-finite redshift, a negative\n"
    "        density, or a model with no expansion history out to z."},

  {"delta_moments_pk", (PyCFunction)py_sif_delta_moments_pk,
    METH_VARARGS | METH_KEYWORDS,
    "delta_moments_pk(k, pk, radii, order, window='top_hat')"
    "\n"
    "--\n\n"
    "Evaluates the spectral moments sigma_0..sigma_order from a tabulated "
    "power spectrum, as the continuum integral.\n"
    "The theory counterpart of measure.delta_moments_grid: no box, no "
    "realisation, no cosmic variance."},

  {"bbks_g", (PyCFunction)py_sif_bbks_g, METH_VARARGS | METH_KEYWORDS,
    "bbks_g(gamma, w, g='fitted')"
    "\n"
    "--\n\n"
    "The BBKS G(gamma, w) function, w = gamma * nu. Fitted form by default "
    "(BBKS 1986 eq. 4.4); g='exact' quadratures the defining integral "
    "(Wu 2020 eqs. 18-19)."},

  {"bbks_number_density_differential",
    (PyCFunction)py_sif_bbks_number_density_differential,
    METH_VARARGS | METH_KEYWORDS,
    "bbks_number_density_differential(nu, gamma, r_star, g='fitted')"
    "\n"
    "--\n\n"
    "Differential number density of maxima of a Gaussian field, per unit "
    "volume per unit nu (BBKS 1986 eq. 4.3).\n"
    "Takes parallel nu, gamma and r_star arrays; gamma and r_star come from "
    "the matching properties of a DeltaMoments."},

  {"bbks_number_density_cumulative",
    (PyCFunction)py_sif_bbks_number_density_cumulative,
    METH_VARARGS | METH_KEYWORDS,
    "bbks_number_density_cumulative(moments, delta, g='fitted')"
    "\n"
    "--\n\n"
    "Number density of Gaussian-field maxima above a density contrast, one "
    "value per smoothing radius of the given DeltaMoments (order >= 2).\n"
    "The threshold is converted per radius as nu_t = delta / sigma_0(R)."},

  {"size_function_bbks", (PyCFunction)py_sif_size_function_bbks,
    METH_VARARGS | METH_KEYWORDS,
    "size_function_bbks(moments, delta, units='ln_r', g='fitted')"
    "\n"
    "--\n\n"
    "Size function of a Gaussian field above a density threshold, as a "
    "SizeFunction over the radii of the given DeltaMoments (order >= 2).\n"
    "units='ln_r' (default) puts -dC/dlnR in vsf, units='r' puts -dC/dR; "
    "options records which. Evaluated as a finite difference over the "
    "moments' radii, which must be strictly increasing."},

  {"delta_sigma_slope_pk", (PyCFunction)py_sif_delta_sigma_slope_pk,
    METH_VARARGS | METH_KEYWORDS,
    "delta_sigma_slope_pk(k, pk, radii, window='top_hat')"
    "\n"
    "--\n\n"
    "Logarithmic slope dln(sigma)/dln(R) from a tabulated power spectrum.\n"
    "Differentiates the window under the integral, so it is exact rather than "
    "a finite difference: the radii need not be ordered or finely spaced."},

  {"spherical_map_nonlinear", (PyCFunction)py_sif_spherical_map_nonlinear,
    METH_VARARGS | METH_KEYWORDS,
    "spherical_map_nonlinear(delta_linear, method='b94')"
    "\n"
    "--\n\n"
    "Non-linear density contrast a void of a given linear contrast evolves "
    "to.\n"
    "method='b94' (default) uses the Bernardeau (1994) fit, method='exact' "
    "solves the Einstein-de Sitter expansion by root-find. The expansion "
    "factor r_NL/r_L is (1 + delta_NL)**(-1/3); delta_L = -2.7 gives ~1.69."},

  {"spherical_map_linear", (PyCFunction)py_sif_spherical_map_linear,
    METH_VARARGS | METH_KEYWORDS,
    "spherical_map_linear(delta_nonlinear, method='b94')"
    "\n"
    "--\n\n"
    "Linear density contrast that evolves into a given non-linear one, the "
    "inverse of delta_nonlinear.\n"
    "Wanted when a barrier is quoted as an observed underdensity rather than "
    "as a linear threshold. Same method= choices."},

  {"svdw_multiplicity_function", (PyCFunction)py_sif_svdw_multiplicity_function,
    METH_VARARGS | METH_KEYWORDS,
    "svdw_multiplicity_function(sigma, delta_v, delta_c)"
    "\n"
    "--\n\n"
    "Excursion-set void multiplicity function f_ln(sigma), Sheth & van de "
    "Weygaert (2004).\n"
    "Shared by the SvdW and Vdn size functions, which differ only in the "
    "Lagrangian-to-Eulerian mapping."},

  {"size_function_svdw", (PyCFunction)py_sif_size_function_svdw,
    METH_VARARGS | METH_KEYWORDS,
    "size_function_svdw(k, pk, radii, delta_v, delta_c, units='ln_r', met"
    "hod='b94')"
    "\n"
    "--\n\n"
    "Sheth & van de Weygaert void size function, as a SizeFunction over the "
    "given Eulerian radii.\n"
    "Number-conserving, so its void volume fraction exceeds one at large "
    "radii; present as the reference rather than the recommendation.\n"
    "The expansion factor is not an argument: delta_v already fixes it. Use "
    "method='b94' (default) or 'exact' to choose how."},

  {"delta_covariance_pk", (PyCFunction)py_sif_delta_covariance_pk,
    METH_VARARGS | METH_KEYWORDS,
    "delta_covariance_pk(k, pk, radii, window='top_hat')"
    "\n"
    "--\n\n"
    "Covariance of the smoothed field between every pair of smoothing radii, "
    "from a tabulated power spectrum.\n"
    "Returns (cov, sigma, high_k_fraction, deriv_variance). cov is the packed "
    "lower triangle in float64, S(i,j) at i*(i+1)/2 + j for j <= i; sigma is "
    "the sqrt of its diagonal, which equals sigma_0 from delta_moments_pk; "
    "deriv_variance is <(d delta / dS)^2> with S = sigma^2, differentiated "
    "under the integral rather than differenced off the matrix, so it does not "
    "depend on how finely radii was sampled. window='top_hat' (default) or "
    "'gaussian'. Requires a P(k) sampled on at least ~1000 log-spaced points."},

  {"ep_barrier_smt", (PyCFunction)py_sif_ep_barrier_smt,
    METH_VARARGS | METH_KEYWORDS,
    "ep_barrier_smt(sigma, alpha, beta, gamma)"
    "\n"
    "--\n\n"
    "Sheth-Mo-Tormen moving barrier, alpha * (1 + (beta/sigma)**gamma).\n"
    "Feed it walk.sigma for the emulator, or the sigma returned by "
    "delta_covariance_pk for the Monte Carlo, so the barrier and the walk "
    "share one variance."},

  {"ep_first_crossing_counts", (PyCFunction)py_sif_ep_first_crossing_counts,
    METH_VARARGS | METH_KEYWORDS,
    "ep_first_crossing_counts(radii, cov, barrier, n_paths, seed=0, collapse="
    "None, return_collapse_counts=False)"
    "\n"
    "--\n\n"
    "Raw first-crossing counts of a correlated random walk against a moving "
    "barrier, one per radius.\n"
    "radii ascending, cov the packed triangle from delta_covariance_pk, "
    "barrier one entry per radius. The result is a deterministic function of "
    "(seed, n_paths), independent of the thread count.\n"
    "collapse adds void-in-cloud: delta_c, positive, as a number or one entry "
    "per radius. A walk that reaches it before the void barrier is inside a "
    "collapsing region and is not counted as a void at any radius. The same "
    "seed gives the same walks with or without it. "
    "return_collapse_counts=True returns (counts, collapse_counts), the "
    "latter the walks discarded at each radius."},

  {"ep_multiplicity_function", (PyCFunction)py_sif_ep_multiplicity_function,
    METH_VARARGS | METH_KEYWORDS,
    "ep_multiplicity_function(radii, cov, barrier, n_paths, seed=0, retur"
    "n_counts=False, collapse=None)"
    "\n"
    "--\n\n"
    "Void multiplicity function from the first crossing of a moving barrier "
    "by a correlated random walk.\n"
    "Returns len(radii) - 1 bin-centred values: a Monte Carlo can only place "
    "a crossing between two consecutive smoothing scales. Same arguments as "
    "ep_first_crossing_counts; collapse adds void-in-cloud.\n"
    "return_counts=True returns (mult, counts) instead, with the raw "
    "len(radii) counts the multiplicity was built from. Prefer it to a second "
    "call to ep_first_crossing_counts, which would run the whole walk again."},

  {"ep_multiplicity_function_emu",
    (PyCFunction)py_sif_ep_multiplicity_function_emu,
    METH_VARARGS | METH_KEYWORDS,
    "ep_multiplicity_function_emu(walk, barrier, radii, collapse=None, "
    "binned=False, return_domain=False)"
    "\n"
    "--\n\n"
    "Emulated void multiplicity function: the same quantity as "
    "ep_multiplicity_function, without the random walks.\n"
    "The up-crossing rate of Verza et al. (2024) eq. (3.15) -- Musso-Sheth "
    "with the exact scale-dependent <(d delta / dS)^2> -- corrected by a "
    "small trained network. About 0.06% of the Monte Carlo it was fitted "
    "against (median, radii below 80 Mpc/h), for walks starting anywhere from "
    "~2 to 250 Mpc/h.\n"
    "walk is kernel(pk) for a VarianceKernel; barrier has one value per "
    "lattice radius, built from walk.sigma (ep_barrier_smt(walk.sigma, "
    "...)); radii, ascending and within the lattice, are where the answer is "
    "wanted. The emulator chooses its own grid, the slice of the lattice from "
    "just inside the smallest radius out to where barrier / sigma reaches "
    "4.5, and interpolates to radii.\n"
    "Returns the first-crossing density per unit R at each radius; "
    "binned=True returns its average over each bin between consecutive "
    "radii instead, len(radii) - 1 values, which is what the Monte Carlo "
    "ep_multiplicity_function measures on the same radii.\n"
    "collapse (delta_c, a number or one per lattice radius) adds "
    "void-in-cloud, as a factor K <= 1 on the single-barrier answer (about "
    "0.09% where it matters); its covariance is built from the walk's "
    "spectrum, which costs milliseconds rather than microseconds. "
    "return_domain=True returns (mult, domain) with in_domain, "
    "n_bins_outside, nu_origin, first_step_mass and expected_error."},

  {"size_function_vdn", (PyCFunction)py_sif_size_function_vdn,
    METH_VARARGS | METH_KEYWORDS,
    "size_function_vdn(k, pk, radii, delta_v, delta_c, units='ln_r', meth"
    "od='b94')"
    "\n"
    "--\n\n"
    "Volume-conserving (Vdn) void size function, as a SizeFunction over the "
    "given Eulerian radii.\n"
    "Identical to SvdW but divided by the Eulerian volume, which keeps the "
    "void volume fraction below unity. Same arguments as size_function_svdw."},

  {"size_function_evm", (PyCFunction)py_sif_size_function_evm,
    METH_VARARGS | METH_KEYWORDS,
    "size_function_evm(k, pk, radii, delta_v, h, units='ln_r', method='b94')"
    "\n"
    "--\n\n"
    "The EVM void size function, as a SizeFunction over the given "
    "Eulerian radii: the moving barrier calibrated at z = 99, its "
    "emulated first crossing, and a fitted volume-conserving mapping.\n"
    "pk is the linear P(k) at the epoch wanted, and no redshift is "
    "passed: the epoch is seen through sigma12, for which h places 12 "
    "Mpc on radii in Mpc/h. delta_v is the linear barrier, as for "
    "size_function_svdw; spherical_map_linear converts a threshold "
    "quoted as an observed underdensity. Calibrated on one cosmology for "
    "sigma12 from 0.39 to 1 and |delta_v| sigma12(z=99) / sigma12 from "
    "0.003 to 0.045; outside them a warning is logged. Radii past the "
    "emulated first crossing give 0.\n"
    "Builds a VarianceKernel on every call, a good part of a second; "
    "size_function_evm_kernel takes one built beforehand."},

  {"size_function_evm_kernel", (PyCFunction)py_sif_size_function_evm_kernel,
    METH_VARARGS | METH_KEYWORDS,
    "size_function_evm_kernel(kernel, pk, radii, delta_v, h, units='ln_r', "
    "method='b94')"
    "\n"
    "--\n\n"
    "size_function_evm on a VarianceKernel built beforehand, for chains: "
    "only pk changes between calls, and it is given at kernel.k.\n"
    "The kernel must have a top-hat window and a lattice holding 12 Mpc "
    "and radii, reaching in to the Lagrangian radii and out to where the "
    "barrier is ~3.5 sigma: VarianceKernel(r_min=0.3, r_max=1000.0) is "
    "what the model was calibrated with. Other arguments as "
    "size_function_evm."},

  {NULL, NULL, 0, NULL}};

static struct PyModuleDef model_module = {PyModuleDef_HEAD_INIT,
  .m_name = "pysif.model",
  .m_doc = "Theoretical predictions.\n\n"
           "Size functions from the BBKS peak statistics, the SvdW and Vdn\n"
           "excursion-set models, and the excursion-set first-crossing\n"
           "problem in both its Monte Carlo and emulated forms. Also the\n"
           "spectral moments and covariance of a model power spectrum, the\n"
           "variance kernel the emulator takes its walk from, and\n"
           "the spherical-evolution map between linear and non-linear\n"
           "density contrast.\n\n"
           "These return the same containers pysif.measure fills, so a\n"
           "prediction and a measurement can be compared directly.",
  .m_size = -1, .m_methods = model_methods};

/* The kernel and the walk it gives, which the emulator takes. */
static const struct {
  const char* name;
  PyTypeObject* type;
} MODEL_TYPES[] = {
  {"VarianceKernel", &sifVarianceKernelType},
  {"Walk", &sifWalkType},
};

/* Submodule exporter called from the parent module initialisation routing */
PyObject* py_sif_init_model(void) {
  const size_t n = sizeof(MODEL_TYPES) / sizeof(MODEL_TYPES[0]);
  for (size_t i = 0; i < n; i++)
    if (PyType_Ready(MODEL_TYPES[i].type) < 0)
      return NULL;

  PyObject* m = PyModule_Create(&model_module);
  if (!m)
    return NULL;

  for (size_t i = 0; i < n; i++) {
    if (PyModule_AddObjectRef(
          m, MODEL_TYPES[i].name, (PyObject*)MODEL_TYPES[i].type) < 0) {
      Py_DECREF(m);
      return NULL;
    }
  }
  return m;
}
