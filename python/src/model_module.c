#define Py_MODULE_HEAD_UNIFIED
#include "model/py_model.h"

static PyMethodDef model_methods[] = {
  {"delta_moments_pk", (PyCFunction)py_sif_delta_moments_pk,
    METH_VARARGS | METH_KEYWORDS,
    "Evaluates the spectral moments sigma_0..sigma_order from a tabulated "
    "power spectrum, as the continuum integral.\n"
    "The theory counterpart of measure.delta_moments_grid: no box, no "
    "realization, no cosmic variance."},

  {"g_bbks", (PyCFunction)py_sif_g_bbks, METH_VARARGS | METH_KEYWORDS,
    "The BBKS G(gamma, w) function, w = gamma * nu. Fitted form by default "
    "(BBKS 1986 eq. 4.4); g='exact' quadratures the defining integral "
    "(Wu 2020 eqs. 18-19)."},

  {"differential_number_density_bbks",
    (PyCFunction)py_sif_differential_number_density_bbks,
    METH_VARARGS | METH_KEYWORDS,
    "Differential number density of maxima of a Gaussian field, per unit "
    "volume per unit nu (BBKS 1986 eq. 4.3).\n"
    "Takes parallel nu, gamma and r_star arrays; gamma and r_star come from "
    "the matching properties of a DeltaMoments."},

  {"cumulative_number_density_bbks",
    (PyCFunction)py_sif_cumulative_number_density_bbks,
    METH_VARARGS | METH_KEYWORDS,
    "Number density of Gaussian-field maxima above a density contrast, one "
    "value per smoothing radius of the given DeltaMoments (order >= 2).\n"
    "The threshold is converted per radius as nu_t = delta / sigma_0(R)."},

  {"size_function_bbks", (PyCFunction)py_sif_size_function_bbks,
    METH_VARARGS | METH_KEYWORDS,
    "Size function of a Gaussian field above a density threshold, as a "
    "SizeFunction over the radii of the given DeltaMoments (order >= 2).\n"
    "units='ln_r' (default) puts -dC/dlnR in vsf, units='r' puts -dC/dR; "
    "options records which. Evaluated as a finite difference over the "
    "moments' radii, which must be strictly increasing."},

  {"sigma_slope_pk", (PyCFunction)py_sif_sigma_slope_pk,
    METH_VARARGS | METH_KEYWORDS,
    "Logarithmic slope dln(sigma)/dln(R) from a tabulated power spectrum.\n"
    "Differentiates the window under the integral, so it is exact rather than "
    "a finite difference: the radii need not be ordered or finely spaced."},

  {"delta_nonlinear", (PyCFunction)py_sif_delta_nonlinear,
    METH_VARARGS | METH_KEYWORDS,
    "Non-linear density contrast a void of a given linear contrast evolves "
    "to.\n"
    "method='b94' (default) uses the Bernardeau (1994) fit, method='exact' "
    "solves the Einstein-de Sitter expansion by root-find. The expansion "
    "factor r_NL/r_L is (1 + delta_NL)**(-1/3); delta_L = -2.7 gives ~1.69."},

  {"delta_linear", (PyCFunction)py_sif_delta_linear,
    METH_VARARGS | METH_KEYWORDS,
    "Linear density contrast that evolves into a given non-linear one, the "
    "inverse of delta_nonlinear.\n"
    "Wanted when a barrier is quoted as an observed underdensity rather than "
    "as a linear threshold. Same method= choices."},

  {"multiplicity_function_svdw",
    (PyCFunction)py_sif_multiplicity_function_svdw,
    METH_VARARGS | METH_KEYWORDS,
    "Excursion-set void multiplicity function f_ln(sigma), Sheth & van de "
    "Weygaert (2004).\n"
    "Shared by the SvdW and Vdn size functions, which differ only in the "
    "Lagrangian-to-Eulerian mapping."},

  {"size_function_svdw", (PyCFunction)py_sif_size_function_svdw,
    METH_VARARGS | METH_KEYWORDS,
    "Sheth & van de Weygaert void size function, as a SizeFunction over the "
    "given Eulerian radii.\n"
    "Number-conserving, so its void volume fraction exceeds one at large "
    "radii; present as the reference rather than the recommendation.\n"
    "The expansion factor is not an argument: delta_v already fixes it. Use "
    "method='b94' (default) or 'exact' to choose how."},

  {"delta_covariance_pk", (PyCFunction)py_sif_delta_covariance_pk,
    METH_VARARGS | METH_KEYWORDS,
    "Covariance of the smoothed field between every pair of smoothing radii, "
    "from a tabulated power spectrum.\n"
    "Returns (cov, sigma, high_k_fraction, deriv_variance). cov is the packed "
    "lower triangle in float64, S(i,j) at i*(i+1)/2 + j for j <= i; sigma is "
    "the sqrt of its diagonal, which equals sigma_0 from delta_moments_pk; "
    "deriv_variance is <(d delta / dS)^2> with S = sigma^2, differentiated "
    "under the integral rather than differenced off the matrix, so it does not "
    "depend on how finely radii was sampled. window='top_hat' (default) or "
    "'gaussian'. Requires a P(k) sampled on at least ~1000 log-spaced points."},

  {"barrier_smt", (PyCFunction)py_sif_barrier_smt,
    METH_VARARGS | METH_KEYWORDS,
    "Sheth-Mo-Tormen moving barrier, alpha * (1 + (beta/sigma)**gamma).\n"
    "Feed it the sigma returned by delta_covariance_pk, so the barrier and "
    "the walk share one variance."},

  {"first_crossing_counts_ep", (PyCFunction)py_sif_first_crossing_counts_ep,
    METH_VARARGS | METH_KEYWORDS,
    "Raw first-crossing counts of a correlated random walk against a moving "
    "barrier, one per radius.\n"
    "radii ascending, cov the packed triangle from delta_covariance_pk, "
    "barrier one entry per radius. The result is a deterministic function of "
    "(seed, n_paths), independent of the thread count."},

  {"multiplicity_function_ep", (PyCFunction)py_sif_multiplicity_function_ep,
    METH_VARARGS | METH_KEYWORDS,
    "Void multiplicity function from the first crossing of a moving barrier "
    "by a correlated random walk.\n"
    "Returns len(radii) - 1 bin-centred values: a Monte Carlo can only place "
    "a crossing between two consecutive smoothing scales. Same arguments as "
    "first_crossing_counts_ep.\n"
    "return_counts=True returns (mult, counts) instead, with the raw "
    "len(radii) counts the multiplicity was built from. Prefer it to a second "
    "call to first_crossing_counts_ep, which would run the whole walk again."},

  {"size_function_vdn", (PyCFunction)py_sif_size_function_vdn,
    METH_VARARGS | METH_KEYWORDS,
    "Volume-conserving (Vdn) void size function, as a SizeFunction over the "
    "given Eulerian radii.\n"
    "Identical to SvdW but divided by the Eulerian volume, which keeps the "
    "void volume fraction below unity. Same arguments as size_function_svdw."},

  {NULL, NULL, 0, NULL}};

static struct PyModuleDef model_module = {PyModuleDef_HEAD_INIT,
  .m_name = "pysif.model",
  .m_doc = "SIF sub-module for theoretical predictions.",
  .m_size = -1, .m_methods = model_methods};

/* Submodule exporter called from the parent module initialization routing */
PyObject* py_sif_init_model(void) {
  return PyModule_Create(&model_module);
}
