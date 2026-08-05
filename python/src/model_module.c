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
    "(BBKS 1986 eq. 4.4); exact=True quadratures the defining integral "
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

  {"expansion_factor", (PyCFunction)py_sif_expansion_factor,
    METH_VARARGS | METH_KEYWORDS,
    "Lagrangian-to-Eulerian expansion factor r_NL/r_L implied by a void "
    "barrier, from the Bernardeau (1994) fit. delta_v = -2.7 gives ~1.69."},

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
    "radii; present as the reference rather than the recommendation."},

  {"size_function_vdn", (PyCFunction)py_sif_size_function_vdn,
    METH_VARARGS | METH_KEYWORDS,
    "Volume-conserving (Vdn) void size function, as a SizeFunction over the "
    "given Eulerian radii.\n"
    "Identical to SvdW but divided by the Eulerian volume, which keeps the "
    "void volume fraction below unity."},

  {NULL, NULL, 0, NULL}};

static struct PyModuleDef model_module = {PyModuleDef_HEAD_INIT,
  .m_name = "pysif.model",
  .m_doc = "SIF sub-module for theoretical predictions.",
  .m_size = -1, .m_methods = model_methods};

/* Submodule exporter called from the parent module initialization routing */
PyObject* py_sif_init_model(void) {
  return PyModule_Create(&model_module);
}
