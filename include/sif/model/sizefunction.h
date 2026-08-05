#ifndef __SIF_MODEL_SIZEFUNCTION_H__
#define __SIF_MODEL_SIZEFUNCTION_H__

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/structures/sizefunction.h"

/* Bernardeau (1994) fit constant linking the linear void barrier to the
 * spherical expansion factor. */
#define SIF_SPHERICAL_EXPANSION_C 1.594

/*
 * @brief Lagrangian-to-Eulerian expansion factor implied by a void barrier.
 *
 * Inverts delta_v = c [1 - (r_NL/r_L)^(3/c)], the fit of Bernardeau (1994) as
 * quoted by Jennings, Li & Hu (2013) eq. (A4), accurate to 0.2%. Shell
 * crossing at delta_v = -2.7 gives about 1.69.
 *
 * @param delta_v Linear void barrier, strictly negative
 *
 * @return r_NL / r_L, or 0 for a non-negative barrier.
 */
real_t sif_expansion_factor(real_t delta_v);

/*
 * @brief Excursion-set void multiplicity function f_ln(sigma).
 *
 * Sheth & van de Weygaert (2004) as given by Jennings, Li & Hu (2013)
 * eq. (8): the analytic small-x limit below x = 0.276, the mode series above
 * it, with x = (D/|delta_v|) sigma and D = |delta_v| / (delta_c + |delta_v|).
 *
 * Shared by the SvdW and Vdn size functions, which differ only in how the
 * result is mapped from Lagrangian to Eulerian radii.
 *
 * @note The series is summed to convergence rather than truncated at the four
 * terms the reference uses; the two agree well inside its quoted 0.2% for
 * D < 3/4, and summing further keeps it valid as D approaches 1.
 *
 * @param sigma R.m.s. density contrast per entry, strictly positive
 * @param n Length of sigma
 * @param delta_v Linear void barrier, strictly negative
 * @param delta_c Collapse barrier, strictly positive
 *
 * @return Newly allocated array of n values, released with sif_free_aligned,
 * or NULL on invalid input.
 */
NODISCARD real_t* sif_multiplicity_function_svdw(
  const real_t* sigma, uint32_t n, real_t delta_v, real_t delta_c);

/*
 * @brief Sheth & van de Weygaert void size function.
 *
 * Jennings, Li & Hu (2013) eqs. (9) and (10): the number-conserving mapping,
 * in which the linear abundance is evaluated at r_L = r / F and reported
 * against r without rescaling its amplitude.
 *
 * @note Number conservation makes this model exceed a void volume fraction of
 * one at large radii, which is why Vdn exists. It is reproduced here as the
 * reference, not as a recommendation.
 *
 * @param k Wavenumbers, strictly positive and strictly increasing
 * @param pk Linear power spectrum sampled at k
 * @param n_points Length of k and pk, at least 2
 * @param radii Eulerian (observed) void radii, strictly positive
 * @param n_radii Number of radii
 * @param delta_v Linear void barrier, strictly negative
 * @param delta_c Collapse barrier, strictly positive
 * @param expansion_factor r_NL / r_L. Pass <= 0 to derive it from delta_v
 * through sif_expansion_factor.
 * @param opt SIF_VSF_BIN_LN (default) or SIF_VSF_BIN_LINEAR for the units.
 * The window is always a top-hat, which is what the barriers are calibrated
 * against.
 *
 * @return Newly allocated size function with n_bins = n_radii, released with
 * sif_size_function_free, or NULL on invalid input.
 */
NODISCARD sif_size_function_t* sif_size_function_svdw(const real_t* k,
  const real_t* pk, uint32_t n_points, const real_t* radii, uint32_t n_radii,
  real_t delta_v, real_t delta_c, real_t expansion_factor, sif_option_t opt);

/*
 * @brief Volume-conserving (Vdn) void size function.
 *
 * Jennings, Li & Hu (2013) eq. (12). Identical to SvdW except that the
 * abundance is divided by the Eulerian volume rather than the Lagrangian one,
 * which conserves volume instead of number and keeps the void volume fraction
 * below unity.
 *
 * Parameters are as sif_size_function_svdw.
 *
 * @return Newly allocated size function with n_bins = n_radii, released with
 * sif_size_function_free, or NULL on invalid input.
 */
NODISCARD sif_size_function_t* sif_size_function_vdn(const real_t* k,
  const real_t* pk, uint32_t n_points, const real_t* radii, uint32_t n_radii,
  real_t delta_v, real_t delta_c, real_t expansion_factor, sif_option_t opt);

#endif /* __SIF_MODEL_SIZEFUNCTION_H__ */
