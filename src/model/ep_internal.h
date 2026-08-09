#ifndef __SIF_EP_INTERNAL_H__
#define __SIF_EP_INTERNAL_H__

/*
 * Internals of the excursion-set first-crossing model. Not a public header;
 * the test suite includes it directly, the way test_fft includes math/fft.h,
 * to assert the factor's invariants without inferring them from a histogram.
 */

#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/model/excursionset.h"

/* Doubles per row of the factor, so each row starts on a cache line and the
 * walk's dot product runs over whole vectors with no masked tail. */
#define __SIF_EP_ROW_PAD ((uint64_t)(__SIF_CACHE_LINE / (int)sizeof(double)))

/*
 * @brief Lower-triangular Cholesky factor of a covariance, L L^T = S.
 *
 * Stored in the DESCENDING radius order the walk runs in: row j is
 * radii[n - 1 - j]. Row-major, each row padded to a multiple of
 * __SIF_EP_ROW_PAD doubles and zero from the diagonal to the end of the
 * padding.
 */
typedef struct {
  uint32_t  n;
  double*   chol;       /* n_packed entries */
  uint64_t* row_offset; /* n + 1 entries; row_offset[n] == n_packed */
  uint64_t  n_packed;
} ep_factor_t;

/*
 * @brief Allocates a factor for n rows, zero-filled.
 *
 * The zero fill is load-bearing: the factorization writes only up to each
 * row's diagonal and relies on the allocation to have zeroed the padding.
 *
 * @return SIF_OK, or SIF_ERR_ALLOC with the struct left safe to free.
 */
int ep_factor_init(ep_factor_t* f, uint32_t n);

void ep_factor_free(ep_factor_t* f);

/*
 * @brief Factorizes a packed covariance into walk order.
 *
 * Doubles as the test that the covariance is positive semi-definite.
 *
 * @param f Factor, already initialized for n rows
 * @param cov Packed lower triangle in ASCENDING radius order
 * @param radii n radii, ascending, used only to name a scale in a diagnostic
 * @param n Number of radii
 *
 * @return SIF_OK, or SIF_ERR_RANGE if a pivot is not positive, having logged
 * the offending radius and the correlation that explains it.
 */
int ep_cholesky(ep_factor_t* f, const double* cov,
  const real_t* radii, uint32_t n);

/*
 * @brief Reads S(i, j) from a packed lower triangle in ascending order.
 *
 * Symmetric in its arguments, so callers need not order them.
 */
static inline double ep_cov_get(
  const double* cov, uint32_t i, uint32_t j) {
  return (i >= j) ? cov[SIF_COV_INDEX(i, j)] : cov[SIF_COV_INDEX(j, i)];
}

/* --- The up-crossing baseline --- */

/*
 * @brief Local description of the walk at each smoothing radius.
 *
 * Everything the semi-analytic first-crossing rate needs, in the walk's own
 * time variable S = sigma^2(R). All arrays are n entries in the caller's
 * ASCENDING radius order.
 */
typedef struct {
  uint32_t n;

  double* S;      /* sigma^2, the walk's time variable; DESCENDS with index */
  double* nu;     /* B / sigma, the barrier in units of the field */
  double* gamma2; /* 1 / (4 S <delta'^2>), the squared correlation between the
                   * walk and its own derivative: 0 uncorrelated, 1 smooth */
  double* y;      /* (dB/dS - mu) / Sigma, how fast the barrier runs away from
                   * the walk, in units of the walk's own slope scatter */
  double* f_up;   /* the up-crossing rate itself, per unit S */
} ep_features_t;

int ep_features_init(ep_features_t* f, uint32_t n);
void ep_features_free(ep_features_t* f);

/*
 * @brief Fills the local description from a covariance and a barrier.
 *
 * @param deriv_variance Optional, n entries: <(d delta / dS)^2> as
 * sif_delta_covariance_pk returns it. Strongly preferred. Pass NULL to fall
 * back on differencing the covariance, which is only FIRST order here -- the
 * covariance carries a |S1 - S2|^3 term across its diagonal, so the leading
 * error of a mixed second difference does not cancel on any grid -- and is
 * wrong by around 10% at 40 radii and 4% at 120. The fallback logs a warning,
 * because the resulting gamma2 describes the caller's grid as much as the
 * field.
 *
 * @return SIF_OK, or SIF_ERR_RANGE if the covariance is not a covariance.
 */
int ep_features_fill(ep_features_t* f, const real_t* radii, uint32_t n,
  const double* cov, const real_t* barrier, const double* deriv_variance);

/*
 * @brief First-crossing multiplicity from the up-crossing rate alone.
 *
 * The Musso-Sheth approximation: the rate at which the walk crosses the
 * barrier upward faster than the barrier itself moves, treated as a hazard and
 * accumulated into a survival curve. Exact in the limit of a high barrier,
 * where a first crossing and any crossing are the same event; it runs a few
 * per cent low for a flat barrier and up to fifteen for a steep moving one.
 *
 * @return Newly allocated array of n - 1 values on the bin centres, released
 * with sif_free_aligned, or NULL on failure.
 */
NODISCARD real_t* ep_multiplicity_upcrossing(const ep_features_t* f,
  const real_t* radii, uint32_t n);

#endif /* __SIF_EP_INTERNAL_H__ */
