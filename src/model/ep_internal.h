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

#endif /* __SIF_EP_INTERNAL_H__ */
