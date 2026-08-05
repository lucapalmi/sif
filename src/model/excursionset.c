/*
 * Excursion-set first crossing: the Sheth & van de Weygaert multiplicity
 * function, the Sheth-Mo-Tormen barrier, and the Monte Carlo first crossing of
 * a moving barrier by a correlated random walk.
 */

#include "sif/model/excursionset.h"

#include "model/ep_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/random.h"

#include <math.h>
#include <stdlib.h>

#define __TAG "ep"

/* Relative jitter added to the diagonal before factorizing, so a marginally
 * rank-deficient radius grid does not fail on its last pivot. Far below the
 * precision at which the covariance is meaningful. */
#define __SIF_EP_JITTER 1e-12

static inline uint64_t __roundup(uint64_t v, uint64_t m) {
  return (v + m - 1) / m * m;
}

/* --- The factor --- */

int ep_factor_init(ep_factor_t* f, uint32_t n) {

  f->n = n;
  f->chol = NULL;
  f->n_packed = 0;

  f->row_offset = sif_malloc_aligned(((size_t)n + 1) * sizeof(uint64_t));
  if (!f->row_offset) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the factor row offsets");
    return SIF_ERR_ALLOC;
  }

  uint64_t at = 0;
  for (uint32_t j = 0; j < n; j++) {
    f->row_offset[j] = at;
    at += __roundup((uint64_t)j + 1, __SIF_EP_ROW_PAD);
  }
  f->row_offset[n] = at;
  f->n_packed = at;

  /* Zeroed, not merely allocated: the sweep below writes only as far as each
   * row's diagonal and leaves the padding for this to have cleared. */
  f->chol = sif_calloc_aligned((size_t)at, sizeof(double));
  if (!f->chol) {
    SIF_LOG_ERROR(__TAG, "failed to allocate a %llu-entry Cholesky factor",
      (unsigned long long)at);
    return SIF_ERR_ALLOC;
  }

  return SIF_OK;
}

void ep_factor_free(ep_factor_t* f) {
  if (!f)
    return;
  sif_free_aligned(f->chol);
  sif_free_aligned(f->row_offset);
  f->chol = NULL;
  f->row_offset = NULL;
  f->n = 0;
  f->n_packed = 0;
}

static inline double __dot(const double* a, const double* b, uint32_t len) {
  double s = 0.0;
  for (uint32_t m = 0; m < len; m++)
    s += a[m] * b[m];
  return s;
}

/* The nearest-neighbour correlation is the number that predicts a failed
 * pivot, so the diagnostic reports it. */
static void __report_pivot_failure(
  const double* cov, const real_t* radii, uint32_t n, uint32_t a, double piv) {

  double worst = 0.0;
  if (a > 0) {
    const double c = ep_cov_get(cov, a, a - 1) /
                     sqrt(ep_cov_get(cov, a, a) *
                          ep_cov_get(cov, a - 1, a - 1));
    if (fabs(c) > worst)
      worst = fabs(c);
  }
  if (a + 1 < n) {
    const double c = ep_cov_get(cov, a + 1, a) /
                     sqrt(ep_cov_get(cov, a, a) *
                          ep_cov_get(cov, a + 1, a + 1));
    if (fabs(c) > worst)
      worst = fabs(c);
  }

  if (worst > 1.0) {
    SIF_LOG_ERROR(__TAG,
      "the covariance is not positive definite: pivot %g at radius %g (index "
      "%u of %u), whose nearest-neighbour correlation is %.6f. A correlation "
      "above one violates Cauchy-Schwarz, so this matrix is not a covariance "
      "at all; check how it was built",
      piv, (double)radii[a], a, n, worst);
  } else {
    SIF_LOG_ERROR(__TAG,
      "the covariance is not positive definite: pivot %g at radius %g (index "
      "%u of %u), whose nearest-neighbour correlation is %.9f. A correlation "
      "this close to one means the radii are sampled more finely than the "
      "covariance can be separated in double precision; use a coarser grid",
      piv, (double)radii[a], a, n, worst);
  }
}

int ep_cholesky(ep_factor_t* f, const double* cov,
  const real_t* radii, uint32_t n) {

  /* Walk index j is ascending index n - 1 - j, so the walk starts at the
   * largest radius, where sigma is smallest. */
  double trace = 0.0;
  for (uint32_t i = 0; i < n; i++)
    trace += ep_cov_get(cov, i, i);
  const double jitter = __SIF_EP_JITTER * trace / (double)n;

  for (uint32_t j = 0; j < n; j++) {
    double* Lj = f->chol + f->row_offset[j];
    const uint32_t aj = n - 1 - j;

    for (uint32_t m = 0; m < j; m++) {
      const double* Lm = f->chol + f->row_offset[m];
      const double s = ep_cov_get(cov, aj, n - 1 - m);
      Lj[m] = (s - __dot(Lj, Lm, m)) / Lm[m];
    }

    const double piv =
      ep_cov_get(cov, aj, aj) + jitter - __dot(Lj, Lj, j);

    /* A sign check before the sqrt, not an isnan/isinf test after it: the
     * release build carries -ffast-math, under which the compiler may fold
     * those predicates to a constant. */
    if (!(piv > 0.0)) {
      __report_pivot_failure(cov, radii, n, aj, piv);
      return SIF_ERR_RANGE;
    }

    Lj[j] = sqrt(piv);
    /* Lj[j+1 ..] is left at the zero the allocation put there. */
  }

  return SIF_OK;
}

/* --- Validation --- */

static int __validate(const real_t* radii, uint32_t n_radii, const double* cov,
  const real_t* barrier, uint64_t n_paths) {

  if (!radii || !cov || !barrier) {
    SIF_LOG_ERROR(__TAG, "radii, cov and barrier are all required");
    return SIF_ERR_INVALID;
  }

  if (n_radii < 2) {
    SIF_LOG_ERROR(__TAG,
      "%u radii; a first-crossing walk needs at least two scales to step "
      "between",
      n_radii);
    return SIF_ERR_INVALID;
  }

  if (n_radii > SIF_COV_MAX_RADII) {
    SIF_LOG_ERROR(__TAG, "%u radii exceeds the maximum of %d", n_radii,
      SIF_COV_MAX_RADII);
    return SIF_ERR_INVALID;
  }

  if (n_paths == 0) {
    SIF_LOG_ERROR(__TAG, "n_paths is zero; there is nothing to sample");
    return SIF_ERR_INVALID;
  }

  for (uint32_t i = 0; i < n_radii; i++) {
    if (!(radii[i] > 0.0f)) {
      SIF_LOG_ERROR(__TAG, "radius %u is %g, must be strictly positive", i,
        (double)radii[i]);
      return SIF_ERR_INVALID;
    }
    if (i > 0 && !(radii[i] > radii[i - 1])) {
      SIF_LOG_ERROR(__TAG,
        "radii are not strictly increasing at index %u (%g after %g); the walk "
        "runs from the largest scale down, but the public order is ascending "
        "like every other entry point",
        i, (double)radii[i], (double)radii[i - 1]);
      return SIF_ERR_INVALID;
    }
    if (!(ep_cov_get(cov, i, i) > 0.0)) {
      SIF_LOG_ERROR(__TAG,
        "the covariance diagonal at radius %u (%g) is %g, must be strictly "
        "positive",
        i, (double)radii[i], ep_cov_get(cov, i, i));
      return SIF_ERR_INVALID;
    }
    if (!isfinite((double)barrier[i])) {
      SIF_LOG_ERROR(__TAG, "the barrier at radius %u (%g) is not finite", i,
        (double)radii[i]);
      return SIF_ERR_INVALID;
    }
  }

  /* Positive semi-definiteness is left to the factorization, which reports
   * the offending scale. */

  return SIF_OK;
}

/* --- The Sheth & van de Weygaert multiplicity function --- */

/* Below this the mode series is replaced by its analytic small-x limit; above
 * it the series converges. Jennings, Li & Hu (2013) eq. (8). */
#define __SVDW_X_SWITCH 0.276

/* The Gaussian factor kills the series long before this; the cap only stops a
 * pathological D from spinning. */
#define __SVDW_MAX_TERMS 64

static double __f_ln_sigma(double sigma, double abs_dv, double dcal) {
  /* D = |delta_v| / (delta_c + |delta_v|), x = (D / |delta_v|) sigma. */
  const double x = dcal / abs_dv * sigma;

  if (x <= __SVDW_X_SWITCH) {
    /* The series does not converge as x -> 0: sum_j j sin(j pi D) diverges,
     * so the limit has to come from the analytic form. */
    return sqrt(2.0 / M_PI) * (abs_dv / sigma) *
           exp(-0.5 * abs_dv * abs_dv / (sigma * sigma));
  }

  double sum = 0.0;
  for (int j = 1; j <= __SVDW_MAX_TERMS; j++) {
    const double jpx = j * M_PI * x;
    const double term =
      exp(-0.5 * jpx * jpx) * j * M_PI * x * x * sin(j * M_PI * dcal);

    sum += term;

    /* The envelope is monotonic past the first term, so once it is negligible
     * everything after it is too, regardless of the sine. */
    if (j > 1 && fabs(exp(-0.5 * jpx * jpx) * j * M_PI * x * x) < 1e-16)
      break;
  }

  return 2.0 * sum;
}

real_t* sif_multiplicity_function_svdw(
  const real_t* sigma, uint32_t n, real_t delta_v, real_t delta_c) {

  if (!sigma || n == 0) {
    SIF_LOG_ERROR(__TAG, "invalid arguments to multiplicity_function_svdw");
    return NULL;
  }

  if (!(delta_v < 0.0f)) {
    SIF_LOG_ERROR(
      __TAG, "delta_v is %g, must be strictly negative", (double)delta_v);
    return NULL;
  }

  if (!(delta_c > 0.0f)) {
    SIF_LOG_ERROR(
      __TAG, "delta_c is %g, must be strictly positive", (double)delta_c);
    return NULL;
  }

  const double abs_dv = fabs((double)delta_v);
  const double dcal = abs_dv / ((double)delta_c + abs_dv);

  if (dcal >= 0.75) {
    SIF_LOG_WARNING(__TAG,
      "D = %g exceeds the 3/4 the reference validates; the series is summed to "
      "convergence but the underlying approximation is outside its tested "
      "range",
      dcal);
  }

  real_t* out = sif_calloc_aligned((size_t)n, sizeof(real_t));
  if (!out) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the multiplicity array");
    return NULL;
  }

  uint32_t n_bad = 0;

#pragma omp parallel for schedule(static) reduction(+ : n_bad)
  for (uint32_t i = 0; i < n; i++) {
    if (!(sigma[i] > 0.0f)) {
      n_bad++;
      continue;
    }
    const double f = __f_ln_sigma((double)sigma[i], abs_dv, dcal);
    out[i] = (real_t)(f > 0.0 ? f : 0.0);
  }

  if (n_bad > 0) {
    SIF_LOG_WARNING(__TAG,
      "%u of %u sigma values were not positive and were reported as zero",
      n_bad, n);
  }

  return out;
}

/* --- The Sheth-Mo-Tormen moving barrier --- */

real_t* sif_barrier_smt(
  const real_t* sigma, uint32_t n, real_t alpha, real_t beta, real_t gamma) {

  if (!sigma || n == 0) {
    SIF_LOG_ERROR(__TAG, "no sigma values to build a barrier over");
    return NULL;
  }

  if (!(alpha > 0.0f)) {
    SIF_LOG_ERROR(
      __TAG, "alpha is %g, must be strictly positive", (double)alpha);
    return NULL;
  }

  /* Strictly positive: (beta/sigma)^gamma is not real for a negative base at
   * the fractional gamma this barrier is always calibrated with. */
  if (!(beta > 0.0f)) {
    SIF_LOG_ERROR(
      __TAG, "beta is %g, must be strictly positive", (double)beta);
    return NULL;
  }

  real_t* out = sif_calloc_aligned((size_t)n, sizeof(real_t));
  if (!out) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the barrier array");
    return NULL;
  }

  const double a = (double)alpha;
  const double b = (double)beta;
  const double g = (double)gamma;

  uint32_t n_bad = 0;

  for (uint32_t i = 0; i < n; i++) {
    if (!(sigma[i] > 0.0f)) {
      n_bad++;
      continue;
    }
    out[i] = (real_t)(a * (1.0 + pow(b / (double)sigma[i], g)));
  }

  if (n_bad > 0) {
    SIF_LOG_WARNING(__TAG,
      "%u of %u sigma values were not positive; their barriers are zero, which "
      "any walk crosses immediately",
      n_bad, n);
  }

  return out;
}

/* --- The walk --- */

/*
 * Per-path seeding, following __fft_seed_mode in src/math/fft.c. Hashing the
 * path index into the seed, rather than drawing from a per-thread stream,
 * makes the counts a pure function of (seed, n_paths).
 */
static inline void __ep_seed_path(
  sif_prng_state_t* prng, uint64_t seed, uint64_t p) {

  uint64_t z = seed + p * 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  sif_prng_init(prng, z ^ (z >> 31));
}

uint64_t* sif_first_crossing_counts_ep(const real_t* radii, uint32_t n_radii,
  const double* cov, const real_t* barrier, uint64_t n_paths, uint64_t seed,
  sif_option_t opt) {

  (void)opt; /* reserved */

  if (__validate(radii, n_radii, cov, barrier, n_paths) != SIF_OK)
    return NULL;

  const uint32_t n = n_radii;

  ep_factor_t f;
  if (ep_factor_init(&f, n) != SIF_OK) {
    ep_factor_free(&f);
    return NULL;
  }
  if (ep_cholesky(&f, cov, radii, n) != SIF_OK) {
    ep_factor_free(&f);
    return NULL;
  }

  /* Reversed into walk order once, and promoted: it is compared against a
   * double accumulator. */
  double* barrier_walk = sif_malloc_aligned((size_t)n * sizeof(double));
  uint64_t* total = sif_calloc_aligned((size_t)n, sizeof(uint64_t));

  if (!barrier_walk || !total) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the walk buffers");
    sif_free_aligned(barrier_walk);
    sif_free_aligned(total);
    ep_factor_free(&f);
    return NULL;
  }

  for (uint32_t j = 0; j < n; j++)
    barrier_walk[j] = (double)barrier[n - 1 - j];

  /* The last row is the longest, and the dot product runs to its padded end. */
  const uint64_t xi_len = __roundup((uint64_t)n, __SIF_EP_ROW_PAD);

  int alloc_failed = 0;

#pragma omp parallel
  {
    /* Per-thread histogram. The critical section below runs once per thread,
     * not once per path -- do not "fix" it into an atomic in the inner loop. */
    uint64_t* local = calloc((size_t)n, sizeof(uint64_t));
    double* xi = sif_calloc_aligned((size_t)xi_len, sizeof(double));

    if (!local || !xi) {
#pragma omp atomic write
      alloc_failed = 1;
    }

    double spare = 0.0;
    int has_spare = 0;

    /* Paths are i.i.d., so equal blocks balance in expectation; static,
     * guided and chunked dynamic measured within a couple of percent of each
     * other. Safe to change at all only because each path seeds itself. */
#pragma omp for schedule(static)
    for (uint64_t p = 0; p < n_paths; p++) {

      /* A thread that failed to allocate still has to reach the worksharing
       * construct, so it enters the loop and declines the body. */
      if (!local || !xi)
        continue;

      sif_prng_state_t prng;
      __ep_seed_path(&prng, seed, p);

      /* Discarded rather than carried across paths, which would couple them
       * and break the per-path reproducibility. */
      has_spare = 0;

      for (uint32_t j = 0; j < n; j++) {

        /* Lazily drawn: a path crossing at step 5 of 100 pays for six
         * normals, not a hundred. */
        if (has_spare) {
          xi[j] = spare;
          has_spare = 0;
        } else {
          double z0, z1;
          sif_prng_next_gaussian_pair(&prng, &z0, &z1);
          xi[j] = z0;
          spare = z1;
          has_spare = 1;
        }

        const double* Lj = f.chol + f.row_offset[j];
        const uint64_t len = f.row_offset[j + 1] - f.row_offset[j];

        /* Runs past the diagonal to the padded end of the row. Those lanes
         * of the factor are zero, so they contribute nothing regardless of
         * what the previous path left in xi beyond step j. */
        double d = 0.0;
        for (uint64_t m = 0; m < len; m++)
          d += Lj[m] * xi[m];

        if (d >= barrier_walk[j]) {
          local[j]++;
          break;
        }
      }
    }

    if (local) {
#pragma omp critical
      {
        for (uint32_t j = 0; j < n; j++)
          total[j] += local[j];
      }
    }

    free(local);
    sif_free_aligned(xi);
  }

  sif_free_aligned(barrier_walk);
  ep_factor_free(&f);

  if (alloc_failed) {
    SIF_LOG_ERROR(__TAG, "a worker failed to allocate its walk buffers");
    sif_free_aligned(total);
    return NULL;
  }

  /* Back to the caller's ascending order. */
  for (uint32_t j = 0; j < n / 2; j++) {
    const uint64_t t = total[j];
    total[j] = total[n - 1 - j];
    total[n - 1 - j] = t;
  }

  return total;
}

/* --- The multiplicity function --- */

real_t* sif_multiplicity_function_ep(const real_t* radii, uint32_t n_radii,
  const double* cov, const real_t* barrier, uint64_t n_paths, uint64_t seed,
  sif_option_t opt) {

  uint64_t* counts = sif_first_crossing_counts_ep(
    radii, n_radii, cov, barrier, n_paths, seed, opt);
  if (!counts)
    return NULL;

  const uint32_t n_bins = n_radii - 1;

  real_t* f = sif_calloc_aligned((size_t)n_bins, sizeof(real_t));
  if (!f) {
    SIF_LOG_ERROR(__TAG, "failed to allocate the multiplicity array");
    sif_free_aligned(counts);
    return NULL;
  }

  uint64_t crossed = 0;
  for (uint32_t i = 0; i < n_radii; i++)
    crossed += counts[i];

  for (uint32_t i = 0; i < n_bins; i++) {
    const double dr = (double)radii[i + 1] - (double)radii[i];
    f[i] = (real_t)((double)counts[i] / ((double)n_paths * dr));
  }

  if (crossed == 0) {
    SIF_LOG_WARNING(__TAG,
      "no path crossed the barrier at any radius, so the multiplicity is "
      "identically zero; the barrier is far above the field or the covariance "
      "is far below it");
  } else {
    /* Crossings at the largest radius happen on the walk's first step, before
     * there is a bin to put them in. A large share of them means the radius
     * grid does not reach far enough out. */
    const uint64_t dropped = counts[n_radii - 1];
    if ((double)dropped > 0.01 * (double)crossed) {
      SIF_LOG_WARNING(__TAG,
        "%llu of %llu crossings (%.1f%%) happened on the walk's first step, at "
        "the largest radius %g, and fall outside every bin; extend the radius "
        "grid upward",
        (unsigned long long)dropped, (unsigned long long)crossed,
        100.0 * (double)dropped / (double)crossed,
        (double)radii[n_radii - 1]);
    }
  }

  sif_free_aligned(counts);
  return f;
}
