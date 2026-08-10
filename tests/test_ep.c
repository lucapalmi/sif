/*
 * Excursion-set first crossing: the covariance across smoothing scales, the
 * correlated random walk it induces, and the multiplicity function built from
 * them.
 *
 * The walk takes its covariance as an input, which is what lets every check on
 * it run against a hand-built matrix with an exact closed-form answer: no
 * quadrature enters those tests, so a failure there is a bug in the walk, the
 * factorization or the binning, never a disagreement about an integral.
 *
 * The covariance itself is checked separately, and mostly against properties
 * that hold whatever the quadrature: its diagonal against the independently
 * tested sigma_0, Cauchy-Schwarz, and the exact self-similarity a power-law
 * spectrum has.
 */
#include "model/ep_internal.h"
#include "sif/core/system.h"
#include "sif/model/deltamoments.h"
#include "sif/model/excursionset.h"
#include "sif/structures/deltamoments.h"
#include "sif/utils/align.h"
#include "sif/utils/random.h"
#include "test_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

/* Standard normal CDF, for the expected bucket occupancies. */
static double __phi(double x) { return 0.5 * erfc(-x / sqrt(2.0)); }

/*
 * The Gaussian generator is the one piece of genuinely new numerical code with
 * no other consumer in the library, so nothing else would catch a bug in it --
 * and it is also the dominant cost of a first-crossing walk, well above the
 * linear algebra, which is why it is worth getting right rather than merely
 * plausible.
 *
 * The seed is fixed, so this test is deterministic: the tolerances below are
 * multiples of the sampling error, but a failure is a real regression and not
 * a bad draw.
 */
#define __N_BUCKETS 40
#define __BUCKET_LO (-4.0)
#define __BUCKET_HI 4.0

static void test_gaussian_generator(void) {
  printf("Gaussian generator\n");

  const uint64_t n_pairs = (uint64_t)SIF_TEST_SCALE(5000000);
  const uint64_t n = 2 * n_pairs;

  sif_prng_state_t prng;
  sif_prng_init(&prng, 20260805u);

  double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
  double cross = 0.0; /* sum of z0 * z1, to catch a correlated pair */
  uint64_t hist[__N_BUCKETS] = {0};
  uint64_t outside = 0;
  uint64_t nonfinite = 0;

  const double width = (__BUCKET_HI - __BUCKET_LO) / __N_BUCKETS;

  for (uint64_t p = 0; p < n_pairs; p++) {
    double z[2];
    sif_prng_next_gaussian_pair(&prng, &z[0], &z[1]);

    cross += z[0] * z[1];

    for (int c = 0; c < 2; c++) {
      const double x = z[c];

      if (!isfinite(x)) {
        nonfinite++;
        continue;
      }

      const double x2 = x * x;
      s1 += x;
      s2 += x2;
      s3 += x2 * x;
      s4 += x2 * x2;

      if (x < __BUCKET_LO || x >= __BUCKET_HI) {
        outside++;
      } else {
        hist[(int)((x - __BUCKET_LO) / width)]++;
      }
    }
  }

  CHECK(nonfinite == 0, "%llu of %llu deviates were not finite; log(0) is not "
                        "being guarded against",
    (unsigned long long)nonfinite, (unsigned long long)n);

  /* Central moments from the raw sums. The mean is O(1/sqrt(n)) from zero, so
   * the subtractions below cancel nothing of consequence. */
  const double dn = (double)n;
  const double mean = s1 / dn;
  const double var = s2 / dn - mean * mean;
  const double m3 = s3 / dn - 3.0 * mean * (s2 / dn) + 2.0 * mean * mean * mean;
  const double m4 = s4 / dn - 4.0 * mean * (s3 / dn) +
                    6.0 * mean * mean * (s2 / dn) -
                    3.0 * mean * mean * mean * mean;
  const double skew = m3 / pow(var, 1.5);
  const double kurt = m4 / (var * var);

  /* Sampling errors for n draws of a standard normal, checked at 5 sigma. */
  const double se_mean = 1.0 / sqrt(dn);
  const double se_var = sqrt(2.0 / dn);
  const double se_skew = sqrt(6.0 / dn);
  const double se_kurt = sqrt(24.0 / dn);

  CHECK(fabs(mean) < 5.0 * se_mean, "mean is %.3e, expected 0 +/- %.3e", mean,
    5.0 * se_mean);
  CHECK(fabs(var - 1.0) < 5.0 * se_var, "variance is %.6f, expected 1 +/- %.3e",
    var, 5.0 * se_var);
  CHECK(fabs(skew) < 5.0 * se_skew, "skewness is %.3e, expected 0 +/- %.3e",
    skew, 5.0 * se_skew);
  CHECK(fabs(kurt - 3.0) < 5.0 * se_kurt,
    "kurtosis is %.6f, expected 3 +/- %.3e", kurt, 5.0 * se_kurt);

  /* The two members of a pair come from the same radius and orthogonal
   * angles, so they must be independent. Reusing one uniform for both, or
   * returning cos twice, shows up here and nowhere in the moments above. */
  const double corr = cross / (double)n_pairs;
  const double se_corr = 1.0 / sqrt((double)n_pairs);
  CHECK(fabs(corr) < 5.0 * se_corr,
    "the two deviates of a pair correlate at %.3e, expected 0 +/- %.3e", corr,
    5.0 * se_corr);

  /* Chi-square over the body of the distribution, plus one category for both
   * tails together. Catches a shape error the first four moments can miss. */
  double chi2 = 0.0;
  for (int b = 0; b < __N_BUCKETS; b++) {
    const double lo = __BUCKET_LO + b * width;
    const double expect = dn * (__phi(lo + width) - __phi(lo));
    const double d = (double)hist[b] - expect;
    chi2 += d * d / expect;
  }
  {
    const double expect = dn * (__phi(__BUCKET_LO) + (1.0 - __phi(__BUCKET_HI)));
    const double d = (double)outside - expect;
    chi2 += d * d / expect;
  }

  /* 40 degrees of freedom: mean 40, sd sqrt(80) ~ 8.9. 100 is roughly six
   * sigma out, far enough that only a real defect reaches it. */
  CHECK(chi2 < 100.0, "chi-square over %d buckets is %.1f, expected ~%d",
    __N_BUCKETS + 1, chi2, __N_BUCKETS);

  printf("  n = %llu, mean %.2e, var %.6f, skew %.2e, kurt %.5f, chi2 %.1f\n",
    (unsigned long long)n, mean, var, skew, kurt, chi2);
}

/*
 * A stand-in for the real thing, with the same structure and none of the
 * quadrature: sigma falling as a power of R, and an exponential correlation in
 * ln R. The exponential kernel is positive definite, so this is a legitimate
 * covariance for any correlation length, and raising `corr_length` tightens
 * the neighbour correlation exactly the way sampling the radii more finely
 * would. Caller frees with sif_free_aligned.
 */
static double* __model_covariance(
  const real_t* radii, uint32_t n, double slope, double corr_length) {

  double* cov = sif_malloc_aligned(SIF_COV_SIZE(n) * sizeof(double));
  if (!cov)
    return NULL;

  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j <= i; j++) {
      const double si = pow((double)radii[i], slope);
      const double sj = pow((double)radii[j], slope);
      const double dx = log((double)radii[i]) - log((double)radii[j]);
      cov[SIF_COV_INDEX(i, j)] = si * sj * exp(-fabs(dx) / corr_length);
    }
  }
  return cov;
}

/*
 * The factorization is the piece with the most invariants worth asserting
 * directly, and the one whose failures would otherwise reach us as a quietly
 * mis-shaped histogram.
 */
static void test_cholesky(void) {
  printf("Cholesky factor\n");

  const uint32_t n = 64;
  real_t* radii = malloc(n * sizeof(real_t));
  for (uint32_t i = 0; i < n; i++)
    radii[i] = (real_t)(2.0 * pow(30.0 / 2.0, (double)i / (n - 1.0)));

  double* cov = __model_covariance(radii, n, -0.5, 1.5);

  sif_ep_factor_t f;
  CHECK(sif_ep_factor_init(&f, n) == SIF_OK, "factor allocation failed");
  CHECK(sif_ep_cholesky(&f, cov, radii, n) == SIF_OK,
    "factorization of a positive definite covariance failed");

  /* Offsets describe the buffer they index. */
  int monotone = 1;
  for (uint32_t j = 0; j < n; j++)
    if (f.row_offset[j + 1] <= f.row_offset[j])
      monotone = 0;
  CHECK(monotone, "row offsets are not strictly increasing");
  CHECK(f.row_offset[n] == f.n_packed,
    "row_offset[n] is %llu but n_packed is %llu",
    (unsigned long long)f.row_offset[n], (unsigned long long)f.n_packed);

  /* A positive diagonal is what makes the back-substitution above it valid. */
  int diag_ok = 1;
  for (uint32_t j = 0; j < n; j++)
    if (!(f.chol[f.row_offset[j] + j] > 0.0))
      diag_ok = 0;
  CHECK(diag_ok, "the factor has a non-positive diagonal entry");

  /*
   * Every padded lane must be exactly zero. The walk runs its dot product past
   * the diagonal to a vector boundary, so a nonzero pad would silently
   * multiply whatever white noise the previous path left in the buffer.
   */
  int pad_ok = 1;
  for (uint32_t j = 0; j < n; j++)
    for (uint64_t m = j + 1; m < f.row_offset[j + 1] - f.row_offset[j]; m++)
      if (f.chol[f.row_offset[j] + m] != 0.0)
        pad_ok = 0;
  CHECK(pad_ok, "the factor has a nonzero entry in its row padding");

  /*
   * L L^T must return the covariance. Row j of the factor is ascending index
   * n - 1 - j, so the reconstruction has to be read back through that
   * reversal -- which is what makes this a test of the ordering as much as of
   * the arithmetic.
   */
  double worst = 0.0;
  for (uint32_t j = 0; j < n; j++) {
    for (uint32_t m = 0; m <= j; m++) {
      const double* Lj = f.chol + f.row_offset[j];
      const double* Lm = f.chol + f.row_offset[m];
      double s = 0.0;
      for (uint32_t q = 0; q <= m; q++)
        s += Lj[q] * Lm[q];

      const uint32_t a = n - 1 - j, b = n - 1 - m;
      const double expect = sif_ep_cov_get(cov, a, b);
      const double scale = sqrt(sif_ep_cov_get(cov, a, a) *
                                sif_ep_cov_get(cov, b, b));
      const double rel = fabs(s - expect) / scale;
      if (rel > worst)
        worst = rel;
    }
  }
  CHECK(worst < 1e-10, "L L^T departs from the covariance by %.3e", worst);
  printf("  n = %u, worst reconstruction error %.3e\n", n, worst);

  /*
   * A rank-one covariance must still factorize: that is what the jitter is
   * for, and the degenerate walk it produces is a physically meaningful limit
   * rather than an error.
   */
  {
    const uint32_t m = 8;
    double* ones = sif_malloc_aligned(SIF_COV_SIZE(m) * sizeof(double));
    for (size_t i = 0; i < SIF_COV_SIZE(m); i++)
      ones[i] = 1.0;

    sif_ep_factor_t g;
    sif_ep_factor_init(&g, m);
    CHECK(sif_ep_cholesky(&g, ones, radii, m) == SIF_OK,
      "the jitter failed to rescue a rank-one covariance");
    sif_ep_factor_free(&g);
    sif_free_aligned(ones);
  }

  /*
   * An indefinite covariance must be reported rather than factorized. A
   * correlation above one is beyond anything the jitter can or should paper
   * over.
   */
  {
    double bad[3] = {1.0, 2.0, 1.0}; /* [[1, 2], [2, 1]], eigenvalues 3 and -1 */
    const real_t two_radii[2] = {1.0f, 2.0f};

    sif_ep_factor_t g;
    sif_ep_factor_init(&g, 2);
    CHECK(sif_ep_cholesky(&g, bad, two_radii, 2) == SIF_ERR_RANGE,
      "an indefinite covariance was accepted by the factorization");
    sif_ep_factor_free(&g);
  }

  sif_ep_factor_free(&f);
  sif_free_aligned(cov);
  free(radii);
}

/* Log-spaced radii; the values are immaterial to a hand-built covariance, but
 * the ordering they declare is not. */
static real_t* __log_radii(uint32_t n, double lo, double hi) {
  real_t* r = malloc(n * sizeof(real_t));
  for (uint32_t i = 0; i < n; i++)
    r[i] = (real_t)(lo * pow(hi / lo, (double)i / (n - 1.0)));
  return r;
}

/* Probability that a standard normal is at or above b. */
static double __tail(double b) { return 0.5 * erfc(b / sqrt(2.0)); }

/*
 * Two correlated steps, where the answer is an exact rational.
 *
 * With S = [[1, 1/2], [1/2, 1]] and a barrier at zero, the orthant probability
 * P(X >= 0, Y >= 0) = 1/4 + arcsin(rho) / 2pi is 1/3 at rho = 1/2. So the walk
 * crosses on its first step with probability 1/2, on its second with
 * 1/2 - 1/3 = 1/6, and never with 1/3.
 *
 * This is the only check that exercises the correlated step, the descending
 * walk order and the barrier comparison against a closed form at once, and the
 * exact rationals make a failure unambiguous. The walk starts at the LARGEST
 * radius, so its first step is the LAST entry of an ascending output -- which
 * is the half of this test that catches a missing reversal.
 */
static void test_two_step_exact(void) {
  printf("two correlated steps against the orthant probability\n");

  const uint64_t n_paths = (uint64_t)SIF_TEST_SCALE(2000000);
  const real_t radii[2] = {5.0f, 10.0f};
  const double cov[3] = {1.0, 0.5, 1.0};
  const real_t barrier[2] = {0.0f, 0.0f};

  uint64_t* c =
    sif_first_crossing_counts_ep(radii, 2, cov, barrier, n_paths, 12345u, SIF_DEFAULT);
  CHECK(c != NULL, "the walk returned NULL on a valid two-step problem");
  if (!c)
    return;

  const double expect_first = 1.0 / 2.0;  /* at radii[1], the walk's step 0 */
  const double expect_second = 1.0 / 6.0; /* at radii[0], the walk's step 1 */

  const double dn = (double)n_paths;
  const double f1 = (double)c[1] / dn;
  const double f0 = (double)c[0] / dn;

  const double se1 = sqrt(expect_first * (1.0 - expect_first) / dn);
  const double se0 = sqrt(expect_second * (1.0 - expect_second) / dn);

  CHECK(fabs(f1 - expect_first) < 5.0 * se1,
    "first crossing at the largest radius is %.6f, expected %.6f +/- %.3e "
    "(a swapped walk order would put %.6f here)",
    f1, expect_first, 5.0 * se1, expect_second);
  CHECK(fabs(f0 - expect_second) < 5.0 * se0,
    "first crossing at the smallest radius is %.6f, expected %.6f +/- %.3e", f0,
    expect_second, 5.0 * se0);

  const double never = 1.0 - f0 - f1;
  CHECK(fabs(never - 1.0 / 3.0) < 5.0 * (se0 + se1),
    "the never-crossing fraction is %.6f, expected 1/3", never);

  printf("  1/2 -> %.6f, 1/6 -> %.6f, 1/3 -> %.6f\n", f1, f0, never);
  sif_free_aligned(c);
}

/*
 * Independent steps make the walk Markovian again, and the first crossing
 * exactly geometric: P(j) = p (1-p)^j with p the barrier tail. Nothing about
 * the covariance, the ordering or the physics is involved, which makes this
 * the sharpest available test of the kernel, the seeding and the lazy
 * generation.
 */
static void test_geometric(void) {
  printf("independent steps against the geometric distribution\n");

  const uint32_t n = 20;
  const uint64_t n_paths = (uint64_t)SIF_TEST_SCALE(2000000);
  const double b = 0.5;
  const double p = __tail(b);

  real_t* radii = __log_radii(n, 1.0, 40.0);
  real_t* barrier = malloc(n * sizeof(real_t));
  double* cov = sif_calloc_aligned(SIF_COV_SIZE(n), sizeof(double));

  for (uint32_t i = 0; i < n; i++) {
    barrier[i] = (real_t)b;
    cov[SIF_COV_INDEX(i, i)] = 1.0;
  }

  uint64_t* c = sif_first_crossing_counts_ep(
    radii, n, cov, barrier, n_paths, 777u, SIF_DEFAULT);
  CHECK(c != NULL, "the walk returned NULL on an identity covariance");

  if (c) {
    double worst_sigma = 0.0;
    for (uint32_t j = 0; j < n; j++) {
      const double expect = (double)n_paths * p * pow(1.0 - p, (double)j);
      const double sd = sqrt(expect * (1.0 - expect / (double)n_paths));
      /* walk step j is ascending index n - 1 - j */
      const double dev = fabs((double)c[n - 1 - j] - expect) / sd;
      if (dev > worst_sigma)
        worst_sigma = dev;
    }
    CHECK(worst_sigma < 4.5,
      "the crossing histogram departs from geometric by %.2f sigma",
      worst_sigma);
    printf("  p = %.4f, worst departure %.2f sigma\n", p, worst_sigma);
    sif_free_aligned(c);
  }

  sif_free_aligned(cov);
  free(barrier);
  free(radii);
}

/*
 * The perfectly correlated limit. With every entry of the covariance equal the
 * matrix is rank one, so the walk has a single degree of freedom: it either
 * clears the barrier on its first step or never moves again. Exercises the
 * jitter, and asserts that the jitter is small enough not to manufacture
 * crossings that the physics does not have.
 */
static void test_degenerate(void) {
  printf("rank-one covariance\n");

  const uint32_t n = 16;
  const uint64_t n_paths = (uint64_t)SIF_TEST_SCALE(1000000);
  const double b = 0.3;

  real_t* radii = __log_radii(n, 1.0, 40.0);
  real_t* barrier = malloc(n * sizeof(real_t));
  double* cov = sif_malloc_aligned(SIF_COV_SIZE(n) * sizeof(double));

  for (uint32_t i = 0; i < n; i++)
    barrier[i] = (real_t)b;
  for (size_t i = 0; i < SIF_COV_SIZE(n); i++)
    cov[i] = 1.0;

  uint64_t* c = sif_first_crossing_counts_ep(
    radii, n, cov, barrier, n_paths, 99u, SIF_DEFAULT);
  CHECK(c != NULL, "the walk returned NULL on a rank-one covariance");

  if (c) {
    const double expect = __tail(b);
    const double got = (double)c[n - 1] / (double)n_paths;
    const double se = sqrt(expect * (1.0 - expect) / (double)n_paths);
    CHECK(fabs(got - expect) < 5.0 * se,
      "first-step crossing fraction is %.6f, expected %.6f +/- %.3e", got,
      expect, 5.0 * se);

    uint64_t later = 0;
    for (uint32_t i = 0; i + 1 < n; i++)
      later += c[i];
    CHECK(later < n_paths / 10000,
      "%llu paths crossed after the first step of a rank-one walk; the jitter "
      "is manufacturing degrees of freedom",
      (unsigned long long)later);
    printf("  first step %.6f (expected %.6f), later steps %llu\n", got, expect,
      (unsigned long long)later);
    sif_free_aligned(c);
  }

  sif_free_aligned(cov);
  free(barrier);
  free(radii);
}

/*
 * A barrier of the shape the model actually uses, B = alpha [1 + (beta/sigma)^
 * gamma], rising towards small sigma. Built here rather than in the library
 * because the barrier is the caller's to choose.
 */
static real_t* __model_barrier(const double* cov, uint32_t n, double alpha,
  double beta, double gamma) {

  real_t* b = malloc(n * sizeof(real_t));
  for (uint32_t i = 0; i < n; i++) {
    const double sigma = sqrt(sif_ep_cov_get(cov, i, i));
    b[i] = (real_t)(alpha * (1.0 + pow(beta / sigma, gamma)));
  }
  return b;
}

/*
 * The counts must not depend on how the work was divided. This is the check
 * that the per-path seeding delivers what it promises: bit-identical, not
 * merely statistically compatible, because integer addition of per-thread
 * histograms is exact and every path's stream is a pure function of its index.
 */
static void test_thread_independence(void) {
  printf("reproducibility across thread counts\n");

#ifdef _OPENMP
  const uint32_t n = 32;
  const uint64_t n_paths = (uint64_t)SIF_TEST_SCALE(400000);

  real_t* radii = __log_radii(n, 1.0, 40.0);
  double* cov = __model_covariance(radii, n, -0.5, 1.5);
  real_t* barrier = __model_barrier(cov, n, 0.2, 0.1, 0.87);

  const int threads[3] = {1, 4, 3};
  uint64_t* ref = NULL;
  int identical = 1;

  for (int t = 0; t < 3; t++) {
    omp_set_num_threads(threads[t]);
    uint64_t* c = sif_first_crossing_counts_ep(
      radii, n, cov, barrier, n_paths, 2024u, SIF_DEFAULT);
    CHECK(c != NULL, "the walk returned NULL at %d threads", threads[t]);
    if (!c)
      continue;

    if (!ref) {
      ref = c;
    } else {
      for (uint32_t i = 0; i < n; i++)
        if (c[i] != ref[i])
          identical = 0;
      sif_free_aligned(c);
    }
  }

  CHECK(identical,
    "the counts changed with the thread count; the per-path seeding is not "
    "isolating paths from the schedule");

  uint64_t crossed = 0;
  if (ref)
    for (uint32_t i = 0; i < n; i++)
      crossed += ref[i];
  printf("  identical at 1, 4 and 3 threads; %llu of %llu paths crossed\n",
    (unsigned long long)crossed, (unsigned long long)n_paths);

  omp_set_num_threads(4);
  sif_free_aligned(ref);
  free(barrier);
  sif_free_aligned(cov);
  free(radii);
#else
  printf("  skipped: built without OpenMP\n");
#endif
}

/*
 * Quadrupling the paths must move the answer by no more than the Monte Carlo
 * error allows. Weak next to the closed forms above, but it is the one check
 * that would notice per-path seeds correlating with each other, which no
 * single-N test can see.
 */
static void test_convergence(void) {
  printf("convergence in the number of paths\n");

  const uint32_t n = 24;
  const uint64_t n1 = (uint64_t)SIF_TEST_SCALE(250000);
  const uint64_t n2 = 4 * n1;

  real_t* radii = __log_radii(n, 1.0, 40.0);
  double* cov = __model_covariance(radii, n, -0.5, 1.5);
  real_t* barrier = __model_barrier(cov, n, 0.2, 0.1, 0.87);

  uint64_t* a =
    sif_first_crossing_counts_ep(radii, n, cov, barrier, n1, 5u, SIF_DEFAULT);
  uint64_t* b =
    sif_first_crossing_counts_ep(radii, n, cov, barrier, n2, 5u, SIF_DEFAULT);

  CHECK(a && b, "the walk returned NULL during the convergence check");

  if (a && b) {
    const double tol = 3.0 * (1.0 / sqrt((double)n1) + 1.0 / sqrt((double)n2));
    double worst = 0.0;
    for (uint32_t i = 0; i < n; i++) {
      const double d =
        fabs((double)a[i] / (double)n1 - (double)b[i] / (double)n2);
      if (d > worst)
        worst = d;
    }
    CHECK(worst < tol, "fractions at N and 4N differ by %.3e, tolerance %.3e",
      worst, tol);
    printf("  worst fractional difference %.3e (tolerance %.3e)\n", worst, tol);
  }

  sif_free_aligned(a);
  sif_free_aligned(b);
  free(barrier);
  sif_free_aligned(cov);
  free(radii);
}

/*
 * An exact inequality that holds for any covariance and any barrier, and so
 * runs on the realistic setup rather than a contrived one.
 *
 * Marginally delta_j is N(0, S_jj), and a walk found above the barrier at step
 * j must have crossed at or before j. So the cumulative crossings down to a
 * radius are at least the one-point tail there. It is the check that would
 * catch a covariance or a barrier off by a constant factor, which every
 * closed-form test above is blind to because they supply both themselves.
 */
static void test_marginal_bound(void) {
  printf("cumulative crossings against the one-point tail\n");

  const uint32_t n = 40;
  const uint64_t n_paths = (uint64_t)SIF_TEST_SCALE(500000);

  real_t* radii = __log_radii(n, 1.0, 40.0);
  double* cov = __model_covariance(radii, n, -0.5, 1.5);
  real_t* barrier = __model_barrier(cov, n, 0.2, 0.1, 0.87);

  uint64_t* c = sif_first_crossing_counts_ep(
    radii, n, cov, barrier, n_paths, 31337u, SIF_DEFAULT);
  CHECK(c != NULL, "the walk returned NULL on the realistic setup");

  if (c) {
    int violated = 0;
    double worst = 0.0;

    /* Walk step j is ascending index n - 1 - j, so cumulating over the walk
     * means cumulating downward from the largest radius. */
    uint64_t cum = 0;
    for (uint32_t a = n; a-- > 0;) {
      cum += c[a];

      const double sigma = sqrt(sif_ep_cov_get(cov, a, a));
      const double p = __tail((double)barrier[a] / sigma);
      const double expect = (double)n_paths * p;
      const double sd = sqrt(expect * (1.0 - p));

      /* The bound is on the expectation; allow the sample five sigma below. */
      if ((double)cum < expect - 5.0 * sd) {
        violated = 1;
        const double miss = (expect - (double)cum) / (sd > 0.0 ? sd : 1.0);
        if (miss > worst)
          worst = miss;
      }
    }

    CHECK(!violated,
      "cumulative crossings fell %.1f sigma below the one-point tail; the "
      "walk is reaching the barrier less often than its own marginals require",
      worst);
    printf("  %llu of %llu paths crossed; bound holds at every radius\n",
      (unsigned long long)cum, (unsigned long long)n_paths);
    sif_free_aligned(c);
  }

  free(barrier);
  sif_free_aligned(cov);
  free(radii);
}

/*
 * The binning, checked against the counts it was built from. No statistics
 * here: whatever the walk produced, the multiplicity has to be that divided by
 * the bin width, bin for bin, with the largest radius dropped.
 */
static void test_binning(void) {
  printf("multiplicity binning against the raw counts\n");

  const uint32_t n = 32;
  const uint64_t n_paths = (uint64_t)SIF_TEST_SCALE(400000);

  real_t* radii = __log_radii(n, 1.0, 40.0);
  double* cov = __model_covariance(radii, n, -0.5, 1.5);
  real_t* barrier = __model_barrier(cov, n, 0.2, 0.1, 0.87);

  uint64_t* c = sif_first_crossing_counts_ep(
    radii, n, cov, barrier, n_paths, 4242u, SIF_DEFAULT);

  uint64_t* c_out = calloc(n, sizeof(uint64_t));
  real_t* f = sif_multiplicity_function_ep(
    radii, n, cov, barrier, n_paths, 4242u, c_out, SIF_DEFAULT);

  CHECK(c && c_out && f, "the counts or the multiplicity returned NULL");

  /* The out-parameter exists so nobody has to run the walk twice to see the
   * counts, which makes "it returns what the second run would have" the whole
   * of its contract. Same seed, so this is exact, not statistical. */
  if (c && c_out) {
    uint32_t mismatched = 0;
    for (uint32_t i = 0; i < n; i++)
      if (c[i] != c_out[i])
        mismatched++;
    CHECK(mismatched == 0,
      "the counts out-parameter disagrees with first_crossing_counts_ep at "
      "%u of %u radii",
      mismatched, n);
  }

  if (c && f) {
    /* The two calls share a seed, so they are the same ensemble. */
    uint64_t crossed = 0;
    for (uint32_t i = 0; i < n; i++)
      crossed += c[i];
    CHECK(crossed <= n_paths,
      "%llu crossings recorded from %llu paths; a path crossed twice",
      (unsigned long long)crossed, (unsigned long long)n_paths);

    /*
     * Bin for bin, not just in aggregate: a reversal or an off-by-one would
     * survive a check on the sum and die here. The tolerance is float
     * roundoff on the stored result, not a statistical allowance.
     */
    double worst = 0.0;
    for (uint32_t i = 0; i + 1 < n; i++) {
      const double dr = (double)radii[i + 1] - (double)radii[i];
      const double recovered = (double)f[i] * dr * (double)n_paths;
      const double scale = c[i] > 0 ? (double)c[i] : 1.0;
      const double rel = fabs(recovered - (double)c[i]) / scale;
      if (rel > worst)
        worst = rel;
    }
    CHECK(worst < 1e-5,
      "the multiplicity does not recover its own counts, worst relative "
      "error %.3e",
      worst);

    printf("  %llu crossings, %llu dropped at the largest radius, worst "
           "recovery error %.2e\n",
      (unsigned long long)crossed, (unsigned long long)c[n - 1], worst);
  }

  sif_free_aligned(c);
  free(c_out);
  sif_free_aligned(f);
  free(barrier);
  sif_free_aligned(cov);
  free(radii);
}

/*
 * Invalid input is reported, not asserted on. Every one of these logs an
 * error on the way past, which is the intended behaviour and why the suite is
 * noisier than its pass count suggests.
 */
static void test_guards(void) {
  printf("input validation\n");

  const uint32_t n = 8;
  real_t* radii = __log_radii(n, 1.0, 40.0);
  double* cov = __model_covariance(radii, n, -0.5, 1.5);
  real_t* barrier = __model_barrier(cov, n, 0.2, 0.1, 0.87);

  uint64_t* r;

  r = sif_first_crossing_counts_ep(NULL, n, cov, barrier, 100, 1u, SIF_DEFAULT);
  CHECK(r == NULL, "NULL radii were accepted");
  sif_free_aligned(r);

  r = sif_first_crossing_counts_ep(radii, n, NULL, barrier, 100, 1u, SIF_DEFAULT);
  CHECK(r == NULL, "a NULL covariance was accepted");
  sif_free_aligned(r);

  r = sif_first_crossing_counts_ep(radii, n, cov, NULL, 100, 1u, SIF_DEFAULT);
  CHECK(r == NULL, "a NULL barrier was accepted");
  sif_free_aligned(r);

  r = sif_first_crossing_counts_ep(radii, 1, cov, barrier, 100, 1u, SIF_DEFAULT);
  CHECK(r == NULL, "a single radius was accepted");
  sif_free_aligned(r);

  r = sif_first_crossing_counts_ep(radii, n, cov, barrier, 0, 1u, SIF_DEFAULT);
  CHECK(r == NULL, "zero paths were accepted");
  sif_free_aligned(r);

  {
    real_t* descending = malloc(n * sizeof(real_t));
    for (uint32_t i = 0; i < n; i++)
      descending[i] = radii[n - 1 - i];
    r = sif_first_crossing_counts_ep(
      descending, n, cov, barrier, 100, 1u, SIF_DEFAULT);
    CHECK(r == NULL,
      "descending radii were accepted; the reference silently produced a "
      "negative multiplicity in exactly this case");
    sif_free_aligned(r);
    free(descending);
  }

  {
    real_t* zeroed = malloc(n * sizeof(real_t));
    for (uint32_t i = 0; i < n; i++)
      zeroed[i] = radii[i];
    zeroed[0] = 0.0f;
    r = sif_first_crossing_counts_ep(
      zeroed, n, cov, barrier, 100, 1u, SIF_DEFAULT);
    CHECK(r == NULL, "a zero radius was accepted");
    sif_free_aligned(r);
    free(zeroed);
  }

  {
    double* flat = __model_covariance(radii, n, -0.5, 1.5);
    flat[SIF_COV_INDEX(0, 0)] = 0.0;
    r = sif_first_crossing_counts_ep(
      radii, n, flat, barrier, 100, 1u, SIF_DEFAULT);
    CHECK(r == NULL, "a zero covariance diagonal was accepted");
    sif_free_aligned(r);
    sif_free_aligned(flat);
  }

  {
    /* Indefinite: a correlation above one. Rejected by the factorization
     * rather than by validation, so this exercises that path end to end. */
    double* bad = __model_covariance(radii, n, -0.5, 1.5);
    bad[SIF_COV_INDEX(1, 0)] *= 50.0;
    r = sif_first_crossing_counts_ep(
      radii, n, bad, barrier, 100, 1u, SIF_DEFAULT);
    CHECK(r == NULL, "an indefinite covariance was accepted");
    sif_free_aligned(r);
    sif_free_aligned(bad);
  }

  {
    real_t* f = sif_multiplicity_function_ep(
      radii, 1, cov, barrier, 100, 1u, NULL, SIF_DEFAULT);
    CHECK(f == NULL, "the multiplicity accepted a single radius");
    sif_free_aligned(f);
  }

  printf("  all guards returned NULL\n");

  free(barrier);
  sif_free_aligned(cov);
  free(radii);
}

/* --- The covariance --- */

#define N_K 4000

static void __power_law(real_t* k, real_t* pk, double slope) {
  const double lo = log(1e-8), hi = log(1e3);
  for (uint32_t i = 0; i < N_K; i++) {
    const double lk = lo + (hi - lo) * i / (N_K - 1.0);
    k[i] = (real_t)exp(lk);
    pk[i] = (real_t)pow(exp(lk), slope);
  }
}

/*
 * The check the whole covariance design rests on.
 *
 * S(R, R) is the definition of sigma_0^2(R), so the diagonal has to reproduce
 * what sif_delta_moments_pk already computes. Because both use the same window
 * and the same trapezoid in log k, that agreement is to rounding rather than to
 * quadrature accuracy -- which turns a vague "roughly consistent" into an
 * assertion sharp enough to catch a real defect, and validates the new code
 * against a module that is already independently tested.
 */
static void test_covariance_diagonal(void) {
  printf("covariance diagonal vs sigma_0 from the moments\n");

  real_t* k = malloc(N_K * sizeof(real_t));
  real_t* pk = malloc(N_K * sizeof(real_t));
  __power_law(k, pk, -2.0);

  const uint32_t n = 24;
  real_t* radii = __log_radii(n, 1.0, 40.0);
  real_t* sigma = malloc(n * sizeof(real_t));

  const sif_option_t windows[2] = {
    SIF_DELTA_FILTER_TOP_HAT, SIF_DELTA_FILTER_GAUSSIAN};
  const char* names[2] = {"top-hat", "Gaussian"};

  for (int w = 0; w < 2; w++) {
    double* cov =
      sif_delta_covariance_pk(k, pk, N_K, radii, n, sigma, NULL, NULL, windows[w]);
    sif_delta_moments_t* m =
      sif_delta_moments_pk(k, pk, N_K, radii, n, 0, windows[w]);

    CHECK(cov && m, "%s: covariance or moments returned NULL", names[w]);

    if (cov && m) {
      const real_t* s0 = sif_delta_moments_sigma(m, 0);
      double worst = 0.0, worst_sigma = 0.0;

      for (uint32_t i = 0; i < n; i++) {
        const double expect = (double)s0[i] * (double)s0[i];
        const double got = cov[SIF_COV_INDEX(i, i)];
        const double rel = fabs(got - expect) / expect;
        if (rel > worst)
          worst = rel;

        /* And the out-param is the sqrt of that same diagonal. */
        const double srel = fabs((double)sigma[i] - (double)s0[i]) / (double)s0[i];
        if (srel > worst_sigma)
          worst_sigma = srel;
      }

      /* Loose enough for the float round-trip through sigma_0, tight enough
       * that a different quadrature could not pass. */
      CHECK(worst < 1e-6,
        "%s: the covariance diagonal departs from sigma_0^2 by %.3e", names[w],
        worst);
      CHECK(worst_sigma < 1e-6,
        "%s: the sigma out-param departs from sigma_0 by %.3e", names[w],
        worst_sigma);
      printf("  %-8s diagonal %.2e, sigma out-param %.2e\n", names[w], worst,
        worst_sigma);
    }

    sif_free_aligned(cov);
    sif_delta_moments_free(m);
  }

  free(sigma);
  free(radii);
  free(k);
  free(pk);
}

/*
 * Two properties that hold whatever the quadrature does.
 *
 * Cauchy-Schwarz is necessary for any covariance and catches a packed-index or
 * transposition bug, which is the likeliest defect in a triangular layout. And
 * for P(k) ~ k^n the covariance is homogeneous of degree -(3+n): R enters only
 * through kR, so scaling every radius by lambda scales the whole matrix by a
 * constant, exactly, for any window. That is the only check here that
 * exercises the off-diagonals against an analytic property.
 */
static void test_covariance_properties(void) {
  printf("covariance invariants\n");

  real_t* k = malloc(N_K * sizeof(real_t));
  real_t* pk = malloc(N_K * sizeof(real_t));
  const double slope = -2.0;
  __power_law(k, pk, slope);

  const uint32_t n = 20;
  real_t* radii = __log_radii(n, 4.0, 20.0);
  real_t* scaled = malloc(n * sizeof(real_t));
  const double lambda = 2.0;
  for (uint32_t i = 0; i < n; i++)
    scaled[i] = (real_t)(lambda * (double)radii[i]);

  double* a = sif_delta_covariance_pk(k, pk, N_K, radii, n, NULL, NULL, NULL, SIF_DEFAULT);
  double* b = sif_delta_covariance_pk(k, pk, N_K, scaled, n, NULL, NULL, NULL, SIF_DEFAULT);

  CHECK(a && b, "the covariance returned NULL");

  if (a && b) {
    /* Cauchy-Schwarz. */
    double worst_cs = 0.0;
    for (uint32_t i = 0; i < n; i++)
      for (uint32_t j = 0; j <= i; j++) {
        const double bound =
          sqrt(a[SIF_COV_INDEX(i, i)] * a[SIF_COV_INDEX(j, j)]);
        const double excess = fabs(a[SIF_COV_INDEX(i, j)]) / bound - 1.0;
        if (excess > worst_cs)
          worst_cs = excess;
      }
    CHECK(worst_cs < 1e-12,
      "an off-diagonal exceeds sqrt(S_ii S_jj) by a relative %.3e", worst_cs);

    /* Self-similarity: every entry scales by the same lambda^-(3+n). */
    const double expect = pow(lambda, -(3.0 + slope));
    double worst_ss = 0.0;
    for (uint32_t i = 0; i < n; i++)
      for (uint32_t j = 0; j <= i; j++) {
        const double ratio = b[SIF_COV_INDEX(i, j)] / a[SIF_COV_INDEX(i, j)];
        const double rel = fabs(ratio - expect) / expect;
        if (rel > worst_ss)
          worst_ss = rel;
      }
    CHECK(worst_ss < 2e-3,
      "scaling the radii by %g did not scale the covariance by %g; worst "
      "relative departure %.3e",
      lambda, expect, worst_ss);

    printf("  Cauchy-Schwarz slack %.2e, self-similarity %.2e (expect ratio "
           "%.4f)\n",
      worst_cs, worst_ss, expect);

    /*
     * And it must factorize. The Gram accumulation makes it positive
     * semi-definite by construction, so this is not a hope but an assertion
     * that the construction is what it claims to be.
     */
    sif_ep_factor_t f;
    sif_ep_factor_init(&f, n);
    CHECK(sif_ep_cholesky(&f, a, radii, n) == SIF_OK,
      "a covariance built from a positive P(k) failed to factorize");
    sif_ep_factor_free(&f);
  }

  sif_free_aligned(a);
  sif_free_aligned(b);
  free(scaled);
  free(radii);
  free(k);
  free(pk);
}

/* --- The barrier --- */

/*
 * The derivative variance, <(d delta / dS)^2>, and the dimensionless quantity
 * it exists to serve:
 *
 *     Gamma^2 = 1 / (4 S <(d delta / dS)^2>)
 *
 * which is the squared correlation between the walk and its own derivative.
 *
 * The point of computing this under the integral rather than differencing the
 * covariance is that the differenced form converges only at second order in
 * the radius spacing, so it reports a property of the caller's grid rather
 * than of the field. All three checks below are about exactly that:
 *
 *   - Gamma^2 is bounded in (0, 1] by Cauchy-Schwarz, since <delta delta'> is
 *     identically 1/2 whatever the spectrum
 *   - for a power-law spectrum the walk is self-similar, so Gamma^2 must not
 *     depend on R at all. No quadrature enters that statement, which makes it
 *     the sharpest available assertion
 *   - the differenced form has to converge TO this one, at second order, or
 *     the two are not computing the same thing
 */
static void test_deriv_variance(void) {
  printf("derivative variance and Gamma^2\n");

  real_t* k = malloc(N_K * sizeof(real_t));
  real_t* pk = malloc(N_K * sizeof(real_t));
  __power_law(k, pk, -2.0);

  /* --- self-similarity, and the bound --- */
  {
    const uint32_t n = 24;
    real_t* radii = __log_radii(n, 2.0, 30.0);
    real_t* sigma = malloc(n * sizeof(real_t));
    double* dvar = malloc(n * sizeof(double));

    double* cov = sif_delta_covariance_pk(
      k, pk, N_K, radii, n, sigma, NULL, dvar, SIF_DEFAULT);
    CHECK(cov != NULL, "the covariance returned NULL");

    if (cov) {
      double g_min = 1e300, g_max = -1e300;
      int out_of_range = 0;

      for (uint32_t i = 0; i < n; i++) {
        const double S = (double)sigma[i] * (double)sigma[i];
        const double g2 = 1.0 / (4.0 * S * dvar[i]);
        if (!(g2 > 0.0 && g2 <= 1.0))
          out_of_range++;
        if (g2 < g_min)
          g_min = g2;
        if (g2 > g_max)
          g_max = g2;
      }

      CHECK(out_of_range == 0,
        "Gamma^2 left (0, 1] at %u of %u radii; <delta delta'> is 1/2 exactly, "
        "so Cauchy-Schwarz forbids it",
        out_of_range, n);

      /*
       * Self-similar, so the spread is quadrature error and nothing else. It
       * bottoms out around 7e-4 and does NOT improve with more k samples:
       * <(d delta / dR)^2> weights the integrand by k^4, which pushes its
       * support to where the top-hat oscillates fastest, and past a point the
       * trapezoid is aliasing those oscillations rather than resolving them.
       * The header already warns about this for the covariance itself. The
       * bound here is set to catch a real defect while sitting above that
       * floor.
       */
      const double spread = (g_max - g_min) / g_max;
      CHECK(spread < 3e-3,
        "Gamma^2 varies by a relative %.3e across the radii of a power-law "
        "spectrum, which is self-similar and must give a constant",
        spread);

      printf("  power law: Gamma^2 = %.6f, spread %.2e over %u radii\n", g_max,
        spread, n);
    }

    sif_free_aligned(cov);
    free(dvar);
    free(sigma);
    free(radii);
  }

  /*
   * The property the exact form exists for: it does not depend on how finely
   * the caller sampled `radii`, and the finite difference it replaces does.
   *
   * The fine grid is chosen so that every fourth of its radii coincides with a
   * coarse one, which makes the comparison exact rather than interpolated.
   */
  {
    const uint32_t n_c = 40;
    const uint32_t n_f = 4 * (n_c - 1) + 1; /* fine[4i] == coarse[i] */

    real_t* rc = __log_radii(n_c, 2.0, 30.0);
    real_t* rf = __log_radii(n_f, 2.0, 30.0);
    real_t* sc = malloc(n_c * sizeof(real_t));
    real_t* sf = malloc(n_f * sizeof(real_t));
    double* dc = malloc(n_c * sizeof(double));
    double* df = malloc(n_f * sizeof(double));

    double* cc =
      sif_delta_covariance_pk(k, pk, N_K, rc, n_c, sc, NULL, dc, SIF_DEFAULT);
    double* cf =
      sif_delta_covariance_pk(k, pk, N_K, rf, n_f, sf, NULL, df, SIF_DEFAULT);

    CHECK(cc && cf, "the covariance returned NULL");

    if (cc && cf) {
      double worst_exact = 0.0, worst_fd = 0.0;

      for (uint32_t i = 1; i + 1 < n_c; i++) {
        const uint32_t j = 4 * i;

        /* The exact form, coarse against fine at the same radius. */
        const double rel = fabs(dc[i] - df[j]) / df[j];
        if (rel > worst_exact)
          worst_exact = rel;

        /* The differenced form on the coarse grid, against the exact value. */
        const double Sp = (double)sc[i + 1] * (double)sc[i + 1];
        const double Sm = (double)sc[i - 1] * (double)sc[i - 1];
        const double h = Sp - Sm;
        const double fd =
          (Sp + Sm - 2.0 * cc[SIF_COV_INDEX(i + 1, i - 1)]) / (h * h);

        const double rel_fd = fabs(fd - dc[i]) / dc[i];
        if (rel_fd > worst_fd)
          worst_fd = rel_fd;
      }

      CHECK(worst_exact < 2e-3,
        "the derivative variance moved by a relative %.3e when the radius grid "
        "was refined 4x; it is supposed to be a property of the field, not of "
        "the grid",
        worst_exact);

      /*
       * And the converse, which is why this parameter exists at all. The
       * stencil is only first order here whatever the spacing -- the
       * covariance carries a |S1 - S2|^3 term across its diagonal, so the
       * leading error of a mixed second difference does not cancel -- and at
       * a realistic radius count it is wrong by percent. If this check ever
       * starts failing because the two agree, the exact path has probably
       * been quietly replaced by a difference.
       */
      CHECK(worst_fd > 20.0 * worst_exact,
        "the differenced derivative variance is within %.3e of the exact one, "
        "against the exact form's own %.3e grid drift; the two are not "
        "supposed to be this close at %u radii",
        worst_fd, worst_exact, n_c);

      printf("  grid refinement 4x: exact moves %.2e, finite difference is "
             "%.2e off\n",
        worst_exact, worst_fd);
    }

    sif_free_aligned(cc);
    sif_free_aligned(cf);
    free(df);
    free(dc);
    free(sf);
    free(sc);
    free(rf);
    free(rc);
  }

  free(pk);
  free(k);
}

static void test_barrier_smt(void) {
  printf("Sheth-Mo-Tormen barrier\n");

  const uint32_t n = 16;
  real_t* sigma = malloc(n * sizeof(real_t));
  for (uint32_t i = 0; i < n; i++)
    sigma[i] = (real_t)(0.1 * pow(30.0, (double)i / (n - 1.0)));

  /* gamma = 0 collapses the bracket to 1 + 1. */
  real_t* flat = sif_barrier_smt(sigma, n, 0.7f, 0.4f, 0.0f);
  CHECK(flat != NULL, "the barrier returned NULL");
  if (flat) {
    double worst = 0.0;
    for (uint32_t i = 0; i < n; i++) {
      const double rel = fabs((double)flat[i] - 1.4) / 1.4;
      if (rel > worst)
        worst = rel;
    }
    CHECK(worst < 1e-6, "gamma = 0 did not give a constant 2 alpha (%.3e)",
      worst);
  }

  /* With gamma > 0 the barrier falls as sigma rises, towards alpha. */
  const double alpha = 0.7;
  real_t* moving = sif_barrier_smt(sigma, n, (real_t)alpha, 0.4f, 0.87f);
  if (moving) {
    int monotone = 1;
    for (uint32_t i = 1; i < n; i++)
      if (!(moving[i] < moving[i - 1]))
        monotone = 0;
    CHECK(monotone,
      "the barrier is not decreasing in sigma; the sign of gamma is inverted");
    CHECK((double)moving[n - 1] > alpha,
      "the barrier fell below alpha, which it approaches from above");
    CHECK(((double)moving[n - 1] - alpha) / alpha < 0.2,
      "the barrier has not approached alpha at the largest sigma");

    /* Against the closed form, at one point, spelled out. */
    const uint32_t mid = n / 2;
    const double expect =
      alpha * (1.0 + pow(0.4 / (double)sigma[mid], 0.87));
    CHECK(fabs((double)moving[mid] - expect) / expect < 1e-6,
      "barrier at sigma=%g is %g, expected %g", (double)sigma[mid],
      (double)moving[mid], expect);

    printf("  sigma %.3f -> B %.4f, sigma %.3f -> B %.4f (alpha = %.2f)\n",
      (double)sigma[0], (double)moving[0], (double)sigma[n - 1],
      (double)moving[n - 1], alpha);
  }

  /* Guards. */
  real_t* bad;
  bad = sif_barrier_smt(NULL, n, 0.7f, 0.4f, 0.87f);
  CHECK(bad == NULL, "a NULL sigma was accepted");
  sif_free_aligned(bad);

  bad = sif_barrier_smt(sigma, n, -0.7f, 0.4f, 0.87f);
  CHECK(bad == NULL, "a negative alpha was accepted");
  sif_free_aligned(bad);

  bad = sif_barrier_smt(sigma, n, 0.7f, -0.4f, 0.87f);
  CHECK(bad == NULL,
    "a negative beta was accepted; (beta/sigma)^gamma is not real there");
  sif_free_aligned(bad);

  sif_free_aligned(flat);
  sif_free_aligned(moving);
  free(sigma);
}

/*
 * The whole pipeline, end to end: P(k) to covariance to sigma to barrier to
 * multiplicity. Not a precision test -- each stage has its own above -- but the
 * one check that the pieces actually compose, with the units and orderings
 * they each claim.
 */
static void test_pipeline(void) {
  printf("P(k) -> covariance -> sigma -> barrier -> multiplicity\n");

  real_t* k = malloc(N_K * sizeof(real_t));
  real_t* pk = malloc(N_K * sizeof(real_t));
  __power_law(k, pk, -2.0);

  /* Normalize so sigma lands where a barrier of order unity does something. */
  {
    const real_t eight[1] = {8.0f};
    real_t s8[1];
    double* c = sif_delta_covariance_pk(k, pk, N_K, eight, 1, s8, NULL, NULL, SIF_DEFAULT);
    if (c) {
      const double scale = (0.8 / (double)s8[0]) * (0.8 / (double)s8[0]);
      for (uint32_t i = 0; i < N_K; i++)
        pk[i] = (real_t)((double)pk[i] * scale);
      sif_free_aligned(c);
    }
  }

  const uint32_t n = 40;
  const uint64_t n_paths = (uint64_t)SIF_TEST_SCALE(400000);

  real_t* radii = __log_radii(n, 1.0, 30.0);
  real_t* sigma = malloc(n * sizeof(real_t));

  double* cov =
    sif_delta_covariance_pk(k, pk, N_K, radii, n, sigma, NULL, NULL, SIF_DEFAULT);
  CHECK(cov != NULL, "the covariance returned NULL");

  real_t* barrier = cov ? sif_barrier_smt(sigma, n, 0.3f, 0.2f, 0.87f) : NULL;
  CHECK(barrier != NULL, "the barrier returned NULL");

  if (cov && barrier) {
    /* sigma must fall with radius for a red spectrum. */
    int falling = 1;
    for (uint32_t i = 1; i < n; i++)
      if (!(sigma[i] < sigma[i - 1]))
        falling = 0;
    CHECK(falling, "sigma does not decrease with radius");

    real_t* f = sif_multiplicity_function_ep(
      radii, n, cov, barrier, n_paths, 1234u, NULL, SIF_DEFAULT);
    CHECK(f != NULL, "the multiplicity returned NULL on the real pipeline");

    if (f) {
      int any = 0, negative = 0;
      double integral = 0.0;
      for (uint32_t i = 0; i + 1 < n; i++) {
        if (f[i] > 0.0f)
          any = 1;
        if (f[i] < 0.0f)
          negative = 1;
        integral += (double)f[i] * ((double)radii[i + 1] - (double)radii[i]);
      }
      CHECK(any, "the multiplicity is identically zero across every bin");
      CHECK(!negative, "the multiplicity went negative");
      CHECK(integral <= 1.0,
        "the multiplicity integrates to %.4f, above the one path per path it "
        "can possibly represent",
        integral);
      printf("  sigma %.3f..%.3f, B %.3f..%.3f, crossing fraction %.3f\n",
        (double)sigma[n - 1], (double)sigma[0], (double)barrier[n - 1],
        (double)barrier[0], integral);
      sif_free_aligned(f);
    }
  }

  sif_free_aligned(cov);
  sif_free_aligned(barrier);
  free(sigma);
  free(radii);
  free(k);
  free(pk);
}

static void test_covariance_guards(void) {
  printf("covariance validation\n");

  real_t* k = malloc(N_K * sizeof(real_t));
  real_t* pk = malloc(N_K * sizeof(real_t));
  __power_law(k, pk, -2.0);

  const real_t radii[3] = {2.0f, 8.0f, 20.0f};
  double* r;

  r = sif_delta_covariance_pk(NULL, pk, N_K, radii, 3, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "a NULL k was accepted");
  sif_free_aligned(r);

  r = sif_delta_covariance_pk(k, pk, N_K, NULL, 3, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "NULL radii were accepted");
  sif_free_aligned(r);

  r = sif_delta_covariance_pk(k, pk, N_K, radii, 0, NULL, NULL, NULL, SIF_DEFAULT);
  CHECK(r == NULL, "zero radii were accepted");
  sif_free_aligned(r);

  {
    const real_t bad_radii[3] = {2.0f, 0.0f, 20.0f};
    r = sif_delta_covariance_pk(
      k, pk, N_K, bad_radii, 3, NULL, NULL, NULL, SIF_DEFAULT);
    CHECK(r == NULL, "a zero radius was accepted");
    sif_free_aligned(r);
  }

  {
    /* Stricter than the moments, and deliberately so: the Gram accumulation
     * takes a square root of k^3 P(k). */
    real_t saved = pk[10];
    pk[10] = -1.0f;
    r = sif_delta_covariance_pk(k, pk, N_K, radii, 3, NULL, NULL, NULL, SIF_DEFAULT);
    CHECK(r == NULL, "a negative P(k) was accepted");
    sif_free_aligned(r);
    pk[10] = saved;
  }

  {
    real_t* k_bad = malloc(N_K * sizeof(real_t));
    memcpy(k_bad, k, N_K * sizeof(real_t));
    k_bad[10] = k_bad[9]; /* not strictly increasing */
    r = sif_delta_covariance_pk(
      k_bad, pk, N_K, radii, 3, NULL, NULL, NULL, SIF_DEFAULT);
    CHECK(r == NULL, "a non-increasing k table was accepted");
    sif_free_aligned(r);
    free(k_bad);
  }

  printf("  all guards returned NULL\n");

  free(k);
  free(pk);
}

int main(void) {
  sif_config_t cfg = {.fft_config = NULL, .omp_config = NULL, .verbose = false,
                      .log_level = SIF_LOG_LEVEL_ERROR};
  sif_init(&cfg);

  test_covariance_diagonal();
  test_covariance_properties();
  test_deriv_variance();
  test_barrier_smt();
  test_covariance_guards();

  test_gaussian_generator();
  test_cholesky();
  test_two_step_exact();
  test_geometric();
  test_degenerate();
  test_thread_independence();
  test_convergence();
  test_marginal_bound();
  test_binning();
  test_guards();
  test_pipeline();

  sif_finalize();

  if (failures) {
    printf("\n%d check(s) failed\n", failures);
    return 1;
  }

  printf("\nall excursion-set checks passed\n");
  return 0;
}
