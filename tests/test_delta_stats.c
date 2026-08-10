/* Validates the Fourier-space delta PDF estimator and the phase-shuffle
 * surrogate. The load-bearing check is that sigma(R) computed from the
 * spectrum agrees with the variance measured on the real-space field: that
 * single equality pins down the filter normalization, the Parseval factors,
 * and -- for a shuffled spectrum -- the Hermitian symmetry, which otherwise
 * fails silently by quietly dropping power in the c2r transform. */
#include "core/get_system.h"
#include "math/fft.h"
#include "test_util.h"
#include "sif/core/system.h"
#include "sif/measure/deltadistribution.h"
#include "sif/measure/deltamoments.h"
#include "sif/model/bbks.h"
#include "sif/structures/sizefunction.h"
#include "sif/model/deltamoments.h"
#include "sif/structures/grid.h"
#include "sif/utils/align.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define BOX_LENGTH 500.0f

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static double uni(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)((rng_state >> 11) & 0x1FFFFFFFFFFFFFULL) /
         (double)0x20000000000000ULL;
}

static inline uint64_t idx3(uint32_t n, uint32_t i, uint32_t j, uint32_t k) {
  return (uint64_t)i * n * n + (uint64_t)j * n + k;
}

/* One periodic box-average pass of half-width h along one axis. */
static void smooth_pass(
  const real_t* src, real_t* dst, uint32_t n, int h, int axis) {
  const double inv = 1.0 / (2 * h + 1);

  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      for (uint32_t k = 0; k < n; k++) {
        double s = 0.0;
        for (int o = -h; o <= h; o++) {
          uint32_t a = i, b = j, c = k;
          uint32_t shifted = (uint32_t)(((int)n * 8 + o) % (int)n);
          if (axis == 0)
            a = (i + shifted) % n;
          else if (axis == 1)
            b = (j + shifted) % n;
          else
            c = (k + shifted) % n;
          s += (double)src[idx3(n, a, b, c)];
        }
        dst[idx3(n, i, j, k)] = (real_t)(s * inv);
      }
    }
  }
}

/*
 * Deterministic lognormal test field.
 *
 * White noise smoothed into a broadband, near-Gaussian field, then passed
 * through exp(). Two properties matter here and neither is incidental: the
 * field is *broadband*, so a phase-randomized surrogate has enough modes to be
 * driven Gaussian by the central limit theorem, and it is *asymmetric*, so it
 * carries the skewness the surrogate is supposed to destroy. An odd
 * perturbation of a symmetric field (say s + s^3) would have neither.
 */
static void fill_field(real_t* d, uint32_t n, real_t box_length) {
  (void)box_length;
  const uint64_t total = (uint64_t)n * n * n;

  real_t* tmp = malloc(total * sizeof(real_t));
  if (!tmp)
    return;

  rng_state = 0x9E3779B97F4A7C15ULL;
  for (uint64_t i = 0; i < total; i++)
    d[i] = (real_t)(uni() - 0.5);

  /* Two passes per axis gives a correlation length of a few cells, which at
   * n=64 in a 500 box is comparable to the radii under test. */
  for (int pass = 0; pass < 2; pass++) {
    for (int axis = 0; axis < 3; axis++) {
      smooth_pass(d, tmp, n, 3, axis);
      memcpy(d, tmp, total * sizeof(real_t));
    }
  }

  free(tmp);

  /* Standardize the Gaussian field before exponentiating. */
  double mean = 0.0;
  for (uint64_t i = 0; i < total; i++)
    mean += (double)d[i];
  mean /= (double)total;

  double var = 0.0;
  for (uint64_t i = 0; i < total; i++) {
    double t = (double)d[i] - mean;
    var += t * t;
  }
  var /= (double)total;

  const double scale = 0.9 / sqrt(var);

  double sum = 0.0;
  for (uint64_t i = 0; i < total; i++) {
    double g = ((double)d[i] - mean) * scale;
    double v = exp(g);
    d[i] = (real_t)v;
    sum += v;
  }

  /* Normalize to an overdensity: zero mean by construction. */
  const double inv_mean = (double)total / sum;
  for (uint64_t i = 0; i < total; i++)
    d[i] = (real_t)((double)d[i] * inv_mean - 1.0);
}

/* Mean and variance of a real-space buffer, in double. */
static void field_moments(
  const real_t* d, uint64_t n, double* mean, double* var) {
  double s = 0.0;
  for (uint64_t i = 0; i < n; i++)
    s += (double)d[i];
  *mean = s / (double)n;

  double v = 0.0;
  for (uint64_t i = 0; i < n; i++) {
    double t = (double)d[i] - *mean;
    v += t * t;
  }
  *var = v / (double)n;
}

/* Half-complex index, mirroring __get_flat_complex_index in fft.c. */
static inline uint64_t cidx(uint32_t ix, uint32_t iy, uint32_t iz, uint32_t n) {
  const uint32_t z_dim = n / 2 + 1;
  return ((uint64_t)ix * n + iy) * z_dim + iz;
}

/* Signed integer wavenumber along one axis. */
static inline int32_t kvec(uint32_t i, uint32_t n) {
  return (i > n / 2) ? (int32_t)i - (int32_t)n : (int32_t)i;
}

/*
 * Applies the window, then multiplies delta_k by a derivative operator, and
 * returns the real-space result. `axis` in 0..2 gives d/dx_axis (multiply by
 * i*k_a); axis == 3 gives the Laplacian (multiply by -k^2); axis == -1 leaves
 * the field alone.
 *
 * The Nyquist component of a first derivative has to be dropped. Multiplying a
 * self-conjugate mode by i turns a real value into an imaginary one, which is
 * not Hermitian-consistent, and c2r would silently discard it anyway. This is
 * exactly the discretization ambiguity that makes the gradient check agree
 * only up to the Nyquist-plane power -- negligible for a well-smoothed field,
 * which is the point.
 */
static real_t* derivative_field(sif_fft_workspace_t* ws, uint32_t n,
  sif_filter_type_t filter, real_t radius, int axis) {

  if (sif_fft_apply_filter(ws, filter, radius, BOX_LENGTH) != SIF_OK)
    return NULL;

  const uint32_t z_dim = n / 2 + 1;
  const double k_unit = 2.0 * M_PI / (double)BOX_LENGTH;

  if (axis >= 0) {
    for (uint32_t ix = 0; ix < n; ix++) {
      const int32_t kx = kvec(ix, n);
      for (uint32_t iy = 0; iy < n; iy++) {
        const int32_t ky = kvec(iy, n);
        for (uint32_t iz = 0; iz < z_dim; iz++) {
          const int32_t kz = (int32_t)iz;
          const uint64_t i = cidx(ix, iy, iz, n);

          double re = ws->delta_k_cpy[i][0];
          double im = ws->delta_k_cpy[i][1];

          if (axis == 3) {
            const double k2 =
              k_unit * k_unit * (double)(kx * kx + ky * ky + kz * kz);
            ws->delta_k_cpy[i][0] = (real_t)(-k2 * re);
            ws->delta_k_cpy[i][1] = (real_t)(-k2 * im);
          } else {
            const int32_t ka = (axis == 0) ? kx : (axis == 1) ? ky : kz;

            /* Drop the Nyquist plane of the derivative. */
            const uint32_t idx_a = (axis == 0) ? ix : (axis == 1) ? iy : iz;
            if ((n % 2 == 0) && idx_a == n / 2) {
              ws->delta_k_cpy[i][0] = 0.0f;
              ws->delta_k_cpy[i][1] = 0.0f;
              continue;
            }

            const double k = k_unit * (double)ka;
            /* (re + i im) * i k = -k im + i k re */
            ws->delta_k_cpy[i][0] = (real_t)(-k * im);
            ws->delta_k_cpy[i][1] = (real_t)(k * re);
          }
        }
      }
    }
  }

  return sif_fft_grid_backward(ws);
}

/*
 * The core invariant, run for every shuffle mode: the k-space moments must
 * reproduce what the real-space field, its gradient and its Laplacian actually
 * do. sigma_0 catches a broken Hermitian plane (it shows up as a variance
 * deficit); sigma_1 and sigma_2 additionally pin down the k^(2j) weighting and
 * the physical units of k, which nothing else in the suite touches.
 *
 * Run with a Gaussian window on purpose: a top-hat leaves enough power near
 * Nyquist that the dropped derivative plane would show up in sigma_1, and the
 * sigma_2 sum would be a statement about the grid rather than the field.
 */
static void test_moments_match_field(
  uint32_t n, sif_option_t shuffle, const char* label) {
  printf("k-space vs real-space moments, n=%u (%s)\n", n, label);

  /* The library owns the FFTW manager: sif_fft_manager_finalize tears FFTW down
   * process-wide, so a test must never create and destroy one of its own. */
  sif_fft_manager_t* mgr = sif_get_system_state()->fft_mgr;
  sif_grid_t* grid = sif_grid_alloc(n, BOX_LENGTH);
  CHECK(mgr && grid, "setup failed");
  if (!mgr || !grid) {
    if (grid)
      sif_grid_free(grid);
    return;
  }

  fill_field(grid->delta, n, BOX_LENGTH);

  sif_fft_workspace_t* ws = sif_fft_workspace_alloc(mgr, n);
  CHECK(ws != NULL, "workspace alloc failed");
  if (!ws) {
    sif_grid_free(grid);
    return;
  }

  sif_fft_grid_forward(ws, grid);

  if (shuffle == SIF_DELTA_SHUFFLE_PHASES) {
    CHECK(sif_fft_randomize_phases(ws, 1234, false) == SIF_OK, "shuffle failed");
  } else if (shuffle == SIF_DELTA_SHUFFLE_GAUSSIAN) {
    CHECK(sif_fft_randomize_phases(ws, 1234, true) == SIF_OK, "shuffle failed");
  }

  CHECK(sif_fft_workspace_init_backward(ws, mgr) == SIF_OK, "init_backward failed");

  const real_t radius = 25.0f;
  const uint64_t total = (uint64_t)n * n * n;

  double predicted[3], high_k[3];
  CHECK(sif_fft_spectral_moments(ws, FILTER_GAUSSIAN, radius, BOX_LENGTH, 2, 0,
          predicted, high_k) == SIF_OK,
    "moment evaluation failed");

  /* The whole comparison is only meaningful if the window really has killed
   * the Nyquist modes, so assert that before trusting the agreement. */
  CHECK(high_k[2] < 1e-6,
    "the Gaussian window left %.3g of the sigma_2 sum above half Nyquist",
    high_k[2]);

  double measured[3] = {0.0, 0.0, 0.0};
  double mean;

  /* sigma_0^2 = <delta_R^2> */
  real_t* f = derivative_field(ws, n, FILTER_GAUSSIAN, radius, -1);
  CHECK(f != NULL, "smoothed field failed");
  if (f)
    field_moments(f, total, &mean, &measured[0]);

  /* sigma_1^2 = <|grad delta_R|^2>, summed over the three components. */
  for (int axis = 0; axis < 3; axis++) {
    real_t* g = derivative_field(ws, n, FILTER_GAUSSIAN, radius, axis);
    CHECK(g != NULL, "gradient component %d failed", axis);
    if (g) {
      double m, v;
      field_moments(g, total, &m, &v);
      /* Each derivative has zero mean, so its variance is its mean square. */
      measured[1] += v + m * m;
    }
  }

  /* sigma_2^2 = <(laplacian delta_R)^2> */
  real_t* l = derivative_field(ws, n, FILTER_GAUSSIAN, radius, 3);
  CHECK(l != NULL, "laplacian failed");
  if (l) {
    double m, v;
    field_moments(l, total, &m, &v);
    measured[2] = v + m * m;
  }

  for (int j = 0; j < 3; j++) {
    CHECK(predicted[j] > 0.0, "sigma_%d^2 from k-space is non-positive (%g)", j,
      predicted[j]);
    if (predicted[j] > 0.0) {
      const double rel = fabs(measured[j] - predicted[j]) / predicted[j];
      CHECK(rel < 1e-3,
        "sigma_%d^2 mismatch: k-space %g vs real-space %g (rel %.3e)", j,
        predicted[j], measured[j], rel);
    }
  }

  sif_fft_workspace_free(ws);
  sif_grid_free(grid);
}

/*
 * Phase randomization must preserve the power spectrum exactly in PHASES mode.
 * Checked mode by mode on |delta_k|, which also catches the redundant planes
 * getting their amplitudes rewritten by the conjugate assignment.
 */
static void test_shuffle_preserves_power(uint32_t n) {
  printf("phase shuffle preserves |delta_k|, n=%u\n", n);

  sif_fft_manager_t* mgr = sif_get_system_state()->fft_mgr;
  sif_grid_t* grid = sif_grid_alloc(n, BOX_LENGTH);
  if (!mgr || !grid) {
    CHECK(0, "setup failed");
    return;
  }

  fill_field(grid->delta, n, BOX_LENGTH);

  sif_fft_workspace_t* ws = sif_fft_workspace_alloc(mgr, n);
  CHECK(ws != NULL, "workspace alloc failed");
  if (!ws) {
    sif_grid_free(grid);
    return;
  }

  sif_fft_grid_forward(ws, grid);

  const uint64_t complex_cells = (uint64_t)n * n * (n / 2 + 1);
  double* before = malloc(complex_cells * sizeof(double));
  CHECK(before != NULL, "scratch alloc failed");

  if (before) {
    for (uint64_t i = 0; i < complex_cells; i++) {
      double re = ws->delta_k[i][0], im = ws->delta_k[i][1];
      before[i] = sqrt(re * re + im * im);
    }

    /* Enforcing Hermitian symmetry replaces one member of each conjugate pair
     * on the redundant planes with the other's amplitude. The input only
     * satisfies that symmetry to the transform's round-off, so the tolerance
     * needs an absolute floor set by the scale of the spectrum: a pure
     * relative test would flag the near-zero high-k modes, where the round-off
     * is comparable to the amplitude itself. */
    double power = 0.0;
    for (uint64_t i = 0; i < complex_cells; i++)
      power += before[i] * before[i];
    const double rms = sqrt(power / (double)complex_cells);

    CHECK(sif_fft_randomize_phases(ws, 99, false) == SIF_OK, "shuffle failed");

    uint64_t bad = 0, moved = 0;
    double power_after = 0.0;

    for (uint64_t i = 1; i < complex_cells; i++) {
      double re = ws->delta_k[i][0], im = ws->delta_k[i][1];
      double after = sqrt(re * re + im * im);
      power_after += after * after;

      if (fabs(after - before[i]) > fmax(1e-4 * before[i], 1e-4 * rms))
        bad++;
      /* Modes whose amplitude is numerically zero cannot move anywhere. */
      if (before[i] > 1e-3 * rms && fabs(im) > 1e-9)
        moved++;
    }

    power_after += before[0] * before[0]; /* k=0 is left alone by design */

    CHECK(bad == 0, "%llu of %llu modes changed amplitude",
      (unsigned long long)bad, (unsigned long long)complex_cells);
    CHECK(moved > 0, "no mode acquired an imaginary part; phases did not move");
    CHECK(fabs(power_after - power) / power < 1e-6,
      "total power moved by %.3e", fabs(power_after - power) / power);

    /* The k = 0 mode carries the mean and must survive untouched. */
    CHECK(fabs((double)ws->delta_k[0][1]) < 1e-9,
      "the k=0 mode acquired an imaginary part");

    free(before);
  }

  sif_fft_workspace_free(ws);
  sif_grid_free(grid);
}

/* End-to-end through the public API, including the normalization convention. */
static void test_stats_api(uint32_t n) {
  printf("delta stats end to end, n=%u\n", n);

  sif_grid_t* grid = sif_grid_alloc(n, BOX_LENGTH);
  CHECK(grid != NULL, "grid alloc failed");
  if (!grid)
    return;

  fill_field(grid->delta, n, BOX_LENGTH);

  const real_t radii[3] = {20.0f, 30.0f, 45.0f};
  /* Wide enough that the lognormal tail does not clip: the estimator counts
   * every cell in the normalization but only histograms the ones in range, so
   * a clipped tail shows up as an integral below one and a depressed sigma. */
  const real_t bounds[2] = {-8.0f, 64.0f};
  const uint32_t n_bins = 720;

  sif_delta_distribution_t* d = sif_delta_distribution_grid(
    grid, radii, 3, n_bins, bounds, 7, SIF_DELTA_KEEP_CIC_WINDOW);
  CHECK(d != NULL, "computation returned NULL");

  /* Every order comes from a single call and a single pass. */
  sif_delta_moments_t* mom = sif_delta_moments_grid(
    grid, radii, 3, 2, 0, 7, SIF_DELTA_KEEP_CIC_WINDOW);
  CHECK(mom != NULL, "moments returned NULL");
  if (!d || !mom) {
    sif_delta_distribution_free(d);
    sif_delta_moments_free(mom);
    sif_grid_free(grid);
    return;
  }

  CHECK(mom->order == 2 && mom->n_moments == 3, "wrong moment count");
  CHECK(mom->n_radii == 3, "wrong radius count");

  /* The offsets must describe the layout they claim to. */
  for (uint8_t j = 0; j <= mom->n_moments; j++) {
    CHECK(mom->offsets[j] == (uint32_t)j * mom->n_radii,
      "offsets[%u] is %u, expected %u", j, mom->offsets[j],
      (uint32_t)j * mom->n_radii);
  }
  for (uint8_t j = 0; j < mom->n_moments; j++) {
    CHECK(sif_delta_moments_sigma(mom, j) == mom->sigma + mom->offsets[j],
      "the sigma accessor disagrees with offsets at order %u", j);
  }
  CHECK(sif_delta_moments_sigma(mom, 3) == NULL,
    "the sigma accessor returned a block for an absent order");

  const real_t* sig0 = sif_delta_moments_sigma(mom, 0);
  const real_t* sig1 = sif_delta_moments_sigma(mom, 1);
  const real_t* sig2 = sif_delta_moments_sigma(mom, 2);

  real_t* gamma_arr = sif_gamma_moments(mom);
  real_t* r_star_arr = sif_r_star_moments(mom);
  CHECK(gamma_arr && r_star_arr, "gamma / R_star computation failed");

  if (d) {
    CHECK(d->n_radii == 3 && d->n_bins == n_bins, "wrong shape");
    CHECK(d->n_samples == grid->total_cells, "wrong sample count");
    CHECK(d->delta_edges[0] == bounds[0] && d->delta_edges[n_bins] == bounds[1],
      "bin edges do not span the requested bounds");

    const real_t bin_width = (bounds[1] - bounds[0]) / n_bins;

    for (uint32_t k = 0; k < 3; k++) {
      CHECK(d->radii[k] == radii[k], "radius %u was not echoed back", k);
      const real_t sigma0 = sig0[k];
      CHECK(sigma0 > 0.0f, "sigma_0 at radius %u is non-positive", k);

      /* The field is well inside the bounds, so the PDF must integrate to 1
       * up to the handful of cells that clip. */
      double integral = 0.0;
      for (uint32_t b = 0; b < n_bins; b++)
        integral += (double)d->distributions[k * n_bins + b] * bin_width;

      CHECK(integral > 0.9999 && integral <= 1.0 + 1e-6,
        "row %u integrates to %g, expected ~1", k, integral);

      /* sigma must be consistent with the second moment of the histogram. */
      double mean = 0.0, m2 = 0.0;
      for (uint32_t b = 0; b < n_bins; b++) {
        double c = 0.5 * ((double)d->delta_edges[b] +
                           (double)d->delta_edges[b + 1]);
        double w = (double)d->distributions[k * n_bins + b] * bin_width;
        mean += w * c;
        m2 += w * c * c;
      }
      double hist_sigma = sqrt(m2 - mean * mean);
      double rel = fabs(hist_sigma - (double)sigma0) / (double)sigma0;

      /* Binning alone costs bin_width^2/12 of variance, so this is loose. */
      CHECK(rel < 0.05,
        "row %u: histogram sigma %g vs spectral sigma %g (rel %.3e)", k,
        hist_sigma, (double)sigma0, rel);

      /* The moments are ordered by construction and gamma is bounded. */
      const real_t s1 = sig1[k];
      const real_t s2 = sig2[k];
      CHECK(s1 > 0.0f && s2 > 0.0f, "row %u has a non-positive higher moment",
        k);
      /* The BBKS parameters come from external functions rather than living
       * in the struct. gamma < 1 is Cauchy-Schwarz on the moment sums, so it
       * is a genuine bound rather than a heuristic. */
      const double gamma = gamma_arr ? (double)gamma_arr[k] : 0.0;
      const double r_star = r_star_arr ? (double)r_star_arr[k] : 0.0;

      CHECK(fabs(gamma - (double)s1 * s1 / ((double)sigma0 * s2)) <
              1e-5 * gamma,
        "row %u: gamma helper disagrees with the definition", k);
      CHECK(fabs(r_star - sqrt(3.0) * (double)s1 / s2) < 1e-5 * r_star,
        "row %u: R_star helper disagrees with the definition", k);

      CHECK(gamma > 0.0 && gamma < 1.0, "row %u: gamma = %g is outside (0, 1)",
        k, gamma);

      /* R_star is only bounded below by zero. The familiar rule of thumb that
       * it lands below R holds for a CDM-like spectrum, where the smoothing
       * sets the coherence scale; it is not an invariant. This test field is
       * white noise driven through several box filters, so its own spectrum
       * cuts off before the window does and R_star exceeds R at small R. */
      CHECK(r_star > 0.0, "row %u: R_star = %g is not positive", k, r_star);

      printf("    R=%5.1f  sigma=(%.5f, %.3e, %.3e)  gamma=%.4f  "
             "R*=%6.2f  high_k(sigma_2)=%.2e\n",
        (double)radii[k], (double)sigma0, (double)s1, (double)s2, gamma,
        r_star, (double)mom->high_k_fraction[mom->offsets[2] + k]);
    }

    /* Smoothing on a larger scale can only reduce the variance. */
    CHECK(sig0[0] > sig0[1] && sig0[1] > sig0[2],
      "sigma_0 is not decreasing with radius: %g %g %g", (double)sig0[0],
      (double)sig0[1], (double)sig0[2]);

    sif_delta_distribution_free(d);
  }

  /* An order-1 set must refuse to produce gamma or R_star. */
  sif_delta_moments_t* low = sif_delta_moments_grid(
    grid, radii, 3, 1, 0, 7, SIF_DELTA_KEEP_CIC_WINDOW);
  CHECK(low != NULL, "order-1 moments returned NULL");
  if (low) {
    CHECK(sif_gamma_moments(low) == NULL,
      "gamma was produced from an order-1 moment set");
    CHECK(sif_r_star_moments(low) == NULL,
      "R_star was produced from an order-1 moment set");
    sif_delta_moments_free(low);
  }

  sif_free_aligned(gamma_arr);
  sif_free_aligned(r_star_arr);
  sif_delta_moments_free(mom);

  sif_grid_free(grid);
}

/* A shuffled field must keep sigma(R) while losing the skewness that the
 * cubed term put into the original. */
static void test_shuffle_removes_skewness(uint32_t n) {
  printf("shuffle preserves sigma and removes skewness, n=%u\n", n);

  const real_t radii[1] = {20.0f};
  /* The surrogate is Gaussian and reaches well below delta = -1, which the
   * original lognormal field never does. Both tails have to fit or the
   * skewness comparison is measuring the clip, not the field. */
  const real_t bounds[2] = {-8.0f, 64.0f};
  const uint32_t n_bins = 720;

  double skew[2] = {0.0, 0.0};
  real_t sigma[2] = {0.0f, 0.0f};

  const sif_option_t modes[2] = {
    SIF_DELTA_SHUFFLE_NONE, SIF_DELTA_SHUFFLE_PHASES};

  for (int m = 0; m < 2; m++) {
    sif_grid_t* grid = sif_grid_alloc(n, BOX_LENGTH);
    if (!grid) {
      CHECK(0, "grid alloc failed");
      return;
    }
    fill_field(grid->delta, n, BOX_LENGTH);

    sif_delta_distribution_t* d = sif_delta_distribution_grid(grid, radii, 1,
      n_bins, bounds, 2024, modes[m] | SIF_DELTA_KEEP_CIC_WINDOW);
    sif_delta_moments_t* mom = sif_delta_moments_grid(
      grid, radii, 1, 0, 0, 2024, modes[m] | SIF_DELTA_KEEP_CIC_WINDOW);
    CHECK(d != NULL && mom != NULL, "computation %d returned NULL", m);

    if (d) {
      const double bw = ((double)bounds[1] - bounds[0]) / n_bins;
      double mean = 0.0, m2 = 0.0, m3 = 0.0;

      for (uint32_t b = 0; b < n_bins; b++) {
        double c =
          0.5 * ((double)d->delta_edges[b] + (double)d->delta_edges[b + 1]);
        double w = (double)d->distributions[b] * bw;
        mean += w * c;
        m2 += w * c * c;
        m3 += w * c * c * c;
      }

      double var = m2 - mean * mean;
      double mu3 = m3 - 3.0 * mean * m2 + 2.0 * mean * mean * mean;
      skew[m] = mu3 / (var * sqrt(var));
      sif_delta_distribution_free(d);
    }

    if (mom) {
      sigma[m] = mom->sigma[0];
      sif_delta_moments_free(mom);
    }

    sif_grid_free(grid);
  }

  const double dsigma = fabs((double)sigma[1] - sigma[0]) / sigma[0];
  CHECK(dsigma < 1e-3, "shuffle changed sigma: %g -> %g (rel %.3e)",
    (double)sigma[0], (double)sigma[1], dsigma);

  CHECK(fabs(skew[0]) > 0.05, "the reference field is not skewed (%g); the "
                              "test cannot detect anything",
    skew[0]);
  CHECK(fabs(skew[1]) < 0.3 * fabs(skew[0]),
    "the surrogate kept its skewness: %g vs %g", skew[1], skew[0]);

  printf("  skewness %.4f -> %.4f, sigma %.5f -> %.5f\n", skew[0], skew[1],
    (double)sigma[0], (double)sigma[1]);
}

/* Radii below the resolution limit or above half the box must be rejected. */
static void test_radius_guards(uint32_t n) {
  printf("resolution and box-size guards, n=%u\n", n);

  sif_grid_t* grid = sif_grid_alloc(n, BOX_LENGTH);
  if (!grid) {
    CHECK(0, "grid alloc failed");
    return;
  }
  fill_field(grid->delta, n, BOX_LENGTH);

  const real_t bounds[2] = {-3.0f, 3.0f};

  const real_t too_small[1] = {grid->cell_length};
  sif_delta_distribution_t* d = sif_delta_distribution_grid(
    grid, too_small, 1, 20, bounds, 1, SIF_DEFAULT);
  CHECK(d == NULL, "a sub-resolution radius was accepted");
  sif_delta_distribution_free(d);

  const real_t too_big[1] = {BOX_LENGTH * 0.75f};
  d = sif_delta_distribution_grid(
    grid, too_big, 1, 20, bounds, 1, SIF_DEFAULT);
  CHECK(d == NULL, "a radius larger than half the box was accepted");
  sif_delta_distribution_free(d);

  const real_t ok[1] = {30.0f};
  const real_t bad_bounds[2] = {2.0f, 1.0f};
  d = sif_delta_distribution_grid(
    grid, ok, 1, 20, bad_bounds, 1, SIF_DEFAULT);
  CHECK(d == NULL, "inverted delta bounds were accepted");
  sif_delta_distribution_free(d);

  /* An out-of-range moment order must be rejected too. */
  sif_delta_moments_t* mom = sif_delta_moments_grid(
    grid, ok, 1, SIF_MAX_MOMENT_ORDER + 1, 0, 1, SIF_DEFAULT);
  CHECK(mom == NULL, "an out-of-range moment order was accepted");
  sif_delta_moments_free(mom);

  sif_grid_free(grid);
}

/*
 * The theory integral and the field sum are the same quantity evaluated two
 * ways: sigma_j^2 = (1/2pi^2) integral dk k^(2j+2) P(k) W^2, versus the same
 * sum over the discrete modes of the box with P(k) replaced by the measured
 * |delta_k|^2. Feeding the *measured* band power of a realization back into
 * the theory route must therefore reproduce the field route, up to the
 * binning of the spectrum and the discreteness of the low-k modes.
 *
 * This is the check that pins the from_pk normalization -- the 1/2pi^2, the
 * k^(2j+2) rather than k^(2j), and the log-k quadrature -- against the
 * already-validated field estimator.
 */
static void test_pk_matches_field(uint32_t n) {
  printf("theory integral vs field sum, n=%u\n", n);

  sif_fft_manager_t* mgr = sif_get_system_state()->fft_mgr;
  sif_grid_t* grid = sif_grid_alloc(n, BOX_LENGTH);
  if (!mgr || !grid) {
    CHECK(0, "setup failed");
    return;
  }
  fill_field(grid->delta, n, BOX_LENGTH);

  sif_fft_workspace_t* ws = sif_fft_workspace_alloc(mgr, n);
  CHECK(ws != NULL, "workspace alloc failed");
  if (!ws) {
    sif_grid_free(grid);
    return;
  }
  sif_fft_grid_forward(ws, grid);

  /* Bin |delta_k|^2 into a spherically averaged P(k) on the box's own mode
   * spacing. P(k) = |delta_k|^2 * V / N^6 in this convention. */
  const uint32_t z_dim = n / 2 + 1;
  const uint32_t n_bins_k = n / 2;
  const double k_f = 2.0 * M_PI / (double)BOX_LENGTH;
  const double volume = (double)BOX_LENGTH * BOX_LENGTH * BOX_LENGTH;
  const double n3 = (double)n * n * n;

  double* p_sum = calloc(n_bins_k, sizeof(double));
  double* p_cnt = calloc(n_bins_k, sizeof(double));
  CHECK(p_sum && p_cnt, "band-power scratch alloc failed");

  if (p_sum && p_cnt) {
    for (uint32_t ix = 0; ix < n; ix++) {
      const int32_t kx = kvec(ix, n);
      for (uint32_t iy = 0; iy < n; iy++) {
        const int32_t ky = kvec(iy, n);
        for (uint32_t iz = 0; iz < z_dim; iz++) {
          const int32_t kz = (int32_t)iz;
          const uint32_t k2 = (uint32_t)(kx * kx + ky * ky + kz * kz);
          if (k2 == 0)
            continue;

          const double kmag = sqrt((double)k2);
          const uint32_t b = (uint32_t)kmag;
          if (b >= n_bins_k)
            continue;

          const uint64_t i = cidx(ix, iy, iz, n);
          const double re = ws->delta_k[i][0], im = ws->delta_k[i][1];
          const double weight = ((2 * iz) % n == 0) ? 1.0 : 2.0;

          p_sum[b] += weight * (re * re + im * im) * volume / (n3 * n3);
          p_cnt[b] += weight;
        }
      }
    }

    /* Bin centres at the mean |k| of each shell; b = 0 is empty by
     * construction since k = 0 was skipped. */
    real_t* kt = malloc(n_bins_k * sizeof(real_t));
    real_t* pt = malloc(n_bins_k * sizeof(real_t));
    uint32_t n_pts = 0;

    if (kt && pt) {
      for (uint32_t b = 1; b < n_bins_k; b++) {
        if (p_cnt[b] <= 0.0)
          continue;
        kt[n_pts] = (real_t)(k_f * ((double)b + 0.5));
        pt[n_pts] = (real_t)(p_sum[b] / p_cnt[b]);
        n_pts++;
      }

      const real_t radii[2] = {30.0f, 45.0f};

      for (uint8_t j = 0; j < 3; j++) {
        sif_delta_moments_t* from_pk = sif_delta_moments_pk(
          kt, pt, n_pts, radii, 2, j, SIF_DELTA_FILTER_GAUSSIAN);
        sif_delta_moments_t* from_field =
          sif_delta_moments_grid(grid, radii, 2, j, 0, 0,
            SIF_DELTA_FILTER_GAUSSIAN | SIF_DELTA_KEEP_CIC_WINDOW);

        CHECK(from_pk != NULL && from_field != NULL,
          "moment %u: one of the two routes returned NULL", j);

        if (from_pk && from_field) {
          CHECK(from_pk->order == j && from_field->order == j,
            "moment %u: order not echoed back", j);

          const real_t* fs = sif_delta_moments_sigma(from_field, j);
          const real_t* ps = sif_delta_moments_sigma(from_pk, j);

          for (uint32_t r = 0; r < 2; r++) {
            const double a = (double)fs[r];
            const double b = (double)ps[r];
            const double rel = fabs(a - b) / a;

            /* A continuum integral cannot reproduce a box exactly, and the
             * gap is worst where the weight sits on the fewest modes: the
             * lowest shells hold only a handful each, and everything below the
             * first bin centre is missing from the integral outright. That
             * hits sigma_0 hardest and sigma_2 least, since higher orders
             * weight the well-populated high-k shells. This bound is therefore
             * loose on purpose -- test_pk_analytic is what pins the
             * normalization; this one only confirms the two routes agree about
             * what they are computing. */
            CHECK(rel < 0.30,
              "sigma_%u at R=%g: field %g vs P(k) %g (rel %.3e)", j,
              (double)radii[r], a, b, rel);
            printf("    sigma_%u  R=%4.0f  field=%.6g  pk=%.6g  rel=%.2e\n", j,
              (double)radii[r], a, b, rel);
          }
        }

        sif_delta_moments_free(from_pk);
        sif_delta_moments_free(from_field);
      }
    }

    free(kt);
    free(pt);
  }

  free(p_sum);
  free(p_cnt);
  sif_fft_workspace_free(ws);
  sif_grid_free(grid);
}

/*
 * Pins the from_pk normalization against a closed form. For a power-law
 * P(k) = A k^n and a Gaussian window, W^2 = exp(-k^2 R^2) and
 *
 *   sigma_j^2 = (A / 4 pi^2) R^-(2j+3+n) Gamma((2j+3+n) / 2)
 *
 * which fixes every factor the implementation could get wrong independently:
 * the 1/2pi^2, the k^(2j+2) rather than k^(2j), and the extra k from the
 * change of variable in the log-k quadrature. Unlike the comparison against
 * the field estimator, this has no box and no mode discreteness, so the
 * tolerance can be tight.
 */
static void test_pk_analytic(void) {
  printf("P(k) integral vs closed form\n");

  const double slope = -1.5;
  const double amplitude = 1.0;

  /* Wide and finely sampled: the table has to run out past k R >> 1 at the
   * top and well below 1/R at the bottom, or the integral is truncated
   * rather than wrong. */
  const uint32_t n_points = 2000;
  real_t* kt = malloc(n_points * sizeof(real_t));
  real_t* pt = malloc(n_points * sizeof(real_t));
  CHECK(kt && pt, "table alloc failed");
  if (!kt || !pt) {
    free(kt);
    free(pt);
    return;
  }

  const double lk_min = log(1e-5), lk_max = log(1e2);
  for (uint32_t i = 0; i < n_points; i++) {
    const double lk = lk_min + (lk_max - lk_min) * i / (double)(n_points - 1);
    const double k = exp(lk);
    kt[i] = (real_t)k;
    pt[i] = (real_t)(amplitude * pow(k, slope));
  }

  const real_t radii[3] = {5.0f, 20.0f, 80.0f};

  sif_delta_moments_t* m = sif_delta_moments_pk(
    kt, pt, n_points, radii, 3, 2, SIF_DELTA_FILTER_GAUSSIAN);
  CHECK(m != NULL, "the analytic table returned NULL");

  for (uint8_t j = 0; m && j <= 2; j++) {
    const real_t* sig = sif_delta_moments_sigma(m, j);
    const real_t* hkf = m->high_k_fraction + m->offsets[j];

    const double p = 2.0 * j + 3.0 + slope;

    for (uint32_t r = 0; r < 3; r++) {
      const double R = (double)radii[r];
      const double expected_sq =
        amplitude / (4.0 * M_PI * M_PI) * pow(R, -p) * tgamma(0.5 * p);
      const double expected = sqrt(expected_sq);
      const double got = (double)sig[r];
      const double rel = fabs(got - expected) / expected;

      CHECK(rel < 1e-3, "sigma_%u at R=%g: got %g, closed form %g (rel %.3e)",
        j, R, got, expected, rel);

      /* A table this wide must not be truncating anything. */
      CHECK(hkf[r] < 1e-3,
        "sigma_%u at R=%g: %.3g of the integral is in the top half of the "
        "table",
        j, R, (double)hkf[r]);
    }

    printf("    order %u: within 1e-3 of the closed form at R = 5, 20, 80\n",
      j);
  }

  sif_delta_moments_free(m);
  free(kt);
  free(pt);
}

/* The P(k) route must reject a table it cannot integrate. */
static void test_pk_guards(void) {
  printf("P(k) table guards\n");

  const real_t radii[1] = {20.0f};
  const real_t good_k[3] = {0.01f, 0.1f, 1.0f};
  const real_t good_p[3] = {100.0f, 50.0f, 1.0f};

  const real_t zero_k[3] = {0.0f, 0.1f, 1.0f};
  sif_delta_moments_t* m = sif_delta_moments_pk(
    zero_k, good_p, 3, radii, 1, 0, SIF_DEFAULT);
  CHECK(m == NULL, "a table containing k = 0 was accepted");
  sif_delta_moments_free(m);

  const real_t unsorted_k[3] = {0.1f, 0.01f, 1.0f};
  m = sif_delta_moments_pk(
    unsorted_k, good_p, 3, radii, 1, 0, SIF_DEFAULT);
  CHECK(m == NULL, "a non-monotonic k table was accepted");
  sif_delta_moments_free(m);

  m = sif_delta_moments_pk(
    good_k, good_p, 1, radii, 1, 0, SIF_DEFAULT);
  CHECK(m == NULL, "a single-point table was accepted");
  sif_delta_moments_free(m);

  const real_t bad_radius[1] = {-1.0f};
  m = sif_delta_moments_pk(
    good_k, good_p, 3, bad_radius, 1, 0, SIF_DEFAULT);
  CHECK(m == NULL, "a negative radius was accepted");
  sif_delta_moments_free(m);

  m = sif_delta_moments_pk(
    good_k, good_p, 3, radii, 1, SIF_MAX_MOMENT_ORDER + 1, SIF_DEFAULT);
  CHECK(m == NULL, "an out-of-range order was accepted");
  sif_delta_moments_free(m);

  /* A well-formed table must succeed and produce a positive sigma. */
  m = sif_delta_moments_pk(
    good_k, good_p, 3, radii, 1, 0, SIF_DELTA_FILTER_GAUSSIAN);
  CHECK(m != NULL, "a well-formed table was rejected");
  if (m) {
    CHECK(m->sigma[0] > 0.0f, "a positive P(k) gave sigma_0 = %g",
      (double)m->sigma[0]);
    sif_delta_moments_free(m);
  }
}

/* --- BBKS peak statistics --- */

/*
 * For large w the exponential term vanishes and the denominator tends to one,
 * so the fitted G must approach its leading polynomial w^3 - 3 gamma^2 w. That
 * is a property of the exact G, not of the fit, so it checks the fit's
 * structure rather than just reproducing its own arithmetic.
 */
static void test_bbks_g_asymptote(void) {
  printf("BBKS G asymptotics\n");

  const double gammas[3] = {0.4, 0.55, 0.7};

  for (int i = 0; i < 3; i++) {
    const double g = gammas[i];
    const double w = 30.0;

    const double got = (double)sif_g_bbks((real_t)g, (real_t)w, SIF_BBKS_G_FITTED);
    const double lead = w * w * w - 3.0 * g * g * w;
    const double rel = fabs(got - lead) / lead;

    CHECK(rel < 1e-6, "gamma=%g: G(%g) = %g, leading term %g (rel %.3e)", g, w,
      got, lead, rel);
  }

  /* G has to be positive across the range peaks are actually counted in. */
  for (int i = 0; i < 3; i++) {
    for (double w = 0.0; w <= 5.0; w += 0.25) {
      const double got = (double)sif_g_bbks((real_t)gammas[i], (real_t)w, SIF_BBKS_G_FITTED);
      CHECK(got > 0.0, "gamma=%g: G(%g) = %g is not positive", gammas[i], w,
        got);
    }
  }
}

/*
 * The number density scales as 1/R_star^3 exactly, and its integral over nu is
 * the total density of maxima -- which for a Gaussian field depends on the
 * field only through R_star, not through gamma. Testing that the integral is
 * gamma-independent exercises the whole normalization of the fitted G at once:
 * a wrong coefficient in the fit would break it, since gamma enters G in five
 * different places.
 */
static void test_bbks_number_density(sif_option_t mode, const char* label) {
  printf("BBKS maxima number density (%s G)\n", label);

  /* Exact 1/R_star^3 scaling at fixed nu and gamma. */
  {
    const real_t nu[2] = {2.0f, 2.0f};
    const real_t gamma[2] = {0.55f, 0.55f};
    const real_t r_star[2] = {10.0f, 20.0f};

    real_t* n = sif_differential_number_density_bbks(
      nu, gamma, r_star, 2, SIF_BBKS_G_FITTED);
    CHECK(n != NULL, "number density returned NULL");
    if (n) {
      const double ratio = (double)n[0] / n[1];
      CHECK(fabs(ratio - 8.0) < 1e-4,
        "doubling R_star changed the density by %g, expected a factor 8",
        ratio);
      sif_free_aligned(n);
    }
  }

  /* Integral over nu, at fixed R_star, for several gamma. */
  {
    const uint32_t n_nu = 40000;
    const double nu_lo = -8.0, nu_hi = 12.0;
    const double dnu = (nu_hi - nu_lo) / (n_nu - 1);
    const double r = 1.0;

    real_t* nu = malloc(n_nu * sizeof(real_t));
    real_t* gm = malloc(n_nu * sizeof(real_t));
    real_t* rs = malloc(n_nu * sizeof(real_t));
    CHECK(nu && gm && rs, "sweep alloc failed");

    if (nu && gm && rs) {
      const double gammas[4] = {0.35, 0.45, 0.55, 0.65};
      double totals[4];

      for (int i = 0; i < 4; i++) {
        for (uint32_t t = 0; t < n_nu; t++) {
          nu[t] = (real_t)(nu_lo + t * dnu);
          gm[t] = (real_t)gammas[i];
          rs[t] = (real_t)r;
        }

        real_t* n = sif_differential_number_density_bbks(
          nu, gm, rs, n_nu, mode);
        CHECK(n != NULL, "number density returned NULL for gamma=%g",
          gammas[i]);

        double total = 0.0;
        if (n) {
          for (uint32_t t = 1; t < n_nu; t++)
            total += 0.5 * ((double)n[t] + n[t - 1]) * dnu;
          sif_free_aligned(n);
        }
        totals[i] = total;

        printf("    gamma=%.2f  n_max * R_star^3 = %.6f\n", gammas[i], total);
      }

      /* The total density of maxima of a Gaussian field is set by R_star
       * alone. The fitted G is only an approximation, so this is a few-percent
       * statement rather than an identity -- but a mistyped coefficient would
       * move it far more than that. */
      double lo = totals[0], hi = totals[0];
      for (int i = 1; i < 4; i++) {
        if (totals[i] < lo)
          lo = totals[i];
        if (totals[i] > hi)
          hi = totals[i];
      }

      CHECK((hi - lo) / lo < 0.05,
        "the total maxima density varies by %.1f%% across gamma; it should "
        "depend only on R_star",
        100.0 * (hi - lo) / lo);
    }

    free(nu);
    free(gm);
    free(rs);
  }

  /* Invalid inputs must be rejected rather than producing a NaN. */
  {
    const real_t nu[1] = {1.0f};
    const real_t good_g[1] = {0.5f};
    const real_t good_r[1] = {10.0f};

    const real_t bad_r[1] = {0.0f};
    real_t* n = sif_differential_number_density_bbks(
      nu, good_g, bad_r, 1, SIF_BBKS_G_FITTED);
    CHECK(n == NULL, "R_star = 0 was accepted");
    sif_free_aligned(n);

    const real_t bad_g[1] = {1.5f};
    n = sif_differential_number_density_bbks(
      nu, bad_g, good_r, 1, SIF_BBKS_G_FITTED);
    CHECK(n == NULL, "gamma > 1 was accepted");
    sif_free_aligned(n);

    n = sif_differential_number_density_bbks(
      NULL, good_g, good_r, 1, SIF_BBKS_G_FITTED);
    CHECK(n == NULL, "a NULL nu array was accepted");
    sif_free_aligned(n);
  }
}

/*
 * The whole reason to carry both forms: the fit and the defining integral have
 * to agree. They share no code and no coefficients -- one is a closed-form
 * approximation, the other a quadrature of eq. (18) -- so agreement is
 * simultaneous evidence that the five fitted coefficients were transcribed
 * correctly and that f(x) was.
 */
static void test_bbks_g_exact_vs_fit(void) {
  printf("BBKS G: fit vs defining integral\n");

  /* The band BBKS calibrated the fit on. */
  const double gammas[4] = {0.4, 0.5, 0.6, 0.7};

  double worst = 0.0;
  double worst_g = 0.0, worst_w = 0.0;

  for (int i = 0; i < 4; i++) {
    const real_t g = (real_t)gammas[i];

    for (double w = 0.5; w <= 8.0; w += 0.1) {
      const double fit = (double)sif_g_bbks(g, (real_t)w, SIF_BBKS_G_FITTED);
      const double exact = (double)sif_g_bbks(g, (real_t)w, SIF_BBKS_G_EXACT);

      CHECK(exact > 0.0, "gamma=%g: exact G(%g) = %g is not positive",
        gammas[i], w, exact);

      if (exact > 0.0) {
        const double rel = fabs(fit - exact) / exact;
        if (rel > worst) {
          worst = rel;
          worst_g = gammas[i];
          worst_w = w;
        }
      }
    }
  }

  printf("    worst relative gap over gamma in [0.4, 0.7], w in [0.5, 8]: "
         "%.3f%% at gamma=%.1f, w=%.1f\n",
    100.0 * worst, worst_g, worst_w);

  /* BBKS quote their fit as good to a few percent in this band. Anything
   * beyond ten would mean a transcription error rather than fit error. */
  CHECK(worst < 0.10,
    "fit and exact G differ by %.1f%% at gamma=%g, w=%g; one of them is wrong",
    100.0 * worst, worst_g, worst_w);

  /* Both must reach the same high-peak limit, eq. (20). */
  for (int i = 0; i < 4; i++) {
    const double w = 25.0;
    const double lead = w * w * w - 3.0 * gammas[i] * gammas[i] * w;
    const double exact = (double)sif_g_bbks((real_t)gammas[i], (real_t)w, SIF_BBKS_G_EXACT);
    const double rel = fabs(exact - lead) / lead;

    CHECK(rel < 1e-3,
      "gamma=%g: exact G(%g) = %g, high-peak limit %g (rel %.3e)", gammas[i], w,
      exact, lead, rel);
  }

  /* The exact form has no calibration band, so it must stay sane well outside
   * the one the fit was tuned on. */
  for (double g = 0.05; g < 0.99; g += 0.05) {
    for (double w = 0.0; w <= 6.0; w += 0.5) {
      const double v = (double)sif_g_bbks((real_t)g, (real_t)w, SIF_BBKS_G_EXACT);
      CHECK(v > 0.0 && v < 1e6, "exact G(gamma=%g, w=%g) = %g is out of range",
        g, w, v);
    }
  }

  /* Unlike the fit, the exact G stays positive below zero. */
  const double below = (double)sif_g_bbks(0.5f, -2.0f, SIF_BBKS_G_EXACT);
  CHECK(below > 0.0, "exact G at w = -2 is %g, expected a small positive",
    below);
  printf("    exact G(0.5, -2) = %.3e, fitted = %.3e\n", below,
    (double)sif_g_bbks(0.5f, -2.0f, SIF_BBKS_G_FITTED));
}

/*
 * The cumulative density has two limits that pin it down from opposite ends,
 * and both are independent of the quadrature that computes it.
 *
 * Far below zero the threshold stops selecting and the integral becomes the
 * total density of maxima, which for a Gaussian field is set by R_star alone.
 *
 * Far above it, G tends to (gamma nu)^3 - 3 gamma nu and the integral closes
 * in elementary form,
 *
 *   C -> exp(-nu_t^2/2) [gamma^3 (nu_t^2 + 2) - 3 gamma] / ((2 pi)^2 R_star^3)
 *
 * using integral_a^inf nu exp(-nu^2/2) = exp(-a^2/2) and
 * integral_a^inf nu^3 exp(-nu^2/2) = (a^2 + 2) exp(-a^2/2). Rewritten in terms
 * of sigma_1, sigma_2 and R this is Wu (2020) eq. (21), derived there by a
 * different route, so agreement checks the normalization against the paper
 * rather than against itself.
 */
static void test_bbks_cumulative(sif_option_t mode, const char* label) {
  printf("BBKS cumulative density (%s G)\n", label);

  /*
   * A P(k) ~ k^-2 power law with a Gaussian window. For a pure power law the
   * R dependence of gamma cancels and it reduces to
   * Gamma((5+n)/2) / sqrt(Gamma((3+n)/2) Gamma((7+n)/2)), which at n = -2 is
   * about 0.577 -- inside the band the BBKS fit was calibrated on, so both
   * forms of G are being used where they are meant to be. A narrow spectrum
   * would instead push gamma towards 1; test_bbks_fit_breaks_at_high_gamma
   * covers that case deliberately.
   */
  const real_t radii[1] = {20.0f};

  const uint32_t n_k = 1000;
  real_t* k_table = malloc(n_k * sizeof(real_t));
  real_t* p_table = malloc(n_k * sizeof(real_t));
  CHECK(k_table && p_table, "table alloc failed");
  if (!k_table || !p_table) {
    free(k_table);
    free(p_table);
    return;
  }

  for (uint32_t i = 0; i < n_k; i++) {
    const double lk = log(1e-4) + (log(1e1) - log(1e-4)) * i / (n_k - 1.0);
    k_table[i] = (real_t)exp(lk);
    p_table[i] = (real_t)pow(exp(lk), -2.0);
  }

  /* Build a real moment set through the public API so the test exercises the
   * same plumbing a caller would. */
  sif_delta_moments_t* m = sif_delta_moments_pk(
    k_table, p_table, n_k, radii, 1, 2, SIF_DELTA_FILTER_GAUSSIAN);
  CHECK(m != NULL, "moment set returned NULL");
  if (!m) {
    free(k_table);
    free(p_table);
    return;
  }

  /* Confirm the spectrum really did land the fit inside its band. */
  {
    real_t* gcheck = sif_gamma_moments(m);
    if (gcheck) {
      CHECK(gcheck[0] > 0.5f && gcheck[0] < 0.65f,
        "the test spectrum gives gamma = %g, outside the intended band",
        (double)gcheck[0]);
      sif_free_aligned(gcheck);
    }
  }

  real_t* gamma = sif_gamma_moments(m);
  real_t* r_star = sif_r_star_moments(m);
  const real_t* s0 = sif_delta_moments_sigma(m, 0);
  CHECK(gamma && r_star && s0, "derived quantities failed");

  if (gamma && r_star && s0) {
    const double g = (double)gamma[0];
    const double rs = (double)r_star[0];
    const double sigma0 = (double)s0[0];

    printf("    sigma_0=%.5g  gamma=%.4f  R_star=%.4g\n", sigma0, g, rs);

    /*
     * --- Limit 1: the threshold enters only through its magnitude. ---
     *
     * The integral runs from |delta|/sigma_0 upwards, so a void threshold and
     * a peak threshold of the same depth give the same count. That is the
     * maxima/minima symmetry of a Gaussian field, and it has to hold exactly
     * rather than approximately.
     */
    {
      const double depths[3] = {0.5, 2.0, 5.0};

      for (int t = 0; t < 3; t++) {
        real_t* neg = sif_cumulative_number_density_bbks(
          (real_t)(-depths[t] * sigma0), m, mode);
        real_t* pos = sif_cumulative_number_density_bbks(
          (real_t)(depths[t] * sigma0), m, mode);

        CHECK(neg != NULL && pos != NULL, "cumulative returned NULL");
        if (neg && pos) {
          CHECK(neg[0] == pos[0],
            "nu_t=%g: C(-delta) = %g but C(+delta) = %g; the threshold is not "
            "entering through its magnitude",
            depths[t], (double)neg[0], (double)pos[0]);
        }
        sif_free_aligned(neg);
        sif_free_aligned(pos);
      }

      /* At zero threshold the integral covers every maximum with nu > 0, so
       * it has to match the differential density integrated over the same
       * range -- a second integrator over the same integrand. */
      real_t* at_zero = sif_cumulative_number_density_bbks(0.0f, m, mode);
      CHECK(at_zero != NULL, "cumulative at delta = 0 returned NULL");

      if (at_zero) {
        const uint32_t n_nu = 20000;
        const double hi = 12.0, dnu = hi / (n_nu - 1);

        double integral = 0.0, prev = 0.0;
        for (uint32_t i = 0; i < n_nu; i++) {
          const double nu = i * dnu;
          const real_t nuf = (real_t)nu, gf = (real_t)g, rf = (real_t)rs;

          real_t* nd =
            sif_differential_number_density_bbks(&nuf, &gf, &rf, 1, mode);
          const double v = nd ? (double)nd[0] : 0.0;
          sif_free_aligned(nd);

          if (i > 0)
            integral += 0.5 * (v + prev) * dnu;
          prev = v;
        }

        const double rel =
          fabs(integral - (double)at_zero[0]) / (double)at_zero[0];
        printf("    delta = 0:  C = %.6e  vs  integral of N over nu > 0 = "
               "%.6e  (rel %.2e)\n",
          (double)at_zero[0], integral, rel);

        CHECK(rel < 1e-3,
          "C(0) = %g does not match the differential density integrated over "
          "nu > 0 (%g, rel %.3e)",
          (double)at_zero[0], integral, rel);

        sif_free_aligned(at_zero);
      }
    }

    /*
     * --- Limit 2: high threshold against the closed form. ---
     *
     * The high-peak form is the leading term of an expansion in 1/(gamma nu),
     * so the two must not agree to a fixed tolerance -- they must *diverge in
     * a specific way*. Dropping the next order leaves a relative error going
     * as (gamma nu)^-2, so the diagnostic quantity is rel * (gamma nu)^2,
     * which has to settle on a constant. That distinguishes "correct up to
     * the expansion's own truncation" from "wrong by a factor", which a plain
     * tolerance cannot do.
     *
     * The paper notes the expansion breaks down entirely once
     * (gamma nu)^3 < 3 gamma nu, i.e. gamma nu < sqrt(3), so the sweep starts
     * comfortably above that.
     */
    for (double nu_t = 5.0; nu_t <= 11.0; nu_t += 1.0) {
      const real_t delta = (real_t)(nu_t * sigma0);

      real_t* c = sif_cumulative_number_density_bbks(delta, m, mode);
      CHECK(c != NULL, "cumulative density returned NULL at nu_t=%g", nu_t);
      if (!c)
        continue;

      const double expected = exp(-0.5 * nu_t * nu_t) *
                              (g * g * g * (nu_t * nu_t + 2.0) - 3.0 * g) /
                              (4.0 * M_PI * M_PI * rs * rs * rs);

      const double got = (double)c[0];
      const double rel = fabs(got - expected) / expected;
      const double w = g * nu_t;
      const double scaled = rel * w * w;

      printf("    nu_t=%4.1f  gamma*nu=%.2f  C=%.4e  high-peak=%.4e  "
             "rel=%.2e  rel*(gamma nu)^2=%.2f\n",
        nu_t, w, got, expected, rel, scaled);

      CHECK(scaled > 1.0 && scaled < 4.0,
        "nu_t=%g: rel * (gamma nu)^2 = %.3f, expected an O(1) constant; the "
        "difference is not behaving like the expansion's truncation error",
        nu_t, scaled);

      sif_free_aligned(c);
    }

    /* --- Monotonicity: deepening the threshold cannot add peaks. --- */
    {
      double prev = 1e30;
      for (double nu_t = 0.0; nu_t <= 5.0; nu_t += 0.5) {
        real_t* c = sif_cumulative_number_density_bbks(
          (real_t)(nu_t * sigma0), m, mode);
        if (!c)
          continue;

        CHECK((double)c[0] <= prev * (1.0 + 1e-6),
          "C rose from %g to %g when the threshold increased to nu_t=%g", prev,
          (double)c[0], nu_t);
        prev = (double)c[0];
        sif_free_aligned(c);
      }
    }
  }

  /* An order-1 set has no gamma, so it cannot produce a cumulative density. */
  sif_delta_moments_t* low = sif_delta_moments_pk(
    k_table, p_table, n_k, radii, 1, 1, SIF_DELTA_FILTER_GAUSSIAN);
  if (low) {
    CHECK(sif_cumulative_number_density_bbks(0.5f, low, mode) == NULL,
      "an order-1 moment set produced a cumulative density");
    sif_delta_moments_free(low);
  }

  CHECK(sif_cumulative_number_density_bbks(0.5f, NULL, mode) == NULL,
    "a NULL moment set was accepted");

  sif_free_aligned(gamma);
  sif_free_aligned(r_star);
  sif_delta_moments_free(m);
  free(k_table);
  free(p_table);
}

/*
 * A narrow spectrum drives gamma towards 1, well outside the band the BBKS fit
 * was calibrated on. The exact G still returns the right total density of
 * maxima there; the fit does not. This is the concrete reason the exact form
 * exists, so it is worth pinning rather than leaving as advice in a comment.
 */
static void test_bbks_fit_breaks_at_high_gamma(void) {
  printf("BBKS fit vs exact outside the calibration band\n");

  const real_t radii[1] = {20.0f};
  const real_t k_narrow[3] = {0.01f, 0.1f, 1.0f};
  const real_t p_narrow[3] = {1.0f, 1.0f, 1.0f};

  sif_delta_moments_t* m = sif_delta_moments_pk(
    k_narrow, p_narrow, 3, radii, 1, 2, SIF_DELTA_FILTER_GAUSSIAN);
  CHECK(m != NULL, "moment set returned NULL");
  if (!m)
    return;

  real_t* gamma = sif_gamma_moments(m);
  real_t* r_star = sif_r_star_moments(m);
  const real_t* s0 = sif_delta_moments_sigma(m, 0);

  if (gamma && r_star && s0) {
    const double rs = (double)r_star[0];
    (void)rs;
    (void)s0;

    CHECK(gamma[0] > 0.9f, "the narrow spectrum gave gamma = %g, expected > 0.9",
      (double)gamma[0]);

    /* A zero threshold puts the whole integral at small w = gamma * nu, which
     * is exactly where the fit degrades once gamma leaves its band. */
    real_t* fitted =
      sif_cumulative_number_density_bbks(0.0f, m, SIF_BBKS_G_FITTED);
    real_t* exact =
      sif_cumulative_number_density_bbks(0.0f, m, SIF_BBKS_G_EXACT);

    if (fitted && exact) {
      const double tf = (double)fitted[0];
      const double te = (double)exact[0];

      printf("    gamma=%.4f  C(delta=0): fitted=%.6e  exact=%.6e  "
             "(%.1f%% apart)\n",
        (double)gamma[0], tf, te, 100.0 * fabs(tf - te) / te);

      CHECK(fabs(tf - te) / te > 0.05,
        "the fit and the exact form agree to %.2f%% at gamma = %g; the "
        "calibration-band warning may no longer be warranted",
        100.0 * fabs(tf - te) / te, (double)gamma[0]);
    }

    sif_free_aligned(fitted);
    sif_free_aligned(exact);
  }

  sif_free_aligned(gamma);
  sif_free_aligned(r_star);
  sif_delta_moments_free(m);
}

/*
 * The size function is the logarithmic derivative of the cumulative density,
 * so it is pinned by two identities that do not involve the differencing
 * scheme at all: integrating it back over ln R has to recover the drop in C,
 * and the two unit conventions have to differ by exactly a factor of R.
 */
static void test_bbks_size_function(sif_option_t mode, const char* label) {
  printf("BBKS size function (%s G)\n", label);

  const uint32_t n_k = 1000;
  real_t* k_table = malloc(n_k * sizeof(real_t));
  real_t* p_table = malloc(n_k * sizeof(real_t));
  if (!k_table || !p_table) {
    CHECK(0, "table alloc failed");
    free(k_table);
    free(p_table);
    return;
  }

  for (uint32_t i = 0; i < n_k; i++) {
    const double lk = log(1e-4) + (log(1e1) - log(1e-4)) * i / (n_k - 1.0);
    k_table[i] = (real_t)exp(lk);
    p_table[i] = (real_t)pow(exp(lk), -2.0);
  }

  const uint32_t n_r = 60;
  real_t* radii = malloc(n_r * sizeof(real_t));
  for (uint32_t i = 0; i < n_r; i++) {
    radii[i] = (real_t)exp(log(5.0) + (log(60.0) - log(5.0)) * i / (n_r - 1.0));
  }

  sif_delta_moments_t* m = sif_delta_moments_pk(
    k_table, p_table, n_k, radii, n_r, 2, SIF_DELTA_FILTER_GAUSSIAN);
  CHECK(m != NULL, "moment set returned NULL");

  if (m) {
    const real_t* s0 = sif_delta_moments_sigma(m, 0);
    const real_t delta = (real_t)(-1.5 * (double)s0[0]);

    real_t* c = sif_cumulative_number_density_bbks(delta, m, mode);
    sif_size_function_t* per_ln =
      sif_size_function_bbks(delta, m, mode | SIF_VSF_BIN_LN);
    sif_size_function_t* per_r =
      sif_size_function_bbks(delta, m, mode | SIF_VSF_BIN_LINEAR);

    CHECK(c && per_ln && per_r, "size function returned NULL");

    if (c && per_ln && per_r) {
      /* The container has to describe itself: bins, centres, and the units
       * the values are in. */
      CHECK(per_ln->n_bins == n_r, "n_bins is %u, expected %u",
        per_ln->n_bins, n_r);
      CHECK((per_ln->options & __SIF_VSF_BIN_MASK) == SIF_VSF_BIN_LN &&
              (per_r->options & __SIF_VSF_BIN_MASK) == SIF_VSF_BIN_LINEAR,
        "the binning convention was not recorded in options");
      CHECK(per_ln->r_min == radii[0] && per_ln->r_max == radii[n_r - 1],
        "r_min / r_max do not match the radius range");

      for (uint32_t r = 0; r < n_r; r++) {
        CHECK(per_ln->r_centers[r] == radii[r],
          "r_centers[%u] is %g, expected %g", r, (double)per_ln->r_centers[r],
          (double)radii[r]);
      }

      /* Reconstructed edges must bracket their centres and stay ordered. */
      for (uint32_t r = 0; r < n_r; r++) {
        CHECK(per_ln->r_edges[r] < per_ln->r_centers[r] &&
                per_ln->r_centers[r] < per_ln->r_edges[r + 1],
          "bin %u: centre %g is not inside [%g, %g]", r,
          (double)per_ln->r_centers[r], (double)per_ln->r_edges[r],
          (double)per_ln->r_edges[r + 1]);
      }

      /* A model counts nothing and carries no Poisson error. */
      for (uint32_t r = 0; r < n_r; r++) {
        CHECK(per_ln->counts[r] == 0 && per_ln->err[r] == 0.0f,
          "bin %u carries counts or an error a model cannot have", r);
      }

      /* A number density cannot be negative, and C is strictly falling here
       * so it cannot be zero either. */
      for (uint32_t r = 0; r < n_r; r++) {
        CHECK(per_ln->vsf[r] > 0.0f, "dn/dlnR at R=%g is %g, expected positive",
          (double)radii[r], (double)per_ln->vsf[r]);
      }

      /* The two conventions are related by dC/dR = (dC/dlnR) / R exactly. */
      double worst = 0.0;
      for (uint32_t r = 0; r < n_r; r++) {
        const double expect = (double)per_ln->vsf[r] / (double)radii[r];
        const double rel = fabs((double)per_r->vsf[r] - expect) / expect;
        if (rel > worst)
          worst = rel;
      }
      CHECK(worst < 1e-5,
        "SIF_VSF_BIN_LINEAR and SIF_VSF_BIN_LN differ by more than a factor "
        "of R (worst relative %.3e)",
        worst);

      /* Integrating the derivative back has to recover the drop in C. The
       * endpoints are one-sided, so this is a trapezoid check rather than an
       * identity, but a sign error or a missing factor would blow it apart. */
      double recovered = 0.0;
      for (uint32_t r = 1; r < n_r; r++) {
        const double dln = log((double)radii[r]) - log((double)radii[r - 1]);
        recovered +=
          0.5 * ((double)per_ln->vsf[r] + per_ln->vsf[r - 1]) * dln;
      }
      const double expected = (double)c[0] - (double)c[n_r - 1];
      const double rel = fabs(recovered - expected) / expected;

      printf("    integral of dn/dlnR = %.6e  vs  C(R_min) - C(R_max) = "
             "%.6e  (rel %.2e)\n",
        recovered, expected, rel);

      CHECK(rel < 0.02,
        "the size function does not integrate back to the drop in C "
        "(%.4e vs %.4e, rel %.3e)",
        recovered, expected, rel);
    }

    sif_free_aligned(c);
    sif_size_function_free(per_ln);
    sif_size_function_free(per_r);

    /* Guards: the derivative needs an ordered grid with more than one point. */
    real_t backwards[3] = {30.0f, 20.0f, 10.0f};
    sif_delta_moments_t* bad = sif_delta_moments_pk(
      k_table, p_table, n_k, backwards, 3, 2, SIF_DELTA_FILTER_GAUSSIAN);
    if (bad) {
      CHECK(sif_size_function_bbks(delta, bad, mode) == NULL,
        "a descending radius grid was accepted");
      sif_delta_moments_free(bad);
    }

    real_t single[1] = {20.0f};
    sif_delta_moments_t* one = sif_delta_moments_pk(
      k_table, p_table, n_k, single, 1, 2, SIF_DELTA_FILTER_GAUSSIAN);
    if (one) {
      CHECK(sif_size_function_bbks(delta, one, mode) == NULL,
        "a single-radius moment set was accepted");
      sif_delta_moments_free(one);
    }

    sif_delta_moments_free(m);
  }

  free(k_table);
  free(p_table);
  free(radii);
}

int main(void) {
  sif_fft_config_t fftcfg = {.skip_tuning = true};
  sif_config_t cfg = {.fft_config = &fftcfg, .omp_config = NULL,
                      .verbose = false, .log_level = SIF_LOG_LEVEL_ERROR};
  sif_init(&cfg);

  const uint32_t n = (uint32_t)SIF_TEST_SCALE(64);

  test_moments_match_field(n, SIF_DELTA_SHUFFLE_NONE, "unshuffled");
  test_moments_match_field(n, SIF_DELTA_SHUFFLE_PHASES, "phases");
  test_moments_match_field(n, SIF_DELTA_SHUFFLE_GAUSSIAN, "gaussian");
  test_shuffle_preserves_power(n);
  test_stats_api(n);
  test_shuffle_removes_skewness(n);
  test_radius_guards(n);
  test_pk_matches_field(n);
  test_pk_analytic();
  test_pk_guards();
  test_bbks_g_asymptote();
  test_bbks_number_density(SIF_BBKS_G_FITTED, "fitted");
  test_bbks_number_density(SIF_BBKS_G_EXACT, "exact");
  test_bbks_g_exact_vs_fit();
  test_bbks_cumulative(SIF_BBKS_G_FITTED, "fitted");
  test_bbks_cumulative(SIF_BBKS_G_EXACT, "exact");
  test_bbks_fit_breaks_at_high_gamma();
  test_bbks_size_function(SIF_BBKS_G_FITTED, "fitted");
  test_bbks_size_function(SIF_BBKS_G_EXACT, "exact");

  sif_finalize();

  if (failures) {
    printf("\n%d check(s) failed\n", failures);
    return 1;
  }

  printf("\nall delta stats checks passed\n");
  return 0;
}
