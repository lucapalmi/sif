#include "fft.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "core/get_system.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/random.h"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* Number of complex slots for an r2c transform. The matching in-place real
 * buffer is 2x this, with the last dimension padded to 2 * (n/2 + 1). */
static inline uint64_t __fft_complex_count(uint32_t n) {
  return (uint64_t)n * n * (uint64_t)(n / 2 + 1);
}

sif_fft_manager_t* sif_fft_manager_init(bool skip_tuning, const char* wisdom_dir) {
  sif_fft_manager_t* mgr = malloc(sizeof(sif_fft_manager_t));
  if (!mgr)
    return NULL;

  mgr->flags = skip_tuning ? FFTW_ESTIMATE : FFTW_MEASURE;
  mgr->wisdom_dir = wisdom_dir ? strdup(wisdom_dir) : NULL;

  if (wisdom_dir && !mgr->wisdom_dir) {
    SIF_LOG_WARNING("fft_manager", "failed to copy the wisdom directory path");
  }

  real_fftw_init_threads();

  return mgr;
}

void sif_fft_manager_finalize(sif_fft_manager_t* mgr) {
  if (!mgr)
    return;

  free(mgr->wisdom_dir);

  /* NOTE: this tears down FFTW state process-wide, not just sif's. Anything
   * else in the same process using FFTW loses its plans and accumulated
   * wisdom. Acceptable while sif owns FFTW initialization, but it is the
   * reason sif_finalize must not be called from a library context. */
  real_fftw_cleanup_threads();
  real_fftw_cleanup();
  free(mgr);
}

sif_fft_workspace_t* sif_fft_workspace_alloc(sif_fft_manager_t* mgr, uint32_t n_cells) {
  if (!mgr || n_cells == 0) {
    SIF_LOG_ERROR("fft_context", "invalid manager or grid size");
    return NULL;
  }

  sif_fft_workspace_t* ws = malloc(sizeof(sif_fft_workspace_t));
  if (!ws)
    return NULL;

  ws->n_cells = n_cells;
  ws->mgr = mgr;
  ws->delta_k = NULL;
  ws->delta_k_cpy = NULL;
  ws->forward_plan = NULL;
  ws->backward_plan = NULL;

  if (mgr->wisdom_dir) {
    char specific_wisdom[512];
    snprintf(specific_wisdom, sizeof(specific_wisdom), "%s/grid_%u.wisdom",
      mgr->wisdom_dir, n_cells);
    if (real_fftw_import_wisdom_from_filename(specific_wisdom) != 0) {
      SIF_LOG_INFO("fft_manager", "loaded wisdom file %s", specific_wisdom);
    }
  }

  const uint64_t complex_cells = __fft_complex_count(n_cells);

  ws->delta_k = real_fftw_malloc(complex_cells * sizeof(sif_real_complex_t));
  if (!ws->delta_k) {
    SIF_LOG_ERROR("fft_context",
      "failed to allocate the spectrum buffer (%" PRIu64 " bytes)",
      complex_cells * sizeof(sif_real_complex_t));
    free(ws);
    return NULL;
  }

  /* The forward plan is created on first use, see __fft_ensure_forward_plan. */
  return ws;
}

/*
 * Plans the forward transform against the buffers it will actually run on.
 *
 * Deferring the plan to the first transform is what keeps the workspace down
 * to a single spectrum-sized allocation. FFTW_MEASURE overwrites its planning
 * input, so planning at construction time needed a throwaway scratch buffer as
 * large as the spectrum itself -- a second 42 GiB at n_cells = 2250, alive at
 * exactly the moment the caller is still holding the density field. Planning
 * here instead lets the caller's own field be the input, at the price of
 * restricting the planner to modes that leave that input intact: a tuned plan
 * when wisdom for this size exists, FFTW_ESTIMATE otherwise.
 *
 * That trade is one-sided in our favour. The forward transform runs once per
 * workspace, and FFTW_MEASURE pays for its better plan by executing the
 * transform repeatedly while planning -- more than an untuned plan costs for a
 * single execution. The backward plan, which runs once per radius, still
 * measures (it plans against its own freshly allocated buffer).
 *
 * FFTW_WISDOM_ONLY returns NULL rather than measuring when no wisdom applies,
 * so neither branch can touch `in`.
 */
static int __fft_ensure_forward_plan(
  sif_fft_workspace_t* ws, const real_t* in) {

  if (ws->forward_plan)
    return SIF_OK;

  const uint32_t n = ws->n_cells;

  real_fftw_plan_with_nthreads(sif_system_get_max_threads());

  ws->forward_plan = real_fftw_plan_dft_r2c_3d(n, n, n, (real_t*)in,
    ws->delta_k, ws->mgr->flags | FFTW_WISDOM_ONLY);

  if (!ws->forward_plan) {
    ws->forward_plan = real_fftw_plan_dft_r2c_3d(
      n, n, n, (real_t*)in, ws->delta_k, FFTW_ESTIMATE);
  }

  if (!ws->forward_plan) {
    SIF_LOG_ERROR("fft_context", "failed to create the forward plan");
    return SIF_ERR_ALLOC;
  }

  return SIF_OK;
}

int sif_fft_workspace_init_backward(sif_fft_workspace_t* ws, sif_fft_manager_t* mgr) {
  if (!ws || !mgr)
    return SIF_ERR_INVALID;

  if (ws->backward_plan)
    return SIF_OK;

  const uint64_t complex_cells = __fft_complex_count(ws->n_cells);

  ws->delta_k_cpy = sif_malloc_aligned(complex_cells * sizeof(sif_real_complex_t));
  if (!ws->delta_k_cpy) {
    SIF_LOG_ERROR("fft_context",
      "failed to allocate the real-space buffer (%" PRIu64 " bytes)",
      complex_cells * sizeof(sif_real_complex_t));
    return SIF_ERR_ALLOC;
  }

  ws->backward_plan = real_fftw_plan_dft_c2r_3d(ws->n_cells, ws->n_cells,
    ws->n_cells, ws->delta_k_cpy, (real_t*)ws->delta_k_cpy, mgr->flags);

  if (!ws->backward_plan) {
    SIF_LOG_ERROR("fft_context", "failed to create the backward plan");
    sif_free_aligned(ws->delta_k_cpy);
    ws->delta_k_cpy = NULL;
    return SIF_ERR_ALLOC;
  }

  return SIF_OK;
}

real_t* sif_fft_workspace_take_real_buffer(sif_fft_workspace_t* ws) {
  if (!ws || !ws->delta_k_cpy)
    return NULL;

  real_t* buffer = (real_t*)ws->delta_k_cpy;
  ws->delta_k_cpy = NULL;

  /* The plan points at a buffer we no longer own, so it must not be reused. */
  if (ws->backward_plan) {
    real_fftw_destroy_plan(ws->backward_plan);
    ws->backward_plan = NULL;
  }

  return buffer;
}

void sif_fft_workspace_free(sif_fft_workspace_t* ws) {
  if (!ws) {
    SIF_LOG_WARNING("fft_context", "cannot free a NULL workspace");
    return;
  }

  if (ws->mgr && ws->mgr->wisdom_dir) {
    char specific_wisdom[512];
    snprintf(specific_wisdom, sizeof(specific_wisdom), "%s/grid_%u.wisdom",
      ws->mgr->wisdom_dir, ws->n_cells);
    if (real_fftw_export_wisdom_to_filename(specific_wisdom) != 0) {
      SIF_LOG_INFO("fft_manager", "exported wisdom to %s", specific_wisdom);
    } else {
      SIF_LOG_WARNING(
        "fft_manager", "failed to export wisdom to %s", specific_wisdom);
    }
  }

  if (ws->forward_plan)
    real_fftw_destroy_plan(ws->forward_plan);
  if (ws->backward_plan)
    real_fftw_destroy_plan(ws->backward_plan);
  if (ws->delta_k)
    real_fftw_free(ws->delta_k);
  if (ws->delta_k_cpy)
    sif_free_aligned(ws->delta_k_cpy);

  free(ws);
}

static inline uint64_t __get_flat_complex_index(
  uint32_t ix, uint32_t iy, uint32_t iz, uint32_t n_cells) {
  uint32_t z_dim = n_cells / 2 + 1;
  return (uint64_t)ix * n_cells * z_dim + (uint64_t)iy * z_dim + (uint64_t)iz;
}

/*
 * Builds the radial filter lookup table, indexed directly by the integer |k|^2.
 * Shared by sif_fft_apply_filter and fft_filtered_variance so the two can never
 * disagree about what the filter actually is.
 */
static real_t* __fft_build_filter_lut(
  sif_filter_type_t filter, real_t r, real_t box_length, uint32_t n_cells) {

  const uint32_t N_half = n_cells >> 1;
  const uint32_t max_k2 = 3 * N_half * N_half;

  real_t* lut = malloc(((size_t)max_k2 + 1) * sizeof(real_t));
  if (!lut) {
    SIF_LOG_ERROR("fft_context",
      "failed to allocate the filter lookup table (%zu bytes)",
      ((size_t)max_k2 + 1) * sizeof(real_t));
    return NULL;
  }

  const real_t factor = (real_t)(r * 2.0 * M_PI / box_length);
  const real_t factor2 = factor * factor;

#pragma omp parallel for schedule(static)
  for (uint32_t k2 = 0; k2 <= max_k2; k2++) {
    if (k2 == 0) {
      lut[k2] = 1.0f;
    } else if (filter == FILTER_TOP_HAT) {
      real_t kr = factor * REAL_SQRT((real_t)k2);
      if (kr > 1e-4f) {
        lut[k2] = 3.0f * (REAL_SIN(kr) - kr * REAL_COS(kr)) / (kr * kr * kr);
      } else {
        lut[k2] = 1.0f;
      }
    } else {
      lut[k2] = REAL_EXP(-0.5f * factor2 * (real_t)k2);
    }
  }

  return lut;
}

int sif_fft_apply_filter(
  sif_fft_workspace_t* ws, sif_filter_type_t filter, real_t r, real_t box_length) {

  if (!ws || !ws->delta_k_cpy) {
    SIF_LOG_ERROR("fft_context",
      "the backward stage must be initialized before filtering");
    return SIF_ERR_INVALID;
  }

  const uint64_t complex_cells = __fft_complex_count(ws->n_cells);

  if (filter == FILTER_NONE) {
    memcpy(
      ws->delta_k_cpy, ws->delta_k, complex_cells * sizeof(sif_real_complex_t));
    SIF_LOG_TRACE("fft_context", "no filter applied");
    return SIF_OK;
  }

  if (filter != FILTER_TOP_HAT && filter != FILTER_GAUSSIAN) {
    /* Returning without writing delta_k_cpy would leave the backward transform
     * operating on whatever was there before, so this has to be an error. */
    SIF_LOG_ERROR("fft_context", "filter type %d is not supported", filter);
    return SIF_ERR_INVALID;
  }

  const uint32_t N = ws->n_cells;
  const uint32_t N_half = N >> 1;

  real_t* lut = __fft_build_filter_lut(filter, r, box_length, N);
  if (!lut)
    return SIF_ERR_ALLOC;

#pragma omp parallel for schedule(static)
  for (uint32_t ix = 0; ix < N; ix++) {
    int32_t kx = (ix > N_half) ? (int32_t)ix - (int32_t)N : (int32_t)ix;
    uint32_t kx2 = (uint32_t)(kx * kx);

    for (uint32_t iy = 0; iy < N; iy++) {
      int32_t ky = (iy > N_half) ? (int32_t)iy - (int32_t)N : (int32_t)iy;
      uint32_t kxy2 = kx2 + (uint32_t)(ky * ky);

      for (uint32_t iz = 0; iz <= N_half; iz++) {
        uint32_t k2 = kxy2 + (iz * iz);

        uint64_t idx = __get_flat_complex_index(ix, iy, iz, N);
        real_t smoothing = lut[k2];

        ws->delta_k_cpy[idx][0] = ws->delta_k[idx][0] * smoothing;
        ws->delta_k_cpy[idx][1] = ws->delta_k[idx][1] * smoothing;
      }
    }
  }

  free(lut);

  SIF_LOG_TRACE("fft_context", "%s smoothing applied",
    filter == FILTER_TOP_HAT ? "top hat" : "gaussian");

  return SIF_OK;
}

int sif_fft_grid_forward(sif_fft_workspace_t* ws, const sif_grid_t* grid) {
  if (!ws || !ws->delta_k || !grid || !grid->delta) {
    SIF_LOG_ERROR("fft_context", "invalid workspace or grid");
    return SIF_ERR_INVALID;
  }

  if (__fft_ensure_forward_plan(ws, grid->delta) != SIF_OK)
    return SIF_ERR_ALLOC;

  real_fftw_execute_dft_r2c(
    ws->forward_plan, (real_t*)grid->delta, ws->delta_k);
  SIF_LOG_TRACE("fft_context", "forward fft on cubic grid executed");

  return SIF_OK;
}

real_t* sif_fft_grid_backward(sif_fft_workspace_t* ws) {
  if (!ws || !ws->backward_plan || !ws->delta_k_cpy) {
    SIF_LOG_ERROR("fft_context", "the backward stage is not initialized");
    return NULL;
  }

  real_fftw_execute_dft_c2r(
    ws->backward_plan, ws->delta_k_cpy, (real_t*)ws->delta_k_cpy);

  real_t* data = (real_t*)ws->delta_k_cpy;

  const uint64_t n = ws->n_cells;
  const uint64_t n_padded = 2 * (n / 2 + 1);
  const uint64_t total_cells = n * n * n;
  const real_t norm = 1.0f / (real_t)total_cells;

  /* An in-place r2c/c2r pair always pads the last dimension, so n_padded > n
   * unconditionally and the rows have to be compacted.
   *
   * The compaction cannot be parallelized row-wise: row r writes into
   * [r*n, (r+1)*n) while row r-1 reads from [(r-1)*n_padded, +n), and those
   * overlap. Sequentially it is safe because the write cursor never overtakes
   * the read cursor. So compact serially with memmove (bandwidth-bound, no
   * per-element loop) and do the arithmetic in a separate parallel pass. */
  for (uint64_t row = 1; row < n * n; row++) {
    memmove(data + row * n, data + row * n_padded, n * sizeof(real_t));
  }

#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < total_cells; i++) {
    data[i] *= norm;
  }

  SIF_LOG_TRACE("fft_context", "backward fft on cubic grid executed");

  return data;
}

/* --- CIC window deconvolution --- */

/* sinc(x) = sin(x)/x, continuous at 0. */
static inline double __fft_sinc(double x) {
  if (x < 1e-9 && x > -1e-9)
    return 1.0;
  return sin(x) / x;
}

int sif_fft_deconvolve_cic(sif_fft_workspace_t* ws) {
  if (!ws || !ws->delta_k) {
    SIF_LOG_ERROR("fft_context", "no spectrum to deconvolve");
    return SIF_ERR_INVALID;
  }

  const uint32_t N = ws->n_cells;
  const uint32_t N_half = N >> 1;
  const uint32_t z_dim = N / 2 + 1;

  /* The window is separable, so the per-axis factor only depends on that
   * axis's index. One table of N entries covers all three. */
  double* axis = malloc((size_t)N * sizeof(double));
  if (!axis) {
    SIF_LOG_ERROR("fft_context", "failed to allocate the CIC window table");
    return SIF_ERR_ALLOC;
  }

  for (uint32_t i = 0; i < N; i++) {
    int32_t k = (i > N_half) ? (int32_t)i - (int32_t)N : (int32_t)i;
    double s = __fft_sinc(M_PI * (double)k / (double)N);
    axis[i] = s * s; /* CIC is the square of the NGP window */
  }

#pragma omp parallel for schedule(static)
  for (uint32_t ix = 0; ix < N; ix++) {
    const double wx = axis[ix];
    for (uint32_t iy = 0; iy < N; iy++) {
      const double wxy = wx * axis[iy];
      for (uint32_t iz = 0; iz < z_dim; iz++) {
        const double w = wxy * axis[iz];
        const double inv = 1.0 / w;

        uint64_t idx = __get_flat_complex_index(ix, iy, iz, N);
        ws->delta_k[idx][0] = (real_t)(ws->delta_k[idx][0] * inv);
        ws->delta_k[idx][1] = (real_t)(ws->delta_k[idx][1] * inv);
      }
    }
  }

  free(axis);

  SIF_LOG_TRACE("fft_context", "CIC window deconvolved");
  return SIF_OK;
}

/* --- Phase randomization --- */

/*
 * Per-mode seeding. Drawing from a sequential stream would make the result
 * depend on how the loop is scheduled across threads; hashing the flat index
 * into the seed instead makes it a pure function of (seed, n_cells).
 */
static inline void __fft_seed_mode(
  sif_prng_state_t* prng, uint64_t seed, uint64_t idx) {
  uint64_t z = seed + idx * 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  sif_prng_init(prng, z ^ (z >> 31));
}

/*
 * Amplitude for a complex mode. |delta_k|^2 of a Gaussian field is exponential
 * with mean equal to the power, so scaling by sqrt(-ln u) turns a fixed
 * amplitude into a correctly distributed Rayleigh one. next_real returns
 * [0, 1), so the draw is taken as 1 - u to keep the log finite.
 */
static inline real_t __fft_amp_complex(
  real_t a, sif_prng_state_t* prng, bool resample) {
  if (!resample)
    return a;
  double u = 1.0 - (double)sif_prng_next_real(prng);
  return (real_t)(a * sqrt(-log(u)));
}

/*
 * Amplitude for a self-conjugate (necessarily real) mode. Those carry the full
 * power in a single real degree of freedom rather than splitting it across a
 * real and an imaginary part, so the Gaussian counterpart is a standard normal
 * scaling, not a Rayleigh one.
 */
static inline real_t __fft_amp_real(
  real_t a, sif_prng_state_t* prng, bool resample) {
  if (!resample)
    return a;
  double u1 = 1.0 - (double)sif_prng_next_real(prng);
  double u2 = (double)sif_prng_next_real(prng);
  return (real_t)(a * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

int sif_fft_randomize_phases(
  sif_fft_workspace_t* ws, uint64_t seed, bool resample_amplitudes) {

  if (!ws || !ws->delta_k) {
    SIF_LOG_ERROR("fft_context", "no spectrum to randomize");
    return SIF_ERR_INVALID;
  }

  const uint32_t N = ws->n_cells;
  const uint32_t z_dim = N / 2 + 1;
  const double two_pi = 2.0 * M_PI;

  /*
   * Pass 1: the interior planes, 0 < k_z < N/2. Their conjugate partners live
   * at k_z' = N - k_z, which the r2c layout does not store, so every entry
   * here is an independent degree of freedom and can be overwritten directly.
   */
#pragma omp parallel for schedule(static)
  for (uint32_t ix = 0; ix < N; ix++) {
    for (uint32_t iy = 0; iy < N; iy++) {
      for (uint32_t iz = 0; iz < z_dim; iz++) {
        /* Skip the self-redundant planes; pass 2 owns them. */
        if ((2 * iz) % N == 0)
          continue;

        uint64_t idx = __get_flat_complex_index(ix, iy, iz, N);

        real_t re = ws->delta_k[idx][0];
        real_t im = ws->delta_k[idx][1];
        real_t amp = (real_t)sqrt((double)re * re + (double)im * im);

        sif_prng_state_t prng;
        __fft_seed_mode(&prng, seed, idx);

        double phase = two_pi * (double)sif_prng_next_real(&prng);
        amp = __fft_amp_complex(amp, &prng, resample_amplitudes);

        ws->delta_k[idx][0] = (real_t)(amp * cos(phase));
        ws->delta_k[idx][1] = (real_t)(amp * sin(phase));
      }
    }
  }

  /*
   * Pass 2: the k_z = 0 plane and, when N is even, the k_z = N/2 plane. Both
   * store k and -k, so they are internally redundant: (ix, iy) pairs with
   * (N-ix, N-iy) mod N and must be its conjugate. Getting this wrong is not
   * loud -- c2r simply discards the inconsistent imaginary part and the
   * surrogate quietly loses power -- so the pairing is explicit here.
   *
   * Exactly one member of each pair is canonical and writes both entries, so
   * there is no write-write race despite the parallel loop.
   */
  for (uint32_t iz = 0; iz < z_dim; iz++) {
    if ((2 * iz) % N != 0)
      continue;

#pragma omp parallel for schedule(static)
    for (uint32_t ix = 0; ix < N; ix++) {
      const uint32_t jx = (N - ix) % N;

      for (uint32_t iy = 0; iy < N; iy++) {
        const uint32_t jy = (N - iy) % N;

        /* Canonical representative of the {k, -k} pair. */
        if (!(ix < jx || (ix == jx && iy <= jy)))
          continue;

        uint64_t idx = __get_flat_complex_index(ix, iy, iz, N);
        uint64_t jdx = __get_flat_complex_index(jx, jy, iz, N);

        real_t re = ws->delta_k[idx][0];
        real_t im = ws->delta_k[idx][1];
        real_t amp = (real_t)sqrt((double)re * re + (double)im * im);

        sif_prng_state_t prng;
        __fft_seed_mode(&prng, seed, idx);

        if (idx == jdx) {
          /* Self-conjugate: k == -k, so the mode has to stay real. The k = 0
           * mode is the mean of the field and is left alone. */
          if (idx == 0)
            continue;

          real_t a = __fft_amp_real(amp, &prng, resample_amplitudes);
          if (!resample_amplitudes && sif_prng_next_real(&prng) < 0.5f)
            a = -a; /* a fixed-amplitude real mode still gets a random sign */

          ws->delta_k[idx][0] = a;
          ws->delta_k[idx][1] = 0.0f;
          continue;
        }

        double phase = two_pi * (double)sif_prng_next_real(&prng);
        amp = __fft_amp_complex(amp, &prng, resample_amplitudes);

        real_t new_re = (real_t)(amp * cos(phase));
        real_t new_im = (real_t)(amp * sin(phase));

        ws->delta_k[idx][0] = new_re;
        ws->delta_k[idx][1] = new_im;
        ws->delta_k[jdx][0] = new_re;
        ws->delta_k[jdx][1] = -new_im;
      }
    }
  }

  SIF_LOG_TRACE("fft_context", "phases randomized (%s amplitudes)",
    resample_amplitudes ? "resampled" : "preserved");
  return SIF_OK;
}

/* --- Fourier-space spectral moments --- */

int sif_fft_spectral_moments(const sif_fft_workspace_t* ws, sif_filter_type_t filter,
  real_t r, real_t box_length, uint8_t max_order, uint64_t n_tracers,
  double* sigma_sq, double* high_k_fraction) {

  if (!ws || !ws->delta_k || !sigma_sq) {
    SIF_LOG_ERROR("fft_context", "no spectrum to evaluate");
    return SIF_ERR_INVALID;
  }

  if (max_order > FFT_MAX_MOMENT_ORDER) {
    SIF_LOG_ERROR("fft_context", "moment order %u exceeds the maximum of %d",
      max_order, FFT_MAX_MOMENT_ORDER);
    return SIF_ERR_INVALID;
  }

  const uint32_t N = ws->n_cells;
  const uint32_t N_half = N >> 1;
  const uint32_t z_dim = N / 2 + 1;
  const int n_moments = (int)max_order + 1;

  real_t* lut = NULL;
  if (filter != FILTER_NONE) {
    lut = __fft_build_filter_lut(filter, r, box_length, N);
    if (!lut)
      return SIF_ERR_ALLOC;
  }

  /*
   * Three accumulators per order -- the measured sum, the window sum behind
   * the shot-noise term, and the part of the measured sum above half Nyquist.
   * Per-thread scratch rather than an OpenMP array reduction, which would
   * require a newer OpenMP than the rest of the library assumes.
   */
  const int n_threads =
    sif_system_get_max_threads() > 0 ? sif_system_get_max_threads() : 1;
  const size_t stride = 3 * (size_t)n_moments;

  double* scratch = calloc((size_t)n_threads * stride, sizeof(double));
  if (!scratch) {
    SIF_LOG_ERROR("fft_context", "failed to allocate the moment scratch");
    free(lut);
    return SIF_ERR_ALLOC;
  }

  /* Isotropic cut at half the Nyquist frequency, in integer mode units. */
  const double high_k2 = (double)N * (double)N / 16.0;

#pragma omp parallel num_threads(n_threads)
  {
    double* local = scratch + (size_t)sif_system_get_thread_num() * stride;
    double* l_sum = local;
    double* l_win = local + n_moments;
    double* l_high = local + 2 * n_moments;

#pragma omp for schedule(static)
    for (uint32_t ix = 0; ix < N; ix++) {
      int32_t kx = (ix > N_half) ? (int32_t)ix - (int32_t)N : (int32_t)ix;
      uint32_t kx2 = (uint32_t)(kx * kx);

      for (uint32_t iy = 0; iy < N; iy++) {
        int32_t ky = (iy > N_half) ? (int32_t)iy - (int32_t)N : (int32_t)iy;
        uint32_t kxy2 = kx2 + (uint32_t)(ky * ky);

        for (uint32_t iz = 0; iz < z_dim; iz++) {
          const uint32_t k2i = kxy2 + iz * iz;

          /* Excluding k = 0 is what makes the j = 0 sum a variance rather
           * than a mean square: the k = 0 term is exactly N^6 <delta>^2. */
          if (k2i == 0)
            continue;

          uint64_t idx = __get_flat_complex_index(ix, iy, iz, N);

          double w = lut ? (double)lut[k2i] : 1.0;
          double re = (double)ws->delta_k[idx][0];
          double im = (double)ws->delta_k[idx][1];

          /* The half-complex layout stores both k and -k on the
           * self-redundant planes but only one of the pair elsewhere, so the
           * interior planes stand in for two physical modes each. */
          double weight = ((2 * iz) % N == 0) ? 1.0 : 2.0;

          const double k2 = (double)k2i;
          const double windowed = weight * w * w;
          const double power = windowed * (re * re + im * im);
          const bool high = k2 > high_k2;

          double k2j = 1.0;
          for (int j = 0; j < n_moments; j++) {
            l_sum[j] += k2j * power;
            l_win[j] += k2j * windowed;
            if (high)
              l_high[j] += k2j * power;
            k2j *= k2;
          }
        }
      }
    }
  }

  free(lut);

  const double total = (double)N * (double)N * (double)N;

  /* k_physical = (2*pi/L) * sqrt(k2_integer), so the k^2j weight carries
   * (2*pi/L)^2j once the integer sum is done. */
  const double k_unit2 =
    (2.0 * M_PI / (double)box_length) * (2.0 * M_PI / (double)box_length);

  double k_scale = 1.0;

  for (int j = 0; j < n_moments; j++) {
    double sum = 0.0, win = 0.0, high = 0.0;

    for (int t = 0; t < n_threads; t++) {
      const double* local = scratch + (size_t)t * stride;
      sum += local[j];
      win += local[n_moments + j];
      high += local[2 * n_moments + j];
    }

    const double norm = k_scale / (total * total);
    const double raw = sum * norm;

    /* Poisson noise contributes a flat |delta_k|^2 = N^6 / n_tracers to every
     * mode, so after the 1/N^6 normalization the subtraction is simply the
     * window sum over n_tracers. Left out of `raw` so that high_k_fraction
     * still describes where the measured power actually sits. */
    const double shot = n_tracers > 0 ? win * norm / (double)n_tracers : 0.0;

    sigma_sq[j] = raw - shot;
    if (high_k_fraction)
      high_k_fraction[j] = raw > 0.0 ? high * norm / raw : 0.0;

    k_scale *= k_unit2;
  }

  free(scratch);

  return SIF_OK;
}
