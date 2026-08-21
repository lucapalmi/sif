/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "fft.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "core/system_internal.h"
#include "sif/core/settings.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "sif/utils/random.h"

/* Number of complex slots for an r2c transform. The matching in-place real
 * buffer is 2x this, with the last dimension padded to 2 * (n/2 + 1). */
static inline uint64_t fft_complex_count(uint32_t n) {
  return (uint64_t)n * n * (uint64_t)(n / 2 + 1);
}

/* Room for a wisdom path. Overlong directories are truncated by snprintf and
 * then simply fail to open, which is the right outcome: no wisdom, no crash. */
#define PATH_CAP 512

sif_fft_manager_t* sif__fft_manager_init(
  bool skip_tuning, const char* wisdom_dir) {
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

void sif__fft_manager_finalize(sif_fft_manager_t* mgr) {
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

/* FFTW_MEASURE is 0, so "is this workspace tuning?" can only be asked the
 * other way round. */
static inline bool fft_is_tuned(const sif_fft_workspace_t* ws) {
  return (ws->plan_flags & FFTW_ESTIMATE) == 0;
}

/*
 * Whether this workspace is allowed to tune, and the one place that decides.
 *
 * Tuning is asked for by the manager and granted here, subject to a size cap
 * (see SIF__FFT_TUNING_MAX_GIB_DEFAULT). It has to be one decision for the
 * whole workspace rather than one per plan: a cap that took tuning away from
 * only the transform that happened to notice it would put back exactly the
 * forward/backward split this file went to some trouble to remove.
 */
static unsigned int fft_resolve_plan_flags(
  const sif_fft_manager_t* mgr, uint32_t n_cells, uint64_t spectrum_bytes) {

  /* Not tuning in the first place; the cap has nothing to take away. */
  if (mgr->flags & FFTW_ESTIMATE)
    return mgr->flags;

  const char* raw =
    sif_setting_get("fft_tuning_max_gib", SIF__FFT_TUNING_MAX_GIB_DEFAULT);

  char* end = NULL;
  double cap_gib = raw ? strtod(raw, &end) : -1.0;

  /* A malformed cap must not read as "no limit": that is the failure mode the
   * cap exists to prevent. Fall back to the documented default instead. */
  if (!raw || end == raw || !(cap_gib >= 0.0)) {
    cap_gib = strtod(SIF__FFT_TUNING_MAX_GIB_DEFAULT, NULL);
    SIF_LOG_WARNING("fft_context",
      "fft_tuning_max_gib is not a non-negative number ('%s'); using %g",
      raw ? raw : "(unset)", cap_gib);
  }

  const double gib = (double)spectrum_bytes / (1024.0 * 1024.0 * 1024.0);
  if (gib <= cap_gib)
    return mgr->flags;

  SIF_LOG_INFO("fft_context",
    "a %u^3 spectrum is %.1f GiB, over the %g GiB fft_tuning_max_gib ceiling; "
    "planning both transforms with FFTW_ESTIMATE. Tuning at this size costs "
    "far more in planning than it returns over a run",
    n_cells, gib, cap_gib);

  return FFTW_ESTIMATE;
}

/*
 * The wisdom file, and why there is only one of it.
 *
 * FFTW's export writes the whole process-global wisdom, not the transform just
 * planned, so the per-grid-size files this used to keep were never per-size in
 * content: each was a full snapshot under a name that described only whichever
 * size happened to finish last. Two sizes in one process wrote two files with
 * the same contents, and a later single-size run overwrote both with less.
 *
 * One cumulative file is what the contents already were. It is re-imported
 * immediately before every write so that a concurrent job -- a job array
 * pointed at one $HOME is the normal case here -- contributes rather than
 * collides: the file that lands is the union of what was on disk and what this
 * process learned, and the rename makes the swap atomic.
 */
static void fft_wisdom_path(
  const char* wisdom_dir, char* out, size_t out_size) {
  snprintf(out, out_size, "%s/%s", wisdom_dir, SIF__FFT_WISDOM_FILE);
}

static void fft_load_wisdom(const sif_fft_manager_t* mgr) {
  if (!mgr || !mgr->wisdom_dir)
    return;

  char path[PATH_CAP];
  fft_wisdom_path(mgr->wisdom_dir, path, sizeof(path));

  if (real_fftw_import_wisdom_from_filename(path) != 0)
    SIF_LOG_INFO("fft_manager", "loaded wisdom from %s", path);
}

/*
 * Writes the accumulated wisdom out.
 *
 * Through a temporary and a rename, for the same reason the settings table and
 * the CIC cache do it: a job array finishes its ranks at roughly the same
 * moment, all of them pointed at one $HOME, and a half-written wisdom file is
 * not merely lost -- FFTW refuses to import it, so every later run silently
 * falls back to an untuned plan until somebody deletes it by hand.
 *
 * Called as soon as a plan has been measured rather than at teardown. Measuring
 * a large transform is the expensive artifact of the whole run -- minutes of it
 * -- and holding it in memory until the workspace is released means a job that
 * is killed, or that dies later for unrelated reasons, throws away everything
 * it paid for and the next run starts from nothing.
 */
static void fft_save_wisdom(const sif_fft_manager_t* mgr) {
  if (!mgr || !mgr->wisdom_dir)
    return;

  char final_path[PATH_CAP];
  char tmp_path[PATH_CAP + 32];

  fft_wisdom_path(mgr->wisdom_dir, final_path, sizeof(final_path));

  int written = snprintf(
    tmp_path, sizeof(tmp_path), "%s.tmp.%ld", final_path, (long)getpid());
  if (written < 0 || (size_t)written >= sizeof(tmp_path))
    return;

  /* Merge whatever another process has contributed since we last looked, so
   * this write adds to the file instead of replacing it. */
  real_fftw_import_wisdom_from_filename(final_path);

  /* FFTW's export returns non-zero on success, unlike most of C. */
  if (real_fftw_export_wisdom_to_filename(tmp_path) == 0) {
    SIF_LOG_WARNING("fft_manager", "failed to export wisdom to %s", tmp_path);
    remove(tmp_path);
    return;
  }

  if (rename(tmp_path, final_path) != 0) {
    SIF_LOG_WARNING(
      "fft_manager", "failed to install the wisdom file %s", final_path);
    remove(tmp_path);
    return;
  }

  SIF_LOG_TRACE("fft_manager", "wisdom saved to %s", final_path);
}

sif_fft_workspace_t* sif__fft_workspace_alloc(
  sif_fft_manager_t* mgr, uint32_t n_cells) {
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
  ws->cic_inv = NULL;
  ws->forward_plan = NULL;
  ws->backward_plan = NULL;

  const uint64_t complex_cells = fft_complex_count(n_cells);
  const uint64_t spectrum_bytes = complex_cells * sizeof(sif_real_complex);

  /* Fixed here, for both transforms, so that the workspace has one planner
   * quality rather than one per plan site. */
  ws->plan_flags = fft_resolve_plan_flags(mgr, n_cells, spectrum_bytes);

  /* After the cap, not before: an untuned workspace has no use for wisdom it
   * cannot be asked to match, and loading it would only announce a file for a
   * run that then estimates everything. */
  if (fft_is_tuned(ws))
    fft_load_wisdom(mgr);

  ws->delta_k = real_fftw_malloc(spectrum_bytes);
  if (!ws->delta_k) {
    SIF_LOG_ERROR("fft_context",
      "failed to allocate the spectrum buffer (%" PRIu64 " bytes)",
      spectrum_bytes);
    free(ws);
    return NULL;
  }

  /* The forward plan is created on first use, see fft_ensure_forward_plan. */
  return ws;
}

/*
 * Plans the forward transform, at the workspace's one planner quality.
 *
 * The awkward part is that measuring means *executing*, repeatedly, over
 * whatever arrays the planner is handed -- and the natural input here is the
 * caller's own density field, which is not ours to destroy. That used to be
 * settled by quietly dropping the forward plan to FFTW_ESTIMATE while the
 * backward plan went on measuring, which made `skip_tuning` a statement about
 * half the workspace. It is settled here instead by giving the planner a
 * scratch input of its own, so a tuned workspace really does tune both
 * transforms.
 *
 * Three paths, in order of preference:
 *
 *   1. Wisdom. FFTW_WISDOM_ONLY returns NULL rather than measuring, so it
 *      cannot touch `in`, and when it succeeds there is nothing left to
 *      measure -- the usual case for a grid size that has been run before.
 *   2. Not tuning. FFTW_ESTIMATE does not execute anything, so it plans
 *      against `in` directly and costs nothing.
 *   3. Tuning, no wisdom. Allocate n^3 reals, measure against that, free it.
 *      The scratch is alive only across the planning call, but it is the size
 *      of the grid (42 GiB at n_cells = 2250) and it is alive at exactly the
 *      moment the caller is still holding the density field -- so this is the
 *      expensive path, and the warning says so.
 *
 * If that scratch cannot be had, the *workspace* drops to FFTW_ESTIMATE rather
 * than this plan alone: the backward transform reads plan_flags too, and the
 * one thing this function must not do is reintroduce the split it exists to
 * remove.
 */
static int fft_ensure_forward_plan(
  sif_fft_workspace_t* ws, const sif_real* in) {

  if (ws->forward_plan)
    return SIF_OK;

  const uint32_t n = ws->n_cells;

  real_fftw_plan_with_nthreads(sif__system_max_threads());

  /* 1. Wisdom, at this workspace's quality. */
  ws->forward_plan = real_fftw_plan_dft_r2c_3d(
    n, n, n, (sif_real*)in, ws->delta_k, ws->plan_flags | FFTW_WISDOM_ONLY);

  /* 3. Tuning with nothing in the wisdom file: buy the planner a scratch. */
  if (!ws->forward_plan && fft_is_tuned(ws)) {
    const uint64_t real_bytes = (uint64_t)n * n * n * sizeof(sif_real);
    const double gib = (double)real_bytes / (1024.0 * 1024.0 * 1024.0);

    /* Only worth interrupting anyone over once the scratch is a real amount of
     * memory. Below that it is an implementation detail. */
    if (real_bytes >= (1ull << 30)) {
      SIF_LOG_WARNING("fft_context",
        "no wisdom for a %u^3 grid, so tuning the forward transform needs a "
        "scratch copy of it (%.1f GiB) alongside the caller's own, for the "
        "duration of the planning; pass skip_tuning to plan both transforms "
        "with FFTW_ESTIMATE instead",
        n, gib);
    } else {
      SIF_LOG_TRACE("fft_context",
        "tuning the forward transform for a %u^3 grid on a %.1f MiB scratch "
        "buffer",
        n, gib * 1024.0);
    }

    sif_real* scratch = sif_malloc_aligned((size_t)real_bytes);

    /* The plan is executed later against `in` through the new-array interface,
     * which only accepts a buffer of the same alignment class as the one it
     * was planned on. Everything sif allocates is cache-line aligned, so this
     * holds -- but it is cheap to confirm and fatal to assume. */
    if (scratch && real_fftw_alignment_of(scratch) ==
                     real_fftw_alignment_of((sif_real*)in)) {
      ws->forward_plan = real_fftw_plan_dft_r2c_3d(
        n, n, n, scratch, ws->delta_k, ws->plan_flags);

      /* Banked before the scratch is even released: this is the only point in
       * the run where the measurement exists and has not yet been paid for
       * twice. */
      if (ws->forward_plan)
        fft_save_wisdom(ws->mgr);
    }

    sif_free_aligned(scratch); /* NULL-safe */

    /* Whichever way it failed -- no scratch, wrong alignment, no plan -- the
     * fall-through below is about to plan against `in`, and that is only safe
     * for a planner that does not execute. Downgrading the workspace rather
     * than this one plan is also what keeps the two transforms in step. */
    if (!ws->forward_plan) {
      SIF_LOG_WARNING("fft_context",
        "could not tune the forward transform; dropping the whole workspace "
        "to FFTW_ESTIMATE so both transforms are planned alike");
      ws->plan_flags = FFTW_ESTIMATE;
    }
  }

  /* 2. Not tuning, either by request or by the downgrade above. */
  if (!ws->forward_plan) {
    ws->forward_plan = real_fftw_plan_dft_r2c_3d(
      n, n, n, (sif_real*)in, ws->delta_k, ws->plan_flags);
  }

  if (!ws->forward_plan) {
    SIF_LOG_ERROR("fft_context", "failed to create the forward plan");
    return SIF_ERR_ALLOC;
  }

  SIF_LOG_TRACE("fft_context", "forward plan created (%s)",
    fft_is_tuned(ws) ? "tuned" : "estimated");

  return SIF_OK;
}

int sif__fft_workspace_init_backward(
  sif_fft_workspace_t* ws, sif_fft_manager_t* mgr) {
  if (!ws || !mgr)
    return SIF_ERR_INVALID;

  if (ws->backward_plan)
    return SIF_OK;

  const uint64_t complex_cells = fft_complex_count(ws->n_cells);

  ws->delta_k_cpy =
    sif_malloc_aligned(complex_cells * sizeof(sif_real_complex));
  if (!ws->delta_k_cpy) {
    SIF_LOG_ERROR("fft_context",
      "failed to allocate the real-space buffer (%" PRIu64 " bytes)",
      complex_cells * sizeof(sif_real_complex));
    return SIF_ERR_ALLOC;
  }

  /*
   * ws->plan_flags, not mgr->flags: the forward plan may have had to downgrade
   * the workspace, and the two are planned alike or not at all. The buffer is
   * the workspace's own, so measuring here is free of the scratch the forward
   * plan needs.
   *
   * Wisdom is asked first even though a plain plan call would consult it
   * anyway, because the two-step is what makes "did we have to measure?"
   * answerable -- and there is nothing to write back for a plan that came out
   * of the file.
   */
  ws->backward_plan = real_fftw_plan_dft_c2r_3d(ws->n_cells, ws->n_cells,
    ws->n_cells, ws->delta_k_cpy, (sif_real*)ws->delta_k_cpy,
    ws->plan_flags | FFTW_WISDOM_ONLY);

  if (!ws->backward_plan) {
    ws->backward_plan = real_fftw_plan_dft_c2r_3d(ws->n_cells, ws->n_cells,
      ws->n_cells, ws->delta_k_cpy, (sif_real*)ws->delta_k_cpy, ws->plan_flags);

    if (ws->backward_plan && fft_is_tuned(ws))
      fft_save_wisdom(ws->mgr);
  }

  if (!ws->backward_plan) {
    SIF_LOG_ERROR("fft_context", "failed to create the backward plan");
    sif_free_aligned(ws->delta_k_cpy);
    ws->delta_k_cpy = NULL;
    return SIF_ERR_ALLOC;
  }

  SIF_LOG_TRACE("fft_context", "backward plan created (%s)",
    fft_is_tuned(ws) ? "tuned" : "estimated");

  return SIF_OK;
}

sif_real* sif__fft_workspace_take_real_buffer(sif_fft_workspace_t* ws) {
  if (!ws || !ws->delta_k_cpy)
    return NULL;

  sif_real* buffer = (sif_real*)ws->delta_k_cpy;
  ws->delta_k_cpy = NULL;

  /* The plan points at a buffer we no longer own, so it must not be reused. */
  if (ws->backward_plan) {
    real_fftw_destroy_plan(ws->backward_plan);
    ws->backward_plan = NULL;
  }

  return buffer;
}

void sif__fft_workspace_free(sif_fft_workspace_t* ws) {
  if (!ws) {
    SIF_LOG_WARNING("fft_context", "cannot free a NULL workspace");
    return;
  }

  /* Nothing to save here: wisdom is written the moment a plan is measured,
   * see fft_save_wisdom(). */

  if (ws->forward_plan)
    real_fftw_destroy_plan(ws->forward_plan);
  if (ws->backward_plan)
    real_fftw_destroy_plan(ws->backward_plan);
  if (ws->delta_k)
    real_fftw_free(ws->delta_k);
  if (ws->delta_k_cpy)
    sif_free_aligned(ws->delta_k_cpy);
  if (ws->cic_inv)
    free(ws->cic_inv);

  free(ws);
}

static inline uint64_t get_flat_complex_index(
  uint32_t ix, uint32_t iy, uint32_t iz, uint32_t n_cells) {
  uint32_t z_dim = n_cells / 2 + 1;
  return (uint64_t)ix * n_cells * z_dim + (uint64_t)iy * z_dim + (uint64_t)iz;
}

/*
 * Builds the radial filter lookup table, indexed directly by the integer |k|^2.
 * Shared by sif__fft_apply_filter and fft_filtered_variance so the two can
 * never disagree about what the filter actually is.
 */
static sif_real* fft_build_filter_lut(
  sif_filter_type_t filter, sif_real r, sif_real box_length, uint32_t n_cells) {

  const uint32_t N_half = n_cells >> 1;
  const uint32_t max_k2 = 3 * N_half * N_half;

  sif_real* lut = malloc(((size_t)max_k2 + 1) * sizeof(sif_real));
  if (!lut) {
    SIF_LOG_ERROR("fft_context",
      "failed to allocate the filter lookup table (%zu bytes)",
      ((size_t)max_k2 + 1) * sizeof(sif_real));
    return NULL;
  }

  const sif_real factor = (sif_real)(r * 2.0 * SIF_PI / box_length);
  const sif_real factor2 = factor * factor;

#pragma omp parallel for schedule(static)
  for (uint32_t k2 = 0; k2 <= max_k2; k2++) {
    if (k2 == 0) {
      /* W(0) = 1 for any normalized window: the k = 0 mode is the mean, which
       * smoothing must leave alone. */
      lut[k2] = (sif_real)1.0;
    } else if (filter == SIF__FILTER_TOP_HAT) {
      const sif_real kr = factor * SIF_REAL_SQRT((sif_real)k2);

      /* The series 3(sin kr - kr cos kr)/(kr)^3 -> 1 as kr -> 0, but evaluated
       * directly it is a difference of two nearly equal terms divided by a
       * cube: at kr = 1e-4 the numerator has already lost most of its
       * significant digits. Below the crossover the limit is the more accurate
       * answer, not merely the cheaper one. */
      if (kr > (sif_real)1e-4) {
        lut[k2] = (sif_real)3.0 * (SIF_REAL_SIN(kr) - kr * SIF_REAL_COS(kr)) /
                  (kr * kr * kr);
      } else {
        lut[k2] = (sif_real)1.0;
      }
    } else {
      lut[k2] = SIF_REAL_EXP((sif_real)-0.5 * factor2 * (sif_real)k2);
    }
  }

  return lut;
}

int sif__fft_apply_filter(sif_fft_workspace_t* ws, sif_filter_type_t filter,
  sif_real r, sif_real box_length) {

  if (!ws || !ws->delta_k_cpy) {
    SIF_LOG_ERROR(
      "fft_context", "the backward stage must be initialized before filtering");
    return SIF_ERR_INVALID;
  }

  const uint64_t complex_cells = fft_complex_count(ws->n_cells);

  if (filter == SIF__FILTER_NONE) {
    memcpy(
      ws->delta_k_cpy, ws->delta_k, complex_cells * sizeof(sif_real_complex));
    SIF_LOG_TRACE("fft_context", "no filter applied");
    return SIF_OK;
  }

  if (filter != SIF__FILTER_TOP_HAT && filter != SIF__FILTER_GAUSSIAN) {
    /* Returning without writing delta_k_cpy would leave the backward transform
     * operating on whatever was there before, so this has to be an error. */
    SIF_LOG_ERROR("fft_context", "filter type %d is not supported", filter);
    return SIF_ERR_INVALID;
  }

  const uint32_t N = ws->n_cells;
  const uint32_t N_half = N >> 1;

  sif_real* lut = fft_build_filter_lut(filter, r, box_length, N);
  if (!lut)
    return SIF_ERR_ALLOC;

  /* Carried through the same multiply as the filter itself: the assignment
   * window is separable and the filter is radial, so neither can be folded
   * into the other's table, but both land on the mode together. */
  const double* cic = ws->cic_inv;

#pragma omp parallel for schedule(static)
  for (uint32_t ix = 0; ix < N; ix++) {
    int32_t kx = (ix > N_half) ? (int32_t)ix - (int32_t)N : (int32_t)ix;
    uint32_t kx2 = (uint32_t)(kx * kx);
    const double cx = cic ? cic[ix] : 1.0;

    for (uint32_t iy = 0; iy < N; iy++) {
      int32_t ky = (iy > N_half) ? (int32_t)iy - (int32_t)N : (int32_t)iy;
      uint32_t kxy2 = kx2 + (uint32_t)(ky * ky);
      const double cxy = cx * (cic ? cic[iy] : 1.0);

      for (uint32_t iz = 0; iz <= N_half; iz++) {
        uint32_t k2 = kxy2 + (iz * iz);

        uint64_t idx = get_flat_complex_index(ix, iy, iz, N);
        sif_real smoothing = lut[k2];

        if (cic)
          smoothing = (sif_real)((double)smoothing * cxy * cic[iz]);

        ws->delta_k_cpy[idx][0] = ws->delta_k[idx][0] * smoothing;
        ws->delta_k_cpy[idx][1] = ws->delta_k[idx][1] * smoothing;
      }
    }
  }

  free(lut);

  SIF_LOG_TRACE("fft_context", "%s smoothing applied",
    filter == SIF__FILTER_TOP_HAT ? "top hat" : "gaussian");

  return SIF_OK;
}

int sif__fft_grid_forward(sif_fft_workspace_t* ws, const sif_grid_t* grid) {
  if (!ws || !ws->delta_k || !grid || !grid->values) {
    SIF_LOG_ERROR("fft_context", "invalid workspace or grid");
    return SIF_ERR_INVALID;
  }

  if (fft_ensure_forward_plan(ws, grid->values) != SIF_OK)
    return SIF_ERR_ALLOC;

  real_fftw_execute_dft_r2c(
    ws->forward_plan, (sif_real*)grid->values, ws->delta_k);
  SIF_LOG_TRACE("fft_context", "forward fft on cubic grid executed");

  return SIF_OK;
}

sif_real* sif__fft_grid_backward(sif_fft_workspace_t* ws) {
  if (!ws || !ws->backward_plan || !ws->delta_k_cpy) {
    SIF_LOG_ERROR("fft_context", "the backward stage is not initialized");
    return NULL;
  }

  real_fftw_execute_dft_c2r(
    ws->backward_plan, ws->delta_k_cpy, (sif_real*)ws->delta_k_cpy);

  sif_real* data = (sif_real*)ws->delta_k_cpy;

  const uint64_t n = ws->n_cells;
  const uint64_t n_padded = 2 * (n / 2 + 1);
  const uint64_t total_cells = n * n * n;
  const sif_real norm = 1.0f / (sif_real)total_cells;

  /* An in-place r2c/c2r pair always pads the last dimension, so n_padded > n
   * unconditionally and the rows have to be compacted.
   *
   * The compaction cannot be parallelized row-wise: row r writes into
   * [r*n, (r+1)*n) while row r-1 reads from [(r-1)*n_padded, +n), and those
   * overlap. Sequentially it is safe because the write cursor never overtakes
   * the read cursor. So compact serially with memmove (bandwidth-bound, no
   * per-element loop) and do the arithmetic in a separate parallel pass. */
  for (uint64_t row = 1; row < n * n; row++) {
    memmove(data + row * n, data + row * n_padded, n * sizeof(sif_real));
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
static inline double fft_sinc(double x) {
  if (x < 1e-9 && x > -1e-9)
    return 1.0;
  return sin(x) / x;
}

/*
 * The window is separable, so the per-axis factor depends only on that axis's
 * index and one table of N entries covers all three. Built as the reciprocal,
 * so the hot loops multiply.
 */
static double* fft_build_cic_inverse_axis(uint32_t n_cells) {
  const uint32_t N_half = n_cells >> 1;

  double* axis = malloc((size_t)n_cells * sizeof(double));
  if (!axis) {
    SIF_LOG_ERROR("fft_context", "failed to allocate the CIC window table");
    return NULL;
  }

  for (uint32_t i = 0; i < n_cells; i++) {
    int32_t k = (i > N_half) ? (int32_t)i - (int32_t)n_cells : (int32_t)i;
    double s = fft_sinc(SIF_PI * (double)k / (double)n_cells);
    axis[i] = 1.0 / (s * s); /* CIC is the NGP window squared */
  }

  return axis;
}

int sif__fft_set_cic_correction(sif_fft_workspace_t* ws, int enable) {
  if (!ws || ws->n_cells == 0) {
    SIF_LOG_ERROR("fft_context", "invalid workspace");
    return SIF_ERR_INVALID;
  }

  if (!enable) {
    free(ws->cic_inv);
    ws->cic_inv = NULL;
    return SIF_OK;
  }

  if (ws->cic_inv)
    return SIF_OK;

  ws->cic_inv = fft_build_cic_inverse_axis(ws->n_cells);
  if (!ws->cic_inv)
    return SIF_ERR_ALLOC;

  SIF_LOG_TRACE("fft_context", "filtering will correct the CIC window");
  return SIF_OK;
}

int sif__fft_deconvolve_cic(sif_fft_workspace_t* ws) {
  if (!ws || !ws->delta_k) {
    SIF_LOG_ERROR("fft_context", "no spectrum to deconvolve");
    return SIF_ERR_INVALID;
  }

  const uint32_t N = ws->n_cells;
  const uint32_t z_dim = N / 2 + 1;

  double* axis = fft_build_cic_inverse_axis(N);
  if (!axis)
    return SIF_ERR_ALLOC;

#pragma omp parallel for schedule(static)
  for (uint32_t ix = 0; ix < N; ix++) {
    const double wx = axis[ix];
    for (uint32_t iy = 0; iy < N; iy++) {
      const double wxy = wx * axis[iy];
      for (uint32_t iz = 0; iz < z_dim; iz++) {
        const double inv = wxy * axis[iz];

        uint64_t idx = get_flat_complex_index(ix, iy, iz, N);
        ws->delta_k[idx][0] = (sif_real)(ws->delta_k[idx][0] * inv);
        ws->delta_k[idx][1] = (sif_real)(ws->delta_k[idx][1] * inv);
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
static inline void fft_seed_mode(
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
static inline sif_real fft_amp_complex(
  sif_real a, sif_prng_state_t* prng, bool resample) {
  if (!resample)
    return a;
  double u = 1.0 - (double)sif_prng_next_real(prng);
  return (sif_real)(a * sqrt(-log(u)));
}

/*
 * Amplitude for a self-conjugate (necessarily real) mode. Those carry the full
 * power in a single real degree of freedom rather than splitting it across a
 * real and an imaginary part, so the Gaussian counterpart is a standard normal
 * scaling, not a Rayleigh one.
 */
static inline sif_real fft_amp_real(
  sif_real a, sif_prng_state_t* prng, bool resample) {
  if (!resample)
    return a;
  double u1 = 1.0 - (double)sif_prng_next_real(prng);
  double u2 = (double)sif_prng_next_real(prng);
  return (sif_real)(a * sqrt(-2.0 * log(u1)) * cos(2.0 * SIF_PI * u2));
}

int sif__fft_randomize_phases(
  sif_fft_workspace_t* ws, uint64_t seed, bool resample_amplitudes) {

  if (!ws || !ws->delta_k) {
    SIF_LOG_ERROR("fft_context", "no spectrum to randomize");
    return SIF_ERR_INVALID;
  }

  const uint32_t N = ws->n_cells;
  const uint32_t z_dim = N / 2 + 1;
  const double two_pi = 2.0 * SIF_PI;

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

        uint64_t idx = get_flat_complex_index(ix, iy, iz, N);

        sif_real re = ws->delta_k[idx][0];
        sif_real im = ws->delta_k[idx][1];
        sif_real amp = (sif_real)sqrt((double)re * re + (double)im * im);

        sif_prng_state_t prng;
        fft_seed_mode(&prng, seed, idx);

        double phase = two_pi * (double)sif_prng_next_real(&prng);
        amp = fft_amp_complex(amp, &prng, resample_amplitudes);

        ws->delta_k[idx][0] = (sif_real)(amp * cos(phase));
        ws->delta_k[idx][1] = (sif_real)(amp * sin(phase));
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

        uint64_t idx = get_flat_complex_index(ix, iy, iz, N);
        uint64_t jdx = get_flat_complex_index(jx, jy, iz, N);

        sif_real re = ws->delta_k[idx][0];
        sif_real im = ws->delta_k[idx][1];
        sif_real amp = (sif_real)sqrt((double)re * re + (double)im * im);

        sif_prng_state_t prng;
        fft_seed_mode(&prng, seed, idx);

        if (idx == jdx) {
          /* Self-conjugate: k == -k, so the mode has to stay real. The k = 0
           * mode is the mean of the field and is left alone. */
          if (idx == 0)
            continue;

          sif_real a = fft_amp_real(amp, &prng, resample_amplitudes);
          if (!resample_amplitudes && sif_prng_next_real(&prng) < 0.5f)
            a = -a; /* a fixed-amplitude real mode still gets a random sign */

          ws->delta_k[idx][0] = a;
          ws->delta_k[idx][1] = 0.0f;
          continue;
        }

        double phase = two_pi * (double)sif_prng_next_real(&prng);
        amp = fft_amp_complex(amp, &prng, resample_amplitudes);

        sif_real new_re = (sif_real)(amp * cos(phase));
        sif_real new_im = (sif_real)(amp * sin(phase));

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

int sif__fft_spectral_moments(const sif_fft_workspace_t* ws,
  sif_filter_type_t filter, sif_real r, sif_real box_length, uint8_t max_order,
  uint64_t n_tracers, double* sigma_sq, double* high_k_fraction) {

  if (!ws || !ws->delta_k || !sigma_sq) {
    SIF_LOG_ERROR("fft_context", "no spectrum to evaluate");
    return SIF_ERR_INVALID;
  }

  if (max_order > SIF__FFT_MAX_MOMENT_ORDER) {
    SIF_LOG_ERROR("fft_context", "moment order %u exceeds the maximum of %d",
      max_order, SIF__FFT_MAX_MOMENT_ORDER);
    return SIF_ERR_INVALID;
  }

  const uint32_t N = ws->n_cells;
  const uint32_t N_half = N >> 1;
  const uint32_t z_dim = N / 2 + 1;
  const int n_moments = (int)max_order + 1;

  sif_real* lut = NULL;
  if (filter != SIF__FILTER_NONE) {
    lut = fft_build_filter_lut(filter, r, box_length, N);
    if (!lut)
      return SIF_ERR_ALLOC;
  }

  /*
   * Three accumulators per order -- the measured sum, the window sum behind
   * the shot-noise term, and the part of the measured sum above half Nyquist.
   *
   * Explicit per-thread scratch rather than an array reduction because the
   * three live in one allocation and are indexed together; the padding below
   * is the part that matters. Every thread updates its row once per mode, and
   * at order 4 three rows of doubles come to 120 bytes -- two threads' rows
   * would share a cache line and bounce it between cores for the whole triple
   * loop. Rounding the stride up to a whole line is the same trick
   * SIF__EP_ROW_PAD plays in the excursion-set walker.
   */
  const int n_threads =
    sif__system_max_threads() > 0 ? sif__system_max_threads() : 1;

  const size_t row = 3 * (size_t)n_moments;
  const size_t per_line = SIF_CACHE_LINE / sizeof(double);
  const size_t stride = ((row + per_line - 1) / per_line) * per_line;

  /* Aligned, so that the padded stride actually starts each row on a line
   * boundary rather than merely spacing the rows apart. */
  double* scratch =
    sif_calloc_aligned((size_t)n_threads * stride, sizeof(double));
  if (!scratch) {
    SIF_LOG_ERROR("fft_context", "failed to allocate the moment scratch");
    free(lut);
    return SIF_ERR_ALLOC;
  }

  /* Isotropic cut at half the Nyquist frequency, in integer mode units. */
  const double high_k2 = (double)N * (double)N / 16.0;

#pragma omp parallel num_threads(n_threads)
  {
    double* local = scratch + (size_t)sif__system_thread_num() * stride;
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

          uint64_t idx = get_flat_complex_index(ix, iy, iz, N);

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
    (2.0 * SIF_PI / (double)box_length) * (2.0 * SIF_PI / (double)box_length);

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

  sif_free_aligned(scratch);

  return SIF_OK;
}
