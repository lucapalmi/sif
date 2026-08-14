/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* Validates the reworked fft wrapper: round-trip fidelity, the rewritten
 * padded->contiguous compaction, filter invariants, and buffer ownership. */
#include "core/system_internal.h"
#include "math/fft.h"
#include "sif/core/settings.h"
#include "sif/structures/grid.h"
#include "sif/utils/align.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

/* Size of a file in bytes, or -1 if it is not there. */
static long file_size(const char* path) {
  struct stat st;
  if (stat(path, &st) != 0)
    return -1;
  return (long)st.st_size;
}

/* Deterministic pseudo-random fill. */
static void fill_random(sif_real* d, uint64_t n, unsigned seed) {
  uint64_t s = seed ? seed : 1;
  for (uint64_t i = 0; i < n; i++) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    d[i] = (sif_real)((double)((s >> 33) & 0xFFFFFF) / 8388608.0 - 1.0);
  }
}

static void test_roundtrip(uint32_t n) {
  printf("fft round-trip, n=%u\n", n);

  sif_fft_manager_t* mgr = sif__fft_manager_init(true /* ESTIMATE */, NULL);
  CHECK(mgr != NULL, "manager init failed");
  if (!mgr)
    return;

  sif_grid_t* grid = sif_grid_alloc(n, 100.0f);
  CHECK(grid != NULL, "grid alloc failed");
  if (!grid) {
    sif__fft_manager_finalize(mgr);
    return;
  }

  const uint64_t total = (uint64_t)n * n * n;
  sif_real* reference = malloc(total * sizeof(sif_real));
  fill_random(grid->values, total, 12345);
  memcpy(reference, grid->values, total * sizeof(sif_real));

  sif_fft_workspace_t* ws = sif__fft_workspace_alloc(mgr, n);
  CHECK(ws != NULL, "workspace alloc failed");

  if (ws) {
    sif__fft_grid_forward(ws, grid);
    sif_free_aligned(grid->values);
    grid->values = NULL;

    CHECK(sif__fft_workspace_init_backward(ws, mgr) == SIF_OK,
      "init_backward failed");

    /* Filtering must be rejected before the backward stage exists. */
    CHECK(sif__fft_apply_filter(ws, (sif_filter_type_t)99, 1.0f, 100.0f) ==
            SIF_ERR_INVALID,
      "an unsupported filter should be rejected");

    CHECK(sif__fft_apply_filter(ws, SIF__FILTER_NONE, 0, 0) == SIF_OK,
      "SIF__FILTER_NONE failed");
    grid->values = sif__fft_grid_backward(ws);
    CHECK(grid->values != NULL, "backward returned NULL");

    if (grid->values) {
      /* Every cell must survive the transform pair. This is what catches a
       * broken row compaction: padding errors misalign whole rows. */
      double max_err = 0.0;
      uint64_t worst = 0;
      for (uint64_t i = 0; i < total; i++) {
        double e = fabs((double)grid->values[i] - (double)reference[i]);
        if (e > max_err) {
          max_err = e;
          worst = i;
        }
      }
      CHECK(max_err < 1e-4, "round-trip max error %.3g at cell %llu", max_err,
        (unsigned long long)worst);
      printf("  round-trip max error: %.3g\n", max_err);
    }

    /* Ownership handoff: the workspace must relinquish the buffer. */
    sif_real* taken = sif__fft_workspace_take_real_buffer(ws);
    CHECK(
      taken == grid->values, "take_real_buffer returned a different pointer");
    CHECK(ws->delta_k_cpy == NULL, "workspace still references the buffer");
    CHECK(sif__fft_workspace_take_real_buffer(ws) == NULL,
      "a second take should return NULL");
    CHECK(sif__fft_grid_backward(ws) == NULL,
      "backward after take should fail cleanly");

    sif__fft_workspace_free(ws); /* must not free the taken buffer */
    CHECK(grid->values[0] == grid->values[0], "buffer freed out from under us");
  }

  free(reference);
  sif_grid_free(grid);
  sif__fft_manager_finalize(mgr);
  printf("  ok\n");
}

/* A constant field is its own top-hat smoothing: W(k=0)=1 and every other
 * mode is zero. */
static void test_tophat_preserves_constant(uint32_t n) {
  printf("top-hat on a constant field, n=%u\n", n);

  sif_fft_manager_t* mgr = sif__fft_manager_init(true, NULL);
  sif_grid_t* grid = sif_grid_alloc(n, 100.0f);
  if (!mgr || !grid) {
    CHECK(0, "setup failed");
    return;
  }

  const uint64_t total = (uint64_t)n * n * n;
  const sif_real value = 2.5f;
  for (uint64_t i = 0; i < total; i++)
    grid->values[i] = value;

  sif_fft_workspace_t* ws = sif__fft_workspace_alloc(mgr, n);
  if (ws) {
    sif__fft_grid_forward(ws, grid);
    sif_free_aligned(grid->values);
    grid->values = NULL;

    CHECK(sif__fft_workspace_init_backward(ws, mgr) == SIF_OK, "init_backward");
    CHECK(
      sif__fft_apply_filter(ws, SIF__FILTER_TOP_HAT, 12.0f, 100.0f) == SIF_OK,
      "top-hat filter failed");
    grid->values = sif__fft_grid_backward(ws);

    double max_err = 0.0;
    for (uint64_t i = 0; i < total; i++) {
      double e = fabs((double)grid->values[i] - (double)value);
      if (e > max_err)
        max_err = e;
    }
    CHECK(
      max_err < 1e-4, "constant field not preserved, max error %.3g", max_err);
    printf("  max deviation from constant: %.3g\n", max_err);

    grid->values = sif__fft_workspace_take_real_buffer(ws);
    sif__fft_workspace_free(ws);
  }

  sif_grid_free(grid);
  sif__fft_manager_finalize(mgr);
  printf("  ok\n");
}

/*
 * The tuned path, which is the one with something to get wrong.
 *
 * With no wisdom directory the forward plan cannot come from wisdom, so a
 * tuning workspace has to measure -- and it measures against a scratch buffer
 * rather than against the caller's field, then executes against the field
 * through the new-array interface. That indirection is the whole risk: a plan
 * measured on one buffer and run on another is only valid while the two agree
 * on alignment, and getting it wrong gives silent garbage rather than a crash.
 *
 * So: same round trip, same tolerance, only with the planner turned up.
 */
static void test_tuned_plan_roundtrip(uint32_t n) {
  printf("tuned (FFTW_MEASURE) round-trip, n=%u\n", n);

  sif_fft_manager_t* mgr = sif__fft_manager_init(false /* MEASURE */, NULL);
  CHECK(mgr != NULL, "manager init failed");
  if (!mgr)
    return;

  sif_grid_t* grid = sif_grid_alloc(n, 100.0f);
  CHECK(grid != NULL, "grid alloc failed");
  if (!grid) {
    sif__fft_manager_finalize(mgr);
    return;
  }

  const uint64_t total = (uint64_t)n * n * n;
  sif_real* reference = malloc(total * sizeof(sif_real));
  fill_random(grid->values, total, 999);
  memcpy(reference, grid->values, total * sizeof(sif_real));

  sif_fft_workspace_t* ws = sif__fft_workspace_alloc(mgr, n);
  CHECK(ws != NULL, "workspace alloc failed");

  if (ws) {
    CHECK(sif__fft_grid_forward(ws, grid) == SIF_OK, "forward failed");

    /* Measuring must not have eaten the caller's field: the scratch buffer
     * exists precisely so that this array is still the input we handed in. */
    int clobbered = 0;
    for (uint64_t i = 0; i < total; i++) {
      if (grid->values[i] != reference[i])
        clobbered++;
    }
    CHECK(clobbered == 0, "planning overwrote %d cells of the caller's grid",
      clobbered);

    sif_free_aligned(grid->values);
    grid->values = NULL;

    CHECK(sif__fft_workspace_init_backward(ws, mgr) == SIF_OK,
      "init_backward failed");
    CHECK(sif__fft_apply_filter(ws, SIF__FILTER_NONE, 0, 0) == SIF_OK,
      "SIF__FILTER_NONE failed");

    grid->values = sif__fft_grid_backward(ws);
    CHECK(grid->values != NULL, "backward returned NULL");

    if (grid->values) {
      double max_err = 0.0;
      for (uint64_t i = 0; i < total; i++) {
        double e = fabs((double)grid->values[i] - (double)reference[i]);
        if (e > max_err)
          max_err = e;
      }
      CHECK(max_err < 1e-4,
        "tuned round-trip max error %.3g -- a plan measured on the scratch "
        "buffer is not running correctly on the real one",
        max_err);
      printf("  round-trip max error: %.3g\n", max_err);
    }

    grid->values = sif__fft_workspace_take_real_buffer(ws);
    sif__fft_workspace_free(ws);
  }

  free(reference);
  sif_grid_free(grid);
  sif__fft_manager_finalize(mgr);
  printf("  ok\n");
}

/*
 * The tuning ceiling, and the wisdom rule that goes with it.
 *
 * Both are about the same failure: FFTW_MEASURE prices its plan by running the
 * transform, so past some size the planning never pays for itself, and an
 * untuned run must not leave wisdom behind that a tuned run will load, fail to
 * match, and re-measure from scratch.
 *
 * A tiny grid is far under any sane ceiling, so the cap is driven by setting it
 * to 0 -- "never tune" -- which is the same code path a 2250^3 grid takes.
 */
static void test_tuning_cap_and_wisdom(void) {
  printf("tuning ceiling and wisdom export\n");

  const uint32_t n = 16;
  char dir[] = "sif_fft_wisdom_test";
  char expected[256];
  snprintf(expected, sizeof(expected), "%s/%s", dir, SIF__FFT_WISDOM_FILE);

  mkdir(dir, 0777);
  remove(expected);

  /* The ceiling is a setting, and sif_setting_set() is a no-op until the table
   * exists. The rest of this file talks to the fft wrapper directly, so bring
   * up just the settings and take them down again -- into the test directory,
   * so nothing lands in a real config file. */
  sif__settings_init(dir);

  /* 1. Capped: the workspace must estimate even though tuning was asked for,
   *    and must leave no wisdom behind. */
  sif_setting_set("fft_tuning_max_gib", "0");
  sif_fft_manager_t* mgr =
    sif__fft_manager_init(false /* wants MEASURE */, dir);
  sif_fft_workspace_t* ws = sif__fft_workspace_alloc(mgr, n);
  CHECK(ws != NULL, "workspace alloc failed under the cap");

  if (ws) {
    CHECK((ws->plan_flags & FFTW_ESTIMATE) != 0,
      "a capped workspace should have dropped to FFTW_ESTIMATE");
    sif_grid_t* g = sif_grid_alloc(n, 100.0f);
    fill_random(g->values, (uint64_t)n * n * n, 7);
    CHECK(sif__fft_grid_forward(ws, g) == SIF_OK, "forward failed");
    sif__fft_workspace_free(ws);
    sif_grid_free(g);
  }
  sif__fft_manager_finalize(mgr);

  FILE* f = fopen(expected, "rb");
  CHECK(f == NULL, "an untuned run must not export wisdom to %s", expected);
  if (f)
    fclose(f);

  /* 2. Uncapped: the same workspace tunes, and now the file is worth writing.
   */
  sif_setting_set("fft_tuning_max_gib", "1024");
  mgr = sif__fft_manager_init(false, dir);
  ws = sif__fft_workspace_alloc(mgr, n);
  CHECK(ws != NULL, "workspace alloc failed without the cap");

  if (ws) {
    CHECK((ws->plan_flags & FFTW_ESTIMATE) == 0,
      "an uncapped workspace should have kept FFTW_MEASURE");
    sif_grid_t* g = sif_grid_alloc(n, 100.0f);
    fill_random(g->values, (uint64_t)n * n * n, 7);
    CHECK(sif__fft_grid_forward(ws, g) == SIF_OK, "forward failed");

    /* The point of saving on measurement rather than on teardown: the file is
     * already there, while the workspace is still alive. A run killed from
     * here on keeps what its planning cost. */
    FILE* early = fopen(expected, "rb");
    CHECK(early != NULL,
      "wisdom should be on disk as soon as a plan is measured, not at "
      "teardown (%s)",
      expected);
    if (early)
      fclose(early);

    CHECK(sif__fft_workspace_init_backward(ws, mgr) == SIF_OK,
      "init_backward failed");

    sif__fft_workspace_free(ws);
    sif_grid_free(g);
  }
  sif__fft_manager_finalize(mgr);

  f = fopen(expected, "rb");
  CHECK(f != NULL, "a tuned run should have exported wisdom to %s", expected);
  if (f)
    fclose(f);

  /*
   * And the file is cumulative rather than per-size: a second size planned into
   * the same directory adds to it instead of displacing the first. This is the
   * property the old grid_<n>.wisdom naming pretended to have and did not --
   * every file held the whole process's wisdom under a single size's name.
   */
  const long size_one = file_size(expected);
  CHECK(size_one > 0, "wisdom file is empty after one size");

  sif_setting_set("fft_tuning_max_gib", "1024");
  mgr = sif__fft_manager_init(false, dir);
  ws = sif__fft_workspace_alloc(mgr, n + 8);
  if (ws) {
    sif_grid_t* g = sif_grid_alloc(n + 8, 100.0f);
    fill_random(g->values, (uint64_t)(n + 8) * (n + 8) * (n + 8), 7);
    CHECK(sif__fft_grid_forward(ws, g) == SIF_OK, "forward failed");
    sif__fft_workspace_free(ws);
    sif_grid_free(g);
  }
  sif__fft_manager_finalize(mgr);

  const long size_two = file_size(expected);
  CHECK(size_two > size_one,
    "a second grid size should have added to the wisdom file, not replaced it "
    "(%ld -> %ld bytes)",
    size_one, size_two);

  /* Still a valid file after all that rewriting -- a truncated one is worse
   * than none, since FFTW refuses it and every later run silently estimates.
   *
   * init_threads first: the manager finalize above ran fftw_cleanup(), which
   * resets the planner, and an import into a torn-down planner fails whatever
   * the file says. The library never hits that -- it loads from
   * workspace_alloc, after a manager exists -- but the test would. */
  real_fftw_init_threads();
  real_fftw_forget_wisdom();
  CHECK(real_fftw_import_wisdom_from_filename(expected) != 0,
    "the accumulated wisdom file no longer imports");

  /* 3. A malformed ceiling must read as the default, never as "no limit". */
  sif_setting_set("fft_tuning_max_gib", "not-a-number");
  mgr = sif__fft_manager_init(false, NULL);
  ws = sif__fft_workspace_alloc(mgr, n);
  CHECK(ws != NULL, "workspace alloc failed with a malformed ceiling");
  if (ws) {
    /* n=16 is well under the default, so the fallback ceiling still tunes. */
    CHECK((ws->plan_flags & FFTW_ESTIMATE) == 0,
      "a malformed ceiling should fall back to the default, not to 0");
    sif__fft_workspace_free(ws);
  }
  sif__fft_manager_finalize(mgr);

  sif__settings_finalize();

  remove(expected);
  {
    char cfg[256];
    snprintf(cfg, sizeof(cfg), "%s/config", dir);
    remove(cfg);
  }
  rmdir(dir);
  printf("  ok\n");
}

int main(void) {
  /* Odd and even n exercise different padding widths. */
  test_roundtrip(16);
  test_roundtrip(15);
  test_roundtrip(32);
  test_tophat_preserves_constant(24);
  test_tuned_plan_roundtrip(16);
  test_tuned_plan_roundtrip(15);
  test_tuning_cap_and_wisdom();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
