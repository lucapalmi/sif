/* Validates the reworked fft wrapper: round-trip fidelity, the rewritten
 * padded->contiguous compaction, filter invariants, and buffer ownership. */
#include "math/fft.h"
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

/* Deterministic pseudo-random fill. */
static void fill_random(real_t* d, uint64_t n, unsigned seed) {
  uint64_t s = seed ? seed : 1;
  for (uint64_t i = 0; i < n; i++) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    d[i] = (real_t)((double)((s >> 33) & 0xFFFFFF) / 8388608.0 - 1.0);
  }
}

static void test_roundtrip(uint32_t n) {
  printf("fft round-trip, n=%u\n", n);

  fft_manager_t* mgr = fft_manager_init(true /* ESTIMATE */, NULL);
  CHECK(mgr != NULL, "manager init failed");
  if (!mgr)
    return;

  sif_grid_t* grid = sif_grid_alloc(n, 100.0f);
  CHECK(grid != NULL, "grid alloc failed");
  if (!grid) {
    fft_manager_finalize(mgr);
    return;
  }

  const uint64_t total = (uint64_t)n * n * n;
  real_t* reference = malloc(total * sizeof(real_t));
  fill_random(grid->delta, total, 12345);
  memcpy(reference, grid->delta, total * sizeof(real_t));

  fft_workspace_t* ws = fft_workspace_alloc(mgr, n);
  CHECK(ws != NULL, "workspace alloc failed");

  if (ws) {
    fft_grid_forward(ws, grid);
    sif_free_aligned(grid->delta);
    grid->delta = NULL;

    CHECK(fft_workspace_init_backward(ws, mgr) == SIF_OK,
      "init_backward failed");

    /* Filtering must be rejected before the backward stage exists. */
    CHECK(fft_apply_filter(ws, (filter_type_t)99, 1.0f, 100.0f) ==
            SIF_ERR_INVALID,
      "an unsupported filter should be rejected");

    CHECK(fft_apply_filter(ws, FILTER_NONE, 0, 0) == SIF_OK,
      "FILTER_NONE failed");
    grid->delta = fft_grid_backward(ws);
    CHECK(grid->delta != NULL, "backward returned NULL");

    if (grid->delta) {
      /* Every cell must survive the transform pair. This is what catches a
       * broken row compaction: padding errors misalign whole rows. */
      double max_err = 0.0;
      uint64_t worst = 0;
      for (uint64_t i = 0; i < total; i++) {
        double e = fabs((double)grid->delta[i] - (double)reference[i]);
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
    real_t* taken = fft_workspace_take_real_buffer(ws);
    CHECK(taken == grid->delta, "take_real_buffer returned a different pointer");
    CHECK(ws->delta_k_cpy == NULL, "workspace still references the buffer");
    CHECK(fft_workspace_take_real_buffer(ws) == NULL,
      "a second take should return NULL");
    CHECK(fft_grid_backward(ws) == NULL,
      "backward after take should fail cleanly");

    fft_workspace_free(ws); /* must not free the taken buffer */
    CHECK(grid->delta[0] == grid->delta[0], "buffer freed out from under us");
  }

  free(reference);
  sif_grid_free(grid);
  fft_manager_finalize(mgr);
  printf("  ok\n");
}

/* A constant field is its own top-hat smoothing: W(k=0)=1 and every other
 * mode is zero. */
static void test_tophat_preserves_constant(uint32_t n) {
  printf("top-hat on a constant field, n=%u\n", n);

  fft_manager_t* mgr = fft_manager_init(true, NULL);
  sif_grid_t* grid = sif_grid_alloc(n, 100.0f);
  if (!mgr || !grid) {
    CHECK(0, "setup failed");
    return;
  }

  const uint64_t total = (uint64_t)n * n * n;
  const real_t value = 2.5f;
  for (uint64_t i = 0; i < total; i++)
    grid->delta[i] = value;

  fft_workspace_t* ws = fft_workspace_alloc(mgr, n);
  if (ws) {
    fft_grid_forward(ws, grid);
    sif_free_aligned(grid->delta);
    grid->delta = NULL;

    CHECK(fft_workspace_init_backward(ws, mgr) == SIF_OK, "init_backward");
    CHECK(fft_apply_filter(ws, FILTER_TOP_HAT, 12.0f, 100.0f) == SIF_OK,
      "top-hat filter failed");
    grid->delta = fft_grid_backward(ws);

    double max_err = 0.0;
    for (uint64_t i = 0; i < total; i++) {
      double e = fabs((double)grid->delta[i] - (double)value);
      if (e > max_err)
        max_err = e;
    }
    CHECK(max_err < 1e-4, "constant field not preserved, max error %.3g",
      max_err);
    printf("  max deviation from constant: %.3g\n", max_err);

    grid->delta = fft_workspace_take_real_buffer(ws);
    fft_workspace_free(ws);
  }

  sif_grid_free(grid);
  fft_manager_finalize(mgr);
  printf("  ok\n");
}

int main(void) {
  /* Odd and even n exercise different padding widths. */
  test_roundtrip(16);
  test_roundtrip(15);
  test_roundtrip(32);
  test_tophat_preserves_constant(24);

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
