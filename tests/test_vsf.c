/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* Covers the simplified histogram-only VSF: no voids dropped at either edge,
 * correct normalization, both binning modes, and the merge path. */
#include "sif/measure/size_function.h"
#include "sif/structures/catalog.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define BOX 100.0f

static sif_catalog_t* make_catalog(uint64_t n, sif_real r_lo, sif_real r_hi) {
  sif_catalog_t* c = sif_catalog_alloc(n);
  for (uint64_t i = 0; i < n; i++) {
    /* Spread radii evenly so both the first and last land exactly on a bound */
    sif_real r = r_lo + (r_hi - r_lo) * ((sif_real)i / (sif_real)(n - 1));
    sif_catalog_append(c, 1.0f, 2.0f, 3.0f, r);
  }
  return c;
}

static void test_no_voids_dropped(const char* label, sif_option bins) {
  printf("%s: every void is binned\n", label);

  const uint64_t n = 1000;
  sif_catalog_t* cat = make_catalog(n, 5.0f, 25.0f);

  sif_size_function_t* vsf =
    sif_size_function_catalog(cat, BOX, 8, bins, 0.0f, 0.0f);
  CHECK(vsf != NULL, "computation returned NULL");

  if (vsf) {
    uint64_t total = 0;
    for (uint32_t b = 0; b < vsf->n_bins; b++)
      total += vsf->counts[b];

    /* The void sitting exactly at r_max used to fall out of linear binning. */
    CHECK(total == n, "binned %llu of %llu voids", (unsigned long long)total,
      (unsigned long long)n);

    CHECK(fabs((double)vsf->r_min - 5.0) < 1e-4, "r_min = %g, expected 5",
      (double)vsf->r_min);
    CHECK(fabs((double)vsf->r_max - 25.0) < 1e-4, "r_max = %g, expected 25",
      (double)vsf->r_max);

    /* Edges must be monotonic and span exactly [r_min, r_max]. */
    int bad_edges = 0;
    for (uint32_t b = 0; b < vsf->n_bins; b++)
      if (!(vsf->r_edges[b + 1] > vsf->r_edges[b]))
        bad_edges++;
    CHECK(bad_edges == 0, "%d non-monotonic bin edges", bad_edges);
    CHECK(fabs((double)vsf->r_edges[vsf->n_bins] - 25.0) < 1e-3,
      "last edge is %g, expected 25", (double)vsf->r_edges[vsf->n_bins]);

    /* Normalization: vsf = counts / (V * bin_width), bin width in r or ln r. */
    const double vol = (double)BOX * BOX * BOX;
    const int is_ln = (bins & SIF__VSF_BIN_MASK) == SIF_VSF_BIN_LN;
    const double lo = is_ln ? log(5.0) : 5.0;
    const double hi = is_ln ? log(25.0) : 25.0;
    const double width = (hi - lo) / vsf->n_bins;

    int bad_norm = 0;
    for (uint32_t b = 0; b < vsf->n_bins; b++) {
      const double expect = (double)vsf->counts[b] / (vol * width);
      if (fabs((double)vsf->vsf[b] - expect) > 1e-9 * (1.0 + fabs(expect)))
        bad_norm++;
    }
    CHECK(bad_norm == 0, "%d bins with wrong normalization", bad_norm);

    /* Poisson error: err/vsf == 1/sqrt(N) wherever N > 0. */
    int bad_err = 0;
    for (uint32_t b = 0; b < vsf->n_bins; b++) {
      if (vsf->counts[b] == 0) {
        if (vsf->err[b] != 0.0f)
          bad_err++;
      } else {
        const double rel = (double)vsf->err[b] / (double)vsf->vsf[b];
        if (fabs(rel - 1.0 / sqrt((double)vsf->counts[b])) > 1e-5)
          bad_err++;
      }
    }
    CHECK(bad_err == 0, "%d bins with wrong Poisson error", bad_err);

    sif_size_function_free(vsf);
  }

  sif_catalog_free(cat);
  printf("  ok\n");
}

static void test_guards(void) {
  printf("input guards\n");

  sif_catalog_t* cat = make_catalog(100, 5.0f, 25.0f);

  CHECK(sif_size_function_catalog(cat, BOX, 0, 0, 0.0f, 0.0f) == NULL,
    "n_bins = 0 should be rejected");
  CHECK(sif_size_function_catalog(cat, -1.0f, 8, 0, 0.0f, 0.0f) == NULL,
    "a non-positive box_length should be rejected");
  CHECK(sif_size_function_catalog(NULL, BOX, 8, 0, 0.0f, 0.0f) == NULL,
    "a NULL catalog should be rejected");
  sif_catalog_free(cat);

  /* A catalog where every void has the same radius has no usable range. */
  sif_catalog_t* flat = sif_catalog_alloc(8);
  for (int i = 0; i < 8; i++)
    sif_catalog_append(flat, 0, 0, 0, 7.0f);
  CHECK(sif_size_function_catalog(flat, BOX, 8, 0, 0.0f, 0.0f) == NULL,
    "a degenerate radius range should be rejected");
  sif_catalog_free(flat);

  sif_size_function_free(NULL); /* must be quiet and not crash */
  printf("  ok\n");
}

static void test_combine(void) {
  printf("combine\n");

  sif_catalog_t* a = make_catalog(500, 5.0f, 15.0f);
  sif_catalog_t* b = make_catalog(500, 12.0f, 25.0f);

  sif_size_function_t* va =
    sif_size_function_catalog(a, BOX, 10, SIF_VSF_BIN_LN, 0.0f, 0.0f);
  sif_size_function_t* vb =
    sif_size_function_catalog(b, BOX, 10, SIF_VSF_BIN_LN, 0.0f, 0.0f);
  CHECK(va && vb, "inputs failed to compute");

  if (va && vb) {
    const sif_size_function_t* arr[2] = {va, vb};
    sif_size_function_t* m = sif_size_function_combine(
      arr, 2, 12, NULL, SIF_VSF_BIN_LN | SIF_VSF_MERGE_MEAN);
    CHECK(m != NULL, "merge returned NULL");
    if (m) {
      CHECK(m->n_bins == 12, "master has %u bins, expected 12", m->n_bins);
      CHECK(fabs((double)m->r_min - 5.0) < 1e-3, "master r_min = %g",
        (double)m->r_min);
      CHECK(fabs((double)m->r_max - 25.0) < 1e-3, "master r_max = %g",
        (double)m->r_max);
      int nonfinite = 0;
      for (uint32_t i = 0; i < m->n_bins; i++)
        if (!isfinite((double)m->vsf[i]) || !isfinite((double)m->err[i]))
          nonfinite++;
      CHECK(nonfinite == 0, "%d non-finite merged values", nonfinite);
      sif_size_function_free(m);
    }
  }

  sif_size_function_free(va);
  sif_size_function_free(vb);
  sif_catalog_free(a);
  sif_catalog_free(b);
  printf("  ok\n");
}

int main(void) {
  test_no_voids_dropped("linear bins", SIF_VSF_BIN_LINEAR);
  test_no_voids_dropped("log bins", SIF_VSF_BIN_LN);
  test_guards();
  test_combine();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
