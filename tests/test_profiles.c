/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* Checks the mesh density profile against a brute-force count.
 * If the `#pragma omp simd` scatter-accumulate in profiles.c loses updates,
 * the enclosed mass comes out systematically low. */
#include "sif/core/system.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalog.h"
#include "sif/structures/field.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "test_util.h"

#define BOX    100.0f
#define N_P    (SIF_TEST_SCALE(120000))
#define N_BINS 10

static uint64_t s = 0x9E3779B97F4A7C15ULL;
static double uni(void) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)((s >> 11) & 0x1FFFFFFFFFFFFFULL) /
         (double)0x20000000000000ULL;
}

int main(void) {
  sif_fft_config_t fftcfg = {.skip_tuning = true};
  sif_config_t cfg = {.fft_config = &fftcfg,
    .omp_config = NULL,
    .verbose = false,
    .log_level = 3};
  sif_init(&cfg);

  sif_real* x = malloc(N_P * sizeof(sif_real));
  sif_real* y = malloc(N_P * sizeof(sif_real));
  sif_real* z = malloc(N_P * sizeof(sif_real));
  for (uint64_t i = 0; i < N_P; i++) {
    x[i] = (sif_real)(uni() * BOX);
    y[i] = (sif_real)(uni() * BOX);
    z[i] = (sif_real)(uni() * BOX);
  }

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  /* One void at the box center. */
  const sif_real vx = 50.0f, vy = 50.0f, vz = 50.0f, vr = 10.0f;
  const sif_real ext = 3.0f;

  sif_catalog_t* cat = sif_catalog_alloc(4);
  sif_catalog_append(cat, vx, vy, vz, vr);

  sif_density_profiles_t* dens = NULL;
  sif_profiles_mesh(cat, f, BOX, ext, N_BINS, SIF_PBC_PERIODIC, &dens, NULL);

  if (!dens) {
    printf("FAIL: profile computation returned NULL\n");
    return 1;
  }

  const sif_real r_max = vr * ext;
  const sif_real half = BOX * 0.5f;
  const double mean_dens = (double)N_P / ((double)BOX * BOX * BOX);

  int failures = 0;

  /* Brute-force cumulative count per bin edge, with the same PBC convention. */
  for (uint32_t b = 0; b < N_BINS; b++) {
    const double r_outer = (double)(b + 1) * (r_max / N_BINS);

    uint64_t n_in = 0;
    for (uint64_t i = 0; i < N_P; i++) {
      double dx = fabs((double)x[i] - vx), dy = fabs((double)y[i] - vy),
             dz = fabs((double)z[i] - vz);
      if (dx > half)
        dx = BOX - dx;
      if (dy > half)
        dy = BOX - dy;
      if (dz > half)
        dz = BOX - dz;
      if (dx * dx + dy * dy + dz * dz <= r_outer * r_outer)
        n_in++;
    }

    const double vol = (4.0 / 3.0) * SIF_PI * r_outer * r_outer * r_outer;
    const double expected = ((double)n_in / vol) / mean_dens - 1.0;
    const double got = (double)sif_density_profiles_get(dens, 0)[b];

    /* Convert the profile back to an enclosed count to make any shortfall
     * legible as "particles lost". */
    const double got_count = (got + 1.0) * mean_dens * vol;

    const double err = fabs(got - expected);
    const char* verdict = (err < 2e-3) ? "ok" : "MISMATCH";
    if (err >= 2e-3)
      failures++;

    printf("  bin %2u  r<%6.2f  brute_n=%6llu  profile_n=%9.1f  "
           "delta=%+.5f vs %+.5f  %s\n",
      b, r_outer, (unsigned long long)n_in, got_count, got, expected, verdict);
  }

  printf("\n%s (%d mismatched bin%s)\n", failures ? "FAILED" : "PASSED",
    failures, failures == 1 ? "" : "s");

  sif_density_profiles_free(dens);
  sif_catalog_free(cat);
  sif_field_free(f);
  free(x);
  free(y);
  free(z);
  sif_finalize();
  return failures != 0;
}
