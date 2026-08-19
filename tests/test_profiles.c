/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* Checks the density profile against a brute-force count.
 * If the scatter-accumulate in profiles.c loses updates -- which is what a
 * `#pragma omp simd` over that loop did -- the enclosed mass comes out
 * systematically low. */
#include "sif/core/system.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
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
    .log_level = SIF_LOG_LEVEL_WARNING};
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

  /* One void at the box center, and a second one placed so that its search
   * sphere reaches past the box face on a coarse mesh -- which is where a
   * cell walk that wraps naively visits the same cell twice. */
  const sif_real vx = 50.0f, vy = 50.0f, vz = 50.0f, vr = 10.0f;
  const sif_real ext = 3.0f;

  sif_catalog_t* cat = sif_catalog_alloc(4);
  sif_catalog_append(cat, vx, vy, vz, vr);
  sif_catalog_append(cat, 20.0f, 20.0f, 20.0f, vr);

  /* The estimator takes the mesh, not the field: it is the caller who decides
   * the resolution, and any of them has to give the same profile. */
  const uint32_t n_cells = sif_profiles_suggest_mesh_cells(N_P);
  sif_chain_mesh_t* mesh =
    sif_chain_mesh_alloc(n_cells, BOX, f, SIF_MESH_DROP_INDICES);
  if (!mesh) {
    printf("FAIL: could not build the chain mesh\n");
    return 1;
  }

  sif_density_profiles_t* dens = NULL;
  if (sif_profiles(cat, mesh, ext, N_BINS, SIF_PBC_PERIODIC, &dens, NULL) !=
      SIF_OK) {
    printf("FAIL: sif_profiles reported an error\n");
    return 1;
  }

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

  /* Resolution invariance. The caller now picks the mesh, and the whole
   * argument for letting them pass one built for something else is that the
   * profile does not depend on it. Counts of unit-weight tracers are exact in
   * float whatever order they are summed in, so this is an equality.
   *
   * The three resolutions bracket the suggested one from both sides: 1^3 and
   * 2^3 put every tracer in a handful of cells, and are coarse enough that a
   * void's search sphere reaches around the box, while 12^3 makes the walk
   * visit hundreds of cells per void. */
  const uint32_t other_cells[3] = {1, 2, 12};

  for (int k = 0; k < 3; k++) {
    sif_chain_mesh_t* other =
      sif_chain_mesh_alloc(other_cells[k], BOX, f, SIF_MESH_DROP_INDICES);
    sif_density_profiles_t* dens_other = NULL;

    if (!other || sif_profiles(cat, other, ext, N_BINS, SIF_PBC_PERIODIC,
                    &dens_other, NULL) != SIF_OK) {
      printf("FAIL: could not profile against a %u^3 mesh\n", other_cells[k]);
      return 1;
    }

    for (uint64_t v = 0; v < cat->n_voids; v++) {
      for (uint32_t b = 0; b < N_BINS; b++) {
        const double ref = (double)sif_density_profiles_get(dens, v)[b];
        const double got = (double)sif_density_profiles_get(dens_other, v)[b];

        if (ref != got) {
          printf("  void %llu resolution bin %2u: %u^3 mesh gives %+.7f, "
                 "%u^3 gives %+.7f  MISMATCH\n",
            (unsigned long long)v, b, n_cells, ref, other_cells[k], got);
          failures++;
        }
      }
    }

    sif_density_profiles_free(dens_other);
    sif_chain_mesh_free(other);
  }

  /* Weights. Every tracer carries the same power of two, so both the binned
   * mass and the mean density it is divided by scale by exactly that factor
   * and the contrast has to come back bit for bit unchanged. That exercises
   * the weighted kernel and the mesh's own weight total together. */
  sif_real* wgt = malloc(N_P * sizeof(sif_real));
  for (uint64_t i = 0; i < N_P; i++)
    wgt[i] = (sif_real)2.0;

  sif_field_t* fw = sif_field_alloc(N_P);
  sif_field_assign_positions(fw, x, y, z);
  sif_field_assign_weights(fw, wgt);

  sif_chain_mesh_t* mesh_w =
    sif_chain_mesh_alloc(n_cells, BOX, fw, SIF_MESH_DROP_INDICES);
  sif_density_profiles_t* dens_w = NULL;

  if (!mesh_w || sif_profiles(cat, mesh_w, ext, N_BINS, SIF_PBC_PERIODIC,
                   &dens_w, NULL) != SIF_OK) {
    printf("FAIL: could not profile a weighted field\n");
    return 1;
  }

  for (uint64_t v = 0; v < cat->n_voids; v++) {
    for (uint32_t b = 0; b < N_BINS; b++) {
      const double ref = (double)sif_density_profiles_get(dens, v)[b];
      const double got = (double)sif_density_profiles_get(dens_w, v)[b];

      if (ref != got) {
        printf("  void %llu weighted bin %2u: %+.7f, unweighted %+.7f  "
               "MISMATCH\n",
          (unsigned long long)v, b, got, ref);
        failures++;
      }
    }
  }

  /* Velocities, which the estimator now reads off the mesh rather than the
   * field. Every tracer is given the unit outward vector from the void centre,
   * through the same nearest image the estimator uses, so the mean radial
   * velocity of every populated shell has to come back as exactly 1. */
  sif_real* vel_x = malloc(N_P * sizeof(sif_real));
  sif_real* vel_y = malloc(N_P * sizeof(sif_real));
  sif_real* vel_z = malloc(N_P * sizeof(sif_real));

  for (uint64_t i = 0; i < N_P; i++) {
    double dx = (double)x[i] - vx, dy = (double)y[i] - vy,
           dz = (double)z[i] - vz;
    if (dx > half)
      dx -= BOX;
    if (dx < -half)
      dx += BOX;
    if (dy > half)
      dy -= BOX;
    if (dy < -half)
      dy += BOX;
    if (dz > half)
      dz -= BOX;
    if (dz < -half)
      dz += BOX;

    const double r = sqrt(dx * dx + dy * dy + dz * dz);
    const double inv_r = (r > 0.0) ? 1.0 / r : 0.0;
    vel_x[i] = (sif_real)(dx * inv_r);
    vel_y[i] = (sif_real)(dy * inv_r);
    vel_z[i] = (sif_real)(dz * inv_r);
  }

  sif_field_t* fv = sif_field_alloc(N_P);
  sif_field_assign_positions(fv, x, y, z);
  sif_field_assign_velocities(fv, vel_x, vel_y, vel_z);

  sif_chain_mesh_t* mesh_v =
    sif_chain_mesh_alloc(n_cells, BOX, fv, SIF_MESH_DROP_INDICES);
  sif_velocity_profiles_t* vel = NULL;

  if (!mesh_v || sif_profiles(cat, mesh_v, ext, N_BINS, SIF_PBC_PERIODIC, NULL,
                   &vel) != SIF_OK) {
    printf("FAIL: could not stack velocity profiles\n");
    return 1;
  }

  for (uint32_t b = 0; b < N_BINS; b++) {
    const double got = (double)sif_velocity_profiles_get(vel, 0)[b];

    if (fabs(got - 1.0) > 1e-5) {
      printf(
        "  velocity bin %2u: v_rad=%+.7f, expected +1  MISMATCH\n", b, got);
      failures++;
    }
  }

  printf("\n%s (%d mismatched bin%s)\n", failures ? "FAILED" : "PASSED",
    failures, failures == 1 ? "" : "s");

  sif_density_profiles_free(dens);
  sif_density_profiles_free(dens_w);
  sif_velocity_profiles_free(vel);
  sif_chain_mesh_free(mesh);
  sif_chain_mesh_free(mesh_v);
  sif_chain_mesh_free(mesh_w);
  sif_catalog_free(cat);
  sif_field_free(f);
  sif_field_free(fv);
  sif_field_free(fw);
  free(x);
  free(y);
  free(z);
  free(vel_x);
  free(vel_y);
  free(vel_z);
  free(wgt);
  sif_finalize();
  return failures != 0;
}
