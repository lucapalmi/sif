/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The sampled tessellation: the two conservation laws it rests on, the volumes
 * it measures for a configuration whose answer is known exactly, and the claim
 * that motivates the whole thing -- that a profile measured from the samples is
 * smoother than one measured by counting tracers.
 *
 * A cubic lattice is the configuration used throughout, because every Voronoi
 * cell of a lattice is a cube of known volume, so the sampler can be checked
 * against arithmetic rather than against another estimator.
 */

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/structures/tessellation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: ");                                                      \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define BOX     100.0f
#define SIDE    20u /* tracers per axis; SIDE^3 in total */
#define SAMPLES 32u
#define SEED    20260819ULL

/* A cubic lattice, offset by half a spacing so no tracer sits on a face. */
static sif_field_t* make_lattice(void) {
  const uint64_t n = (uint64_t)SIDE * SIDE * SIDE;
  sif_field_t* f = sif_field_alloc(n);
  if (!f)
    return NULL;

  if (sif_field_reserve_positions(f) != SIF_OK) {
    sif_field_free(f);
    return NULL;
  }

  const double step = (double)BOX / (double)SIDE;

  for (uint64_t i = 0; i < SIDE; i++) {
    for (uint64_t j = 0; j < SIDE; j++) {
      for (uint64_t k = 0; k < SIDE; k++) {
        const uint64_t p = (i * SIDE + j) * SIDE + k;
        f->x[p] = (sif_real)(((double)i + 0.5) * step);
        f->y[p] = (sif_real)(((double)j + 0.5) * step);
        f->z[p] = (sif_real)(((double)k + 0.5) * step);
      }
    }
  }

  return f;
}

/* Root-mean-square of a profile that should be flat at zero. */
static double profile_rms(const sif_density_profiles_t* p, uint32_t n_bins) {
  double sum = 0.0;
  const sif_real* row = sif_density_profiles_get(p, 0);
  for (uint32_t b = 0; b < n_bins; b++)
    sum += (double)row[b] * (double)row[b];
  return sqrt(sum / (double)n_bins);
}

int main(void) {
  sif_fft_config_t fftcfg = {.skip_tuning = true};
  sif_config_t cfg = {.fft_config = &fftcfg,
    .omp_config = NULL,
    .log_level = SIF_LOG_LEVEL_WARNING};
  sif_init(&cfg);

  const uint64_t n_tracers = (uint64_t)SIDE * SIDE * SIDE;
  const double box_volume = (double)BOX * BOX * BOX;

  sif_field_t* f = make_lattice();
  CHECK(f != NULL, "lattice allocation failed");
  if (!f)
    return 1;

  sif_chain_mesh_t* mesh =
    sif_chain_mesh_alloc(8, BOX, f, SIF_MESH_DROP_INDICES);
  CHECK(mesh != NULL, "mesh construction failed");
  if (!mesh)
    return 1;

  printf(
    "sampling %" PRIu64 " tracers at %u samples each\n", n_tracers, SAMPLES);

  sif_tessellation_t* tess =
    sif_tessellation_alloc(mesh, SAMPLES, SEED, SIF_PBC_PERIODIC);
  CHECK(tess != NULL, "tessellation failed");
  if (!tess)
    return 1;

  /* --- what the sampler must conserve --- */

  CHECK(tess->n_samples == n_tracers * SAMPLES,
    "threw %" PRIu64 " samples, expected %" PRIu64, tess->n_samples,
    n_tracers * SAMPLES);

  /* Every sample lands in exactly one cell, so the cells account for the whole
   * box however noisy an individual one is.
   *
   * Weak on its own: the sum is n_samples * (box / n_samples) whatever the
   * samples did, so it holds even if they all landed in one corner. It catches
   * a mis-scaled sample_volume and nothing else, which is why the coverage
   * check below exists. */
  double total_volume = 0.0;
  for (uint64_t i = 0; i < tess->n_tracers; i++)
    total_volume += (double)tess->volumes[i];

  CHECK(fabs(total_volume / box_volume - 1.0) < 1e-6,
    "the cell volumes sum to %.6f of the box, not 1",
    total_volume / box_volume);

  /* That the samples actually fill the box. A stratification that goes wrong
   * -- too few samples for the cells, a flat index unpacked in the wrong order
   * -- leaves them in a slab or a pencil while every sum above still balances,
   * so this is the check that would notice. */
  sif_real lo[3] = {tess->box_length, tess->box_length, tess->box_length};
  sif_real hi[3] = {0, 0, 0};
  const sif_real* axis[3] = {
    tess->samples->x, tess->samples->y, tess->samples->z};

  for (int a = 0; a < 3; a++) {
    for (uint64_t j = 0; j < tess->samples->n_particles; j++) {
      if (axis[a][j] < lo[a])
        lo[a] = axis[a][j];
      if (axis[a][j] > hi[a])
        hi[a] = axis[a][j];
    }
  }

  for (int a = 0; a < 3; a++) {
    const double span = (double)(hi[a] - lo[a]) / (double)tess->box_length;
    CHECK(span > 0.99,
      "the samples span only %.3f of the box on axis %d; they are not spread "
      "over it",
      span, a);
  }

  /* And that they fill it evenly rather than merely reaching the far side:
   * every octant should hold an eighth of them. */
  uint64_t octant[8] = {0};
  const sif_real half = tess->box_length * (sif_real)0.5;
  for (uint64_t j = 0; j < tess->samples->n_particles; j++) {
    const int o = (tess->samples->x[j] >= half ? 1 : 0) |
                  (tess->samples->y[j] >= half ? 2 : 0) |
                  (tess->samples->z[j] >= half ? 4 : 0);
    octant[o]++;
  }

  const double expect = (double)tess->n_samples / 8.0;
  for (int o = 0; o < 8; o++) {
    const double ratio = (double)octant[o] / expect;
    CHECK(fabs(ratio - 1.0) < 0.05,
      "octant %d holds %.3f of its share of the samples", o, ratio);
  }

  /* The identity the profiles rest on: the samples carry the tracer mass. */
  double total_weight = 0.0;
  for (uint64_t j = 0; j < tess->samples->n_particles; j++)
    total_weight += (double)tess->samples->weights[j];

  CHECK(fabs(total_weight / (double)n_tracers - 1.0) < 1e-5,
    "the sample weights sum to %.6f tracers, not %" PRIu64, total_weight,
    n_tracers);

  CHECK(tess->n_empty == 0,
    "%" PRIu64 " lattice cells caught no sample, which should not happen at %u "
    "samples per tracer",
    tess->n_empty, SAMPLES);

  /* --- volumes against arithmetic --- */

  /* Every cell of a cubic lattice is a cube of exactly this volume, so the
   * sampler is being checked against a known answer rather than against
   * another estimator. The scatter is the sampling error, 1/sqrt(SAMPLES). */
  const double exact_volume = box_volume / (double)n_tracers;

  double mean = 0.0, mean_sq = 0.0;
  for (uint64_t i = 0; i < tess->n_tracers; i++) {
    const double v = (double)tess->volumes[i] / exact_volume;
    mean += v;
    mean_sq += v * v;
  }
  mean /= (double)tess->n_tracers;
  mean_sq /= (double)tess->n_tracers;

  const double scatter = sqrt(mean_sq - mean * mean);
  const double expected_scatter = 1.0 / sqrt((double)SAMPLES);

  printf("  cell volume: mean %.5f of exact, scatter %.4f (expected ~%.4f)\n",
    mean, scatter, expected_scatter);

  CHECK(fabs(mean - 1.0) < 5e-3, "cell volumes are biased: mean %.5f of exact",
    mean);

  /* Loose bounds: the point is that the scatter is the sampling error and not
   * something an order of magnitude away from it. */
  CHECK(scatter > 0.5 * expected_scatter && scatter < 2.0 * expected_scatter,
    "cell volume scatter %.4f is nowhere near the expected %.4f", scatter,
    expected_scatter);

  /* --- the profile the whole thing exists for --- */

  /* A sphere centred on a corner between eight tracers, so the configuration
   * is symmetric and the true enclosed density is uniform at every radius. */
  const uint32_t n_bins = 12;
  const sif_real ext = 1.5f;

  sif_catalog_t* cat = sif_catalog_alloc(2);
  sif_catalog_append(
    cat, (sif_real)0.0, (sif_real)0.0, (sif_real)0.0, (sif_real)20.0);

  sif_density_profiles_t* counted = NULL;
  CHECK(sif_profiles(
          cat, mesh, ext, n_bins, SIF_PBC_PERIODIC, &counted, NULL) == SIF_OK,
    "counting profile failed");

  sif_chain_mesh_t* smesh = sif_chain_mesh_alloc_tessellation(tess,
    sif_profiles_suggest_mesh_cells(tess->n_samples), SIF_MESH_DROP_INDICES);
  CHECK(smesh != NULL, "sample mesh construction failed");

  sif_density_profiles_t* sampled = NULL;
  if (smesh)
    CHECK(sif_profiles(cat, smesh, ext, n_bins, SIF_PBC_PERIODIC, &sampled,
            NULL) == SIF_OK,
      "tessellation profile failed");

  if (counted && sampled) {
    /* The lattice is exactly uniform, so the true enclosed density is the mean
     * at every radius and both profiles are measuring zero.
     *
     * Counting can only answer in whole tracers: it reads -1 inside the first
     * bin, which holds no tracer at all, and then swings by order unity as
     * each shell of the lattice crosses the boundary. The samples resolve that
     * boundary to a fraction of a cell, and what is left is sampling noise
     * that falls as one over the root of the samples the sphere contains. */
    const double n_per_volume = (double)tess->n_samples / box_volume;

    printf("  %4s %8s %10s %10s %9s\n", "bin", "r", "counting", "sampled",
      "tolerance");

    for (uint32_t b = 0; b < n_bins; b++) {
      const double r = (double)(b + 1) * 20.0 * (double)ext / (double)n_bins;
      const double volume = (4.0 / 3.0) * SIF_PI * r * r * r;
      const double n_samples_in = volume * n_per_volume;

      /* Five sigma of the count that produced it. Wide for the first bin,
       * which is a small sphere holding few samples, and tight by the last. */
      const double tolerance = 5.0 / sqrt(n_samples_in);

      const double count_val = (double)sif_density_profiles_get(counted, 0)[b];
      const double sample_val = (double)sif_density_profiles_get(sampled, 0)[b];

      printf("  %4u %8.2f %10.4f %10.4f %9.4f\n", b, r, count_val, sample_val,
        tolerance);

      CHECK(fabs(sample_val) < tolerance,
        "sampled bin %u reads %+.4f, outside %.4f of the zero it is measuring",
        b, sample_val, tolerance);
    }

    /* The failure mode the tessellation exists to fix, asserted rather than
     * described: no tracer lies within the first bin of a lattice sampled at
     * its own corner, so counting can only call it empty. */
    CHECK(fabs((double)sif_density_profiles_get(counted, 0)[0] + 1.0) < 1e-6,
      "the counted first bin should read exactly -1 on this lattice");

    const double rms_counted = profile_rms(counted, n_bins);
    const double rms_sampled = profile_rms(sampled, n_bins);

    printf("  rms about zero: counting %.5f, samples %.5f (%.1fx flatter)\n",
      rms_counted, rms_sampled, rms_counted / rms_sampled);

    CHECK(rms_sampled < 0.5 * rms_counted,
      "the sampled profile (rms %.5f) is no flatter than the counted one "
      "(rms %.5f)",
      rms_sampled, rms_counted);
  }

  /* --- velocities --- */

  /* Carried from the owner unchanged, so a mean over samples weights by volume
   * rather than by tracer. Given every tracer the same velocity, any mean of
   * any subset has to return it, which is enough to catch a payload that was
   * dropped, mis-strided, or shared out like a weight. */
  {
    sif_field_t* fv = sif_field_alloc(n_tracers);
    if (fv && sif_field_reserve_positions(fv) == SIF_OK &&
        sif_field_reserve_velocities(fv) == SIF_OK) {
      for (uint64_t i = 0; i < n_tracers; i++) {
        fv->x[i] = f->x[i];
        fv->y[i] = f->y[i];
        fv->z[i] = f->z[i];
        fv->vx[i] = (sif_real)3.0;
        fv->vy[i] = (sif_real)-4.0;
        fv->vz[i] = (sif_real)0.5;
      }

      sif_chain_mesh_t* mv =
        sif_chain_mesh_alloc(8, BOX, fv, SIF_MESH_DROP_INDICES);
      sif_tessellation_t* tv =
        mv ? sif_tessellation_alloc(mv, SAMPLES, SEED, SIF_PBC_PERIODIC) : NULL;

      CHECK(tv != NULL, "tessellating a field with velocities failed");

      if (tv) {
        CHECK(tv->samples->vx != NULL,
          "the samples dropped the velocities the mesh carried");

        int exact = 1;
        for (uint64_t j = 0; j < tv->samples->n_particles; j++) {
          if (tv->samples->vx[j] != (sif_real)3.0 ||
              tv->samples->vy[j] != (sif_real)-4.0 ||
              tv->samples->vz[j] != (sif_real)0.5)
            exact = 0;
        }
        CHECK(exact, "a sample does not carry its owner's velocity");
      }

      sif_tessellation_free(tv);
      sif_chain_mesh_free(mv);
    }
    sif_field_free(fv);
  }

  /* A tessellation of a field without velocities must not invent any. */
  CHECK(tess->samples->vx == NULL,
    "the samples carry velocities the tracers never had");

  /* --- depositing onto a grid --- */

  /* No lattice cell went unsampled, so the samples carry the whole tracer
   * weight and the two deposits have to agree on the total. What they will not
   * agree on is where that weight sits, which is the point. */
  sif_grid_t* g_tracers = sif_grid_alloc(16, BOX);
  sif_grid_t* g_tess = sif_grid_alloc(16, BOX);

  if (g_tracers && g_tess) {
    sif_grid_assign_cic(g_tracers, f);
    sif_grid_assign_cic_tessellation(g_tess, tess);

    CHECK(g_tess->content == SIF_GRID_DENSITY,
      "the deposited grid does not report holding a density");

    double sum_tracers = 0.0, sum_tess = 0.0;
    for (uint64_t c = 0; c < g_tracers->total_cells; c++) {
      sum_tracers += (double)g_tracers->values[c];
      sum_tess += (double)g_tess->values[c];
    }

    printf(
      "  grid total: tracers %.6e, tessellation %.6e\n", sum_tracers, sum_tess);

    CHECK(fabs(sum_tess / sum_tracers - 1.0) < 1e-5,
      "the two deposits disagree on the total density by %.2e",
      fabs(sum_tess / sum_tracers - 1.0));
  }

  /* The case the deposit exists for: tracers too sparse for the grid. Counting
   * them leaves most cells holding nothing at all, and a cell that was never
   * sampled cannot be smoothed back into existence. */
  {
    const uint32_t n_sparse = 4000;
    const uint32_t grid_side = 24;

    sif_field_t* sparse = sif_field_alloc(n_sparse);
    if (sparse && sif_field_reserve_positions(sparse) == SIF_OK) {
      uint64_t r = 0x2545F4914F6CDD1DULL;
      for (uint32_t i = 0; i < n_sparse; i++) {
        sif_real* axis[3] = {sparse->x, sparse->y, sparse->z};
        for (int a = 0; a < 3; a++) {
          r = r * 6364136223846793005ULL + 1442695040888963407ULL;
          const double u = (double)((r >> 11) & 0x1FFFFFFFFFFFFFULL) /
                           (double)0x20000000000000ULL;
          axis[a][i] = (sif_real)(u * (double)BOX * 0.999999);
        }
      }

      sif_chain_mesh_t* smesh_sparse =
        sif_chain_mesh_alloc(8, BOX, sparse, SIF_MESH_DROP_INDICES);
      sif_tessellation_t* stess = smesh_sparse
                                    ? sif_tessellation_alloc(smesh_sparse,
                                        SAMPLES, SEED, SIF_PBC_PERIODIC)
                                    : NULL;

      sif_grid_t* g_count = sif_grid_alloc(grid_side, BOX);
      sif_grid_t* g_vtfe = sif_grid_alloc(grid_side, BOX);

      if (stess && g_count && g_vtfe) {
        sif_grid_assign_cic(g_count, sparse);
        sif_grid_to_density_contrast(g_count);

        sif_grid_assign_cic_tessellation(g_vtfe, stess);
        sif_grid_to_density_contrast(g_vtfe);

        uint64_t empty_count = 0, empty_vtfe = 0;
        double sq_count = 0.0, sq_vtfe = 0.0;

        for (uint64_t c = 0; c < g_count->total_cells; c++) {
          const double dc = (double)g_count->values[c];
          const double dv = (double)g_vtfe->values[c];
          sq_count += dc * dc;
          sq_vtfe += dv * dv;
          if (dc < -0.999999)
            empty_count++;
          if (dv < -0.999999)
            empty_vtfe++;
        }

        const double n = (double)g_count->total_cells;
        const double rms_count = sqrt(sq_count / n);
        const double rms_vtfe = sqrt(sq_vtfe / n);

        printf("  %u tracers on a %u^3 grid (%.2f per cell):\n", n_sparse,
          grid_side, (double)n_sparse / n);
        printf("    tracers      rms(delta) %6.3f, %5.1f%% of cells empty\n",
          rms_count, 100.0 * (double)empty_count / n);
        printf("    tessellation rms(delta) %6.3f, %5.1f%% of cells empty\n",
          rms_vtfe, 100.0 * (double)empty_vtfe / n);

        /* The underlying density is uniform, so every delta is an error and a
         * smaller spread is a better field, not merely a smoother one. */
        CHECK(rms_vtfe < rms_count,
          "the tessellation grid (rms %.3f) is no better than the tracer grid "
          "(rms %.3f)",
          rms_vtfe, rms_count);
        CHECK(empty_vtfe < empty_count,
          "the tessellation left %" PRIu64 " cells empty against the tracers' "
          "%" PRIu64,
          empty_vtfe, empty_count);
      }

      sif_grid_free(g_count);
      sif_grid_free(g_vtfe);
      sif_tessellation_free(stess);
      sif_chain_mesh_free(smesh_sparse);
    }
    sif_field_free(sparse);
  }

  /* A grid spanning a different box would not fail, it would just put the
   * density on the wrong scale. */
  sif_grid_t* wrong_box = sif_grid_alloc(8, BOX * 2.0f);
  if (wrong_box) {
    sif_grid_assign_cic_tessellation(wrong_box, tess);
    CHECK(wrong_box->content == SIF_GRID_EMPTY,
      "a grid spanning a different box should have been refused");
    sif_grid_free(wrong_box);
  }

  /* --- handing the samples over --- */

  sif_chain_mesh_t* consumed =
    sif_chain_mesh_alloc_tessellation_consume(tess, 8, SIF_MESH_DROP_INDICES);

  CHECK(consumed != NULL, "consuming mesh construction failed");
  if (consumed) {
    CHECK(consumed->n_particles == tess->n_samples,
      "the consumed mesh holds %" PRIu64 " samples, expected %" PRIu64,
      consumed->n_particles, tess->n_samples);
    CHECK(tess->samples->n_particles == 0,
      "the samples were copied rather than taken over");
    CHECK(tess->volumes != NULL,
      "consuming the samples should leave the volumes alone");
  }

  /* Nor can a consumed tessellation be deposited. */
  sif_grid_t* after_consume = sif_grid_alloc(8, BOX);
  if (after_consume) {
    sif_grid_assign_cic_tessellation(after_consume, tess);
    CHECK(after_consume->content == SIF_GRID_EMPTY,
      "consumed samples should not deposit onto a grid");
    sif_grid_free(after_consume);
  }

  /* Asking twice has nothing left to give, and has to say so. */
  sif_chain_mesh_t* twice =
    sif_chain_mesh_alloc_tessellation(tess, 8, SIF_MESH_DROP_INDICES);
  CHECK(twice == NULL, "consumed samples should not produce a second mesh");
  sif_chain_mesh_free(twice);

  CHECK(sif_tessellation_alloc(mesh, 0, SEED, SIF_PBC_PERIODIC) == NULL,
    "zero samples per tracer should be refused");

  /* A mesh with more cells than the run would have samples cannot be
   * stratified one sample per cell, and sampling it anyway would quietly
   * cover a corner of the box. */
  {
    sif_chain_mesh_t* fine =
      sif_chain_mesh_alloc(64, BOX, f, SIF_MESH_DROP_INDICES);
    if (fine) {
      CHECK(fine->total_cells > n_tracers,
        "this check needs a mesh finer than one sample per tracer");
      CHECK(sif_tessellation_alloc(fine, 1, SEED, SIF_PBC_PERIODIC) == NULL,
        "a mesh with more cells than samples should be refused");
      sif_chain_mesh_free(fine);
    }
  }
  CHECK(sif_tessellation_alloc(NULL, SAMPLES, SEED, SIF_PBC_PERIODIC) == NULL,
    "a NULL mesh should be refused");

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");

  sif_grid_free(g_tracers);
  sif_grid_free(g_tess);
  sif_chain_mesh_free(consumed);
  sif_chain_mesh_free(smesh);
  sif_density_profiles_free(counted);
  sif_density_profiles_free(sampled);
  sif_catalog_free(cat);
  sif_tessellation_free(tess);
  sif_chain_mesh_free(mesh);
  sif_field_free(f);
  sif_finalize();

  return failures ? 1 : 0;
}
