/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* Covers the field/grid rework: Morton permutation coherence, the require_*
 * helpers, CIC mass weighting and the box contract. */
#include "sif/core/settings.h"
#include "sif/core/system.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/structures/octree.h"
#include "sif/utils/align.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define N_P (SIF_TEST_SCALE(4096) < 512 ? 512 : SIF_TEST_SCALE(4096))
#define BOX 100.0f

static void make_positions(sif_real* x, sif_real* y, sif_real* z) {
  uint64_t s = 99;
  for (uint64_t i = 0; i < N_P; i++) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    x[i] = (sif_real)((double)((s >> 33) & 0xFFFF) / 65536.0 * (double)BOX);
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    y[i] = (sif_real)((double)((s >> 33) & 0xFFFF) / 65536.0 * (double)BOX);
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    z[i] = (sif_real)((double)((s >> 33) & 0xFFFF) / 65536.0 * (double)BOX);
  }
}

/* The headline coherence bug: assigning velocities AFTER a Morton sort used to
 * pair every particle with someone else's velocity. */
static void test_velocity_permutation(void) {
  printf("velocity assignment after morton sort\n");

  sif_real *x = malloc(N_P * sizeof(sif_real)),
           *y = malloc(N_P * sizeof(sif_real)),
           *z = malloc(N_P * sizeof(sif_real));
  make_positions(x, y, z);

  /* Tag each particle's velocity with its position so a mismatch is visible. */
  sif_real *vx = malloc(N_P * sizeof(sif_real)),
           *vy = malloc(N_P * sizeof(sif_real)),
           *vz = malloc(N_P * sizeof(sif_real));
  sif_real* m = malloc(N_P * sizeof(sif_real));
  for (uint64_t i = 0; i < N_P; i++) {
    vx[i] = x[i];
    vy[i] = y[i];
    vz[i] = z[i];
    m[i] = x[i];
  }

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  CHECK(sif_field_sort_morton(f) == SIF_OK, "sort_morton failed");
  CHECK((f->state_flags & SIF_FIELD_STATE_MORTON_SORTED) != 0, "flag not set");

  /* Assign in ORIGINAL order, after the sort. */
  CHECK(sif_field_assign_velocities(f, vx, vy, vz) == SIF_OK,
    "assign_velocities failed");
  CHECK(sif_field_assign_weights(f, m) == SIF_OK, "assign_weights failed");

  int mismatched = 0;
  for (uint64_t i = 0; i < N_P; i++) {
    if (f->vx[i] != f->x[i] || f->vy[i] != f->y[i] || f->vz[i] != f->z[i])
      mismatched++;
    if (f->weights[i] != f->x[i])
      mismatched++;
  }
  CHECK(
    mismatched == 0, "%d particles carry the wrong velocity/mass", mismatched);

  /* The permutation is applied on the way in, so the caller's arrays are left
   * in their original order. */
  int caller_disturbed = 0;
  for (uint64_t i = 0; i < N_P; i++) {
    if (vx[i] != x[i] || vy[i] != y[i] || vz[i] != z[i] || m[i] != x[i])
      caller_disturbed++;
  }
  CHECK(
    caller_disturbed == 0, "%d caller entries were modified", caller_disturbed);

  sif_field_free(f);
  free(x);
  free(y);
  free(z);
  free(vx);
  free(vy);
  free(vz);
  free(m);
  printf("  ok\n");
}

/*
 * Moving the positions after a sort -- translating them, or wrapping one back
 * into the box -- clears the Morton flag but leaves the arrays in sorted
 * order, so data assigned afterwards in the caller's order still has to be
 * permuted on the way in. It used not to be once the flag was gone, and every
 * particle got someone else's weight.
 */
static void test_move_after_sort_keeps_permutation(void) {
  printf("translate and wrap after a morton sort keep the permutation\n");

  sif_real *x = malloc(N_P * sizeof(sif_real)),
           *y = malloc(N_P * sizeof(sif_real)),
           *z = malloc(N_P * sizeof(sif_real));
  make_positions(x, y, z);

  sif_real* m = malloc(N_P * sizeof(sif_real));
  for (uint64_t i = 0; i < N_P; i++)
    m[i] = (sif_real)i;

  for (int how = 0; how < 2; how++) {
    sif_field_t* f = sif_field_alloc(N_P);
    sif_field_assign_positions(f, x, y, z);
    CHECK(sif_field_sort_morton(f) == SIF_OK, "sort_morton failed");

    if (how == 0) {
      sif_real* before = malloc(N_P * sizeof(sif_real));
      memcpy(before, f->x, N_P * sizeof(sif_real));

      const sif_real offset[3] = {12.5f, -3.25f, 1000.0f};
      CHECK(sif_field_translate(f, offset) == SIF_OK, "translate failed");

      int moved_wrong = 0;
      for (uint64_t i = 0; i < N_P; i++)
        moved_wrong += f->x[i] != before[i] + offset[0];
      CHECK(moved_wrong == 0, "%d positions were not shifted by the offset",
        moved_wrong);
      free(before);
    } else {
      /* One coordinate nudged out of the box, so the wrap actually moves
       * something and drops the flag. */
      f->x[N_P / 2] = -1.0f;
      CHECK(
        sif_field_wrap_periodic(f, BOX, NULL, NULL) == SIF_OK, "wrap failed");
    }

    CHECK((f->state_flags & SIF_FIELD_STATE_MORTON_SORTED) == 0,
      "%s should drop the Morton flag", how ? "wrap" : "translate");
    CHECK((f->state_flags & SIF_FIELD_STATE_BOUNDS_VALID) == 0,
      "%s should invalidate the bounds", how ? "wrap" : "translate");

    CHECK(sif_field_assign_weights(f, m) == SIF_OK, "assign_weights failed");

    int wrong = 0;
    for (uint64_t i = 0; i < N_P; i++)
      wrong += f->weights[i] != m[f->original_indices[i]];
    CHECK(wrong == 0, "after %s, %d particles carry another's weight",
      how ? "wrap" : "translate", wrong);

    sif_field_free(f);
  }

  CHECK(sif_field_translate(NULL, (sif_real[3]){0, 0, 0}) == SIF_ERR_INVALID,
    "translate(NULL) should be SIF_ERR_INVALID");

  free(x);
  free(y);
  free(z);
  free(m);
  printf("  ok\n");
}

/* Assigning BEFORE the sort must also stay coherent: the sort permutes
 * everything together. */
static void test_sort_permutes_everything(void) {
  printf("morton sort permutes velocities with positions\n");

  sif_real *x = malloc(N_P * sizeof(sif_real)),
           *y = malloc(N_P * sizeof(sif_real)),
           *z = malloc(N_P * sizeof(sif_real));
  make_positions(x, y, z);

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);
  CHECK(sif_field_assign_velocities(f, x, y, z) == SIF_OK,
    "assign_velocities failed");
  CHECK(sif_field_sort_morton(f) == SIF_OK, "sort failed");

  int mismatched = 0;
  for (uint64_t i = 0; i < N_P; i++)
    if (f->vx[i] != f->x[i] || f->vy[i] != f->y[i] || f->vz[i] != f->z[i])
      mismatched++;
  CHECK(mismatched == 0, "%d particles desynced during the sort", mismatched);

  /* original_indices must be a genuine permutation. */
  uint8_t* seen = calloc(N_P, 1);
  int bad = 0;
  for (uint64_t i = 0; i < N_P; i++) {
    uint64_t o = f->original_indices[i];
    if (o >= N_P || seen[o])
      bad++;
    else
      seen[o] = 1;
  }
  CHECK(bad == 0, "original_indices is not a permutation (%d bad)", bad);
  free(seen);

  sif_field_free(f);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

static void test_require_helpers(void) {
  printf("require_morton / require_bounds\n");

  sif_real *x = malloc(N_P * sizeof(sif_real)),
           *y = malloc(N_P * sizeof(sif_real)),
           *z = malloc(N_P * sizeof(sif_real));
  make_positions(x, y, z);

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  CHECK(sif_field_require_bounds(f) == SIF_OK, "require_bounds failed");
  CHECK(sif_field_require_bounds(f) == SIF_OK, "require_bounds not idempotent");
  CHECK(f->max_p[0] > f->min_p[0], "bounds look wrong");

  /* The octree used to hard-fail on an unsorted field; it must now sort it. */
  CHECK((f->state_flags & SIF_FIELD_STATE_MORTON_SORTED) == 0,
    "field should not be sorted yet");

  sif_octree_t* tree = sif_octree_alloc(f, 16);
  CHECK(tree != NULL, "octree build should succeed on an unsorted field now");
  if (tree) {
    CHECK((f->state_flags & SIF_FIELD_STATE_MORTON_SORTED) != 0,
      "octree build should have sorted the field");
    sif_octree_free(tree);
  }

  CHECK(sif_field_require_morton(f) == SIF_OK, "require_morton not idempotent");

  sif_field_free(f);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

/* CIC must now honour per-particle weights and conserve total mass. */
static void test_cic_mass(void) {
  printf("cic mass weighting and conservation\n");

  sif_real *x = malloc(N_P * sizeof(sif_real)),
           *y = malloc(N_P * sizeof(sif_real)),
           *z = malloc(N_P * sizeof(sif_real));
  make_positions(x, y, z);

  sif_real* m = malloc(N_P * sizeof(sif_real));
  for (uint64_t i = 0; i < N_P; i++)
    m[i] = 2.0f; /* uniform but != 1, so ignoring weights is detectable */

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);
  sif_field_assign_weights(f, m);

  const uint32_t n = 16;
  sif_grid_t* g = sif_grid_alloc(n, BOX);
  CHECK(sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");

  const uint64_t total = (uint64_t)n * n * n;
  const sif_real cell_vol = g->cell_length * g->cell_length * g->cell_length;
  double sum = 0.0;
  for (uint64_t i = 0; i < total; i++)
    sum += (double)g->values[i];
  sum *= (double)cell_vol; /* delta holds density, undo the 1/V */

  const double expected = 2.0 * (double)N_P;
  CHECK(fabs(sum - expected) / expected < 1e-4,
    "mass not conserved: got %.6g, expected %.6g", sum, expected);
  printf("  deposited mass: %.6g (expected %.6g)\n", sum, expected);

  /* Overdensity must average to ~0. */
  CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");
  double mean = 0.0;
  for (uint64_t i = 0; i < total; i++)
    mean += (double)g->values[i];
  mean /= (double)total;
  CHECK(fabs(mean) < 1e-3, "mean overdensity is %.3g, expected ~0", mean);

  sif_grid_free(g);
  sif_field_free(f);
  free(x);
  free(y);
  free(z);
  free(m);
  printf("  ok\n");
}

/*
 * The counting sort runs a tile at a time so its index array does not scale
 * with the field. How many tiles that takes is a memory decision and must not
 * be a physics one: the same particles have to land in the same cells whether
 * they are sorted in one pass or twenty.
 *
 * Exact equality is not the claim -- weight accumulates into a cell in tile
 * order, so the last bits move, which is why GRID_CIC_VERSION was bumped. What
 * must hold is that no particle is dropped, duplicated, or deposited into the
 * wrong cell, and a tile-boundary bug shows up as all three.
 */
static void test_cic_tiling_is_invisible(void) {
  printf("cic tiling does not change the field\n");

  sif_real *x = malloc(N_P * sizeof(sif_real)),
           *y = malloc(N_P * sizeof(sif_real)),
           *z = malloc(N_P * sizeof(sif_real));
  make_positions(x, y, z);

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  const uint32_t n = 16;
  const uint64_t total = (uint64_t)n * n * n;

  /* One tile for the whole field, against tiles small enough that a boundary
   * falls inside almost every chunk. */
  const char* tilings[] = {"100000000", "997", "64", "7"};
  sif_real* reference = malloc(total * sizeof(sif_real));

  for (int t = 0; t < 4; t++) {
    sif_setting_set("cic_tile_particles", tilings[t]);

    sif_grid_t* g = sif_grid_alloc(n, BOX);
    CHECK(sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");

    const sif_real cell_vol = g->cell_length * g->cell_length * g->cell_length;
    double sum = 0.0;
    for (uint64_t i = 0; i < total; i++)
      sum += (double)g->values[i];
    sum *= (double)cell_vol;

    CHECK(fabs(sum - (double)N_P) / (double)N_P < 1e-4,
      "tile size %s deposited %.6g particles, expected %llu", tilings[t], sum,
      (unsigned long long)N_P);

    if (t == 0) {
      memcpy(reference, g->values, total * sizeof(sif_real));
    } else {
      double worst = 0.0;
      for (uint64_t i = 0; i < total; i++) {
        double d = fabs((double)g->values[i] - (double)reference[i]);
        if (d > worst)
          worst = d;
      }
      /* Against the mean cell occupancy, so the tolerance means something. */
      const double mean = (double)N_P / (double)total / (double)cell_vol;
      CHECK(worst < 1e-4 * mean,
        "tile size %s moved a cell by %.3g (mean occupancy %.3g)", tilings[t],
        worst, mean);
    }

    sif_grid_free(g);
  }

  sif_setting_set("cic_tile_particles", SIF__CIC_TILE_PARTICLES_DEFAULT);

  free(reference);
  sif_field_free(f);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

/*
 * The .xgrid cache has to key on what actually determines the grid.
 *
 * It used to key on the particle count, the box, a flag saying whether weights
 * existed, and five sampled positions -- so two runs over identical positions
 * with different mass *values* produced the same key, and the second silently
 * received the first one's grid. That is the case pinned here: no collision is
 * involved, it was a guaranteed wrong answer.
 */
static void test_cic_cache_keys_on_content(void) {
  printf("cic cache distinguishes fields by content\n");

  sif_real *x = malloc(N_P * sizeof(sif_real)),
           *y = malloc(N_P * sizeof(sif_real)),
           *z = malloc(N_P * sizeof(sif_real));
  make_positions(x, y, z);

  sif_real* m1 = malloc(N_P * sizeof(sif_real));
  sif_real* m2 = malloc(N_P * sizeof(sif_real));
  for (uint64_t i = 0; i < N_P; i++) {
    m1[i] = 1.0f;
    m2[i] = 3.0f; /* same positions, three times the weight */
  }

  /* The cache is opt-in; HOME points into the build tree for the test run, so
   * this writes nothing outside it. */
  sif_setting_set("grid_cache_enabled", "1");

  const uint32_t n = 16;
  const uint64_t total = (uint64_t)n * n * n;

  sif_field_t* f1 = sif_field_alloc(N_P);
  sif_field_assign_positions(f1, x, y, z);
  sif_field_assign_weights(f1, m1);

  sif_grid_t* g1 = sif_grid_alloc(n, BOX);
  CHECK(sif_grid_assign_cic(g1, f1) == SIF_OK, "CIC assignment failed");

  double sum1 = 0.0;
  for (uint64_t i = 0; i < total; i++)
    sum1 += (double)g1->values[i];

  sif_field_t* f2 = sif_field_alloc(N_P);
  sif_field_assign_positions(f2, x, y, z);
  sif_field_assign_weights(f2, m2);

  sif_grid_t* g2 = sif_grid_alloc(n, BOX);
  CHECK(sif_grid_assign_cic(g2, f2) == SIF_OK, "CIC assignment failed");

  double sum2 = 0.0;
  for (uint64_t i = 0; i < total; i++)
    sum2 += (double)g2->values[i];

  CHECK(fabs(sum2 / sum1 - 3.0) < 1e-3,
    "three times the mass gave %.6g times the grid, not 3 -- the second run "
    "was served the first one's cache entry",
    sum2 / sum1);

  /* And the cache must still work: the same field again is a hit, and a hit
   * has to reproduce the computed grid exactly. */
  sif_grid_t* g1_again = sif_grid_alloc(n, BOX);
  CHECK(sif_grid_assign_cic(g1_again, f1) == SIF_OK, "CIC assignment failed");

  int identical = 1;
  for (uint64_t i = 0; i < total; i++)
    if (g1->values[i] != g1_again->values[i])
      identical = 0;
  CHECK(identical, "a cache hit did not reproduce the grid bit for bit");

  sif_setting_set("grid_cache_enabled", "0");

  sif_grid_free(g1_again);
  sif_grid_free(g2);
  sif_grid_free(g1);
  sif_field_free(f2);
  sif_field_free(f1);
  free(x);
  free(y);
  free(z);
  free(m1);
  free(m2);
  printf("  ok\n");
}

static void test_cic_rejects_out_of_box(void) {
  printf("cic box contract\n");

  sif_real *x = malloc(N_P * sizeof(sif_real)),
           *y = malloc(N_P * sizeof(sif_real)),
           *z = malloc(N_P * sizeof(sif_real));
  make_positions(x, y, z);
  x[7] = BOX + 5.0f; /* used to underflow the local slab index */

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  sif_grid_t* g = sif_grid_alloc(16, BOX);

  /* Something in the cells beforehand, so "left as it was" means something:
   * a refusal must not zero the grid, and must not claim it holds a
   * density. */
  for (uint64_t i = 0; i < g->total_cells; i++)
    g->values[i] = 7.0f;

  CHECK(sif_grid_assign_cic(g, f) == SIF_ERR_RANGE,
    "an out-of-box particle should be SIF_ERR_RANGE");

  int touched = 0;
  for (uint64_t i = 0; i < g->total_cells; i++)
    touched += g->values[i] != 7.0f;
  CHECK(touched == 0, "a refused assignment changed %d cells", touched);
  CHECK(g->content == SIF_GRID_EMPTY,
    "a refused assignment should not mark the grid as a density");

  /* Nothing to deposit is a refusal too, not a grid of zeros. */
  sif_field_t* empty = sif_field_alloc(0);
  CHECK(sif_grid_assign_cic(g, empty) == SIF_ERR_INVALID,
    "an empty field should be SIF_ERR_INVALID");
  sif_field_free(empty);

  CHECK(sif_grid_assign_cic(g, NULL) == SIF_ERR_INVALID,
    "a NULL field should be SIF_ERR_INVALID");
  CHECK(sif_grid_assign_cic(NULL, f) == SIF_ERR_INVALID,
    "a NULL grid should be SIF_ERR_INVALID");

  /* And a field that is fine deposits, with SIF_OK. */
  x[7] = 1.0f;
  sif_field_assign_positions(f, x, y, z);
  CHECK(sif_grid_assign_cic(g, f) == SIF_OK && g->content == SIF_GRID_DENSITY,
    "a valid field should deposit");

  /* Freeing a grid whose buffer was taken must not leak the struct. */
  sif_free_aligned(g->values);
  g->values = NULL;
  sif_grid_free(g);

  sif_field_free(f);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

/*
 * The wrap has to satisfy the binning contract exactly, not approximately: its
 * whole purpose is to turn a field the validators reject into one they accept.
 */
static void test_wrap_periodic(void) {
  printf("periodic wrap\n");

  sif_real *x = malloc(N_P * sizeof(sif_real)),
           *y = malloc(N_P * sizeof(sif_real)),
           *z = malloc(N_P * sizeof(sif_real));
  make_positions(x, y, z);

  /* The rounding artifact this exists for: stored exactly on the edge. */
  x[3] = BOX;
  y[9] = BOX;

  /* A negative small enough that -eps + BOX is not representable and rounds
   * back to BOX. Without the final clamp the wrap would return it still out of
   * range, which is the trap the implementation guards. */
  z[11] = -1e-6f;

  /* Genuinely outside, in both directions. */
  x[21] = BOX * 2.5f;
  y[33] = -BOX * 1.5f;

  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);

  /* A stale bounds cache must not survive the move. */
  CHECK(sif_field_require_bounds(f) == SIF_OK, "require_bounds failed");

  uint64_t boundary = 0, wrapped = 0;
  CHECK(sif_field_wrap_periodic(f, BOX, &boundary, &wrapped) == SIF_OK,
    "wrap_periodic failed");

  CHECK(boundary == 2, "expected 2 boundary folds, got %llu",
    (unsigned long long)boundary);
  CHECK(wrapped == 3, "expected 3 genuine wraps, got %llu",
    (unsigned long long)wrapped);

  CHECK(
    f->x[3] == 0.0f, "x on the edge should fold to 0, got %g", (double)f->x[3]);
  CHECK(
    f->y[9] == 0.0f, "y on the edge should fold to 0, got %g", (double)f->y[9]);

  /* The clamp case: whatever it lands on, it must be inside the box. */
  CHECK(z[11] + BOX == BOX, "test premise: -1e-6 + BOX should round to BOX");
  CHECK(f->z[11] >= 0.0f && f->z[11] < BOX,
    "a tiny negative must land inside the box, got %g", (double)f->z[11]);

  CHECK((f->state_flags & SIF_FIELD_STATE_BOUNDS_VALID) == 0,
    "the bounds cache should be invalidated by a wrap");

  /* The contract itself: every coordinate now satisfies what the CIC and the
   * chain mesh require. */
  uint64_t bad = 0;
  for (uint64_t i = 0; i < N_P; i++) {
    if (!(f->x[i] >= 0.0f && f->x[i] < BOX) ||
        !(f->y[i] >= 0.0f && f->y[i] < BOX) ||
        !(f->z[i] >= 0.0f && f->z[i] < BOX))
      bad++;
  }
  CHECK(bad == 0, "%llu coordinates still outside the box after wrapping",
    (unsigned long long)bad);

  /* And the field the validators used to reject now assigns. */
  sif_grid_t* g = sif_grid_alloc(16, BOX);
  CHECK(sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
  double sum = 0.0;
  for (uint64_t i = 0; i < g->total_cells; i++)
    sum += (double)g->values[i];
  CHECK(sum > 0.0, "a wrapped field should assign, got total mass %g", sum);
  sif_grid_free(g);

  /* Idempotent: a field already inside the box is left alone. */
  boundary = wrapped = 1;
  CHECK(sif_field_wrap_periodic(f, BOX, &boundary, &wrapped) == SIF_OK,
    "second wrap failed");
  CHECK(boundary == 0 && wrapped == 0,
    "wrapping an in-box field should move nothing, got %llu/%llu",
    (unsigned long long)boundary, (unsigned long long)wrapped);

  CHECK(sif_field_wrap_periodic(f, 0.0f, NULL, NULL) == SIF_ERR_INVALID,
    "a non-positive box should be rejected");

  sif_field_free(f);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

static void test_grid_content_tag(void) {
  printf("grid content tag\n");

  sif_grid_t* g = sif_grid_alloc(8, 100.0f);
  if (!g)
    return;
  CHECK(g->content == SIF_GRID_EMPTY, "a fresh grid should be EMPTY");

  for (uint64_t i = 0; i < g->total_cells; i++)
    g->values[i] = (sif_real)(1.0 + 0.01 * (double)i);
  g->content = SIF_GRID_DENSITY;

  CHECK(sif_grid_to_density_contrast(g) == SIF_OK,
    "conversion of a positive density should succeed");
  CHECK(g->content == SIF_GRID_DENSITY_CONTRAST,
    "conversion should retag the grid");

  /* The second call must refuse rather than renormalize a field whose mean is
   * now zero. Snapshot first so we can prove nothing was touched. */
  sif_real* before = malloc((size_t)g->total_cells * sizeof(sif_real));
  memcpy(before, g->values, (size_t)g->total_cells * sizeof(sif_real));
  CHECK(sif_grid_to_density_contrast(g) == SIF_ERR_INVALID,
    "a second conversion should be SIF_ERR_INVALID");
  CHECK(
    memcmp(before, g->values, (size_t)g->total_cells * sizeof(sif_real)) == 0,
    "a second conversion should leave the values untouched");
  free(before);

  /* A grid with nothing in it has no mean to divide by: refused, untouched,
   * still not a contrast. */
  sif_grid_t* blank = sif_grid_alloc(8, 100.0f);
  if (blank) {
    blank->content = SIF_GRID_DENSITY;
    CHECK(sif_grid_to_density_contrast(blank) == SIF_ERR_RANGE,
      "a grid of zeros should be SIF_ERR_RANGE");
    CHECK(blank->content == SIF_GRID_DENSITY && blank->values[0] == 0.0f,
      "a refused conversion should leave the grid as it was");
    sif_grid_free(blank);
  }
  CHECK(sif_grid_to_density_contrast(NULL) == SIF_ERR_INVALID,
    "a NULL grid should be SIF_ERR_INVALID");

  sif_grid_free(g);
}

int main(void) {
  /* Needed by the cache test and nothing else here: the grid cache is driven
   * through the settings table, which does not exist until the library is
   * initialized -- so without this sif_setting_set() is a silent no-op and
   * the cache test passes without ever enabling the cache. */
  if (sif_init(SIF_CONFIG_QUIET) != SIF_OK) {
    printf("FAIL: sif_init\n");
    return 1;
  }

  test_velocity_permutation();
  test_move_after_sort_keeps_permutation();
  test_sort_permutes_everything();
  test_require_helpers();
  test_cic_mass();
  test_cic_tiling_is_invisible();
  test_cic_cache_keys_on_content();
  test_cic_rejects_out_of_box();
  test_wrap_periodic();
  test_grid_content_tag();

  sif_finalize();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
