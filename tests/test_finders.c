/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * End-to-end check for both void finders.
 *
 * Builds a uniform field with four evacuated spheres carved out, runs the full
 * pipeline (CIC -> overdensity -> finder) and asserts that each finder
 * recovers exactly those four voids, at the right places and with sensible
 * radii. Exercises the options that take different code paths.
 */
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/finder/spherical_finder.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"

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

#define BOX    200.0f
#define N_GRID 32
/* NOT scaled under sanitizers: the finder needs a physically meaningful
 * density field, and thinning the particles would make the voids undetectable.
 * The instrumented build drops option cases instead (see main). */
#define N_P 60000

#define CELL (BOX / (sif_real)N_GRID)

/* Mesh resolution for the exodus finder. The finder used to pick this
 * itself; now that the caller owns the mesh, sif_finder_suggest_mesh_cells
 * does, which also keeps this test honest about the recommended path. */
#define MESH_CELLS sif_finder_suggest_mesh_cells(N_P, BOX, radii[0])

static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static double next_uniform(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)((rng_state >> 11) & 0x1FFFFFFFFFFFFFULL) /
         (double)0x20000000000000ULL;
}

typedef struct {
  double x, y, z, r;
} hole_t;

static const hole_t holes[] = {
  {50.0, 50.0, 50.0, 26.0},
  {150.0, 140.0, 60.0, 21.0},
  {70.0, 160.0, 150.0, 18.0},
  {160.0, 40.0, 170.0, 15.0},
};
static const int n_holes = (int)(sizeof(holes) / sizeof(holes[0]));

static const sif_real radii[] = {30.0f, 25.0f, 20.0f, 16.0f, 13.0f, 10.0f};
static const uint32_t n_radii = (uint32_t)(sizeof(radii) / sizeof(radii[0]));

static int inside_a_hole(double x, double y, double z) {
  for (int h = 0; h < n_holes; h++) {
    const double dx = x - holes[h].x, dy = y - holes[h].y, dz = z - holes[h].z;
    if (dx * dx + dy * dy + dz * dz < holes[h].r * holes[h].r)
      return 1;
  }
  return 0;
}

static void make_field(sif_real* x, sif_real* y, sif_real* z) {
  rng_state = 0x243F6A8885A308D3ULL;
  uint64_t n = 0;
  while (n < N_P) {
    const double px = next_uniform() * BOX;
    const double py = next_uniform() * BOX;
    const double pz = next_uniform() * BOX;
    /* Keep a 3% residual inside the holes so they are underdense, not empty. */
    if (inside_a_hole(px, py, pz) && next_uniform() > 0.03)
      continue;
    x[n] = (sif_real)px;
    y[n] = (sif_real)py;
    z[n] = (sif_real)pz;
    n++;
  }
}

/*
 * Each recovered void must sit on a distinct injected hole. The finder snaps
 * centers to grid cells, so allow a couple of cells of slack.
 */
static void check_catalog(const char* label, const sif_catalog_t* cat,
  sif_real r_lo_factor, sif_real r_hi_factor) {

  CHECK(cat != NULL, "%s: finder returned NULL", label);
  if (!cat)
    return;

  CHECK(cat->n_voids == (uint64_t)n_holes, "%s: found %llu voids, expected %d",
    label, (unsigned long long)cat->n_voids, n_holes);

  int matched[4] = {0};

  for (uint64_t i = 0; i < cat->n_voids; i++) {
    int best = -1;
    double best_d = 1e300;

    for (int h = 0; h < n_holes; h++) {
      const double dx = (double)cat->cx[i] - holes[h].x;
      const double dy = (double)cat->cy[i] - holes[h].y;
      const double dz = (double)cat->cz[i] - holes[h].z;
      const double d = sqrt(dx * dx + dy * dy + dz * dz);
      if (d < best_d) {
        best_d = d;
        best = h;
      }
    }

    CHECK(best_d <= 2.0 * (double)CELL,
      "%s: void %llu at (%.1f, %.1f, %.1f) is %.2f from the nearest hole "
      "(tolerance %.2f)",
      label, (unsigned long long)i, (double)cat->cx[i], (double)cat->cy[i],
      (double)cat->cz[i], best_d, 2.0 * (double)CELL);

    if (best >= 0) {
      CHECK(matched[best] == 0, "%s: two voids matched hole %d", label, best);
      matched[best] = 1;

      const double r = (double)cat->radii[i];
      CHECK(
        r >= r_lo_factor * holes[best].r && r <= r_hi_factor * holes[best].r,
        "%s: void on hole %d has r = %.2f, expected within [%.2f, %.2f]", label,
        best, r, r_lo_factor * holes[best].r, r_hi_factor * holes[best].r);
    }
  }

  for (int h = 0; h < n_holes; h++)
    CHECK(matched[h] == 1, "%s: hole %d was not recovered", label, h);
}

static void run_case(const char* label, sif_option opts, sif_real overlap) {
  printf("%s\n", label);

  sif_real* x = malloc(N_P * sizeof(sif_real));
  sif_real* y = malloc(N_P * sizeof(sif_real));
  sif_real* z = malloc(N_P * sizeof(sif_real));
  make_field(x, y, z);

  /* --- spherical finder --- */
  {
    sif_field_t* f = sif_field_alloc(N_P);
    sif_field_assign_positions(f, x, y, z);
    sif_grid_t* g = sif_grid_alloc(N_GRID, BOX);
    CHECK(sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
    CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");

    /* Snapshot the input so the grid-restoration contract can be checked:
     * the finder smooths in place, so a correct run has to put the original
     * field back unless SIF_FINDER_CONSUME_GRID was asked for. */
    sif_real* before = malloc((size_t)g->total_cells * sizeof(sif_real));
    memcpy(before, g->values, (size_t)g->total_cells * sizeof(sif_real));

    sif_catalog_t* cat =
      sif_finder_spherical(g, radii, n_radii, -0.7f, overlap, opts);
    /* Fixed radii: the finder can only report one of the requested sizes, so
     * it lands on the rung nearest the hole rather than on the hole itself.
     * That rung can exceed the hole: an empty hole of radius p smoothed at R
     * reads as -(p/R)^3, so it still clears a threshold t while
     * R <= p * |t|^(-1/3), which at -0.7 is 1.13 p. The upper factor allows
     * that plus a little for the grid. */
    check_catalog("spherical", cat, 0.55f, 1.15f);
    sif_catalog_free(cat);

    if (!(opts & SIF_FINDER_CONSUME_GRID)) {
      /* Round trip through the FFT, so exact equality is not on offer. */
      sif_real worst = 0.0f;
      for (uint64_t i = 0; i < g->total_cells; i++) {
        const sif_real d = SIF_REAL_ABS(g->values[i] - before[i]);
        if (d > worst)
          worst = d;
      }
      CHECK(worst < 1e-3f, "grid should be restored (worst drift %g)",
        (double)worst);
    }
    free(before);

    sif_grid_free(g);
    sif_field_free(f);
  }

  /* --- exodus finder --- */
  {
    sif_field_t* f = sif_field_alloc(N_P);
    sif_field_assign_positions(f, x, y, z);
    sif_grid_t* g = sif_grid_alloc(N_GRID, BOX);
    CHECK(sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
    CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");

    /* The finder borrows the mesh, so the field it was built from is dead
     * weight from here on and is released before the run. The index map is
     * dead weight too -- no finder reads it -- so this doubles as coverage
     * that the rescaling really does not need it. */
    sif_chain_mesh_t* mesh =
      sif_chain_mesh_alloc(MESH_CELLS, BOX, f, SIF_MESH_DROP_INDICES);
    CHECK(mesh != NULL, "chain mesh construction failed");
    sif_field_free(f);

    sif_catalog_t* cat =
      sif_finder_exodus(g, mesh, radii, n_radii, -0.7f, overlap, opts);
    /* Rescaling grows the void until the enclosed density crosses the
     * threshold, which overshoots the geometric hole edge somewhat. */
    check_catalog("exodus   ", cat, 0.9f, 1.5f);
    sif_catalog_free(cat);

    sif_chain_mesh_free(mesh);
    sif_grid_free(g);
  }

  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

/* --- weights --- */

/*
 * One full exodus pipeline -- grid, mesh, finder -- on the given columns, with
 * the weights assigned when `w` is non-NULL. Both halves of the finder read
 * the weights (the grid through the CIC deposit, the rescaling through the
 * mesh), so this is what a caller with a weighted field actually runs.
 */
static sif_field_t* weighted_field(
  const sif_real* x, const sif_real* y, const sif_real* z, const sif_real* w) {
  sif_field_t* f = sif_field_alloc(N_P);
  sif_field_assign_positions(f, x, y, z);
  if (w)
    sif_field_assign_weights(f, w);
  return f;
}

/* The grid from one set of weights and the mesh from another, so that the
 * finder's own refusal can be tested with a grid that is fine. */
static sif_catalog_t* exodus_on_split(const sif_real* x, const sif_real* y,
  const sif_real* z, const sif_real* grid_w, const sif_real* mesh_w) {

  sif_field_t* fg = weighted_field(x, y, z, grid_w);
  sif_grid_t* g = sif_grid_alloc(N_GRID, BOX);
  CHECK(sif_grid_assign_cic(g, fg) == SIF_OK, "CIC assignment failed");
  CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");
  sif_field_free(fg);

  sif_field_t* fm = weighted_field(x, y, z, mesh_w);
  sif_chain_mesh_t* mesh =
    sif_chain_mesh_alloc(MESH_CELLS, BOX, fm, SIF_MESH_DROP_INDICES);
  sif_field_free(fm);

  sif_catalog_t* cat =
    mesh ? sif_finder_exodus(g, mesh, radii, n_radii, -0.7f, 0.0f, 0) : NULL;

  sif_chain_mesh_free(mesh);
  sif_grid_free(g);
  return cat;
}

static sif_catalog_t* exodus_on(
  const sif_real* x, const sif_real* y, const sif_real* z, const sif_real* w) {
  return exodus_on_split(x, y, z, w, w);
}

/* Bit for bit: same voids, same order, same centres and radii. */
static void check_same_catalog(
  const char* label, const sif_catalog_t* got, const sif_catalog_t* ref) {

  CHECK(got && ref, "%s: a run returned NULL", label);
  if (!got || !ref)
    return;

  CHECK(got->n_voids == ref->n_voids, "%s: %llu voids, reference has %llu",
    label, (unsigned long long)got->n_voids, (unsigned long long)ref->n_voids);
  if (got->n_voids != ref->n_voids)
    return;

  for (uint64_t i = 0; i < got->n_voids; i++) {
    CHECK(got->cx[i] == ref->cx[i] && got->cy[i] == ref->cy[i] &&
            got->cz[i] == ref->cz[i] && got->radii[i] == ref->radii[i],
      "%s: void %llu is (%g, %g, %g; r=%.6f), reference (%g, %g, %g; "
      "r=%.6f)",
      label, (unsigned long long)i, (double)got->cx[i], (double)got->cy[i],
      (double)got->cz[i], (double)got->radii[i], (double)ref->cx[i],
      (double)ref->cy[i], (double)ref->cz[i], (double)ref->radii[i]);
  }
}

static void run_weighted_cases(void) {
  printf("weights\n");

  sif_real* x = malloc(N_P * sizeof(sif_real));
  sif_real* y = malloc(N_P * sizeof(sif_real));
  sif_real* z = malloc(N_P * sizeof(sif_real));
  sif_real* w = malloc(N_P * sizeof(sif_real));

#if !SIF_TEST_INSTRUMENTED
  /*
   * A uniform weight has to change nothing. The density is normalized to the
   * mean weight, so a constant factor cancels -- and with a power of two it
   * cancels exactly, in every sum and every comparison, so the catalogue has
   * to come back bit for bit. A weight of 1 checks the weighted path against
   * the unweighted one; a weight of 4 checks that nothing in either half of
   * the finder forgot to normalize.
   */
  make_field(x, y, z);
  sif_catalog_t* ref = exodus_on(x, y, z, NULL);

  for (uint64_t i = 0; i < N_P; i++)
    w[i] = 1.0f;
  sif_catalog_t* ones = exodus_on(x, y, z, w);
  check_same_catalog("weights = 1", ones, ref);

  for (uint64_t i = 0; i < N_P; i++)
    w[i] = 4.0f;
  sif_catalog_t* fours = exodus_on(x, y, z, w);
  check_same_catalog("weights = 4", fours, ref);

  sif_catalog_free(ref);
  sif_catalog_free(ones);
  sif_catalog_free(fours);
#endif

  /*
   * Voids that exist only in the weights. The tracers are spread uniformly,
   * holes included, and the ones inside a hole weigh nothing -- so counted,
   * the box is featureless, and weighted it holds the same four voids the
   * position-only field does. Only a finder that reads the weights on both
   * sides can recover them with the right radii: a weightless tracer is also
   * what exercises the bins the walk skips for carrying no weight.
   */
  rng_state = 0x243F6A8885A308D3ULL;
  for (uint64_t i = 0; i < N_P; i++) {
    const double px = next_uniform() * BOX;
    const double py = next_uniform() * BOX;
    const double pz = next_uniform() * BOX;
    x[i] = (sif_real)px;
    y[i] = (sif_real)py;
    z[i] = (sif_real)pz;
    w[i] = inside_a_hole(px, py, pz) ? 0.0f : 1.0f;
  }

  sif_catalog_t* cat = exodus_on(x, y, z, w);
  check_catalog("weight-only voids", cat, 0.9f, 1.5f);
  sif_catalog_free(cat);

  /* The walk can only rule shells out while the enclosed weight grows with
   * the radius, so anything that breaks that is refused rather than run. */
  w[N_P / 2] = -1.0f;
  cat = exodus_on(x, y, z, w);
  CHECK(cat == NULL, "a negative weight should be refused");
  sif_catalog_free(cat);

  /* A NaN is refused twice over. The grid cannot be normalized, since its
   * mean is NaN -- and the finder, handed a grid built from clean weights,
   * still refuses the mesh that carries it. */
  sif_real* clean = malloc(N_P * sizeof(sif_real));
  memcpy(clean, w, N_P * sizeof(sif_real));
  clean[N_P / 2] = 1.0f;
  w[N_P / 2] = 0.0f / 0.0f; /* NaN */

  sif_field_t* fnan = weighted_field(x, y, z, w);
  sif_grid_t* gnan = sif_grid_alloc(N_GRID, BOX);
  CHECK(sif_grid_assign_cic(gnan, fnan) == SIF_OK, "CIC assignment failed");
  CHECK(sif_grid_to_density_contrast(gnan) == SIF_ERR_RANGE,
    "a grid holding a NaN should not normalize");
  sif_grid_free(gnan);
  sif_field_free(fnan);

  cat = exodus_on_split(x, y, z, clean, w);
  CHECK(cat == NULL, "a NaN weight should be refused");
  sif_catalog_free(cat);
  free(clean);

  free(x);
  free(y);
  free(z);
  free(w);
  printf("  ok\n");
}

int main(void) {
  sif_fft_config_t fftcfg = {.skip_tuning = true};
  sif_config_t cfg = {.fft_config = &fftcfg,
    .omp_config = NULL,
    .log_level = SIF_LOG_LEVEL_WARNING};
  if (sif_init(&cfg) != SIF_OK) {
    printf("FAIL: sif_init\n");
    return 1;
  }

  run_case("defaults", 0, 0.0f);

#if !SIF_TEST_INSTRUMENTED
  /* Each case is a full pipeline run; under a sanitizer the default case alone
     is enough to cover the pipeline without a multi-minute test. */
  run_case("overlap 0.2", 0, 0.2f);
  run_case("consume_grid", SIF_FINDER_CONSUME_GRID, 0.0f);
#endif

  run_weighted_cases();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");

  sif_finalize();
  return failures != 0;
}
