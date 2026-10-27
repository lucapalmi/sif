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
#include "sif/core/settings.h"
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/finder/spherical_finder.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"

#include "finder/exodus_checkpoint.h"
#include "finder/utils.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
#define MESH_CELLS sif_finder_suggest_mesh_cells(N_P, BOX, radii[0], 0)

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
 * Each recovered void must sit on a distinct injected hole. A centre is where
 * the smoothed field puts it, which shot noise moves by more than a cell, so
 * allow a couple of cells of slack.
 */
static void check_catalogue(const char* label, const sif_catalogue_t* cat,
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
    CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
      "sif_field_assign_positions failed");
    sif_grid_t* g = sif_grid_alloc(N_GRID, BOX);
    CHECK(sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
    CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");

    /* Snapshot the input so the grid-restoration contract can be checked:
     * the finder smooths in place, so a correct run has to put the original
     * field back unless SIF_FINDER_CONSUME_GRID was asked for. */
    sif_real* before = malloc((size_t)g->total_cells * sizeof(sif_real));
    memcpy(before, g->values, (size_t)g->total_cells * sizeof(sif_real));

    sif_catalogue_t* cat =
      sif_finder_spherical(g, radii, n_radii, -0.7f, overlap, opts);
    /* Fixed radii: the finder can only report one of the requested sizes, so
     * it lands on the rung nearest the hole rather than on the hole itself.
     * That rung can exceed the hole: an empty hole of radius p smoothed at R
     * reads as -(p/R)^3, so it still clears a threshold t while
     * R <= p * |t|^(-1/3), which at -0.7 is 1.13 p. The upper factor allows
     * that plus a little for the grid. */
    check_catalogue("spherical", cat, 0.55f, 1.15f);
    sif_catalogue_free(cat);

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
    CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
      "sif_field_assign_positions failed");
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

    sif_catalogue_t* cat = sif_finder_exodus_raw(
      g, mesh, radii, n_radii, -0.7f, overlap, opts, NULL);
    /* Rescaling grows the void until the enclosed density crosses the
     * threshold, which overshoots the geometric hole edge somewhat. */
    check_catalogue("exodus   ", cat, 0.9f, 1.5f);
    sif_catalogue_free(cat);

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
  CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
    "sif_field_assign_positions failed");
  if (w)
    CHECK(sif_field_assign_weights(f, w) == SIF_OK,
      "sif_field_assign_weights failed");
  return f;
}

/* The grid from one set of weights and the mesh from another, so that the
 * finder's own refusal can be tested with a grid that is fine. */
static sif_catalogue_t* exodus_on_split(const sif_real* x, const sif_real* y,
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

  sif_catalogue_t* cat =
    mesh ? sif_finder_exodus_raw(g, mesh, radii, n_radii, -0.7f, 0.0f, 0, NULL)
         : NULL;

  sif_chain_mesh_free(mesh);
  sif_grid_free(g);
  return cat;
}

static sif_catalogue_t* exodus_on(
  const sif_real* x, const sif_real* y, const sif_real* z, const sif_real* w) {
  return exodus_on_split(x, y, z, w, w);
}

/* Bit for bit: same voids, same order, same centres and radii. */
static void check_same_catalogue(
  const char* label, const sif_catalogue_t* got, const sif_catalogue_t* ref) {

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
   * A uniform weight has to change nothing. The density is normalised to the
   * mean weight, so a constant factor cancels -- and with a power of two it
   * cancels exactly, in every sum and every comparison, so the catalogue has
   * to come back bit for bit. A weight of 1 checks the weighted path against
   * the unweighted one; a weight of 4 checks that nothing in either half of
   * the finder forgot to normalise.
   */
  make_field(x, y, z);
  sif_catalogue_t* ref = exodus_on(x, y, z, NULL);

  for (uint64_t i = 0; i < N_P; i++)
    w[i] = 1.0f;
  sif_catalogue_t* ones = exodus_on(x, y, z, w);
  check_same_catalogue("weights = 1", ones, ref);

  for (uint64_t i = 0; i < N_P; i++)
    w[i] = 4.0f;
  sif_catalogue_t* fours = exodus_on(x, y, z, w);
  check_same_catalogue("weights = 4", fours, ref);

  sif_catalogue_free(ref);
  sif_catalogue_free(ones);
  sif_catalogue_free(fours);
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

  sif_catalogue_t* cat = exodus_on(x, y, z, w);
  check_catalogue("weight-only voids", cat, 0.9f, 1.5f);
  sif_catalogue_free(cat);

  /* The walk can only rule shells out while the enclosed weight grows with
   * the radius, so anything that breaks that is refused rather than run. */
  w[N_P / 2] = -1.0f;
  cat = exodus_on(x, y, z, w);
  CHECK(cat == NULL, "a negative weight should be refused");
  sif_catalogue_free(cat);

  /* A NaN is refused twice over. The grid cannot be normalised, since its
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
    "a grid holding a NaN should not normalise");
  sif_grid_free(gnan);
  sif_field_free(fnan);

  cat = exodus_on_split(x, y, z, clean, w);
  CHECK(cat == NULL, "a NaN weight should be refused");
  sif_catalogue_free(cat);
  free(clean);

  free(x);
  free(y);
  free(z);
  free(w);
  printf("  ok\n");
}

/* --- the radius, recomputed --- */

static int cmp_double(const void* a, const void* b) {
  const double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

/*
 * The radius by definition, from every particle in the box: the outermost
 * particle distance d in (rung, r_search] at which the particles strictly
 * inside -- the one on the surface not counted -- are at most (1 + t) times
 * the mean count of a sphere of radius d. Nothing here shares code with the
 * finder's histogram walk, which is what the comparison is for. Returns -1
 * when no distance in the range qualifies.
 */
static double brute_force_radius(const sif_real* x, const sif_real* y,
  const sif_real* z, double cx, double cy, double cz, double rung,
  double r_search, double threshold, double* scratch) {

  const double half = 0.5 * (double)BOX;
  uint64_t n_core = 0, n_shell = 0;

  for (uint64_t p = 0; p < N_P; p++) {
    double dx = fabs((double)x[p] - cx);
    double dy = fabs((double)y[p] - cy);
    double dz = fabs((double)z[p] - cz);
    dx = dx > half ? (double)BOX - dx : dx;
    dy = dy > half ? (double)BOX - dy : dy;
    dz = dz > half ? (double)BOX - dz : dz;
    const double d = sqrt(dx * dx + dy * dy + dz * dz);
    if (d <= rung)
      n_core++;
    else if (d <= r_search)
      scratch[n_shell++] = d;
  }
  qsort(scratch, n_shell, sizeof(double), cmp_double);

  /* In double: SIF_PI is a sif_real, and M_PI is not C99. */
  const double pi = 3.14159265358979323846;
  const double k = (1.0 + threshold) * (double)N_P /
                   ((double)BOX * (double)BOX * (double)BOX) * 4.0 / 3.0 * pi;
  for (uint64_t i = n_shell; i-- > 0;) {
    const double d = scratch[i];
    if ((double)(n_core + i) <= k * d * d * d)
      return d;
  }
  return -1.0;
}

/*
 * One rung at a time, so that each void's search range is known exactly, and
 * every radius in the catalogue checked against brute_force_radius(). A
 * radius is a particle distance, so the two agree to float rounding or the
 * walk is wrong -- the bin skipping, the refinement window, the periodic
 * images. The search factors move r_search and with it the histogram.
 */
static void run_radius_oracle(void) {
  printf("radius against brute force\n");

  sif_real* x = malloc(N_P * sizeof(sif_real));
  sif_real* y = malloc(N_P * sizeof(sif_real));
  sif_real* z = malloc(N_P * sizeof(sif_real));
  double* scratch = malloc(N_P * sizeof(double));
  make_field(x, y, z);

  const struct {
    sif_real rung;
    sif_option opt;
    double factor;
  } cases[] = {
    {16.0f, 0, 1.5},
    {20.0f, SIF_FINDER_SEARCH_2_00, 2.0},
    {13.0f, SIF_FINDER_SEARCH_1_25, 1.25},
    /* Centres off the nodes: the radius reported is still exactly the
     * tracers' at the centre reported. */
    {16.0f, SIF_FINDER_SEEDS_MINIMA, 1.5},
  };

  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    sif_field_t* f = sif_field_alloc(N_P);
    CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
      "sif_field_assign_positions failed");
    sif_grid_t* g = sif_grid_alloc(N_GRID, BOX);
    CHECK(sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
    CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");
    sif_chain_mesh_t* mesh =
      sif_chain_mesh_alloc(MESH_CELLS, BOX, f, SIF_MESH_DROP_INDICES);
    sif_field_free(f);

    const sif_real rung[] = {cases[c].rung};
    sif_catalogue_t* cat =
      sif_finder_exodus_raw(g, mesh, rung, 1, -0.7f, 0.0f, cases[c].opt, NULL);
    CHECK(cat && cat->n_voids > 0, "rung %.0f: no voids to check",
      (double)cases[c].rung);

    /* The same reach the finder computes for a first rung. */
    const double r0 = (double)cases[c].rung;
    const double r_search = r0 + fmax((cases[c].factor - 1.0) * r0, 3.0 * CELL);

    for (uint64_t i = 0; cat && i < cat->n_voids; i++) {
      const double want = brute_force_radius(x, y, z, (double)cat->cx[i],
        (double)cat->cy[i], (double)cat->cz[i], r0, r_search, -0.7, scratch);
      const double got = (double)cat->radii[i];
      CHECK(want > 0.0 && fabs(got - want) <= 1e-5 * want,
        "rung %.0f, void %llu at (%g, %g, %g): r = %.6f, brute force %.6f", r0,
        (unsigned long long)i, (double)cat->cx[i], (double)cat->cy[i],
        (double)cat->cz[i], got, want);
    }

    sif_catalogue_free(cat);
    sif_chain_mesh_free(mesh);
    sif_grid_free(g);
  }

  free(scratch);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

/* --- what the catalogue must not depend on --- */

static sif_catalogue_t* exodus_with_mesh(const sif_real* x, const sif_real* y,
  const sif_real* z, uint32_t mesh_cells) {

  sif_field_t* f = sif_field_alloc(N_P);
  CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
    "sif_field_assign_positions failed");
  sif_grid_t* g = sif_grid_alloc(N_GRID, BOX);
  CHECK(sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
  CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");
  sif_chain_mesh_t* mesh =
    sif_chain_mesh_alloc(mesh_cells, BOX, f, SIF_MESH_DROP_INDICES);
  sif_field_free(f);

  sif_catalogue_t* cat =
    mesh ? sif_finder_exodus_raw(g, mesh, radii, n_radii, -0.7f, 0.0f, 0, NULL)
         : NULL;

  sif_chain_mesh_free(mesh);
  sif_grid_free(g);
  return cat;
}

/*
 * The mesh only indexes the particles and the speculative batches only group
 * the candidates, so neither may change a single bit of the catalogue. The
 * thread count is fixed at sif_system_init(), so the single-threaded run
 * brings the system down and up again around it, and restores the original
 * configuration afterwards.
 */
static void run_invariance(sif_config_t* cfg) {
  printf("mesh and thread invariance\n");

  sif_real* x = malloc(N_P * sizeof(sif_real));
  sif_real* y = malloc(N_P * sizeof(sif_real));
  sif_real* z = malloc(N_P * sizeof(sif_real));
  make_field(x, y, z);

  sif_catalogue_t* ref = exodus_with_mesh(x, y, z, MESH_CELLS);

  /* From the coarsest mesh the search sphere allows to several times the
   * suggestion: the walk's cost moves, the answer may not. */
  const uint32_t meshes[] = {4, 7, 2 * MESH_CELLS, 4 * MESH_CELLS};
  for (size_t i = 0; i < sizeof(meshes) / sizeof(meshes[0]); i++) {
    char label[32];
    snprintf(label, sizeof(label), "mesh %u", meshes[i]);
    sif_catalogue_t* cat = exodus_with_mesh(x, y, z, meshes[i]);
    check_same_catalogue(label, cat, ref);
    sif_catalogue_free(cat);
  }

  sif_omp_config_t one = {.n_threads = 1};
  sif_config_t serial = *cfg;
  serial.omp_config = &one;
  sif_system_finalise();
  CHECK(sif_system_init(&serial) == SIF_OK, "re-init with one thread failed");
  sif_catalogue_t* cat = exodus_with_mesh(x, y, z, MESH_CELLS);
  check_same_catalogue("1 thread", cat, ref);
  sif_catalogue_free(cat);
  sif_system_finalise();
  CHECK(sif_system_init(cfg) == SIF_OK, "re-init with defaults failed");

  sif_catalogue_free(ref);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

/* --- checkpoints and prior catalogues --- */

/* The test's own grid and mesh, built once: a run hands the grid back as it
 * came, so every run below can share them. */
typedef struct {
  sif_grid_t* grid;
  sif_chain_mesh_t* mesh;
  sif_field_t* field;
} setup_t;

/*
 * The grid as the tracers make it. A finder hands its grid back through an
 * inverse transform, which restores it to float precision but not to the
 * bit; centres placed between the nodes follow the field's last bits, so
 * runs compared bit for bit each start from a grid built afresh -- as a
 * resumed run in a new process does.
 */
static void regrid(const setup_t* s) {
  CHECK(sif_grid_assign_cic(s->grid, s->field) == SIF_OK &&
          sif_grid_to_density_contrast(s->grid) == SIF_OK,
    "rebuilding the grid failed");
}

static sif_catalogue_t* exodus_x(const setup_t* s, const sif_real* r,
  uint32_t n, sif_real threshold, sif_option opt, const char* checkpoint,
  const sif_catalogue_t* prior) {

  regrid(s);
  const sif_finder_exodus_extras_t extras = {
    .checkpoint = checkpoint, .prior = prior};
  return sif_finder_exodus_raw(
    s->grid, s->mesh, r, n, threshold, 0.0f, opt, &extras);
}

static int file_exists(const char* path) { return access(path, F_OK) == 0; }

/* Checkpoints a run named for itself, in the working directory. */
static int auto_checkpoints(char* first, size_t size) {
  int n = 0;
  DIR* dir = opendir(".");
  if (!dir)
    return -1;
  struct dirent* e;
  while ((e = readdir(dir)) != NULL) {
    const size_t len = strlen(e->d_name);
    if (strncmp(e->d_name, "exodus-", 7) == 0 && len > 5 &&
        strcmp(e->d_name + len - 5, ".ckpt") == 0) {
      if (n++ == 0 && first)
        snprintf(first, size, "%s", e->d_name);
    }
  }
  closedir(dir);
  return n;
}

/*
 * A run stopped after any rung and resumed from its checkpoint returns the
 * uninterrupted catalogue, bit for bit; a checkpoint of another run is
 * refused and left alone; a run handed the voids of the larger rungs and
 * the ladder below them returns the single run's catalogue too.
 */
static void run_checkpoints(void) {
  printf("checkpoints and prior catalogues\n");

  sif_real* x = malloc(N_P * sizeof(sif_real));
  sif_real* y = malloc(N_P * sizeof(sif_real));
  sif_real* z = malloc(N_P * sizeof(sif_real));
  make_field(x, y, z);

  sif_field_t* f = sif_field_alloc(N_P);
  CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
    "sif_field_assign_positions failed");
  setup_t s = {sif_grid_alloc(N_GRID, BOX), NULL, f};
  s.mesh = sif_chain_mesh_alloc(MESH_CELLS, BOX, f, SIF_MESH_DROP_INDICES);
  free(x);
  free(y);
  free(z);

  const sif_option none = SIF_FINDER_NO_CHECKPOINT;
  const char* path = "test_finders.ckpt";
  unlink(path);

  sif_catalogue_t* ref = exodus_x(&s, radii, n_radii, -0.7f, none, NULL, NULL);
  CHECK(ref && ref->n_voids > 0, "reference run found nothing");

  const int auto_before = auto_checkpoints(NULL, 0);

  /* A plain run checkpoints under a name of its own, and clears it up. */
  regrid(&s);
  sif_catalogue_t* cat =
    sif_finder_exodus_raw(s.grid, s.mesh, radii, n_radii, -0.7f, 0.0f, 0, NULL);
  check_same_catalogue("plain run", cat, ref);
  sif_catalogue_free(cat);
  CHECK(auto_checkpoints(NULL, 0) == auto_before,
    "a run that finished left its checkpoint behind");

  /* Stopped after the first rung, part-way, and after the last. */
  const uint32_t cuts[] = {1, 3, n_radii};
  for (size_t c = 0; c < sizeof(cuts) / sizeof(cuts[0]); c++) {
    const uint32_t k = cuts[c];
    char label[48];
    snprintf(label, sizeof(label), "resumed after rung %u", k);

    sif__exodus_halt_after_set(k);
    sif_error_clear();
    cat = exodus_x(&s, radii, n_radii, -0.7f, 0, path, NULL);
    sif__exodus_halt_after_set(0);
    CHECK(!cat, "%s: the halted run returned a catalogue", label);
    sif_catalogue_free(cat);
    CHECK(file_exists(path), "%s: no checkpoint kept", label);

    /* What it holds is what the first k rungs find on their own. */
    sif_finder_exodus_checkpoint_t h;
    sif_catalogue_t* saved = sif_finder_exodus_checkpoint_read(path, &h);
    sif_catalogue_t* head = exodus_x(&s, radii, k, -0.7f, none, NULL, NULL);
    CHECK(saved && h.next_rung == k && h.n_radii == n_radii,
      "%s: header says %u of %u rungs", label, h.next_rung, h.n_radii);
    check_same_catalogue(label, saved, head);
    sif_catalogue_free(saved);
    sif_catalogue_free(head);

    cat = exodus_x(&s, radii, n_radii, -0.7f, 0, path, NULL);
    check_same_catalogue(label, cat, ref);
    sif_catalogue_free(cat);
    CHECK(!file_exists(path), "%s: checkpoint left after finishing", label);
  }

  /* Another run's checkpoint: refused, named, and not touched. */
  sif__exodus_halt_after_set(2);
  cat = exodus_x(&s, radii, n_radii, -0.7f, 0, path, NULL);
  sif__exodus_halt_after_set(0);
  sif_catalogue_free(cat);

  sif_error_clear();
  cat = exodus_x(&s, radii, n_radii, -0.6f, 0, path, NULL);
  CHECK(!cat && error_has("not this run: threshold"),
    "another threshold was not refused (%s)", sif_error_message());
  sif_catalogue_free(cat);

  sif_error_clear();
  cat = exodus_x(&s, radii, n_radii - 1, -0.7f, 0, path, NULL);
  CHECK(!cat && error_has("not this run: ladder"),
    "another ladder was not refused (%s)", sif_error_message());
  sif_catalogue_free(cat);

  sif_error_clear();
  cat = exodus_x(&s, radii, n_radii, -0.7f, SIF_FINDER_SEARCH_2_00, path, NULL);
  CHECK(!cat && error_has("not this run: search factor"),
    "another search factor was not refused (%s)", sif_error_message());
  sif_catalogue_free(cat);

  sif_error_clear();
  cat =
    exodus_x(&s, radii, n_radii, -0.7f, SIF_FINDER_SEEDS_MINIMA, path, NULL);
  CHECK(!cat && error_has("not this run: centres every cell, on its node"),
    "minima centres were not refused (%s)", sif_error_message());
  sif_catalogue_free(cat);

  sif_error_clear();
  cat =
    exodus_x(&s, radii, n_radii, -0.7f, SIF_FINDER_ALIASING_WINDOW, path, NULL);
  CHECK(!cat && error_has("not this run: window deconvolved"),
    "the aliasing-aware window was not refused (%s)", sif_error_message());
  sif_catalogue_free(cat);

  /* The same tracers on a TSC grid: another field, so another run. */
  sif_error_clear();
  CHECK(sif_grid_assign(s.grid, s.field, SIF_ASSIGNMENT_TSC, 0) == SIF_OK &&
          sif_grid_to_density_contrast(s.grid) == SIF_OK,
    "TSC grid failed");
  {
    const sif_finder_exodus_extras_t extras = {.checkpoint = path};
    cat = sif_finder_exodus_raw(
      s.grid, s.mesh, radii, n_radii, -0.7f, 0.0f, 0, &extras);
  }
  CHECK(!cat && error_has("not this run: grid assigned with CIC (this run's: TSC)"),
    "a TSC grid was not refused (%s)", sif_error_message());
  sif_catalogue_free(cat);
  CHECK(file_exists(path), "a refused resume removed the checkpoint");

  /* One byte changed anywhere fails the checksum. */
  FILE* fp = fopen(path, "r+b");
  CHECK(fp != NULL, "cannot reopen the checkpoint");
  if (fp) {
    fseek(fp, -3, SEEK_END);
    const int byte = fgetc(fp);
    fseek(fp, -3, SEEK_END);
    fputc(byte ^ 0x10, fp);
    fclose(fp);
  }
  sif_error_clear();
  cat = exodus_x(&s, radii, n_radii, -0.7f, 0, path, NULL);
  CHECK(!cat && error_has("corrupt"), "a damaged checkpoint was resumed (%s)",
    sif_error_message());
  sif_catalogue_free(cat);
  unlink(path);

  /* A run stopped without a name for its checkpoint keeps the one it made. */
  sif__exodus_halt_after_set(2);
  cat = exodus_x(&s, radii, n_radii, -0.7f, 0, NULL, NULL);
  sif__exodus_halt_after_set(0);
  sif_catalogue_free(cat);
  char kept[256] = "";
  CHECK(auto_checkpoints(kept, sizeof(kept)) == auto_before + 1,
    "a halted run kept no checkpoint of its own");
  if (kept[0]) {
    cat = exodus_x(&s, radii, n_radii, -0.7f, 0, kept, NULL);
    check_same_catalogue("resumed from its own name", cat, ref);
    sif_catalogue_free(cat);
    unlink(kept);
  }

  sif_error_clear();
  cat = exodus_x(&s, radii, n_radii, -0.7f, none, path, NULL);
  CHECK(!cat && error_has("SIF_FINDER_NO_CHECKPOINT"),
    "a checkpoint path with checkpoints off was not refused");
  sif_catalogue_free(cat);

  /* The larger rungs' voids, extended with the rest of the ladder. The test
   * ladder steps by at most 1.3, under the default search factor. */
  const uint32_t split = 3;
  sif_catalogue_t* prior = exodus_x(&s, radii, split, -0.7f, none, NULL, NULL);
  cat = exodus_x(&s, radii + split, n_radii - split, -0.7f, none, NULL, prior);
  check_same_catalogue("extended", cat, ref);
  sif_catalogue_free(cat);

  /* Extended, stopped, and resumed: the prior has to be the same one. */
  sif__exodus_halt_after_set(1);
  cat = exodus_x(&s, radii + split, n_radii - split, -0.7f, 0, path, prior);
  sif__exodus_halt_after_set(0);
  sif_catalogue_free(cat);

  sif_error_clear();
  cat = exodus_x(&s, radii + split, n_radii - split, -0.7f, 0, path, NULL);
  CHECK(!cat && error_has("prior voids"),
    "a resume without its prior was not refused (%s)", sif_error_message());
  sif_catalogue_free(cat);

  cat = exodus_x(&s, radii + split, n_radii - split, -0.7f, 0, path, prior);
  check_same_catalogue("extended and resumed", cat, ref);
  sif_catalogue_free(cat);
  CHECK(!file_exists(path), "extended run left its checkpoint");
  sif_catalogue_free(prior);

  /* Any catalogue will do as a prior; it comes back first and as given, and
   * nothing found overlaps it. */
  prior = sif_catalogue_alloc(1);
  CHECK(sif_catalogue_append(prior, 100.0f, 100.0f, 100.0f, 35.0f) == SIF_OK,
    "append failed");
  cat = exodus_x(&s, radii, n_radii, -0.7f, none, NULL, prior);
  CHECK(
    cat && cat->n_voids >= 1 && cat->cx[0] == 100.0f && cat->radii[0] == 35.0f,
    "prior void not returned first");
  for (uint64_t i = 1; cat && i < cat->n_voids; i++) {
    const double dx = cat->cx[i] - 100.0, dy = cat->cy[i] - 100.0,
                 dz = cat->cz[i] - 100.0;
    CHECK(sqrt(dx * dx + dy * dy + dz * dz) >= 35.0 + cat->radii[i] - 1e-3,
      "void %llu overlaps the prior one", (unsigned long long)i);
  }
  sif_catalogue_free(cat);

  /* Priors the finder cannot place. */
  prior->radii[0] = 0.0f;
  sif_error_clear();
  cat = exodus_x(&s, radii, n_radii, -0.7f, none, NULL, prior);
  CHECK(!cat && error_has("prior: void 0: radius"),
    "a zero prior radius was not refused (%s)", sif_error_message());
  sif_catalogue_free(cat);

  prior->radii[0] = 10.0f;
  prior->cx[0] = BOX + 1.0f;
  sif_error_clear();
  cat = exodus_x(&s, radii, n_radii, -0.7f, none, NULL, prior);
  CHECK(!cat && error_has("outside the box"),
    "a prior outside the box was not refused (%s)", sif_error_message());
  sif_catalogue_free(cat);
  sif_catalogue_free(prior);

  sif_catalogue_free(ref);
  sif_chain_mesh_free(s.mesh);
  sif_grid_free(s.grid);
  sif_field_free(s.field);
  printf("  ok\n");
}

/* --- centres between the nodes --- */

/* Periodic displacement a - b on a box of side L, into [-L/2, L/2). */
static double wrap_delta(double a, double b, double L) {
  double d = fmod(a - b, L);
  if (d < -0.5 * L)
    d += L;
  if (d >= 0.5 * L)
    d -= L;
  return d;
}

/*
 * The scan on fields whose minimum is known: a well, anisotropic and turned
 * off the axes, centred between the nodes -- and across the box face, where
 * the stencil and the centre both wrap. The extra Gaussian the scan smooths
 * with is symmetric, so it moves nothing about a symmetric well. A flat field
 * has one minimum, by the tie rule, and no curvature to place it by.
 */
static void run_minimum_scan(void) {
  printf("minimum scan\n");

  const uint32_t n = 32;
  const double L = 32.0; /* cells of one */
  sif_grid_t* g = sif_grid_alloc(n, (sif_real)L);
  sif_bitmask_t* mask = sif_bitmask_alloc(g->total_cells);
  sif_bitmask_t* below = sif_bitmask_alloc(g->total_cells);
  sif_candidate_buffer_t cands = {0};
  sif_offset_buffer_t offs = {0};
  sif_refine_stats_t st;
  sif_real* well = malloc(g->total_cells * sizeof(sif_real));

  /* Q = R diag(1/9, 1/16, 1/25) R^T, R a turn about z then about x. */
  const double a = 0.6, b = 0.9, ca = cos(a), sa = sin(a), cb = cos(b),
               sb = sin(b);
  const double R[3][3] = {
    {ca, -sa, 0.0}, {sa * cb, ca * cb, -sb}, {sa * sb, ca * sb, cb}};
  const double inv_s2[3] = {1.0 / 9.0, 1.0 / 16.0, 1.0 / 25.0};
  double Q[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Q[i][j] = 0.0;
      for (int k = 0; k < 3; k++)
        Q[i][j] += R[i][k] * inv_s2[k] * R[j][k];
    }

  const double centres[][3] = {
    {10.3, 20.7, 5.5}, {0.2, 31.9, 15.0}, {16.49, 16.51, 16.0}};
  for (int c = 0; c < 3; c++) {
    for (uint32_t i = 0; i < n; i++)
      for (uint32_t j = 0; j < n; j++)
        for (uint32_t k = 0; k < n; k++) {
          const double d[3] = {wrap_delta(i, centres[c][0], L),
            wrap_delta(j, centres[c][1], L), wrap_delta(k, centres[c][2], L)};
          double q = 0.0;
          for (int u = 0; u < 3; u++)
            for (int v = 0; v < 3; v++)
              q += d[u] * Q[u][v] * d[v];
          g->values[((uint64_t)i * n + j) * n + k] =
            (sif_real)(-0.95 * exp(-0.5 * q));
        }

    /* The scan smooths the grid in place; the minima-only runs below start
     * from the well again. */
    memcpy(well, g->values, g->total_cells * sizeof(sif_real));

    CHECK(sif__finder_scan_refined(
            g, mask, below, -0.5f, 0, 1, 0, &cands, &offs, &st, NULL, NULL) == SIF_OK,
      "well %d: scan failed", c);
    CHECK(cands.count > 1 && st.n_minima == 1 && st.n_fallback == 0,
      "well %d: %llu candidates, %llu minima, %llu left on their node", c,
      (unsigned long long)cands.count, (unsigned long long)st.n_minima,
      (unsigned long long)st.n_fallback);

    /* The deepest candidate is the minimum, and the only one refined. */
    uint64_t n_refined = 0;
    for (uint64_t k = 0; k < cands.count; k++)
      n_refined += cands.items[k].refined != 0;
    CHECK(n_refined == 1 && cands.items[0].refined != 0,
      "well %d: %llu candidates refined, the deepest %s", c,
      (unsigned long long)n_refined,
      cands.items[0].refined ? "among them" : "not");
    if (cands.count == 0 || cands.items[0].refined == 0)
      continue;

    const sif_offset_t* m = &offs.items[cands.items[0].refined - 1];
    const uint64_t f = cands.items[0].flat_idx;
    const double got[3] = {(double)(f / ((uint64_t)n * n)) + m->dx,
      (double)((f / n) % n) + m->dy, (double)(f % n) + m->dz};
    double err = 0.0;
    for (int u = 0; u < 3; u++)
      err = fmax(err, fabs(wrap_delta(got[u], centres[c][u], L)));
    CHECK(err < 0.05,
      "well %d: centre (%.3f, %.3f, %.3f), expected (%.3f, %.3f, %.3f)", c,
      got[0], got[1], got[2], centres[c][0], centres[c][1], centres[c][2]);
    CHECK(fabs(m->dx) <= 0.5f && fabs(m->dy) <= 0.5f && fabs(m->dz) <= 0.5f,
      "well %d: offset (%g, %g, %g) is off its own node", c, (double)m->dx,
      (double)m->dy, (double)m->dz);

    /* Minima only: the minimum alone, refined to the same centre -- or, not
     * refined, on the node it was found at. */
    const sif_offset_t first = *m;
    memcpy(g->values, well, g->total_cells * sizeof(sif_real));
    CHECK(sif__finder_scan_refined(
            g, mask, below, -0.5f, 0, 1, 1, &cands, &offs, &st, NULL, NULL) == SIF_OK &&
            cands.count == 1 && cands.items[0].flat_idx == f &&
            cands.items[0].refined != 0 &&
            offs.items[cands.items[0].refined - 1].dx == first.dx,
      "well %d, minima only: %llu candidates", c,
      (unsigned long long)cands.count);
    memcpy(g->values, well, g->total_cells * sizeof(sif_real));
    CHECK(sif__finder_scan_refined(
            g, mask, below, -0.5f, 0, 0, 1, &cands, &offs, &st, NULL, NULL) == SIF_OK &&
            cands.count == 1 && cands.items[0].refined == 0 && offs.count == 0,
      "well %d, minima only on the nodes: %llu candidates, %llu offsets", c,
      (unsigned long long)cands.count, (unsigned long long)offs.count);
  }
  free(well);

  for (uint64_t i = 0; i < g->total_cells; i++)
    g->values[i] = -0.9f;
  CHECK(sif__finder_scan_refined(
          g, mask, below, -0.5f, 0, 1, 0, &cands, &offs, &st, NULL, NULL) == SIF_OK &&
          cands.count == g->total_cells && st.n_minima == 1 &&
          st.n_fallback == 1,
    "flat field: %llu candidates, %llu minima, %llu left on their node",
    (unsigned long long)cands.count, (unsigned long long)st.n_minima,
    (unsigned long long)st.n_fallback);
  for (uint64_t k = 0; k < cands.count; k++)
    CHECK(cands.items[k].refined == 0, "flat field: candidate %llu refined",
      (unsigned long long)k);

  sif__candidate_buffer_free(&cands);
  sif__offset_buffer_free(&offs);
  sif_bitmask_free(below);
  sif_bitmask_free(mask);
  sif_grid_free(g);
  printf("  ok\n");
}

/* The grid the shift test runs on: twice the usual resolution, so the
 * smallest rung spans three cells -- inside the two or more a radius is meant
 * to have, where the usual grid's 1.6 is not. */
#define SHIFT_GRID (2 * N_GRID)
#define SHIFT_CELL (BOX / (sif_real)SHIFT_GRID)

/* Whether void i sits off the grid. A centre left on its node is exactly
 * i * SHIFT_CELL, so any fraction of a cell in it was put there by the
 * refinement. */
static int off_the_nodes(const sif_catalogue_t* c, uint64_t i) {
  const double p[3] = {
    c->cx[i] / SHIFT_CELL, c->cy[i] / SHIFT_CELL, c->cz[i] / SHIFT_CELL};
  for (int a = 0; a < 3; a++)
    if (fabs(p[a] - nearbyint(p[a])) > 1e-4)
      return 1;
  return 0;
}

/* The median of how far each void of `a` -- each one off the nodes, if
 * `refined_only` -- moved by `shift`, is from the nearest void of `b`, over
 * those within 1.5 cells of one. */
static double shift_residual(const sif_catalogue_t* a, const sif_catalogue_t* b,
  const double shift[3], int refined_only, uint64_t* n_matched) {

  double* r = malloc(a->n_voids * sizeof(double));
  uint64_t m = 0;
  for (uint64_t i = 0; r && i < a->n_voids; i++) {
    if (refined_only && !off_the_nodes(a, i))
      continue;
    double best = 1e300;
    for (uint64_t j = 0; j < b->n_voids; j++) {
      const double dx = wrap_delta(b->cx[j], a->cx[i] + shift[0], BOX);
      const double dy = wrap_delta(b->cy[j], a->cy[i] + shift[1], BOX);
      const double dz = wrap_delta(b->cz[j], a->cz[i] + shift[2], BOX);
      best = fmin(best, sqrt(dx * dx + dy * dy + dz * dz));
    }
    if (best < 1.5 * SHIFT_CELL)
      r[m++] = best / SHIFT_CELL;
  }
  qsort(r, m, sizeof(double), cmp_double);
  const double med = m ? r[m / 2] : 1e300;
  free(r);
  *n_matched = m;
  return med;
}

/*
 * Moving every tracer by a fraction of a cell should move every centre by the
 * same. Centres on cells sit on nodes and cannot; minima are placed between
 * the nodes, to a tenth or two of a cell. On a Poisson field full of
 * marginal voids.
 */
static void run_grid_shift(void) {
  printf("grid shift\n");

  /* A quarter of the usual tracers, so a Poisson field reaches the threshold
   * often enough to give a few dozen voids to compare. */
  const uint64_t np = N_P / 4;
  const double shift_cells[3] = {0.37, 0.61, 0.13};
  const double shift[3] = {shift_cells[0] * SHIFT_CELL,
    shift_cells[1] * SHIFT_CELL, shift_cells[2] * SHIFT_CELL};

  sif_real* x = malloc(2 * 3 * np * sizeof(sif_real));
  rng_state = 0x9E3779B97F4A7C15ULL;
  for (uint64_t i = 0; i < 3 * np; i++)
    x[i] = (sif_real)(next_uniform() * BOX);
  for (uint64_t i = 0; i < np; i++)
    for (int a = 0; a < 3; a++) {
      double v = (double)x[a * np + i] + shift[a];
      if (v >= BOX)
        v -= BOX;
      x[3 * np + a * np + i] = (sif_real)v;
    }

  const struct {
    const char* name;
    sif_option opt;
  } modes[2] = {
    {"cells", SIF_FINDER_SEEDS_CELLS}, {"minima", SIF_FINDER_SEEDS_MINIMA}};
  sif_catalogue_t* cat[2][2] = {{NULL, NULL}, {NULL, NULL}};
  for (int moved = 0; moved < 2; moved++) {
    const sif_real* p = x + 3 * np * moved;
    sif_field_t* f = sif_field_alloc(np);
    CHECK(sif_field_assign_positions(f, p, p + np, p + 2 * np) == SIF_OK,
      "positions failed");
    sif_grid_t* g = sif_grid_alloc(SHIFT_GRID, BOX);
    sif_chain_mesh_t* mesh =
      sif_chain_mesh_alloc(MESH_CELLS, BOX, f, SIF_MESH_DROP_INDICES);
    for (int k = 0; k < 2; k++) {
      CHECK(sif_grid_assign_cic(g, f) == SIF_OK &&
              sif_grid_to_density_contrast(g) == SIF_OK,
        "grid failed");
      cat[moved][k] = sif_finder_exodus_raw(g, mesh, radii, n_radii, -0.7f,
        0.0f, SIF_FINDER_NO_CHECKPOINT | modes[k].opt, NULL);
      CHECK(cat[moved][k] && cat[moved][k]->n_voids >= 20,
        "%s centres%s: %llu voids, too few to measure by", modes[k].name,
        moved ? ", moved" : "",
        (unsigned long long)(cat[moved][k] ? cat[moved][k]->n_voids : 0));
    }
    sif_chain_mesh_free(mesh);
    sif_grid_free(g);
    sif_field_free(f);
  }

  double miss[2] = {1e300, 1e300};
  for (int k = 0; k < 2; k++) {
    if (!cat[0][k] || !cat[1][k])
      continue;
    uint64_t m;
    miss[k] = shift_residual(cat[0][k], cat[1][k], shift, k == 1, &m);
    printf("  %-9s %3llu voids, median miss %.3f cells (%llu matched)\n",
      modes[k].name, (unsigned long long)cat[0][k]->n_voids, miss[k],
      (unsigned long long)m);
  }
  CHECK(miss[1] < 0.25 && miss[1] < 0.5 * miss[0],
    "minima missed the shift by a median %.3f cells (cells: %.3f)", miss[1],
    miss[0]);

  for (int a = 0; a < 2; a++)
    for (int b = 0; b < 2; b++)
      sif_catalogue_free(cat[a][b]);
  free(x);
  printf("  ok\n");
}

/*
 * The exact overlap test reaches every void it can collide with -- including
 * when the candidate is larger than every void accepted so far, as a rescaled
 * radius often is. A void of 10 centred 55 inside a candidate of 60 collides
 * at any overlap fraction; the search used to stop at 10 + 60 * (1 - f).
 */
static void run_overlap_reach(void) {
  printf("overlap reach\n");

  const sif_real L = 1000.0f;
  sif_catalogue_t* cat = sif_catalogue_alloc(1);
  sif_cell_linked_list_t* cll =
    sif_cell_linked_list_alloc(32, L, 1, SIF_PBC_PERIODIC);
  CHECK(cat && cll && sif_catalogue_append(cat, 500, 500, 500, 10) == SIF_OK &&
          sif_cell_linked_list_insert(cll, 0, 500, 500, 500) == SIF_OK,
    "setup failed");

  const sif_real fractions[] = {0.0f, 0.5f, 1.0f};
  for (int k = 0; cat && cll && k < 3; k++) {
    const sif_real f = fractions[k];
    const sif_real collide = 60.0f + 10.0f - f * 10.0f;
    for (sif_real sep = 5.0f; sep < 80.0f; sep += 2.5f) {
      const int hit =
        sif__overlap_exact(cat, 500 + sep, 500, 500, 60, 10, L, cll, 0, f);
      CHECK(hit == (sep < collide),
        "f = %.1f: a void of 60 at %.1f from one of 10 %s (collides below "
        "%.1f)",
        (double)f, (double)sep, hit ? "collided" : "did not collide",
        (double)collide);
    }
  }

  sif_cell_linked_list_free(cll);
  sif_catalogue_free(cat);
  printf("  ok\n");
}

/* --- refusals --- */

/*
 * Bad arguments are refused up front with the argument named, and a run that
 * fails partway returns NULL -- not the voids found so far -- with the grid
 * handed back as it came.
 */
static void run_refusals(void) {
  printf("refusals\n");

  sif_real* x = malloc(N_P * sizeof(sif_real));
  sif_real* y = malloc(N_P * sizeof(sif_real));
  sif_real* z = malloc(N_P * sizeof(sif_real));
  make_field(x, y, z);
  sif_field_t* f = sif_field_alloc(N_P);
  CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
    "sif_field_assign_positions failed");
  sif_grid_t* g = sif_grid_alloc(N_GRID, BOX);
  CHECK(sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
  CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");
  sif_chain_mesh_t* mesh = sif_chain_mesh_alloc(MESH_CELLS, BOX, f, 0);
  CHECK(mesh != NULL, "chain mesh construction failed");

  const sif_real bad_radii[] = {20.0f, -3.0f};
  const struct {
    const sif_real* radii;
    uint32_t n;
    sif_real threshold, overlap;
    const char* reason;
  } cases[] = {
    {radii, 0, -0.7f, 0.0f, "radii: none given"},
    {bad_radii, 2, -0.7f, 0.0f, "radii[1] = -3: not a positive radius"},
    {radii, n_radii, 0.5f, 0.0f, "threshold 0.5: not a density contrast"},
    {radii, n_radii, -1.0f, 0.0f, "threshold -1: not a density contrast"},
    {radii, n_radii, -0.7f, 1.5f, "overlap_fraction 1.5: not in [0, 1]"},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    sif_error_clear();
    sif_catalogue_t* cat = sif_finder_exodus_raw(g, mesh, cases[i].radii,
      cases[i].n, cases[i].threshold, cases[i].overlap, 0, NULL);
    CHECK(!cat && error_has(cases[i].reason) &&
            sif_error_status() == SIF_ERR_INVALID,
      "exodus case %zu: '%s', status %d", i, sif_error_message(),
      sif_error_status());
    sif_catalogue_free(cat);

    sif_error_clear();
    cat = sif_finder_spherical(
      g, cases[i].radii, cases[i].n, cases[i].threshold, cases[i].overlap, 0);
    CHECK(!cat && error_has(cases[i].reason) &&
            sif_error_status() == SIF_ERR_INVALID,
      "spherical case %zu: '%s', status %d", i, sif_error_message(),
      sif_error_status());
    sif_catalogue_free(cat);
  }

  /* Variants: no third one. */
  sif_error_clear();
  sif_catalogue_t* bad = sif_finder_exodus(
    g, mesh, radii, n_radii, -0.7f, SIF__EXODUS_N_VARIANTS, NULL);
  CHECK(!bad && error_has("variant: 2 (expected 0 to 1)") &&
          sif_error_status() == SIF_ERR_INVALID,
    "an unknown variant: '%s'", sif_error_message());
  sif_catalogue_free(bad);

  /* A search sphere wider than the mesh: the first rung smooths the grid, then
   * fails building its template. */
  sif_real* before = malloc((size_t)g->total_cells * sizeof(sif_real));
  memcpy(before, g->values, (size_t)g->total_cells * sizeof(sif_real));
  const sif_real too_wide[] = {150.0f};
  sif_error_clear();
  sif_catalogue_t* cat =
    sif_finder_exodus_raw(g, mesh, too_wide, 1, -0.7f, 0, 0, NULL);
  CHECK(!cat, "a failed run should return NULL, not a partial catalogue");
  CHECK(error_has("wider than the mesh") && sif_error_status() == SIF_ERR_RANGE,
    "failed run: '%s', status %d", sif_error_message(), sif_error_status());
  sif_catalogue_free(cat);

  sif_real worst = 0.0f;
  for (uint64_t i = 0; g->values && i < g->total_cells; i++)
    worst = SIF_REAL_MAX(worst, SIF_REAL_ABS(g->values[i] - before[i]));
  CHECK(g->values && worst < 1e-3f,
    "a failed run should restore the grid (worst drift %g)", (double)worst);
  free(before);

  sif_chain_mesh_free(mesh);
  sif_grid_free(g);
  sif_field_free(f);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

/*
 * Cell size to grid cells: rounded up, except where the size divides the box
 * and float rounding alone would push the count one over.
 */
static void run_grid_cells(void) {
  printf("grid cells\n");

  const struct {
    uint64_t n;
    sif_real box, cell;
    sif_finder_units_t units;
    uint32_t want;
    sif_real volume;
  } cases[] = {
    {0, 200.0f, 6.25f, SIF_FINDER_UNITS_PHYSICAL, 32},
    {0, 200.0f, 6.0f, SIF_FINDER_UNITS_PHYSICAL, 34}, /* 33.3, up */
    /* 1000 / 11 rounds down in float, and 1000 over it is 11.0000003 */
    {0, 1000.0f, 1000.0f / 11.0f, SIF_FINDER_UNITS_PHYSICAL, 11},
    {0, 200.0f, 500.0f, SIF_FINDER_UNITS_PHYSICAL, 1},
    {256ull * 256 * 256, 1000.0f, 1.0f, SIF_FINDER_UNITS_MPS, 256},
    {256ull * 256 * 256, 1000.0f, 0.5f, SIF_FINDER_UNITS_MPS, 512},
    {1000ull * 1000 * 1000, 2000.0f, 1.0f, SIF_FINDER_UNITS_MPS, 1000},
    {N_P, BOX, 1.0f, SIF_FINDER_UNITS_MPS, 40}, /* 60000^(1/3) = 39.1 */
    /* Tracers in an eighth of the box: half the separation, twice the
     * cells. */
    {256ull * 256 * 256, 1000.0f, 1.0f, SIF_FINDER_UNITS_MPS, 512, 1.25e8f},
    /* The volume is only read for mps. */
    {0, 200.0f, 6.25f, SIF_FINDER_UNITS_PHYSICAL, 32, -1.0f},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    const uint32_t got = sif_finder_grid_cells_from_size(
      cases[i].n, cases[i].box, cases[i].cell, cases[i].units, cases[i].volume);
    CHECK(got == cases[i].want, "case %zu: %u cells, expected %u", i, got,
      cases[i].want);
  }

  const struct {
    uint64_t n;
    sif_real box, cell;
    sif_finder_units_t units;
    const char* reason;
    sif_real volume;
  } bad[] = {
    {N_P, 0.0f, 1.0f, SIF_FINDER_UNITS_MPS, "box_length: 0"},
    {N_P, BOX, -1.0f, SIF_FINDER_UNITS_MPS, "cell_size: -1"},
    {N_P, BOX, (sif_real)NAN, SIF_FINDER_UNITS_MPS, "cell_size: nan"},
    {N_P, BOX, (sif_real)INFINITY, SIF_FINDER_UNITS_PHYSICAL, "cell_size: inf"},
    {0, BOX, 1.0f, SIF_FINDER_UNITS_MPS, "n_particles: 0"},
    {N_P, BOX, 1.0f, (sif_finder_units_t)7, "units: 7"},
    {N_P, BOX, 1e-8f, SIF_FINDER_UNITS_PHYSICAL, "cells per side"},
    {N_P, BOX, 1.0f, SIF_FINDER_UNITS_MPS, "volume: -1", -1.0f},
    {N_P, BOX, 1.0f, SIF_FINDER_UNITS_MPS, "volume: nan", (sif_real)NAN},
  };
  for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
    sif_error_clear();
    const uint32_t got = sif_finder_grid_cells_from_size(
      bad[i].n, bad[i].box, bad[i].cell, bad[i].units, bad[i].volume);
    CHECK(got == 0 && error_has(bad[i].reason), "refusal %zu: %u cells, '%s'",
      i, got, sif_error_message());
  }

  printf("  ok\n");
}

/*
 * The variants: each is sif_finder_exodus_raw() at its preset, bit for bit,
 * the presets are the two agreed combinations, and the names go both ways.
 */
static void run_variants(void) {
  printf("variants\n");

  static const struct {
    sif_exodus_variant_t v;
    const char* name;
    sif_real overlap;
    sif_option opt;
  } want[] = {
    {SIF_EXODUS_EXCURSION_SET, "excursion-set", 0.0f, SIF_FINDER_SEEDS_CELLS},
    {SIF_EXODUS_EXCURSION_PEAK, "excursion-peak", 1.0f,
      SIF_FINDER_SEEDS_MINIMA},
  };
  const int n_want = (int)(sizeof(want) / sizeof(want[0]));

  for (int k = 0; k < n_want; k++) {
    sif_real overlap = -1.0f;
    sif_option opt = 99;
    sif_exodus_variant_t back = SIF__EXODUS_N_VARIANTS;
    CHECK(sif_finder_exodus_preset(want[k].v, &overlap, &opt) == SIF_OK &&
            overlap == want[k].overlap && opt == want[k].opt,
      "%s: preset overlap %g, options %u", want[k].name, (double)overlap,
      (unsigned)opt);
    const char* name = sif_finder_exodus_variant_name(want[k].v);
    CHECK(name && strcmp(name, want[k].name) == 0, "variant %d named %s",
      (int)want[k].v, name ? name : "(null)");
    CHECK(sif_finder_exodus_variant_from_name(want[k].name, &back) == SIF_OK &&
            back == want[k].v,
      "%s: read back as %d", want[k].name, (int)back);
  }
  sif_exodus_variant_t untouched = SIF_EXODUS_EXCURSION_PEAK;
  CHECK(sif_finder_exodus_variant_from_name("excursion-peek", &untouched) ==
            SIF_ERR_INVALID &&
          untouched == SIF_EXODUS_EXCURSION_PEAK,
    "an unknown name was taken");
  CHECK(sif_finder_exodus_variant_name(SIF__EXODUS_N_VARIANTS) == NULL,
    "a variant out of range has a name");

  sif_real* x = malloc(N_P * sizeof(sif_real));
  sif_real* y = malloc(N_P * sizeof(sif_real));
  sif_real* z = malloc(N_P * sizeof(sif_real));
  make_field(x, y, z);
  sif_field_t* f = sif_field_alloc(N_P);
  CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
    "sif_field_assign_positions failed");
  setup_t s = {sif_grid_alloc(N_GRID, BOX), NULL, f};
  s.mesh = sif_chain_mesh_alloc(MESH_CELLS, BOX, f, SIF_MESH_DROP_INDICES);
  free(x);
  free(y);
  free(z);

  const char* path = "test_finders_variant.ckpt";
  const sif_finder_exodus_extras_t extras = {.checkpoint = path};
  for (int k = 0; k < n_want; k++) {
    unlink(path);
    regrid(&s);
    sif_catalogue_t* raw = sif_finder_exodus_raw(s.grid, s.mesh, radii, n_radii,
      -0.7f, want[k].overlap, want[k].opt | SIF_FINDER_NO_CHECKPOINT, NULL);
    regrid(&s);
    sif_catalogue_t* cat = sif_finder_exodus(
      s.grid, s.mesh, radii, n_radii, -0.7f, want[k].v, &extras);
    CHECK(cat && cat->n_voids > 0, "%s: found nothing", want[k].name);
    check_same_catalogue(want[k].name, cat, raw);
    CHECK(!file_exists(path), "%s: left its checkpoint", want[k].name);
    sif_catalogue_free(raw);
    sif_catalogue_free(cat);
  }

  sif_catalogue_free(sif_finder_exodus_checkpoint_read(path, NULL));
  sif_chain_mesh_free(s.mesh);
  sif_grid_free(s.grid);
  sif_field_free(f);
}

/* Files left in @p path, other than . and .. */
static int dir_entries(const char* path) {
  int n = 0;
  DIR* dir = opendir(path);
  if (!dir)
    return -1;
  struct dirent* e;
  while ((e = readdir(dir)) != NULL)
    if (strcmp(e->d_name, ".") != 0 && strcmp(e->d_name, "..") != 0)
      n++;
  closedir(dir);
  return n;
}

/*
 * save_memory changes the memory and nothing else: the catalogue and the
 * grid handed back are those of a run without it, bit for bit, for both kinds
 * of candidate -- the minima smooth twice a rung. Its scratch file is gone
 * when the run is, and a scratch directory it cannot use fails the run before
 * it starts, with the grid as it was.
 */
static void run_save_memory(void) {
  printf("save_memory\n");

  sif_real* x = malloc(N_P * sizeof(sif_real));
  sif_real* y = malloc(N_P * sizeof(sif_real));
  sif_real* z = malloc(N_P * sizeof(sif_real));
  make_field(x, y, z);
  sif_field_t* f = sif_field_alloc(N_P);
  CHECK(sif_field_assign_positions(f, x, y, z) == SIF_OK,
    "sif_field_assign_positions failed");
  setup_t s = {sif_grid_alloc(N_GRID, BOX), NULL, f};
  s.mesh = sif_chain_mesh_alloc(MESH_CELLS, BOX, f, SIF_MESH_DROP_INDICES);
  free(x);
  free(y);
  free(z);

  const char* scratch = "test_finders_scratch";
  CHECK(sif_setting_override("scratch_directory", scratch) == SIF_OK,
    "pointing scratch_directory at the test failed");

  const uint64_t cells = s.grid->total_cells;
  sif_real* back = malloc(cells * sizeof(sif_real));
  const sif_finder_exodus_extras_t lean = {.save_memory = 1};

  static const struct {
    const char* label;
    sif_option opt;
    sif_real overlap;
  } cases[] = {
    {"cells", SIF_FINDER_SEEDS_CELLS, 0.0f},
    {"minima", SIF_FINDER_SEEDS_MINIMA, 1.0f},
  };
  for (int k = 0; k < 2; k++) {
    const sif_option opt = cases[k].opt | SIF_FINDER_NO_CHECKPOINT;

    regrid(&s);
    sif_catalogue_t* ref = sif_finder_exodus_raw(
      s.grid, s.mesh, radii, n_radii, -0.7f, cases[k].overlap, opt, NULL);
    memcpy(back, s.grid->values, cells * sizeof(sif_real));

    regrid(&s);
    sif_catalogue_t* cat = sif_finder_exodus_raw(
      s.grid, s.mesh, radii, n_radii, -0.7f, cases[k].overlap, opt, &lean);
    CHECK(
      ref && ref->n_voids > 0, "%s: reference found nothing", cases[k].label);
    check_same_catalogue(cases[k].label, cat, ref);
    CHECK(memcmp(back, s.grid->values, cells * sizeof(sif_real)) == 0,
      "%s: the grid handed back differs from a run without save_memory",
      cases[k].label);
    CHECK(dir_entries(scratch) == 0, "%s: the scratch file outlived the run",
      cases[k].label);

    sif_catalogue_free(ref);
    sif_catalogue_free(cat);
  }

  /* A directory naming an unset variable: refused before the grid is
   * touched. */
  unsetenv("SIF_TEST_NO_SUCH_VAR");
  CHECK(sif_setting_override("scratch_directory", "$SIF_TEST_NO_SUCH_VAR/x") ==
          SIF_OK,
    "overriding scratch_directory failed");
  regrid(&s);
  memcpy(back, s.grid->values, cells * sizeof(sif_real));
  sif_error_clear();
  sif_catalogue_t* none = sif_finder_exodus_raw(s.grid, s.mesh, radii, n_radii,
    -0.7f, 0.0f, SIF_FINDER_NO_CHECKPOINT, &lean);
  CHECK(!none && sif_error_status() == SIF_ERR_INVALID,
    "an unusable scratch directory should fail the run with SIF_ERR_INVALID");
  CHECK(s.grid->values &&
          memcmp(back, s.grid->values, cells * sizeof(sif_real)) == 0,
    "a run refused for its scratch directory changed the grid");
  sif_catalogue_free(none);

  CHECK(sif_setting_override("scratch_directory", NULL) == SIF_OK,
    "clearing the scratch_directory override failed");
  rmdir(scratch);
  free(back);
  sif_chain_mesh_free(s.mesh);
  sif_grid_free(s.grid);
  sif_field_free(f);
  printf("  ok\n");
}

int main(void) {
  sif_fft_config_t fftcfg = {.skip_tuning = true};
  sif_config_t cfg = {.fft_config = &fftcfg,
    .omp_config = NULL,
    .log_level = SIF_LOG_LEVEL_WARNING};
  if (sif_system_init(&cfg) != SIF_OK) {
    printf("FAIL: sif_system_init\n");
    return 1;
  }

  run_case("defaults", 0, 0.0f);

#if !SIF_TEST_INSTRUMENTED
  /* Each case is a full pipeline run; under a sanitizer the default case alone
     is enough to cover the pipeline without a multi-minute test. */
  run_case("overlap 0.2", 0, 0.2f);
  run_case("consume_grid", SIF_FINDER_CONSUME_GRID, 0.0f);
  run_case("minima centres", SIF_FINDER_SEEDS_MINIMA, 0.0f);
#endif

  run_weighted_cases();
  run_radius_oracle();
#if !SIF_TEST_INSTRUMENTED
  run_invariance(&cfg);
#endif
  run_checkpoints();
  run_minimum_scan();
  run_overlap_reach();
#if !SIF_TEST_INSTRUMENTED
  run_grid_shift();
#endif
  run_refusals();
  run_grid_cells();
#if !SIF_TEST_INSTRUMENTED
  run_variants();
#endif
  run_save_memory();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");

  sif_system_finalise();
  return failures != 0;
}
