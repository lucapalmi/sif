/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Nothing sif computes may depend on how many threads computed it.
 *
 * Every product a pipeline passes along -- the CIC grid, the density contrast,
 * the chain mesh and its weight sums, the tessellation, both finders'
 * catalogues -- is built once on one thread and then again on several others,
 * and has to come back bit for bit.
 *
 * Bitwise, and stage by stage, because a last-bit difference is all a
 * thread-dependent order of summation produces -- and an end-to-end check on a
 * clean field never sees one. The CIC grid used to cut its slabs per thread,
 * so the planes on slab boundaries were summed in a different order; on four
 * well-separated voids the catalogue came out the same, while on a real field
 * the finder's greedy acceptance turned the same bits into a few hundred
 * different voids.
 *
 * The thread counts include ones that divide nothing in sight, so a partition
 * derived from the thread count cannot line up with a grid's by accident. Each
 * count is a fresh sif_system_init(), since that is where the library reads
 * the thread count it sizes its work from; omp_set_num_threads() alone would
 * miss anything that reads the cached value.
 */
#include "sif/core/settings.h"
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/finder/spherical_finder.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/structures/tessellation.h"

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

#define BOX 200.0f
/* A Poisson field rather than carved holes: it is full of marginal voids, the
 * candidates a last-bit change in the field can tip either way. */
#define N_P SIF_TEST_SCALE(100000)

/* 40 is not a multiple of anything a slab width is likely to be; 64 is a power
 * of two, as most production grids are. */
static const uint32_t grid_sizes[] = {40, 64};
#define N_GRIDS (sizeof(grid_sizes) / sizeof(grid_sizes[0]))

#define FINDER_GRID 40
static const sif_real radii[] = {12.0f, 10.0f, 8.0f, 7.0f, 6.0f};
/* Loose enough that a Poisson field has plenty of voids at these radii. */
#define THRESHOLD -0.5f
static const uint32_t n_radii = (uint32_t)(sizeof(radii) / sizeof(radii[0]));

static const uint32_t thread_counts[] = {1, 2, 3, 5, 8};
#define N_COUNTS (sizeof(thread_counts) / sizeof(thread_counts[0]))

#define TESS_SAMPLES 8u
#define TESS_SEED    0x5eedULL

/* --- the field --- */

static uint64_t rng_state;
static double next_uniform(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)((rng_state >> 11) & 0x1FFFFFFFFFFFFFULL) /
         (double)0x20000000000000ULL;
}

static sif_real *xs, *ys, *zs, *ws;

static void make_field(void) {
  xs = malloc(N_P * sizeof(sif_real));
  ys = malloc(N_P * sizeof(sif_real));
  zs = malloc(N_P * sizeof(sif_real));
  ws = malloc(N_P * sizeof(sif_real));

  rng_state = 0x243F6A8885A308D3ULL;
  for (uint64_t i = 0; i < N_P; i++) {
    xs[i] = (sif_real)(next_uniform() * BOX);
    ys[i] = (sif_real)(next_uniform() * BOX);
    zs[i] = (sif_real)(next_uniform() * BOX);
    /* Not powers of two, so every sum rounds and the order shows. */
    ws[i] = (sif_real)(0.5 + next_uniform());
  }
}

static sif_field_t* field(int weighted) {
  sif_field_t* f = sif_field_alloc(N_P);
  CHECK(f && sif_field_assign_positions(f, xs, ys, zs) == SIF_OK,
    "building the field failed");
  if (f && weighted)
    CHECK(sif_field_assign_weights(f, ws) == SIF_OK,
      "assigning the weights failed");
  return f;
}

/* --- what one run produces --- */

typedef struct {
  sif_real* values;
  uint64_t n;
} column_t;

typedef struct {
  uint64_t n_voids;
  column_t cx, cy, cz, r;
} voids_t;

typedef struct {
  column_t density[N_GRIDS];
  column_t density_weighted[N_GRIDS];
  column_t contrast[N_GRIDS];
  column_t tess_density;

  column_t mesh_x, mesh_w, mesh_cell_w;
  uint64_t* mesh_offsets;
  uint64_t* mesh_indices;
  uint64_t mesh_cells;
  double mesh_total, mesh_total_unweighted;

  column_t tess_volumes;

  voids_t spherical, exodus;
} products_t;

static column_t keep(const sif_real* v, uint64_t n) {
  column_t c = {NULL, n};
  if (v && n) {
    c.values = malloc(n * sizeof(sif_real));
    memcpy(c.values, v, n * sizeof(sif_real));
  }
  return c;
}

static uint64_t* keep_u64(const uint64_t* v, uint64_t n) {
  if (!v)
    return NULL;
  uint64_t* c = malloc(n * sizeof(uint64_t));
  memcpy(c, v, n * sizeof(uint64_t));
  return c;
}

static voids_t keep_voids(sif_catalogue_t* cat) {
  voids_t v = {0};
  CHECK(cat != NULL, "a finder returned NULL");
  if (!cat)
    return v;
  v.n_voids = cat->n_voids;
  v.cx = keep(cat->cx, cat->n_voids);
  v.cy = keep(cat->cy, cat->n_voids);
  v.cz = keep(cat->cz, cat->n_voids);
  v.r = keep(cat->radii, cat->n_voids);
  sif_catalogue_free(cat);
  return v;
}

/* A grid of the field: CIC density, weighted or not. */
static sif_grid_t* grid_of(uint32_t n, int weighted) {
  sif_field_t* f = field(weighted);
  sif_grid_t* g = sif_grid_alloc(n, BOX);
  CHECK(g && sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
  sif_field_free(f);
  return g;
}

static void compute(products_t* p) {
  memset(p, 0, sizeof(*p));

  for (size_t i = 0; i < N_GRIDS; i++) {
    sif_grid_t* g = grid_of(grid_sizes[i], 0);
    p->density[i] = keep(g->values, g->total_cells);
    CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");
    p->contrast[i] = keep(g->values, g->total_cells);
    sif_grid_free(g);

    g = grid_of(grid_sizes[i], 1);
    p->density_weighted[i] = keep(g->values, g->total_cells);
    sif_grid_free(g);
  }

  /* The mesh the finder would get, kept with its indices so the order inside
   * every cell is compared too, not only the sums. */
  const uint32_t mesh_cells = sif_finder_suggest_mesh_cells(N_P, BOX, radii[0], 0);
  sif_field_t* f = field(1);
  sif_chain_mesh_t* m = sif_chain_mesh_alloc(mesh_cells, BOX, f, 0);
  sif_field_free(f);
  CHECK(m != NULL, "weighted chain mesh failed");
  if (m) {
    p->mesh_cells = m->total_cells;
    p->mesh_x = keep(m->x, m->n_particles);
    p->mesh_w = keep(m->weights, m->n_particles);
    p->mesh_cell_w = keep(m->cell_weights, m->total_cells);
    p->mesh_offsets = keep_u64(m->cell_offsets, m->total_cells + 1);
    p->mesh_indices = keep_u64(m->original_indices, m->n_particles);
    p->mesh_total = m->total_weight;

    sif_tessellation_t* t =
      sif_tessellation_alloc(m, TESS_SAMPLES, TESS_SEED, SIF_PBC_PERIODIC);
    CHECK(t != NULL, "tessellation failed");
    if (t) {
      p->tess_volumes = keep(t->volumes, m->n_particles);
      sif_grid_t* g = sif_grid_alloc(FINDER_GRID, BOX);
      CHECK(g && sif_grid_assign_cic_tessellation(g, t) == SIF_OK,
        "tessellation CIC failed");
      if (g)
        p->tess_density = keep(g->values, g->total_cells);
      sif_grid_free(g);
    }
    sif_tessellation_free(t);
  }
  sif_chain_mesh_free(m);

  /* The finders, end to end, on an unweighted field. */
  f = field(0);
  m = sif_chain_mesh_alloc(mesh_cells, BOX, f, SIF_MESH_DROP_INDICES);
  sif_field_free(f);
  CHECK(m != NULL, "chain mesh failed");
  if (m)
    p->mesh_total_unweighted = m->total_weight;

  sif_grid_t* g = grid_of(FINDER_GRID, 0);
  CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");

  p->spherical =
    keep_voids(sif_finder_spherical(g, radii, n_radii, THRESHOLD, 0.0f, 0));
  if (m)
    p->exodus = keep_voids(sif_finder_exodus(
      g, m, radii, n_radii, THRESHOLD, 0.0f, SIF_FINDER_NO_CHECKPOINT));

  sif_grid_free(g);
  sif_chain_mesh_free(m);
}

static void release_voids(voids_t* v) {
  free(v->cx.values);
  free(v->cy.values);
  free(v->cz.values);
  free(v->r.values);
}

static void release(products_t* p) {
  for (size_t i = 0; i < N_GRIDS; i++) {
    free(p->density[i].values);
    free(p->density_weighted[i].values);
    free(p->contrast[i].values);
  }
  free(p->tess_density.values);
  free(p->mesh_x.values);
  free(p->mesh_w.values);
  free(p->mesh_cell_w.values);
  free(p->mesh_offsets);
  free(p->mesh_indices);
  free(p->tess_volumes.values);
  release_voids(&p->spherical);
  release_voids(&p->exodus);
}

/* --- comparison --- */

/* Bits, not values: a difference in the last place is exactly the defect. */
static void same_column(uint32_t threads, const char* what, const column_t* got,
  const column_t* ref) {

  CHECK(got->n == ref->n, "%u threads: %s has %llu values, reference %llu",
    threads, what, (unsigned long long)got->n, (unsigned long long)ref->n);
  if (got->n != ref->n || !got->values || !ref->values)
    return;

  uint64_t n_diff = 0, first = 0;
  for (uint64_t i = 0; i < got->n; i++) {
    if (memcmp(&got->values[i], &ref->values[i], sizeof(sif_real)) != 0) {
      if (n_diff++ == 0)
        first = i;
    }
  }
  CHECK(n_diff == 0,
    "%u threads: %s differs in %llu of %llu values (first at %llu: %.9g, "
    "reference %.9g)",
    threads, what, (unsigned long long)n_diff, (unsigned long long)got->n,
    (unsigned long long)first, n_diff ? (double)got->values[first] : 0.0,
    n_diff ? (double)ref->values[first] : 0.0);
}

static void same_voids(
  uint32_t threads, const char* what, const voids_t* got, const voids_t* ref) {

  CHECK(got->n_voids == ref->n_voids, "%u threads: %s found %llu voids, "
                                      "reference %llu",
    threads, what, (unsigned long long)got->n_voids,
    (unsigned long long)ref->n_voids);
  if (got->n_voids != ref->n_voids)
    return;

  char label[64];
  snprintf(label, sizeof(label), "%s centres x", what);
  same_column(threads, label, &got->cx, &ref->cx);
  snprintf(label, sizeof(label), "%s centres y", what);
  same_column(threads, label, &got->cy, &ref->cy);
  snprintf(label, sizeof(label), "%s centres z", what);
  same_column(threads, label, &got->cz, &ref->cz);
  snprintf(label, sizeof(label), "%s radii", what);
  same_column(threads, label, &got->r, &ref->r);
}

static void compare(uint32_t t, const products_t* got, const products_t* ref) {
  char label[64];

  for (size_t i = 0; i < N_GRIDS; i++) {
    snprintf(label, sizeof(label), "CIC density %u^3", grid_sizes[i]);
    same_column(t, label, &got->density[i], &ref->density[i]);
    snprintf(label, sizeof(label), "weighted CIC density %u^3", grid_sizes[i]);
    same_column(t, label, &got->density_weighted[i], &ref->density_weighted[i]);
    snprintf(label, sizeof(label), "density contrast %u^3", grid_sizes[i]);
    same_column(t, label, &got->contrast[i], &ref->contrast[i]);
  }

  same_column(t, "mesh positions", &got->mesh_x, &ref->mesh_x);
  same_column(t, "mesh weights", &got->mesh_w, &ref->mesh_w);
  same_column(t, "mesh cell weights", &got->mesh_cell_w, &ref->mesh_cell_w);
  CHECK(got->mesh_cells == ref->mesh_cells && got->mesh_offsets &&
          ref->mesh_offsets &&
          memcmp(got->mesh_offsets, ref->mesh_offsets,
            (ref->mesh_cells + 1) * sizeof(uint64_t)) == 0,
    "%u threads: mesh cell offsets differ", t);
  CHECK(got->mesh_indices && ref->mesh_indices &&
          memcmp(got->mesh_indices, ref->mesh_indices,
            N_P * sizeof(uint64_t)) == 0,
    "%u threads: mesh particle order differs", t);
  CHECK(memcmp(&got->mesh_total, &ref->mesh_total, sizeof(double)) == 0,
    "%u threads: mesh total weight %.17g, reference %.17g", t, got->mesh_total,
    ref->mesh_total);
  CHECK(memcmp(&got->mesh_total_unweighted, &ref->mesh_total_unweighted,
          sizeof(double)) == 0,
    "%u threads: unweighted mesh total %.17g, reference %.17g", t,
    got->mesh_total_unweighted, ref->mesh_total_unweighted);

  same_column(t, "tessellation volumes", &got->tess_volumes, &ref->tess_volumes);
  same_column(
    t, "tessellation CIC density", &got->tess_density, &ref->tess_density);

  same_voids(t, "spherical", &got->spherical, &ref->spherical);
  same_voids(t, "exodus", &got->exodus, &ref->exodus);
}

/* --- driver --- */

static int start(uint32_t threads) {
  sif_fft_config_t fft = {.skip_tuning = true};
  sif_omp_config_t omp = {.n_threads = threads};
  /* Errors only: more threads than processors is the point here, and the
   * warning saying so would be noise. */
  sif_config_t cfg = {
    .fft_config = &fft, .omp_config = &omp, .log_level = SIF_LOG_LEVEL_ERROR};

  if (sif_system_init(&cfg) != SIF_OK)
    return SIF_ERR_INVALID;

  /* A cached grid would be read back rather than recomputed, and would pass
   * whatever the deposit did. */
  return sif_setting_set("grid_cache_enabled", "0");
}

int main(void) {
  make_field();

  printf("thread-count invariance, reference on 1 thread\n");
  if (start(thread_counts[0]) != SIF_OK) {
    printf("FAIL: sif_system_init\n");
    return 1;
  }
  products_t ref;
  compute(&ref);
  sif_system_finalise();

  /* A test that finds nothing cannot fail on what it finds. */
  CHECK(ref.exodus.n_voids >= 10 && ref.spherical.n_voids >= 10,
    "the reference field gave %llu exodus and %llu spherical voids; too few "
    "for the catalogues to show anything",
    (unsigned long long)ref.exodus.n_voids,
    (unsigned long long)ref.spherical.n_voids);
  printf("  %llu exodus, %llu spherical voids\n",
    (unsigned long long)ref.exodus.n_voids,
    (unsigned long long)ref.spherical.n_voids);

  for (size_t c = 1; c < N_COUNTS; c++) {
    printf("%u threads\n", thread_counts[c]);
    if (start(thread_counts[c]) != SIF_OK) {
      printf("  FAIL: sif_system_init with %u threads\n", thread_counts[c]);
      failures++;
      continue;
    }
    products_t got;
    compute(&got);
    sif_system_finalise();

    const int before = failures;
    compare(thread_counts[c], &got, &ref);
    if (failures == before)
      printf("  ok\n");
    release(&got);
  }

  release(&ref);
  free(xs);
  free(ys);
  free(zs);
  free(ws);

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
