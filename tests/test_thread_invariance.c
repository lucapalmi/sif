/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Nothing sif computes may depend on how many threads computed it, or on how
 * those threads happened to be scheduled.
 *
 * Every public path that runs in parallel is exercised once on one thread,
 * and every output it produces is kept as bytes. The same work is then done
 * again on several other thread counts, and repeatedly on the largest, and
 * each output has to come back bit for bit:
 *
 *   - a thread-count dependence -- a partition of the work, or an order of
 *     summation, that follows the number of threads -- shows up as a
 *     difference between counts;
 *   - a race -- an order that follows which thread got there first -- shows
 *     up as a difference between two runs on the same count, which is what
 *     the repeats are for.
 *
 * Bitwise, and stage by stage, because a last-bit difference is all a
 * reordered sum produces, and an end-to-end check never sees one on a clean
 * field. The CIC grid used to cut its slabs per thread: the planes on slab
 * boundaries summed in a different order, and on the four well-separated voids
 * of test_finders the catalogue still came out the same -- while on a real
 * field the finder's greedy acceptance turned those bits into a few hundred
 * different voids. Every intermediate product is therefore compared on its
 * own, and the finders run on a Poisson field full of marginal voids.
 *
 * The thread counts include ones that divide nothing in sight, so a partition
 * derived from the thread count cannot line up with a grid's by accident, and
 * one above any slab or block count the library uses. Each count is a fresh
 * sif_system_init(), since that is where the library reads the thread count
 * it sizes its work from; omp_set_num_threads() alone would miss anything
 * reading the cached value.
 *
 * To add a stage: compute it in compute(), and put() each output under a name
 * of its own. Nothing else needs to change.
 *
 * SIF_TEST_THREAD_REPEATS sets how many times the largest count is rerun
 * (default 2); raise it to hunt for a race that only shows up rarely.
 */
#include "sif/core/settings.h"
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/finder/spherical_finder.h"
#include "sif/measure/delta_distribution.h"
#include "sif/measure/delta_moments.h"
#include "sif/measure/profiles.h"
#include "sif/measure/size_function.h"
#include "sif/measure/survey_volume.h"
#include "sif/model/bbks.h"
#include "sif/model/cosmology.h"
#include "sif/model/delta_moments.h"
#include "sif/model/excursion_set.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/delta_distribution.h"
#include "sif/structures/delta_moments.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/structures/size_function.h"
#include "sif/structures/tessellation.h"
#include "sif/utils/align.h"
#include "sif/utils/array.h"

#include <math.h>
#include <stdarg.h>
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

/* Fewer particles than the largest thread count, so a per-thread split of the
 * particles leaves threads with nothing to do. */
#define N_TINY 5

/* 8 is narrower than one CIC slab, 17 and 40 are multiples of nothing a slab
 * or block is likely to be, and 64 is a power of two, as most production grids
 * are. */
static const uint32_t grid_sizes[] = {8, 17, 40, 64};
#define N_GRIDS (sizeof(grid_sizes) / sizeof(grid_sizes[0]))

#define FINDER_GRID 40
static const sif_real radii[] = {12.0f, 10.0f, 8.0f, 7.0f, 6.0f};
static const uint32_t n_radii = (uint32_t)(sizeof(radii) / sizeof(radii[0]));
/* Loose enough that a Poisson field has plenty of voids at these radii. */
#define THRESHOLD -0.5f

static const uint32_t thread_counts[] = {1, 2, 3, 5, 8, 13};
#define N_COUNTS (sizeof(thread_counts) / sizeof(thread_counts[0]))

#define TESS_SAMPLES 8u
#define SEED         0x5eedULL

/* --- inputs, the same for every run --- */

static uint64_t rng_state;
static double next_uniform(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)((rng_state >> 11) & 0x1FFFFFFFFFFFFFULL) /
         (double)0x20000000000000ULL;
}

typedef struct {
  sif_real *x, *y, *z;
  uint64_t n;
} points_t;

static points_t make_points(uint64_t n, double lo, double hi) {
  points_t p = {malloc(n * sizeof(sif_real)), malloc(n * sizeof(sif_real)),
    malloc(n * sizeof(sif_real)), n};
  for (uint64_t i = 0; i < n; i++) {
    p.x[i] = (sif_real)(lo + next_uniform() * (hi - lo));
    p.y[i] = (sif_real)(lo + next_uniform() * (hi - lo));
    p.z[i] = (sif_real)(lo + next_uniform() * (hi - lo));
  }
  return p;
}

static void free_points(points_t* p) {
  free(p->x);
  free(p->y);
  free(p->z);
}

/* The survey: a ball well inside its box, so the finder has the padding it
 * needs, with randoms three times as dense. */
#define SURVEY_BOX   300.0f
#define SURVEY_R     110.0
#define SURVEY_GRID  48
#define N_SURVEY     SIF_TEST_SCALE(60000)
#define N_SURVEY_RND (3 * N_SURVEY)

static points_t make_ball(uint64_t n) {
  points_t p = {malloc(n * sizeof(sif_real)), malloc(n * sizeof(sif_real)),
    malloc(n * sizeof(sif_real)), n};
  const double c = 0.5 * (double)SURVEY_BOX;
  uint64_t k = 0;
  while (k < n) {
    const double dx = (2.0 * next_uniform() - 1.0) * SURVEY_R;
    const double dy = (2.0 * next_uniform() - 1.0) * SURVEY_R;
    const double dz = (2.0 * next_uniform() - 1.0) * SURVEY_R;
    if (dx * dx + dy * dy + dz * dz > SURVEY_R * SURVEY_R)
      continue;
    p.x[k] = (sif_real)(c + dx);
    p.y[k] = (sif_real)(c + dy);
    p.z[k] = (sif_real)(c + dz);
    k++;
  }
  return p;
}

static points_t box_pts, tiny_pts, survey_pts, random_pts, sky_pts;
static sif_real *weights, *vx, *vy, *vz, *values;
#define N_VALUES SIF_TEST_SCALE(1000000)

/* A smooth, positive P(k) for the model stages. Any shape does: the point is
 * that the same input gives the same output. */
#define N_K 400
static sif_real pk_k[N_K], pk_p[N_K];

/* Radii for the excursion-set stages, ascending. The emulator's void-in-cloud
 * hazard only runs in parallel above 64 radii. */
#define N_MC  24
#define N_EMU 96
static sif_real mc_radii[N_MC], emu_radii[N_EMU];

static void make_inputs(void) {
  rng_state = 0x243F6A8885A308D3ULL;

  box_pts = make_points(N_P, 0.0, (double)BOX);
  tiny_pts = make_points(N_TINY, 0.0, (double)BOX);
  survey_pts = make_ball(N_SURVEY);
  random_pts = make_ball(N_SURVEY_RND);

  /* RA, Dec and redshift, as a sky catalogue holds them. */
  sky_pts = make_points(N_P, 0.0, 1.0);
  for (uint64_t i = 0; i < N_P; i++) {
    sky_pts.x[i] *= 360.0f;
    sky_pts.y[i] = (sif_real)(asin(2.0 * sky_pts.y[i] - 1.0) * 180.0 / M_PI);
    sky_pts.z[i] = 0.01f + 0.5f * sky_pts.z[i];
  }

  weights = malloc(N_P * sizeof(sif_real));
  vx = malloc(N_P * sizeof(sif_real));
  vy = malloc(N_P * sizeof(sif_real));
  vz = malloc(N_P * sizeof(sif_real));
  for (uint64_t i = 0; i < N_P; i++) {
    /* Spread over twenty octaves. Weights of one magnitude add up exactly
     * even in a float-fed double sum -- 1e5 of them in [0.5, 1.5] need 41 of
     * a double's 53 bits -- so every order would give the same total and a
     * reordered sum could not show. Spread like this, every sum rounds. */
    weights[i] = (sif_real)ldexp(
      0.5 + next_uniform(), (int)(next_uniform() * 21.0) - 10);
    vx[i] = (sif_real)(300.0 * (next_uniform() - 0.5));
    vy[i] = (sif_real)(300.0 * (next_uniform() - 0.5));
    vz[i] = (sif_real)(300.0 * (next_uniform() - 0.5));
  }

  values = malloc(N_VALUES * sizeof(sif_real));
  for (uint64_t i = 0; i < N_VALUES; i++)
    values[i] = (sif_real)(next_uniform() * 2.0 - 0.7);

  for (uint32_t i = 0; i < N_K; i++) {
    const double k = 1e-3 * pow(1e4, (double)i / (N_K - 1));
    const double q = k / 0.02;
    pk_k[i] = (sif_real)k;
    pk_p[i] = (sif_real)(2e4 * k / pow(1.0 + q * q, 1.5));
  }
  for (uint32_t i = 0; i < N_MC; i++)
    mc_radii[i] = (sif_real)(2.0 * pow(15.0, (double)i / (N_MC - 1)));
  for (uint32_t i = 0; i < N_EMU; i++)
    emu_radii[i] = (sif_real)(1.0 * pow(30.0, (double)i / (N_EMU - 1)));
}

static void free_inputs(void) {
  free_points(&box_pts);
  free_points(&tiny_pts);
  free_points(&survey_pts);
  free_points(&random_pts);
  free_points(&sky_pts);
  free(weights);
  free(vx);
  free(vy);
  free(vz);
  free(values);
}

static sif_field_t* field_of(const points_t* p, int weighted, int velocities) {
  sif_field_t* f = sif_field_alloc(p->n);
  CHECK(f && sif_field_assign_positions(f, p->x, p->y, p->z) == SIF_OK,
    "building a field failed");
  if (f && weighted)
    CHECK(sif_field_assign_weights(f, weights) == SIF_OK,
      "assigning the weights failed");
  if (f && velocities)
    CHECK(sif_field_assign_velocities(f, vx, vy, vz) == SIF_OK,
      "assigning the velocities failed");
  return f;
}

/* --- what one run produces: named blobs of bytes --- */

typedef struct {
  char name[80];
  unsigned char* bytes;
  size_t size;
  /** Width of one element, for saying where a difference is. */
  size_t elem;
} blob_t;

typedef struct {
  blob_t* items;
  size_t n, cap;
} products_t;

static void put(products_t* p, const void* data, size_t n_elem, size_t elem,
  const char* fmt, ...) {

  if (p->n == p->cap) {
    p->cap = p->cap ? 2 * p->cap : 64;
    p->items = realloc(p->items, p->cap * sizeof(blob_t));
  }
  blob_t* b = &p->items[p->n++];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(b->name, sizeof(b->name), fmt, ap);
  va_end(ap);

  b->elem = elem;
  b->size = data ? n_elem * elem : 0;
  b->bytes = NULL;
  if (b->size) {
    b->bytes = malloc(b->size);
    memcpy(b->bytes, data, b->size);
  }
}

#define PUT_REALS(p, v, n, ...)  put(p, v, n, sizeof(sif_real), __VA_ARGS__)
#define PUT_U64S(p, v, n, ...)   put(p, v, n, sizeof(uint64_t), __VA_ARGS__)
#define PUT_DOUBLES(p, v, n, ...) put(p, v, n, sizeof(double), __VA_ARGS__)

static void release(products_t* p) {
  for (size_t i = 0; i < p->n; i++)
    free(p->items[i].bytes);
  free(p->items);
  p->items = NULL;
  p->n = p->cap = 0;
}

static void put_catalogue(
  products_t* p, const char* what, const sif_catalogue_t* cat) {

  const uint64_t n = cat ? cat->n_voids : 0;
  PUT_U64S(p, &n, 1, "%s: void count", what);
  if (!cat)
    return;
  PUT_REALS(p, cat->cx, n, "%s: centres x", what);
  PUT_REALS(p, cat->cy, n, "%s: centres y", what);
  PUT_REALS(p, cat->cz, n, "%s: centres z", what);
  PUT_REALS(p, cat->radii, n, "%s: radii", what);
  if (cat->footprint)
    PUT_REALS(p, cat->footprint, n, "%s: footprint", what);
  if (cat->footprint_shell)
    PUT_REALS(p, cat->footprint_shell, n, "%s: footprint shell", what);
}

/* --- the stages --- */

static void stage_arrays(products_t* p) {
  const sif_real s = sif_array_sum(values, N_VALUES);
  const sif_real lo = sif_array_min(values, N_VALUES);
  const sif_real hi = sif_array_max(values, N_VALUES);
  PUT_REALS(p, &s, 1, "array sum");
  PUT_REALS(p, &lo, 1, "array min");
  PUT_REALS(p, &hi, 1, "array max");

  sif_real* lin = sif_array_linspace(-3.0f, 7.0f, N_VALUES);
  sif_real* lg = sif_array_logspace(-2.0f, 3.0f, N_VALUES, 10.0f);
  PUT_REALS(p, lin, lin ? N_VALUES : 0, "array linspace");
  PUT_REALS(p, lg, lg ? N_VALUES : 0, "array logspace");
  sif_free_aligned(lin);
  sif_free_aligned(lg);
}

static void stage_field(products_t* p) {
  /* Shifted half a box, so a quarter of every axis has to wrap. */
  sif_field_t* f = field_of(&box_pts, 1, 1);
  const sif_real shift[3] = {0.5f * BOX, -0.25f * BOX, 0.125f * BOX};
  CHECK(sif_field_translate(f, shift) == SIF_OK, "translate failed");
  uint64_t counts[3] = {0, 0, 0};
  CHECK(sif_field_wrap_periodic(f, BOX, &counts[0], &counts[1], &counts[2]) ==
          SIF_OK,
    "wrap failed");
  PUT_U64S(p, counts, 3, "wrap counts");
  PUT_REALS(p, f->x, N_P, "wrapped x");
  PUT_REALS(p, f->y, N_P, "wrapped y");
  PUT_REALS(p, f->z, N_P, "wrapped z");

  CHECK(sif_field_sort_morton(f) == SIF_OK, "Morton sort failed");
  PUT_REALS(p, f->x, N_P, "Morton-sorted x");
  PUT_REALS(p, f->y, N_P, "Morton-sorted y");
  PUT_REALS(p, f->z, N_P, "Morton-sorted z");
  PUT_REALS(p, f->weights, N_P, "Morton-sorted weights");
  PUT_REALS(p, f->vx, N_P, "Morton-sorted vx");
  sif_field_free(f);

  /* Sky coordinates to positions, which also builds the distance table. */
  f = field_of(&sky_pts, 0, 0);
  f->units = SIF_COORDINATES_SKY;
  CHECK(sif_field_convert_sky_coordinates(f, SIF_COSMOLOGY_FLAT_LCDM(0.31)) ==
          SIF_OK,
    "sky conversion failed");
  PUT_REALS(p, f->x, N_P, "sky to Cartesian x");
  PUT_REALS(p, f->y, N_P, "sky to Cartesian y");
  PUT_REALS(p, f->z, N_P, "sky to Cartesian z");
  sif_field_free(f);
}

static sif_grid_t* grid_of(const points_t* pts, uint32_t n, int weighted) {
  sif_field_t* f = field_of(pts, weighted, 0);
  sif_grid_t* g = sif_grid_alloc(n, BOX);
  CHECK(g && sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
  sif_field_free(f);
  return g;
}

static void stage_grids(products_t* p) {
  for (size_t i = 0; i < N_GRIDS; i++) {
    const uint32_t n = grid_sizes[i];
    sif_grid_t* g = grid_of(&box_pts, n, 0);
    PUT_REALS(p, g->values, g->total_cells, "CIC density %u^3", n);
    CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");
    PUT_REALS(p, g->values, g->total_cells, "density contrast %u^3", n);
    sif_grid_free(g);

    g = grid_of(&box_pts, n, 1);
    PUT_REALS(p, g->values, g->total_cells, "weighted CIC density %u^3", n);
    CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");
    PUT_REALS(
      p, g->values, g->total_cells, "weighted density contrast %u^3", n);
    sif_grid_free(g);
  }

  sif_grid_t* g = grid_of(&tiny_pts, 8, 0);
  PUT_REALS(p, g->values, g->total_cells, "CIC density of %d particles",
    N_TINY);
  sif_grid_free(g);
}

static void stage_mesh(products_t* p) {
  const uint32_t cells =
    sif_finder_suggest_mesh_cells(N_P, BOX, radii[0], 0);
  sif_field_t* f = field_of(&box_pts, 1, 1);
  sif_chain_mesh_t* m = sif_chain_mesh_alloc(cells, BOX, f, 0);
  sif_field_free(f);
  CHECK(m != NULL, "weighted chain mesh failed");
  if (!m)
    return;

  /* The indices too, so the order inside every cell is compared, not only
   * what the cells add up to. */
  PUT_U64S(p, m->cell_offsets, m->total_cells + 1, "mesh cell offsets");
  PUT_U64S(p, m->original_indices, m->n_particles, "mesh particle order");
  PUT_REALS(p, m->x, m->n_particles, "mesh x");
  PUT_REALS(p, m->weights, m->n_particles, "mesh weights");
  PUT_REALS(p, m->vx, m->n_particles, "mesh vx");
  PUT_REALS(p, m->cell_weights, m->total_cells, "mesh cell weights");
  PUT_DOUBLES(p, &m->total_weight, 1, "mesh total weight");

  sif_tessellation_t* t =
    sif_tessellation_alloc(m, TESS_SAMPLES, SEED, SIF_PBC_PERIODIC);
  CHECK(t != NULL, "tessellation failed");
  if (t) {
    PUT_REALS(p, t->volumes, m->n_particles, "tessellation volumes");
    PUT_REALS(p, t->samples->weights, t->samples->n_particles,
      "tessellation sample weights");
    sif_grid_t* g = sif_grid_alloc(FINDER_GRID, BOX);
    CHECK(g && sif_grid_assign_cic_tessellation(g, t) == SIF_OK,
      "tessellation CIC failed");
    if (g)
      PUT_REALS(p, g->values, g->total_cells, "tessellation CIC density");
    sif_grid_free(g);
  }
  sif_tessellation_free(t);
  sif_chain_mesh_free(m);

  f = field_of(&tiny_pts, 1, 0);
  m = sif_chain_mesh_alloc(2, BOX, f, 0);
  sif_field_free(f);
  CHECK(m != NULL, "tiny chain mesh failed");
  if (m) {
    PUT_U64S(p, m->original_indices, m->n_particles, "tiny mesh order");
    PUT_DOUBLES(p, &m->total_weight, 1, "tiny mesh total weight");
  }
  sif_chain_mesh_free(m);
}

static void stage_delta_stats(products_t* p) {
  sif_grid_t* g = grid_of(&box_pts, 64, 0);
  CHECK(sif_grid_to_density_contrast(g) == SIF_OK, "density contrast failed");

  const sif_real r[] = {10.0f, 15.0f, 20.0f};
  const sif_real bounds[2] = {-1.0f, 3.0f};
  const sif_option shuffles[] = {SIF_DELTA_SHUFFLE_NONE,
    SIF_DELTA_SHUFFLE_PHASES, SIF_DELTA_SHUFFLE_GAUSSIAN};
  const char* names[] = {"", ", phases shuffled", ", Gaussianised"};

  for (int s = 0; s < 3; s++) {
    sif_delta_distribution_t* d =
      sif_delta_distribution_grid(g, r, 3, 40, bounds, SEED, shuffles[s]);
    CHECK(d != NULL, "delta distribution failed");
    if (d)
      PUT_REALS(p, d->distributions, (uint64_t)d->n_radii * d->n_bins,
        "delta distribution%s", names[s]);
    sif_delta_distribution_free(d);

    sif_delta_moments_t* m =
      sif_delta_moments_grid(g, r, 3, 2, N_P, SEED, shuffles[s]);
    CHECK(m != NULL, "delta moments failed");
    if (m) {
      PUT_REALS(p, m->sigma, (uint64_t)m->n_moments * m->n_radii,
        "spectral moments%s", names[s]);
      PUT_REALS(p, m->high_k_fraction, (uint64_t)m->n_moments * m->n_radii,
        "high-k fractions%s", names[s]);

      if (s == 0) {
        sif_real* nb = sif_bbks_number_density_cumulative(-0.5f, m, 0);
        CHECK(nb != NULL, "BBKS cumulative density failed");
        PUT_REALS(p, nb, nb ? m->n_radii : 0, "BBKS cumulative density");
        sif_free_aligned(nb);
      }
    }
    sif_delta_moments_free(m);
  }
  sif_grid_free(g);
}

static void stage_models(products_t* p) {
  /* Monte Carlo first crossings: per-path seeding, per-thread histograms. */
  sif_real sigma[N_EMU];
  double dvar[N_EMU];
  double* cov = sif_delta_covariance_pk(
    pk_k, pk_p, N_K, mc_radii, N_MC, sigma, NULL, NULL, SIF_DEFAULT);
  CHECK(cov != NULL, "covariance failed");
  if (cov) {
    sif_real* barrier = sif_ep_barrier_smt(sigma, N_MC, 0.5f, 0.3f, 0.8f);
    sif_real collapse[N_MC];
    for (uint32_t i = 0; i < N_MC; i++)
      collapse[i] = 1.686f;
    uint64_t collapsed[N_MC];
    uint64_t* counts = barrier ? sif_ep_first_crossing_counts(mc_radii, N_MC,
                                   cov, barrier, collapse,
                                   SIF_TEST_SCALE(40000), SEED, collapsed, 0)
                               : NULL;
    CHECK(counts != NULL, "first-crossing counts failed");
    PUT_U64S(p, counts, counts ? N_MC : 0, "first-crossing counts");
    PUT_U64S(p, collapsed, counts ? N_MC : 0, "first-crossing collapses");
    sif_free_aligned(counts);
    sif_free_aligned(barrier);
    sif_free_aligned(cov);
  }

  /* The emulator with void-in-cloud, at enough radii to go parallel. */
  cov = sif_delta_covariance_pk(
    pk_k, pk_p, N_K, emu_radii, N_EMU, sigma, NULL, dvar, SIF_DEFAULT);
  CHECK(cov != NULL, "covariance failed");
  if (cov) {
    PUT_DOUBLES(p, cov, (uint64_t)N_EMU * (N_EMU + 1) / 2, "covariance");
    sif_real* barrier = sif_ep_barrier_smt(sigma, N_EMU, 0.5f, 0.3f, 0.8f);
    sif_real collapse[N_EMU];
    for (uint32_t i = 0; i < N_EMU; i++)
      collapse[i] = 1.686f;
    sif_real* f = barrier
                    ? sif_ep_multiplicity_function_emu(emu_radii, N_EMU, sigma,
                        barrier, dvar, collapse, cov, NULL, SIF_DEFAULT)
                    : NULL;
    CHECK(f != NULL, "void-in-cloud emulator failed");
    PUT_REALS(p, f, f ? N_EMU : 0, "void-in-cloud multiplicity");
    sif_free_aligned(f);
    sif_free_aligned(barrier);
    sif_free_aligned(cov);
  }
}

static void stage_finders(products_t* p, uint64_t* n_found) {
  const uint32_t cells =
    sif_finder_suggest_mesh_cells(N_P, BOX, radii[0], 0);
  const sif_option no_ckpt = SIF_FINDER_NO_CHECKPOINT;

  for (int weighted = 0; weighted <= 1; weighted++) {
    const char* w = weighted ? "weighted " : "";

    sif_field_t* f = field_of(&box_pts, weighted, 1);
    sif_chain_mesh_t* m =
      sif_chain_mesh_alloc(cells, BOX, f, SIF_MESH_DROP_INDICES);
    sif_grid_t* g = sif_grid_alloc(FINDER_GRID, BOX);
    CHECK(g && sif_grid_assign_cic(g, f) == SIF_OK, "CIC assignment failed");
    sif_field_free(f);
    CHECK(m && sif_grid_to_density_contrast(g) == SIF_OK,
      "finder inputs failed");

    if (!weighted) {
      sif_catalogue_t* sph =
        sif_finder_spherical(g, radii, n_radii, THRESHOLD, 0.0f, 0);
      put_catalogue(p, "spherical", sph);
      n_found[0] = sph ? sph->n_voids : 0;
      sif_catalogue_free(sph);
    }

    sif_catalogue_t* cat = m ? sif_finder_exodus(g, m, radii, n_radii,
                                 THRESHOLD, 0.0f, no_ckpt)
                             : NULL;
    char what[32];
    snprintf(what, sizeof(what), "%sexodus", w);
    put_catalogue(p, what, cat);

    if (cat && !weighted) {
      n_found[1] = cat->n_voids;

      sif_density_profiles_t* dens = NULL;
      sif_velocity_profiles_t* vel = NULL;
      CHECK(sif_profiles(cat, m, 3.0f, 30, SIF_PROFILES_DIFFERENTIAL, &dens,
              &vel) == SIF_OK,
        "profiles failed");
      if (dens)
        PUT_REALS(p, dens->profiles, dens->n_voids * dens->n_bins,
          "density profiles");
      if (vel)
        PUT_REALS(p, vel->v_rad, vel->n_voids * vel->n_bins,
          "velocity profiles");
      sif_density_profiles_free(dens);
      sif_velocity_profiles_free(vel);

      sif_size_function_t* vsf =
        sif_size_function_catalogue(cat, BOX, 12, 0, 0.0f, 0.0f);
      CHECK(vsf != NULL, "size function failed");
      if (vsf) {
        PUT_U64S(p, vsf->counts, vsf->n_bins, "size function counts");
        PUT_REALS(p, vsf->vsf, vsf->n_bins, "size function");
      }
      sif_size_function_free(vsf);

      /* To the sky and back: the distance table, and both conversions. */
      const sif_cosmology_t* cosmo = SIF_COSMOLOGY_FLAT_LCDM(0.31);
      CHECK(sif_catalogue_to_sky(cat, cosmo) == SIF_OK, "to_sky failed");
      put_catalogue(p, "exodus on the sky", cat);
      CHECK(sif_catalogue_to_cartesian(cat, cosmo) == SIF_OK,
        "to_cartesian failed");
      put_catalogue(p, "exodus back from the sky", cat);
    }

    sif_catalogue_free(cat);
    sif_grid_free(g);
    sif_chain_mesh_free(m);
  }
}

static void stage_survey(products_t* p, uint64_t* n_found) {
  static const sif_real sr[] = {12.0f, 10.0f, 8.5f, 7.0f};
  const uint32_t n_sr = (uint32_t)(sizeof(sr) / sizeof(sr[0]));

  sif_field_t* fd = field_of(&survey_pts, 0, 0);
  sif_field_t* fr = field_of(&random_pts, 0, 0);

  sif_real volume = 0.0f;
  CHECK(sif_survey_volume(fr, NULL, &volume) == SIF_OK, "survey volume failed");
  PUT_REALS(p, &volume, 1, "survey volume");

  sif_grid_t* gd = sif_grid_alloc(SURVEY_GRID, SURVEY_BOX);
  sif_grid_t* gr = sif_grid_alloc(SURVEY_GRID, SURVEY_BOX);
  CHECK(gd && gr && sif_grid_assign_cic(gd, fd) == SIF_OK &&
          sif_grid_assign_cic(gr, fr) == SIF_OK,
    "survey CIC failed");
  sif_chain_mesh_t* md = sif_chain_mesh_alloc(
    sif_finder_suggest_mesh_cells(N_SURVEY, SURVEY_BOX, sr[0], 0), SURVEY_BOX,
    fd, SIF_MESH_DROP_INDICES);
  sif_chain_mesh_t* mr = sif_chain_mesh_alloc(
    sif_finder_suggest_mesh_cells(N_SURVEY_RND, SURVEY_BOX, sr[0], 0),
    SURVEY_BOX, fr, SIF_MESH_DROP_INDICES);
  sif_field_free(fd);
  sif_field_free(fr);
  CHECK(md && mr, "survey meshes failed");

  sif_catalogue_t* cat = md && mr ? sif_finder_exodus_survey(gd, gr, md, mr,
                                      NULL, sr, n_sr, THRESHOLD, 0.0f,
                                      SIF_FINDER_NO_CHECKPOINT)
                                  : NULL;
  CHECK(cat != NULL, "survey finder failed");
  put_catalogue(p, "survey exodus", cat);
  n_found[2] = cat ? cat->n_voids : 0;

  sif_catalogue_free(cat);
  sif_chain_mesh_free(md);
  sif_chain_mesh_free(mr);
  sif_grid_free(gd);
  sif_grid_free(gr);
}

/* Everything, in a fixed order. n_found: spherical, exodus, survey. */
static void compute(products_t* p, uint64_t n_found[3]) {
  memset(p, 0, sizeof(*p));
  stage_arrays(p);
  stage_field(p);
  stage_grids(p);
  stage_mesh(p);
  stage_delta_stats(p);
  stage_models(p);
  stage_finders(p, n_found);
  stage_survey(p, n_found);
}

/* --- comparison --- */

/* Bits, not values: a difference in the last place is exactly the defect. */
static void compare(const char* run, const products_t* got,
  const products_t* ref) {

  CHECK(got->n == ref->n, "%s: %zu outputs, reference has %zu", run, got->n,
    ref->n);

  const size_t n = got->n < ref->n ? got->n : ref->n;
  for (size_t i = 0; i < n; i++) {
    const blob_t* a = &got->items[i];
    const blob_t* b = &ref->items[i];

    if (strcmp(a->name, b->name) != 0) {
      CHECK(0, "%s: output %zu is '%s', reference has '%s'", run, i, a->name,
        b->name);
      return;
    }
    if (a->size != b->size) {
      CHECK(0, "%s: %s has %zu bytes, reference %zu", run, a->name, a->size,
        b->size);
      continue;
    }

    const size_t n_elem = a->elem ? a->size / a->elem : 0;
    size_t n_diff = 0, first = 0;
    for (size_t e = 0; e < n_elem; e++) {
      if (memcmp(a->bytes + e * a->elem, b->bytes + e * a->elem, a->elem)) {
        if (n_diff++ == 0)
          first = e;
      }
    }
    CHECK(n_diff == 0, "%s: %s differs in %zu of %zu values (first at %zu)",
      run, a->name, n_diff, n_elem, first);
  }
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

static int run_and_compare(
  uint32_t threads, const char* label, const products_t* ref) {

  printf("%s\n", label);
  if (start(threads) != SIF_OK) {
    printf("  FAIL: sif_system_init with %u threads\n", threads);
    failures++;
    return 0;
  }
  products_t got;
  uint64_t found[3] = {0, 0, 0};
  compute(&got, found);
  sif_system_finalise();

  const int before = failures;
  compare(label, &got, ref);
  if (failures == before)
    printf("  ok (%zu outputs)\n", got.n);
  release(&got);
  return 1;
}

int main(void) {
  make_inputs();

  printf("reference on 1 thread\n");
  if (start(thread_counts[0]) != SIF_OK) {
    printf("FAIL: sif_system_init\n");
    return 1;
  }
  products_t ref;
  uint64_t found[3] = {0, 0, 0};
  compute(&ref, found);
  sif_system_finalise();
  printf("  %zu outputs; %llu spherical, %llu exodus, %llu survey voids\n",
    ref.n, (unsigned long long)found[0], (unsigned long long)found[1],
    (unsigned long long)found[2]);

#if !SIF_TEST_INSTRUMENTED
  /* A finder that finds nothing cannot fail on what it finds. Not checked
   * under a sanitizer, whose scaled-down fields are too sparse to hold
   * voids at these radii. */
  CHECK(found[0] >= 10 && found[1] >= 10 && found[2] >= 10,
    "too few voids for the catalogues to show anything");
#endif

  for (size_t c = 1; c < N_COUNTS; c++) {
    char label[48];
    snprintf(label, sizeof(label), "%u threads", thread_counts[c]);
    run_and_compare(thread_counts[c], label, &ref);
  }

  /* The same count again: anything that differs now depends on timing. */
  const char* env = getenv("SIF_TEST_THREAD_REPEATS");
  const int repeats = env ? atoi(env) : 2;
  const uint32_t top = thread_counts[N_COUNTS - 1];
  for (int r = 0; r < repeats; r++) {
    char label[48];
    snprintf(label, sizeof(label), "%u threads, repeat %d", top, r + 1);
    run_and_compare(top, label, &ref);
  }

  release(&ref);
  free_inputs();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
