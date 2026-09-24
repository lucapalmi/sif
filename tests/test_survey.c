/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * End-to-end check for the survey finder.
 *
 * A synthetic survey: a spherical footprint inside a padded box, with a
 * rectangular stripe masked out of it, data and randoms drawn over it, and
 * voids carved into the data -- three well inside, one cut by the edge. Run
 * once with uniform selection and once with a density that falls by a factor
 * of 2.5 across the footprint, which the randoms carry and the finder has to
 * divide back out. Then weights on both sides, which must change nothing when
 * they only rescale, and the refusals: a survey too close to the box, and a
 * grid that is already a contrast.
 */
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
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

#define BOX    400.0
#define N_GRID 64
#define CELL   (BOX / N_GRID)

/* The footprint: a ball, less a stripe that runs through it along z. */
#define FOOT_X 200.0
#define FOOT_Y 200.0
#define FOOT_Z 200.0
#define FOOT_R 140.0

#define N_DATA    100000
#define N_RANDOMS (10 * N_DATA)

static const sif_real radii[] = {30.0f, 24.0f, 19.0f, 15.0f};
static const uint32_t n_radii = sizeof(radii) / sizeof(radii[0]);

typedef struct {
  double x, y, z, r;
} hole_t;

/* Three voids well inside the footprint, clear of the stripe and of the edge
 * by more than twice their radius. */
static const hole_t inner[] = {
  {160.0, 160.0, 160.0, 26.0},
  {240.0, 160.0, 240.0, 22.0},
  {150.0, 240.0, 220.0, 20.0},
};
#define N_INNER 3

/* One cut by the footprint's edge: its centre is observed, its top is not. */
static const hole_t edge = {200.0, 200.0, 325.0, 25.0};

static uint64_t rng_state;
static double next_uniform(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)((rng_state >> 11) & 0x1FFFFFFFFFFFFFULL) /
         (double)0x20000000000000ULL;
}

static int in_footprint(double x, double y, double z) {
  const double dx = x - FOOT_X, dy = y - FOOT_Y, dz = z - FOOT_Z;
  if (dx * dx + dy * dy + dz * dz > FOOT_R * FOOT_R)
    return 0;
  /* The masked stripe, as a bright star or a bad field would leave. */
  if (fabs(x - 200.0) < 12.0 && fabs(y - 260.0) < 12.0)
    return 0;
  return 1;
}

static int in_hole(const hole_t* h, double x, double y, double z) {
  const double dx = x - h->x, dy = y - h->y, dz = z - h->z;
  return dx * dx + dy * dy + dz * dz < h->r * h->r;
}

/* The radial selection, as a fraction kept: 1 at the bottom of the footprint
 * and 0.4 at the top. Flat when `selection` is off. */
static double kept_fraction(double z, int selection) {
  if (!selection)
    return 1.0;
  return 1.0 - 0.6 * (z - (FOOT_Z - FOOT_R)) / (2.0 * FOOT_R);
}

/*
 * n points over the footprint under the selection. Data (`carve` set) lose all
 * but 3% of what falls in a hole, so the holes are underdense but not empty.
 */
static void draw(sif_real* x, sif_real* y, sif_real* z, uint64_t n,
  int selection, int carve, sif_real offset) {

  uint64_t k = 0;
  while (k < n) {
    const double px = FOOT_X + (2.0 * next_uniform() - 1.0) * FOOT_R;
    const double py = FOOT_Y + (2.0 * next_uniform() - 1.0) * FOOT_R;
    const double pz = FOOT_Z + (2.0 * next_uniform() - 1.0) * FOOT_R;

    if (!in_footprint(px, py, pz))
      continue;
    if (next_uniform() > kept_fraction(pz, selection))
      continue;

    if (carve) {
      int carved = in_hole(&edge, px, py, pz);
      for (int h = 0; h < N_INNER && !carved; h++)
        carved = in_hole(&inner[h], px, py, pz);
      if (carved && next_uniform() > 0.03)
        continue;
    }

    x[k] = (sif_real)px + offset;
    y[k] = (sif_real)py + offset;
    z[k] = (sif_real)pz + offset;
    k++;
  }
}

typedef struct {
  sif_grid_t* grid;
  sif_chain_mesh_t* mesh;
} tracers_t;

/* `weight` > 0 gives every tracer that weight; 0 leaves the field unweighted.
 */
static tracers_t build_weighted(const sif_real* x, const sif_real* y,
  const sif_real* z, uint64_t n, sif_real weight) {

  sif_field_t* f = sif_field_alloc(n);
  sif_field_assign_positions(f, x, y, z);
  if (weight > 0.0f) {
    sif_real* w = malloc(n * sizeof(sif_real));
    for (uint64_t i = 0; i < n; i++)
      w[i] = weight;
    sif_field_assign_weights(f, w);
    free(w);
  }

  tracers_t t;
  t.grid = sif_grid_alloc(N_GRID, (sif_real)BOX);
  CHECK(sif_grid_assign_cic(t.grid, f) == SIF_OK, "CIC assignment failed");
  t.mesh = sif_chain_mesh_alloc(
    sif_finder_suggest_mesh_cells(n, (sif_real)BOX, radii[0]), (sif_real)BOX, f,
    SIF_MESH_DROP_INDICES);
  sif_field_free(f);
  return t;
}

static tracers_t build(
  const sif_real* x, const sif_real* y, const sif_real* z, uint64_t n) {
  return build_weighted(x, y, z, n, 0.0f);
}

static void release(tracers_t* t) {
  sif_grid_free(t->grid);
  sif_chain_mesh_free(t->mesh);
}

/* The void nearest a hole, or -1. */
static int64_t nearest(const sif_catalog_t* cat, const hole_t* h, double* d) {
  int64_t best = -1;
  double best_d = 1e300;
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    const double dx = (double)cat->cx[i] - h->x;
    const double dy = (double)cat->cy[i] - h->y;
    const double dz = (double)cat->cz[i] - h->z;
    const double di = sqrt(dx * dx + dy * dy + dz * dz);
    if (di < best_d) {
      best_d = di;
      best = (int64_t)i;
    }
  }
  *d = best_d;
  return best;
}

static void run_case(const char* label, int selection) {
  printf("%s\n", label);

  sif_real* dx = malloc(N_DATA * sizeof(sif_real));
  sif_real* dy = malloc(N_DATA * sizeof(sif_real));
  sif_real* dz = malloc(N_DATA * sizeof(sif_real));
  sif_real* rx = malloc(N_RANDOMS * sizeof(sif_real));
  sif_real* ry = malloc(N_RANDOMS * sizeof(sif_real));
  sif_real* rz = malloc(N_RANDOMS * sizeof(sif_real));

  rng_state = 0x243F6A8885A308D3ULL;
  draw(dx, dy, dz, N_DATA, selection, 1, 0.0f);
  draw(rx, ry, rz, N_RANDOMS, selection, 0, 0.0f);

  tracers_t data = build(dx, dy, dz, N_DATA);
  tracers_t rand = build(rx, ry, rz, N_RANDOMS);
  CHECK(data.mesh && rand.mesh, "%s: mesh construction failed", label);

  /* Snapshots, for the restoration contract. */
  const uint64_t cells = data.grid->total_cells;
  sif_real* data_before = malloc(cells * sizeof(sif_real));
  sif_real* rand_before = malloc(cells * sizeof(sif_real));
  memcpy(data_before, data.grid->values, cells * sizeof(sif_real));
  memcpy(rand_before, rand.grid->values, cells * sizeof(sif_real));

  sif_catalog_t* cat = sif_finder_exodus_survey(data.grid, rand.grid, data.mesh,
    rand.mesh, radii, n_radii, -0.7f, 0.0f, SIF_DEFAULT);

  CHECK(cat != NULL, "%s: the survey finder returned NULL", label);
  if (cat) {
    CHECK(cat->footprint && cat->footprint_shell,
      "%s: the catalogue carries no footprint columns", label);

    /* Every void found is one that was carved: the selection gradient must
     * not show up as voids of its own. */
    CHECK(cat->n_voids == N_INNER || cat->n_voids == N_INNER + 1,
      "%s: found %llu voids, expected the %d inner ones and perhaps the edge "
      "one",
      label, (unsigned long long)cat->n_voids, N_INNER);

    for (int h = 0; h < N_INNER; h++) {
      double d;
      const int64_t i = nearest(cat, &inner[h], &d);
      CHECK(i >= 0 && d <= 2.0 * CELL,
        "%s: inner hole %d not recovered (nearest void %.1f away)", label, h,
        d);
      if (i < 0 || d > 2.0 * CELL)
        continue;

      const double r = (double)cat->radii[i];
      CHECK(r >= 0.9 * inner[h].r && r <= 1.5 * inner[h].r,
        "%s: inner hole %d has r = %.2f, expected within [%.1f, %.1f]", label,
        h, r, 0.9 * inner[h].r, 1.5 * inner[h].r);

      if (cat->footprint) {
        CHECK(cat->footprint[i] > 0.99f && cat->footprint_shell[i] > 0.99f,
          "%s: inner hole %d is fully observed but reads %.3f / %.3f", label, h,
          (double)cat->footprint[i], (double)cat->footprint_shell[i]);
      }
    }

    /* The edge void, if it was kept, is centred on it and reports that part
     * of it was never observed. */
    for (uint64_t i = 0; i < cat->n_voids && cat->footprint; i++) {
      const sif_real f = cat->footprint[i], fs = cat->footprint_shell[i];
      CHECK(f >= 0.0f && f <= 1.0f && fs >= 0.0f && fs <= 1.0f,
        "%s: void %llu has footprint %.3f / %.3f, outside [0, 1]", label,
        (unsigned long long)i, (double)f, (double)fs);
    }

    double d;
    const int64_t e = nearest(cat, &edge, &d);
    if (e >= 0 && d <= 2.0 * CELL && cat->footprint) {
      printf("  edge void: r = %.2f, footprint %.3f, shell %.3f\n",
        (double)cat->radii[e], (double)cat->footprint[e],
        (double)cat->footprint_shell[e]);
      CHECK(cat->footprint[e] < 0.99f && cat->footprint[e] > 0.3f,
        "%s: the edge void reads a footprint of %.3f", label,
        (double)cat->footprint[e]);
      CHECK(cat->footprint_shell[e] < cat->footprint[e],
        "%s: the edge void's shell (%.3f) should be less observed than its "
        "sphere (%.3f)",
        label, (double)cat->footprint_shell[e], (double)cat->footprint[e]);
    } else {
      CHECK(cat->n_voids == N_INNER,
        "%s: an extra void that is not the edge one", label);
    }
  }

  /* Both grids come back as they went in, up to the FFT round trip. */
  double worst_d = 0.0, worst_r = 0.0, scale_d = 0.0, scale_r = 0.0;
  for (uint64_t c = 0; c < cells; c++) {
    worst_d =
      fmax(worst_d, fabs((double)data.grid->values[c] - data_before[c]));
    worst_r =
      fmax(worst_r, fabs((double)rand.grid->values[c] - rand_before[c]));
    scale_d = fmax(scale_d, fabs((double)data_before[c]));
    scale_r = fmax(scale_r, fabs((double)rand_before[c]));
  }
  CHECK(worst_d <= 1e-3 * scale_d && worst_r <= 1e-3 * scale_r,
    "%s: grids not restored (worst drift %.3g of %.3g, %.3g of %.3g)", label,
    worst_d, scale_d, worst_r, scale_r);

  sif_catalog_free(cat);
  free(data_before);
  free(rand_before);
  release(&data);
  release(&rand);
  free(dx);
  free(dy);
  free(dz);
  free(rx);
  free(ry);
  free(rz);
  printf("  ok\n");
}

/*
 * Weights on both sides. Data weighing 2 and randoms weighing 0.5 scale alpha
 * by exactly 4 and every sum the finder compares by a power of two, so the
 * weighted run has to reproduce the unweighted one bit for bit -- footprint
 * included. That is what says the weights reach the grid stage, the
 * normalization and both histograms, and that nothing was left counting.
 */
static void run_weights(void) {
  printf("weights\n");

  sif_real* dx = malloc(N_DATA * sizeof(sif_real));
  sif_real* dy = malloc(N_DATA * sizeof(sif_real));
  sif_real* dz = malloc(N_DATA * sizeof(sif_real));
  sif_real* rx = malloc(N_RANDOMS * sizeof(sif_real));
  sif_real* ry = malloc(N_RANDOMS * sizeof(sif_real));
  sif_real* rz = malloc(N_RANDOMS * sizeof(sif_real));

  rng_state = 0x243F6A8885A308D3ULL;
  draw(dx, dy, dz, N_DATA, 1, 1, 0.0f);
  draw(rx, ry, rz, N_RANDOMS, 1, 0, 0.0f);

  sif_catalog_t* cat[2] = {NULL, NULL};
  for (int weighted = 0; weighted < 2; weighted++) {
    tracers_t data = build_weighted(dx, dy, dz, N_DATA, weighted ? 2.0f : 0.0f);
    tracers_t rand =
      build_weighted(rx, ry, rz, N_RANDOMS, weighted ? 0.5f : 0.0f);
    cat[weighted] = sif_finder_exodus_survey(data.grid, rand.grid, data.mesh,
      rand.mesh, radii, n_radii, -0.7f, 0.0f, SIF_DEFAULT);
    release(&data);
    release(&rand);
  }

  CHECK(cat[0] && cat[1], "a run returned NULL");
  if (cat[0] && cat[1]) {
    CHECK(cat[0]->n_voids == cat[1]->n_voids,
      "weighted run found %llu voids, unweighted %llu",
      (unsigned long long)cat[1]->n_voids, (unsigned long long)cat[0]->n_voids);

    int differ = 0;
    for (uint64_t i = 0; i < cat[0]->n_voids && i < cat[1]->n_voids; i++) {
      differ += cat[0]->cx[i] != cat[1]->cx[i] ||
                cat[0]->cy[i] != cat[1]->cy[i] ||
                cat[0]->cz[i] != cat[1]->cz[i] ||
                cat[0]->radii[i] != cat[1]->radii[i] ||
                cat[0]->footprint[i] != cat[1]->footprint[i] ||
                cat[0]->footprint_shell[i] != cat[1]->footprint_shell[i];
    }
    CHECK(differ == 0,
      "%d voids differ between the weighted and unweighted "
      "runs",
      differ);
  }

  sif_catalog_free(cat[0]);
  sif_catalog_free(cat[1]);
  free(dx);
  free(dy);
  free(dz);
  free(rx);
  free(ry);
  free(rz);
  printf("  ok\n");
}

/*
 * The workflow a caller actually runs: a survey in its own frame, nowhere near
 * a box, placed by sif_finder_exodus_survey_box(), moved in with
 * sif_field_translate(), and its voids moved back with
 * sif_catalog_translate(). They have to come out where the holes are in the
 * caller's frame.
 */
static void run_workflow(void) {
  printf("box helper and translation\n");

  /* A frame far from the origin and from any box, on every axis. */
  const double frame[3] = {-3000.5, 1250.25, 7000.0};

  sif_real* dx = malloc(N_DATA * sizeof(sif_real));
  sif_real* dy = malloc(N_DATA * sizeof(sif_real));
  sif_real* dz = malloc(N_DATA * sizeof(sif_real));
  sif_real* rx = malloc(N_RANDOMS * sizeof(sif_real));
  sif_real* ry = malloc(N_RANDOMS * sizeof(sif_real));
  sif_real* rz = malloc(N_RANDOMS * sizeof(sif_real));

  rng_state = 0x243F6A8885A308D3ULL;
  draw(dx, dy, dz, N_DATA, 0, 1, 0.0f);
  draw(rx, ry, rz, N_RANDOMS, 0, 0, 0.0f);
  for (uint64_t i = 0; i < N_DATA; i++) {
    dx[i] += (sif_real)frame[0];
    dy[i] += (sif_real)frame[1];
    dz[i] += (sif_real)frame[2];
  }
  for (uint64_t i = 0; i < N_RANDOMS; i++) {
    rx[i] += (sif_real)frame[0];
    ry[i] += (sif_real)frame[1];
    rz[i] += (sif_real)frame[2];
  }

  sif_field_t* fd = sif_field_alloc(N_DATA);
  sif_field_assign_positions(fd, dx, dy, dz);
  sif_field_t* fr = sif_field_alloc(N_RANDOMS);
  sif_field_assign_positions(fr, rx, ry, rz);

  sif_real offset[3], box = 0.0f;
  CHECK(sif_finder_exodus_survey_box(
          fr, radii, n_radii, N_GRID, SIF_DEFAULT, offset, &box) == SIF_OK,
    "the box helper failed");

  CHECK(sif_field_translate(fd, offset) == SIF_OK &&
          sif_field_translate(fr, offset) == SIF_OK,
    "translation failed");

  sif_grid_t* gd = sif_grid_alloc(N_GRID, box);
  sif_grid_t* gr = sif_grid_alloc(N_GRID, box);
  CHECK(sif_grid_assign_cic(gd, fd) == SIF_OK, "CIC assignment failed");
  CHECK(sif_grid_assign_cic(gr, fr) == SIF_OK, "CIC assignment failed");

  /* Sized for the density inside the footprint, which is what the box
   * average would badly understate. */
  const double foot_vol = (4.0 / 3.0) * 3.14159265358979 * pow(FOOT_R, 3);
  const double box_vol = (double)box * (double)box * (double)box;
  sif_chain_mesh_t* md = sif_chain_mesh_alloc(
    sif_finder_suggest_mesh_cells(
      (uint64_t)(N_DATA * box_vol / foot_vol), box, radii[0]),
    box, fd, SIF_MESH_DROP_INDICES);
  sif_chain_mesh_t* mr = sif_chain_mesh_alloc(
    sif_finder_suggest_mesh_cells(
      (uint64_t)(N_RANDOMS * box_vol / foot_vol), box, radii[0]),
    box, fr, SIF_MESH_DROP_INDICES);
  sif_field_free(fd);
  sif_field_free(fr);

  sif_catalog_t* cat = sif_finder_exodus_survey(
    gd, gr, md, mr, radii, n_radii, -0.7f, 0.0f, SIF_DEFAULT);
  CHECK(cat != NULL, "the finder refused the box the helper chose");

  if (cat) {
    const sif_real back[3] = {-offset[0], -offset[1], -offset[2]};
    CHECK(
      sif_catalog_translate(cat, back) == SIF_OK, "translating back failed");

    const double cell = (double)box / N_GRID;
    for (int h = 0; h < N_INNER; h++) {
      const hole_t in_frame = {inner[h].x + frame[0], inner[h].y + frame[1],
        inner[h].z + frame[2], inner[h].r};
      double d;
      const int64_t i = nearest(cat, &in_frame, &d);
      CHECK(i >= 0 && d <= 2.0 * cell,
        "inner hole %d not found in the caller's frame (nearest %.1f away)", h,
        d);
      if (i >= 0 && d <= 2.0 * cell)
        CHECK(cat->radii[i] >= 0.9 * inner[h].r &&
                cat->radii[i] <= 1.5 * inner[h].r && cat->footprint[i] > 0.99f,
          "inner hole %d: r = %.2f, footprint %.3f", h, (double)cat->radii[i],
          (double)cat->footprint[i]);
    }
  }

  sif_catalog_free(cat);
  sif_chain_mesh_free(md);
  sif_chain_mesh_free(mr);
  sif_grid_free(gd);
  sif_grid_free(gr);
  free(dx);
  free(dy);
  free(dz);
  free(rx);
  free(ry);
  free(rz);
  printf("  ok\n");
}

static void run_refusals(void) {
  printf("refusals\n");

  const uint64_t n = 20000;
  sif_real* x = malloc(n * sizeof(sif_real));
  sif_real* y = malloc(n * sizeof(sif_real));
  sif_real* z = malloc(n * sizeof(sif_real));
  sif_real* rx = malloc(4 * n * sizeof(sif_real));
  sif_real* ry = malloc(4 * n * sizeof(sif_real));
  sif_real* rz = malloc(4 * n * sizeof(sif_real));

  /* Shifted down by 50: the footprint then reaches to 10 from the faces, far
   * inside the padding the largest search sphere needs. */
  rng_state = 0x9E3779B97F4A7C15ULL;
  draw(x, y, z, n, 0, 0, -50.0f);
  draw(rx, ry, rz, 4 * n, 0, 0, -50.0f);

  tracers_t data = build(x, y, z, n);
  tracers_t rand = build(rx, ry, rz, 4 * n);

  CHECK(sif_finder_exodus_survey(data.grid, rand.grid, data.mesh, rand.mesh,
          radii, n_radii, -0.7f, 0.0f, SIF_DEFAULT) == NULL,
    "a survey without padding should be refused");

  /* A grid already turned into a contrast against the box mean. */
  CHECK(sif_grid_to_density_contrast(data.grid) == SIF_OK,
    "density contrast failed");
  CHECK(sif_finder_exodus_survey(data.grid, rand.grid, data.mesh, rand.mesh,
          radii, n_radii, -0.7f, 0.0f, SIF_DEFAULT) == NULL,
    "a density contrast should be refused as the data grid");

  CHECK(sif_finder_exodus_survey(data.grid, rand.grid, data.mesh, NULL, radii,
          n_radii, -0.7f, 0.0f, SIF_DEFAULT) == NULL,
    "a missing random mesh should be refused");

  release(&data);
  release(&rand);
  free(x);
  free(y);
  free(z);
  free(rx);
  free(ry);
  free(rz);
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

  run_case("uniform selection", 0);
#if !SIF_TEST_INSTRUMENTED
  run_case("radial selection", 1);
  run_weights();
#endif
  run_workflow();
  run_refusals();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");

  sif_finalize();
  return failures != 0;
}
