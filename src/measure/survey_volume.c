/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The volume of a separable footprint, Omega (r_max^3 - r_min^3) / 3, with
 * the solid angle Omega measured from the randoms' directions.
 *
 * Why the sky and not the volume itself: a fully three-dimensional estimate
 * has to tell, from the randoms alone, a density that is falling (the tail of
 * n(z)) from a footprint that is ending, and where the randoms are sparse it
 * cannot -- measured against shells and wedges with a steep n(z), the best
 * such estimate was biased by several per cent. On the sky that ambiguity is
 * gone: the randoms' angular density is uniform over the mask by
 * construction, and all of them sample it, at every distance.
 *
 * Omega = sum_i 1 / sigma(theta_i), sigma the randoms' density on the sky.
 * Summed cell by cell that is sum_c n_c / lambda_c times the cell's area,
 * with n_c the randoms a cell holds and lambda_c what a cell entirely inside
 * the mask would hold there. A cell the edge cuts through then counts by the
 * share of it the randoms reach, so the edges are resolved statistically and
 * not to the nearest cell. lambda_c is measured only over cells whose eight
 * neighbours all hold randoms: those lie wholly inside the mask, and an edge
 * or a hole cannot dilute the estimate.
 *
 * The sky grid is equal-area -- uniform in mu = cos(theta) and in phi --
 * after a rotation that puts the footprint's mean direction on its equator,
 * where the cells are square; only a footprint reaching 90 degrees from its
 * own centre sees the elongated cells near the poles.
 */

#include "sif/measure/survey_volume.h"

#include "sif/core/macros.h"
#include "sif/structures/bitmask.h"
#include "sif/utils/logger.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TAG "survey_volume"

/* In double: SIF_PI is a sif_real, and the sky grid is laid out in double. */
#define PI 3.14159265358979323846

/*
 * Randoms a sky cell holds at the mask's density. Small, so that holes are
 * resolved down to a few randoms' separation: at 30 a catalogue of 10^4
 * randoms missed 1.5-degree holes outright. It can be this small because the
 * angular density is uniform -- a cell's chance of being judged interior
 * then does not depend on where it is, and the estimate is not biased
 * towards the denser parts of the sky.
 */
#define RANDOMS_PER_CELL 5.0

/* Randoms behind each density estimate: a 2% estimate per cell, and the
 * errors average down further over the many cells of a footprint. */
#define SUPPORT_RANDOMS 2000.0

/* Cap on the sky grid, 17 bytes a cell: 71 MiB at the cap, where a full-sky
 * cell is about 6 arcmin. Only catalogues of tens of millions of randoms over
 * much of the sky reach it, and they then hold more than RANDOMS_PER_CELL. */
#define MAX_CELLS (1u << 22)

/* Fixed blocks for the reductions, so no result depends on the thread
 * count. */
#define REDUCE_BLOCKS 64

/* --- the separability check --- */

/* Distance bins for the randoms' radial distribution. */
#define RADIAL_BINS 4096

/* Randoms a sky patch aims to hold, and a cap on the patches. */
#define PATCH_RANDOMS 2000.0
#define MAX_PATCHES   4096

/* ln of the chance below which a patch is judged to stop short of the
 * catalogue's range: 1e-6, so that a few thousand patches raise no false
 * alarm. */
#define SHORT_LOG_CHANCE (-13.8)

static void block_bounds(uint64_t n, int b, uint64_t* lo, uint64_t* hi) {
  const uint64_t chunk = n / REDUCE_BLOCKS;
  const uint64_t rem = n % REDUCE_BLOCKS;
  const uint64_t ub = (uint64_t)b;
  *lo = ub * chunk + (ub < rem ? ub : rem);
  *hi = *lo + chunk + (ub < rem ? 1 : 0);
}

/*
 * By its bits rather than isfinite(): the release build's -ffast-math lets
 * the compiler assume no NaN or infinity exists and fold the test away.
 */
static int real_is_finite(sif_real v) {
  if (sizeof(sif_real) == sizeof(uint32_t)) {
    uint32_t b;
    memcpy(&b, &v, sizeof(b));
    return ((b >> 23) & 0xFFu) != 0xFFu;
  }
  uint64_t b;
  memcpy(&b, &v, sizeof(b));
  return ((b >> 52) & 0x7FFu) != 0x7FFu;
}

/* --- directions --- */

typedef struct {
  double rot[3][3]; /* takes the footprint's mean direction onto +x */
  double obs[3];
} sky_frame_t;

/*
 * The distance of random i from the observer, and its direction as mu and
 * phi in the rotated frame. 0 for a random at the observer, which has none.
 */
static double sky_coords(const sky_frame_t* f, const sif_field_t* p, uint64_t i,
  double* mu, double* phi) {

  const double d[3] = {(double)p->x[i] - f->obs[0], (double)p->y[i] - f->obs[1],
    (double)p->z[i] - f->obs[2]};
  const double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  if (!(r > 0.0))
    return 0.0;

  double v[3];
  for (int a = 0; a < 3; a++)
    v[a] =
      (f->rot[a][0] * d[0] + f->rot[a][1] * d[1] + f->rot[a][2] * d[2]) / r;
  *mu = v[2] < -1.0 ? -1.0 : v[2] > 1.0 ? 1.0 : v[2];
  *phi = atan2(v[1], v[0]);
  return r;
}

/* Rodrigues' rotation taking the unit vector m onto +x. */
static void rotation_onto_x(const double m[3], double rot[3][3]) {
  /* v = m x e_x, sin and cos of the angle between them */
  const double v[3] = {0.0, m[2], -m[1]};
  const double s2 = v[1] * v[1] + v[2] * v[2];
  const double c = m[0];
  const double k[3][3] = {
    {0.0, -v[2], v[1]}, {v[2], 0.0, -v[0]}, {-v[1], v[0], 0.0}};

  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++)
      rot[a][b] = a == b ? 1.0 : 0.0;

  if (s2 < 1e-24) {
    if (c < 0.0) /* opposite: half a turn about z */
      rot[0][0] = rot[1][1] = -1.0;
    return;
  }

  const double w = (1.0 - c) / s2;
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++) {
      double kk = 0.0;
      for (int e = 0; e < 3; e++)
        kk += k[a][e] * k[e][b];
      rot[a][b] += k[a][b] + kk * w;
    }
}

/* --- the sky grid --- */

typedef struct {
  double mu0, dmu, phi0, dphi;
  uint32_t n_mu, n_phi;
  int periodic; /* phi goes all the way round */
} sky_grid_t;

/*
 * A grid of cells about delta on a side over the window the randoms span,
 * with two empty cells of margin so that a cell on the edge of the mask has
 * an empty neighbour. In mu the margin stops at the poles, where the grid is
 * cut to end exactly; in phi the window is kept unless the margin would
 * reach round to the far side, and then the grid wraps.
 */
static sky_grid_t sky_grid_layout(const double window[4], double delta) {
  sky_grid_t g;

  const double mu_lo = fmax(-1.0, window[0] - 2.0 * delta);
  const double mu_hi = fmin(1.0, window[1] + 2.0 * delta);
  g.n_mu = (uint32_t)fmax(3.0, ceil((mu_hi - mu_lo) / delta));
  g.mu0 = mu_lo;
  g.dmu = (mu_hi - mu_lo) / g.n_mu;

  const double phi_lo = window[2] - 2.0 * delta;
  const double phi_hi = window[3] + 2.0 * delta;
  g.periodic = !(phi_lo > -PI && phi_hi < PI);
  if (g.periodic) {
    g.n_phi = (uint32_t)fmax(3.0, round(2.0 * PI / delta));
    g.phi0 = -PI;
    g.dphi = 2.0 * PI / g.n_phi;
  } else {
    g.n_phi = (uint32_t)fmax(3.0, ceil((phi_hi - phi_lo) / delta));
    g.phi0 = phi_lo;
    g.dphi = (phi_hi - phi_lo) / g.n_phi;
  }
  return g;
}

/* The same, coarsened until it fits under MAX_CELLS. */
static sky_grid_t sky_grid_capped(const double window[4], double delta) {
  sky_grid_t g = sky_grid_layout(window, delta);
  while ((uint64_t)g.n_mu * g.n_phi > MAX_CELLS) {
    delta *= 1.01 * sqrt((double)g.n_mu * g.n_phi / MAX_CELLS);
    g = sky_grid_layout(window, delta);
  }
  return g;
}

static void sky_cell(
  const sky_grid_t* g, double mu, double phi, uint32_t* i, uint32_t* j) {

  double a = floor((mu - g->mu0) / g->dmu);
  *i = a < 0.0 ? 0 : a >= g->n_mu ? g->n_mu - 1 : (uint32_t)a;

  double b = floor((phi - g->phi0) / g->dphi);
  if (g->periodic) {
    b = fmod(b, (double)g->n_phi);
    if (b < 0.0)
      b += g->n_phi;
  }
  *j = b < 0.0 ? 0 : b >= g->n_phi ? g->n_phi - 1 : (uint32_t)b;
}

/*
 * Each cell's sum over the (2k + 1)^2 cells around it, in place: along mu,
 * which ends at the grid's edges, then along phi, which wraps when the grid
 * goes all the way round. The values are counts, so the running sums in
 * double are exact; a float holds the result to far better than it is
 * known.
 */
static void box_sum(float* a, float* tmp, const sky_grid_t* g, uint32_t k) {
  const uint32_t nm = g->n_mu, np = g->n_phi;

#pragma omp parallel for schedule(static)
  for (uint32_t j = 0; j < np; j++) {
    double s = 0.0;
    for (uint32_t i = 0; i < k && i < nm; i++)
      s += a[(uint64_t)i * np + j];
    for (uint32_t i = 0; i < nm; i++) {
      if ((uint64_t)i + k < nm)
        s += a[((uint64_t)i + k) * np + j];
      if (i >= k + 1)
        s -= a[((uint64_t)i - k - 1) * np + j];
      tmp[(uint64_t)i * np + j] = (float)s;
    }
  }

#pragma omp parallel for schedule(static)
  for (uint32_t i = 0; i < nm; i++) {
    const float* x = tmp + (uint64_t)i * np;
    float* out = a + (uint64_t)i * np;

    if (g->periodic && 2 * (uint64_t)k + 1 >= np) {
      double s = 0.0;
      for (uint32_t j = 0; j < np; j++)
        s += x[j];
      for (uint32_t j = 0; j < np; j++)
        out[j] = (float)s;
    } else if (g->periodic) {
      double s = 0.0;
      for (int64_t d = -(int64_t)k; d <= (int64_t)k; d++)
        s += x[(d + np) % np];
      for (uint32_t j = 0; j < np; j++) {
        out[j] = (float)s;
        s += x[(j + k + 1) % np];
        s -= x[(j + np - k) % np];
      }
    } else {
      double s = 0.0;
      for (uint32_t j = 0; j < k && j < np; j++)
        s += x[j];
      for (uint32_t j = 0; j < np; j++) {
        if ((uint64_t)j + k < np)
          s += x[j + k];
        if (j >= k + 1)
          s -= x[j - k - 1];
        out[j] = (float)s;
      }
    }
  }
}

/* Whether a cell's eight neighbours all hold randoms; its own count is left
 * out, so that the cells chosen are not chosen for holding more. */
static int sky_interior(
  const uint32_t* counts, const sky_grid_t* g, uint32_t i, uint32_t j) {

  if (i == 0 || i + 1 >= g->n_mu)
    return 0;
  if (!g->periodic && (j == 0 || j + 1 >= g->n_phi))
    return 0;

  for (int di = -1; di <= 1; di++)
    for (int dj = -1; dj <= 1; dj++) {
      if (di == 0 && dj == 0)
        continue;
      const uint32_t jj = (uint32_t)(((int64_t)j + g->n_phi + dj) % g->n_phi);
      if (counts[(uint64_t)(i + di) * g->n_phi + jj] == 0)
        return 0;
    }
  return 1;
}

/* --- the scan --- */

typedef struct {
  double dir[3]; /* sum of unit vectors */
  double r_min, r_max;
  uint64_t n_bad, n_at_observer;
} scan_block_t;

static void scan_randoms(
  const sif_field_t* p, const double obs[3], scan_block_t* out) {

  scan_block_t blocks[REDUCE_BLOCKS];

#pragma omp parallel for schedule(static)
  for (int b = 0; b < REDUCE_BLOCKS; b++) {
    uint64_t lo, hi;
    block_bounds(p->n_particles, b, &lo, &hi);
    scan_block_t s = {{0.0, 0.0, 0.0}, INFINITY, 0.0, 0, 0};

    for (uint64_t i = lo; i < hi; i++) {
      if (!real_is_finite(p->x[i]) || !real_is_finite(p->y[i]) ||
          !real_is_finite(p->z[i])) {
        s.n_bad++;
        continue;
      }
      const double d[3] = {(double)p->x[i] - obs[0], (double)p->y[i] - obs[1],
        (double)p->z[i] - obs[2]};
      const double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (!(r > 0.0)) {
        s.n_at_observer++;
        continue;
      }
      for (int a = 0; a < 3; a++)
        s.dir[a] += d[a] / r;
      s.r_min = fmin(s.r_min, r);
      s.r_max = fmax(s.r_max, r);
    }
    blocks[b] = s;
  }

  scan_block_t t = {{0.0, 0.0, 0.0}, INFINITY, 0.0, 0, 0};
  for (int b = 0; b < REDUCE_BLOCKS; b++) {
    for (int a = 0; a < 3; a++)
      t.dir[a] += blocks[b].dir[a];
    t.r_min = fmin(t.r_min, blocks[b].r_min);
    t.r_max = fmax(t.r_max, blocks[b].r_max);
    t.n_bad += blocks[b].n_bad;
    t.n_at_observer += blocks[b].n_at_observer;
  }
  *out = t;
}

/* The smallest mu-phi box holding every direction, in the rotated frame. */
static void sky_window(
  const sif_field_t* p, const sky_frame_t* f, double window[4]) {

  double w[REDUCE_BLOCKS][4];

#pragma omp parallel for schedule(static)
  for (int b = 0; b < REDUCE_BLOCKS; b++) {
    uint64_t lo, hi;
    block_bounds(p->n_particles, b, &lo, &hi);
    double s[4] = {INFINITY, -INFINITY, INFINITY, -INFINITY};
    for (uint64_t i = lo; i < hi; i++) {
      double mu, phi;
      if (sky_coords(f, p, i, &mu, &phi) > 0.0) {
        s[0] = fmin(s[0], mu);
        s[1] = fmax(s[1], mu);
        s[2] = fmin(s[2], phi);
        s[3] = fmax(s[3], phi);
      }
    }
    memcpy(w[b], s, sizeof(s));
  }

  window[0] = window[2] = INFINITY;
  window[1] = window[3] = -INFINITY;
  for (int b = 0; b < REDUCE_BLOCKS; b++) {
    window[0] = fmin(window[0], w[b][0]);
    window[1] = fmax(window[1], w[b][1]);
    window[2] = fmin(window[2], w[b][2]);
    window[3] = fmax(window[3], w[b][3]);
  }
}

/* Sky cells any random reaches, on a grid built for counting them. */
static int sky_occupied(const sif_field_t* p, const sky_frame_t* f,
  const sky_grid_t* g, uint64_t* n_occupied) {

  sif_bitmask_t* occ = sif_bitmask_alloc((uint64_t)g->n_mu * g->n_phi);
  if (!occ) {
    SIF_LOG_ERROR(
      TAG, "out of memory: sky grid of %u x %u cells", g->n_mu, g->n_phi);
    return SIF_ERR_ALLOC;
  }

#pragma omp parallel for schedule(static)
  for (uint64_t n = 0; n < p->n_particles; n++) {
    double mu, phi;
    uint32_t i, j;
    if (sky_coords(f, p, n, &mu, &phi) > 0.0) {
      sky_cell(g, mu, phi, &i, &j);
      sif_bitmask_set_atomic(occ, (uint64_t)i * g->n_phi + j);
    }
  }

  *n_occupied = sif_bitmask_count_set(occ);
  sif_bitmask_free(occ);
  return SIF_OK;
}

/* --- separability --- */

typedef struct {
  uint32_t n;
  float r_min, r_max;
} patch_t;

typedef struct {
  uint32_t side; /* cells per patch side */
  uint32_t n_mu, n_phi;
  patch_t* patches; /* [REDUCE_BLOCKS][n_mu * n_phi] */
  uint32_t* radial; /* [REDUCE_BLOCKS][RADIAL_BINS] */
} patch_grid_t;

static uint32_t radial_bin(double r, double r_min, double r_max) {
  const double t = (r - r_min) / (r_max - r_min) * RADIAL_BINS;
  return t <= 0.0 ? 0 : t >= RADIAL_BINS ? RADIAL_BINS - 1 : (uint32_t)t;
}

/*
 * Whether some part of the sky stops short of the catalogue's radial range.
 *
 * Under separability a patch's randoms are drawn from the catalogue's own
 * radial distribution F, so the chance that all n of them fall at or below
 * the patch's furthest is F(r_far)^n -- tiny if the patch ends well short of
 * where the rest of the sky goes on. That needs no model of n(z): F is the
 * catalogue's own, and a sparse tail simply makes the test forgiving there.
 * The same on the near side. F is read at the upper edge of a bin, which can
 * only make the test more forgiving.
 */
static void check_separable(const patch_grid_t* pg, const sky_grid_t* g,
  const sky_frame_t* f, uint64_t n_dir, double r_min, double r_max) {

  uint64_t cum[RADIAL_BINS];
  uint64_t acc = 0;
  for (uint32_t k = 0; k < RADIAL_BINS; k++) {
    for (int b = 0; b < REDUCE_BLOCKS; b++)
      acc += pg->radial[(uint64_t)b * RADIAL_BINS + k];
    cum[k] = acc;
  }

  const uint64_t n_patches = (uint64_t)pg->n_mu * pg->n_phi;
  uint32_t n_seen = 0, n_far = 0, n_near = 0;
  double worst_far = 0.0, worst_near = 0.0;
  uint64_t at_far = 0, at_near = 0;
  float r_far = 0.0f, r_near = 0.0f;

  for (uint64_t q = 0; q < n_patches; q++) {
    patch_t s = {0, INFINITY, -INFINITY};
    for (int b = 0; b < REDUCE_BLOCKS; b++) {
      const patch_t* t = &pg->patches[(uint64_t)b * n_patches + q];
      s.n += t->n;
      s.r_min = fminf(s.r_min, t->r_min);
      s.r_max = fmaxf(s.r_max, t->r_max);
    }
    if (s.n == 0)
      continue;
    n_seen++;

    const uint32_t hi = radial_bin(s.r_max, r_min, r_max);
    const double below = (double)cum[hi] / (double)n_dir;
    const double log_far = below < 1.0 ? s.n * log(below) : 0.0;
    if (log_far < SHORT_LOG_CHANCE) {
      n_far++;
      if (log_far < worst_far) {
        worst_far = log_far;
        at_far = q;
        r_far = s.r_max;
      }
    }

    const uint32_t lo = radial_bin(s.r_min, r_min, r_max);
    const double above =
      1.0 - (lo > 0 ? (double)cum[lo - 1] / (double)n_dir : 0.0);
    const double log_near = above < 1.0 ? s.n * log(above) : 0.0;
    if (log_near < SHORT_LOG_CHANCE) {
      n_near++;
      if (log_near < worst_near) {
        worst_near = log_near;
        at_near = q;
        r_near = s.r_min;
      }
    }
  }

  /* A patch's centre, back in the caller's frame, to say where it is. */
  const struct {
    uint32_t n;
    uint64_t at;
    const char* what;
    double expected;
    float found;
  } short_of[2] = {{n_far, at_far, "end short of r_max", r_max, r_far},
    {n_near, at_near, "start past r_min", r_min, r_near}};

  for (int s = 0; s < 2; s++) {
    if (short_of[s].n == 0)
      continue;
    const uint64_t q = short_of[s].at;
    const double mu =
      g->mu0 + ((q / pg->n_phi) * pg->side + 0.5 * pg->side) * g->dmu;
    const double phi =
      g->phi0 + ((q % pg->n_phi) * pg->side + 0.5 * pg->side) * g->dphi;
    const double sin_t = sqrt(fmax(0.0, 1.0 - mu * mu));
    const double v[3] = {sin_t * cos(phi), sin_t * sin(phi), mu};
    double u[3];
    for (int a = 0; a < 3; a++) /* rot is orthogonal: its inverse is rot^T */
      u[a] = f->rot[0][a] * v[0] + f->rot[1][a] * v[1] + f->rot[2][a] * v[2];

    SIF_LOG_WARNING(TAG,
      "randoms: radial range differs across the sky: %u of %u patches %s "
      "(expected %g, found %g towards (%.2f, %.2f, %.2f)); the volume "
      "assumes one range everywhere",
      short_of[s].n, n_seen, short_of[s].what, short_of[s].expected,
      (double)short_of[s].found, u[0], u[1], u[2]);
  }
}

/* --- counting --- */

/*
 * Every random into its sky cell, and, for the separability check, into its
 * patch and its distance bin. The cells are shared and counted atomically;
 * the patches and bins are per block, merged afterwards in block order.
 */
static void count_randoms(const sif_field_t* p, const sky_frame_t* f,
  const sky_grid_t* g, uint32_t* counts, patch_grid_t* pg, double r_min,
  double r_max) {

  const uint64_t n_patches = (uint64_t)pg->n_mu * pg->n_phi;

#pragma omp parallel for schedule(static)
  for (int b = 0; b < REDUCE_BLOCKS; b++) {
    uint64_t lo, hi;
    block_bounds(p->n_particles, b, &lo, &hi);
    patch_t* patches = pg->patches + (uint64_t)b * n_patches;
    uint32_t* radial = pg->radial + (uint64_t)b * RADIAL_BINS;

    for (uint64_t q = 0; q < n_patches; q++)
      patches[q] = (patch_t){0, INFINITY, -INFINITY};

    for (uint64_t n = lo; n < hi; n++) {
      double mu, phi;
      uint32_t i, j;
      const double r = sky_coords(f, p, n, &mu, &phi);
      if (!(r > 0.0))
        continue;
      sky_cell(g, mu, phi, &i, &j);

#pragma omp atomic
      counts[(uint64_t)i * g->n_phi + j]++;

      patch_t* t =
        &patches[(uint64_t)(i / pg->side) * pg->n_phi + j / pg->side];
      t->n++;
      t->r_min = fminf(t->r_min, (float)r);
      t->r_max = fmaxf(t->r_max, (float)r);
      radial[radial_bin(r, r_min, r_max)]++;
    }
  }
}

/* --- the solid angle --- */

/*
 * sum_c n_c / lambda_c times the cell area, lambda_c from the interior cells
 * in the smallest box around c that holds about SUPPORT_RANDOMS of them. The
 * box is chosen by how many interior cells it holds, never by what they
 * count, so the choice cannot favour boxes that happened to count high.
 */
static int solid_angle(
  const uint32_t* counts, const sky_grid_t* g, uint64_t n_dir, double* omega) {

  const uint64_t n_cells = (uint64_t)g->n_mu * g->n_phi;
  uint8_t* interior = malloc(n_cells);
  uint8_t* done = calloc(n_cells, 1);
  float* num = malloc(n_cells * sizeof(float));
  float* den = malloc(n_cells * sizeof(float));
  float* tmp = malloc(n_cells * sizeof(float));
  int status = SIF_OK;

  if (!interior || !done || !num || !den || !tmp) {
    SIF_LOG_ERROR(
      TAG, "out of memory: sky grid of %u x %u cells", g->n_mu, g->n_phi);
    status = SIF_ERR_ALLOC;
    goto out;
  }

  uint64_t n_interior = 0, in_interior = 0, n_occupied = 0;
#pragma omp parallel for schedule(static)                                      \
  reduction(+ : n_interior, in_interior, n_occupied)
  for (uint64_t c = 0; c < n_cells; c++) {
    interior[c] = (uint8_t)sky_interior(
      counts, g, (uint32_t)(c / g->n_phi), (uint32_t)(c % g->n_phi));
    n_interior += interior[c];
    in_interior += interior[c] ? counts[c] : 0;
    n_occupied += counts[c] > 0;
  }

  if (n_interior == 0 || in_interior == 0) {
    const double cell_deg2 = g->dmu * g->dphi * pow(180.0 / PI, 2);
    SIF_LOG_ERROR(TAG,
      "randoms: %" PRIu64 " too few to resolve the mask: no sky cell of "
      "%.3g deg^2 lies wholly inside it",
      n_dir, cell_deg2);
    status = SIF_ERR_INVALID;
    goto out;
  }

  const double lambda_mean = (double)in_interior / (double)n_interior;
  const double support = fmin(SUPPORT_RANDOMS, 0.5 * (double)in_interior);
  const uint32_t k_cover = g->n_mu > g->n_phi ? g->n_mu : g->n_phi;

  double total = 0.0;
  uint64_t remaining = n_occupied;
  for (uint32_t k = 1; remaining > 0; k *= 2) {
#pragma omp parallel for schedule(static)
    for (uint64_t c = 0; c < n_cells; c++) {
      num[c] = interior[c] ? (float)counts[c] : 0.0f;
      den[c] = interior[c] ? 1.0f : 0.0f;
    }
    box_sum(num, tmp, g, k);
    box_sum(den, tmp, g, k);

    double partial[REDUCE_BLOCKS];
    uint64_t taken = 0;
#pragma omp parallel for schedule(static) reduction(+ : taken)
    for (int b = 0; b < REDUCE_BLOCKS; b++) {
      uint64_t lo, hi;
      block_bounds(n_cells, b, &lo, &hi);
      double s = 0.0;
      for (uint64_t c = lo; c < hi; c++) {
        if (counts[c] == 0 || done[c])
          continue;
        if ((double)den[c] * lambda_mean < support || !(num[c] > 0.0f))
          continue;
        s += (double)counts[c] * (double)den[c] / (double)num[c];
        done[c] = 1;
        taken++;
      }
      partial[b] = s;
    }
    for (int b = 0; b < REDUCE_BLOCKS; b++)
      total += partial[b];
    remaining -= taken;

    /* Once the box covers the grid every cell has the whole of the interior
     * behind it, which holds at least `support` by construction. */
    SIF_ASSERT(remaining == 0 || k < k_cover);
    if (k >= k_cover)
      break;
  }

  *omega = total * g->dmu * g->dphi;

out:
  free(interior);
  free(done);
  free(num);
  free(den);
  free(tmp);
  return status;
}

/* --- entry point --- */

int sif_survey_volume(
  const sif_field_t* randoms, const sif_real observer[3], sif_real* volume) {

  if (!randoms || !volume) {
    SIF_LOG_ERROR(TAG, "%s: NULL", !randoms ? "randoms" : "volume");
    return SIF_ERR_INVALID;
  }
  if (randoms->n_particles == 0 || !randoms->x || !randoms->y || !randoms->z) {
    SIF_LOG_ERROR(TAG, "randoms: no positions");
    return SIF_ERR_INVALID;
  }
  if (randoms->units == SIF_COORDINATES_SKY) {
    SIF_LOG_ERROR(TAG, "randoms: sky coordinates, expected Cartesian (convert "
                       "them with sif_field_convert_sky_coordinates())");
    return SIF_ERR_INVALID;
  }

  sky_frame_t frame = {
    {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {0.0, 0.0, 0.0}};
  if (observer) {
    for (int a = 0; a < 3; a++) {
      if (!real_is_finite(observer[a])) {
        SIF_LOG_ERROR(
          TAG, "observer[%d]: %g, expected finite", a, (double)observer[a]);
        return SIF_ERR_INVALID;
      }
      frame.obs[a] = (double)observer[a];
    }
  }

  /* 1. Distances, and the mean direction the sky grid is centred on. */
  scan_block_t scan;
  scan_randoms(randoms, frame.obs, &scan);
  if (scan.n_bad > 0) {
    SIF_LOG_ERROR(TAG,
      "randoms: %" PRIu64 " of %" PRIu64 " with a position that is not finite",
      scan.n_bad, randoms->n_particles);
    return SIF_ERR_INVALID;
  }
  const uint64_t n_dir = randoms->n_particles - scan.n_at_observer;
  if (n_dir == 0) {
    SIF_LOG_ERROR(
      TAG, "randoms: all %" PRIu64 " at the observer", randoms->n_particles);
    return SIF_ERR_INVALID;
  }
  if (scan.n_at_observer > 0)
    SIF_LOG_WARNING(TAG,
      "randoms: %" PRIu64 " at the observer, with no direction: left out of "
      "the solid angle",
      scan.n_at_observer);

  /* A footprint that covers the sky evenly has no mean direction, and needs
   * none: any orientation serves. */
  const double norm =
    sqrt(scan.dir[0] * scan.dir[0] + scan.dir[1] * scan.dir[1] +
         scan.dir[2] * scan.dir[2]);
  if (norm > 1e-6 * (double)n_dir) {
    const double m[3] = {
      scan.dir[0] / norm, scan.dir[1] / norm, scan.dir[2] / norm};
    rotation_onto_x(m, frame.rot);
  }

  /* 2. The grid. Sized first as if the randoms filled their whole window, so
   * that every cell holds at least RANDOMS_PER_CELL and the share the mask
   * covers is counted reliably; then refined by that share. */
  double window[4];
  sky_window(randoms, &frame, window);
  const double window_area = (window[1] - window[0]) * (window[3] - window[2]);
  if (!(window_area > 0.0)) {
    SIF_LOG_ERROR(TAG, "randoms: all in one direction, spanning no solid "
                       "angle");
    return SIF_ERR_INVALID;
  }

  const sky_grid_t coarse = sky_grid_capped(
    window, sqrt(window_area * RANDOMS_PER_CELL / (double)n_dir));
  uint64_t n_occupied;
  int status = sky_occupied(randoms, &frame, &coarse, &n_occupied);
  if (status != SIF_OK)
    return status;
  const double mask_area = (double)n_occupied * coarse.dmu * coarse.dphi;
  const sky_grid_t g =
    sky_grid_capped(window, sqrt(mask_area * RANDOMS_PER_CELL / (double)n_dir));

  /* Patches of about PATCH_RANDOMS each, at the density the grid ended up
   * with -- above RANDOMS_PER_CELL where the cap coarsened it. */
  patch_grid_t pg;
  const double per_cell = (double)n_dir * g.dmu * g.dphi / mask_area;
  pg.side = (uint32_t)fmax(1.0, round(sqrt(PATCH_RANDOMS / per_cell)));
  for (;;) {
    pg.n_mu = (g.n_mu + pg.side - 1) / pg.side;
    pg.n_phi = (g.n_phi + pg.side - 1) / pg.side;
    if ((uint64_t)pg.n_mu * pg.n_phi <= MAX_PATCHES)
      break;
    pg.side++;
  }

  const uint64_t n_cells = (uint64_t)g.n_mu * g.n_phi;
  uint32_t* counts = calloc(n_cells, sizeof(uint32_t));
  pg.patches =
    malloc((uint64_t)REDUCE_BLOCKS * pg.n_mu * pg.n_phi * sizeof(patch_t));
  pg.radial = calloc((uint64_t)REDUCE_BLOCKS * RADIAL_BINS, sizeof(uint32_t));
  if (!counts || !pg.patches || !pg.radial) {
    SIF_LOG_ERROR(
      TAG, "out of memory: sky grid of %u x %u cells", g.n_mu, g.n_phi);
    status = SIF_ERR_ALLOC;
    goto out;
  }

  /* 3. Count, measure the solid angle, check the range is one range. */
  count_randoms(randoms, &frame, &g, counts, &pg, scan.r_min, scan.r_max);

  double omega;
  status = solid_angle(counts, &g, n_dir, &omega);
  if (status != SIF_OK)
    goto out;

  if (scan.r_max > scan.r_min)
    check_separable(&pg, &g, &frame, n_dir, scan.r_min, scan.r_max);

  const double v = omega *
                   (scan.r_max * scan.r_max * scan.r_max -
                     scan.r_min * scan.r_min * scan.r_min) /
                   3.0;
  SIF_LOG_INFO(TAG,
    "randoms: %" PRIu64 " over %.6g sr (%.1f deg^2), r in [%g, %g]: "
    "volume %.6g",
    n_dir, omega, omega * pow(180.0 / PI, 2), scan.r_min, scan.r_max, v);
  *volume = (sif_real)v;

out:
  free(counts);
  free(pg.patches);
  free(pg.radial);
  return status;
}

#undef PI
