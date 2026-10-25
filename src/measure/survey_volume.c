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
 * gone: the directions are drawn independently of the distances, and all of
 * the randoms sample the mask, at every distance.
 *
 * Omega = sum_i 1 / sigma(theta_i), sigma the randoms' density on the sky,
 * which need not be uniform: randoms drawn with an angular completeness are
 * denser where it is higher. Cell by cell, a cell whose eight neighbours all
 * hold randoms lies wholly inside the mask and counts as one cell; any other
 * counts as n_c / lambda_c, n_c the randoms it holds and lambda_c what a cell
 * wholly inside the mask would hold there, measured over the interior cells
 * nearest it. A cell the edge cuts through then counts by the share of it
 * the randoms reach, so the edges are resolved statistically and not to the
 * nearest cell, and only those cells depend on how the density varies.
 *
 * Telling interior cells reliably needs several randoms in each, which in
 * sparse parts of the sky only larger cells hold. The grid is therefore a
 * quadtree of levels, cells merged 2 x 2 from one to the next, and each part
 * of the sky is measured on the finest level dense enough there.
 *
 * The sky grid is equal-area -- uniform in mu = cos(theta) and in phi --
 * after a rotation that puts the footprint's mean direction on its equator,
 * where the cells are square, and the poles where the footprint extends
 * least; only a footprint reaching 90 degrees from its own centre sees the
 * elongated cells near the poles.
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
 * Randoms a cell of the finest level holds at the mask's mean density. It
 * sets how finely the densest parts of the sky are resolved, and holes with
 * them: a little above LEVEL_RANDOMS, so that a uniform catalogue is
 * measured on the finest level, in cells of about a dozen randoms.
 */
#define RANDOMS_PER_CELL 12.0

/*
 * Randoms a cell must be expected to hold, at its own level, for the sky
 * there to be measured on that level and not a coarser one. A cell in the
 * mask is judged interior with chance (1 - e^-lambda)^8: 0.997 at 8, so the
 * interior cells sample the density without favouring its denser parts.
 */
#define LEVEL_RANDOMS 8.0

/* Interior cells behind each density estimate. Few, so that the estimate is
 * local and does not mix regions of different density; its inverse is formed
 * so as to be unbiased however few randoms it rests on (see cell_share). */
#define SUPPORT_CELLS 4.0

/* Levels of the sky grid, and the fewest cells a side of the coarsest. */
#define MAX_LEVELS     8
#define LEVEL_MIN_SIDE 4

/* Cap on the sky grid, 28 bytes a cell: 112 MiB at the cap, where a full-sky
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

/* The eigenvalues and unit eigenvectors (vec[k] for val[k]) of the
 * symmetric matrix a, by cyclic Jacobi rotations. */
static void sym3_eigen(const double a[3][3], double val[3], double vec[3][3]) {
  double m[3][3], v[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  memcpy(m, a, sizeof(m));

  for (int sweep = 0; sweep < 32; sweep++) {
    const double off =
      m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[1][2] * m[1][2];
    if (!(off > 1e-30))
      break;
    for (int p = 0; p < 2; p++)
      for (int q = p + 1; q < 3; q++) {
        if (m[p][q] == 0.0)
          continue;
        const double theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                         (fabs(theta) + sqrt(theta * theta + 1.0));
        const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < 3; k++) { /* m <- m J, then J^T m */
          const double kp = m[k][p], kq = m[k][q];
          m[k][p] = c * kp - s * kq;
          m[k][q] = s * kp + c * kq;
        }
        for (int k = 0; k < 3; k++) {
          const double pk = m[p][k], qk = m[q][k];
          m[p][k] = c * pk - s * qk;
          m[q][k] = s * pk + c * qk;
        }
        for (int k = 0; k < 3; k++) {
          const double kp = v[k][p], kq = v[k][q];
          v[k][p] = c * kp - s * kq;
          v[k][q] = s * kp + c * kq;
        }
      }
  }

  for (int k = 0; k < 3; k++) {
    val[k] = m[k][k];
    for (int e = 0; e < 3; e++)
      vec[k][e] = v[e][k];
  }
}

/*
 * The sky frame: the footprint's mean direction onto +x, on the grid's
 * equator, and onto the poles the axis perpendicular to it that the
 * footprint extends least along, so that a long footprint lies along the
 * equator rather than across a pole. A footprint with no mean direction --
 * the whole sky, or two opposite caps -- has the axis it extends most along
 * put on +x, and the one it extends least along on the poles.
 *
 * dir is the sum of the unit vectors to the randoms, mom the sum of their
 * outer products, over n of them.
 */
static void sky_rotation(
  const double dir[3], const double mom[3][3], double n, double rot[3][3]) {

  double val[3], vec[3][3], x[3], z[3];
  const double norm =
    sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);

  if (norm > 1e-6 * n) {
    /* mom projected off x, with x itself pushed to the top of the spectrum
     * so that the least eigenvalue is the least perpendicular to it. */
    for (int a = 0; a < 3; a++)
      x[a] = dir[a] / norm;
    double px[3][3], proj[3][3];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
        px[a][b] = (a == b ? 1.0 : 0.0) - x[a] * x[b];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        double t = 0.0;
        for (int e = 0; e < 3; e++)
          for (int f = 0; f < 3; f++)
            t += px[a][e] * mom[e][f] * px[f][b];
        proj[a][b] = t + 2.0 * n * x[a] * x[b];
      }
    sym3_eigen(proj, val, vec);
  } else
    sym3_eigen(mom, val, vec);

  int lo = 0, hi = 0;
  for (int k = 1; k < 3; k++) {
    if (val[k] < val[lo])
      lo = k;
    if (val[k] > val[hi])
      hi = k;
  }
  if (lo == hi) /* all equal: any frame */
    hi = (lo + 1) % 3;
  if (!(norm > 1e-6 * n))
    memcpy(x, vec[hi], sizeof(x));
  memcpy(z, vec[lo], sizeof(z));

  /* z made exactly perpendicular to x, then y = z x x. */
  const double zx = z[0] * x[0] + z[1] * x[1] + z[2] * x[2];
  for (int a = 0; a < 3; a++)
    z[a] -= zx * x[a];
  const double zn = sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
  for (int a = 0; a < 3; a++)
    z[a] /= zn;
  const double y[3] = {z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2],
    z[0] * x[1] - z[1] * x[0]};

  memcpy(rot[0], x, sizeof(x));
  memcpy(rot[1], y, sizeof(y));
  memcpy(rot[2], z, sizeof(z));
}

/* --- the sky grid --- */

typedef struct {
  double mu0, dmu, phi0, dphi;
  uint32_t n_mu, n_phi;
  int periodic;    /* phi goes all the way round */
  uint32_t levels; /* coarser grids, of cells merged 2^l x 2^l, l < levels */
} sky_grid_t;

static uint32_t round_up(uint32_t n, uint32_t b) {
  return (n + b - 1) / b * b;
}

/*
 * A grid of cells about delta on a side over the window the randoms span,
 * with two empty cells of margin so that a cell on the edge of the mask has
 * an empty neighbour. In mu the margin stops at the poles, where the grid is
 * cut to end exactly; in phi the window is kept unless the margin would
 * reach round to the far side, and then the grid wraps. Both counts are
 * rounded up to a multiple of the coarsest level's cell, so that every level
 * tiles the same window.
 */
static sky_grid_t sky_grid_layout(const double window[4], double delta) {
  sky_grid_t g;

  const double mu_lo = fmax(-1.0, window[0] - 2.0 * delta);
  const double mu_hi = fmin(1.0, window[1] + 2.0 * delta);
  g.n_mu = (uint32_t)fmax(3.0, ceil((mu_hi - mu_lo) / delta));

  const double phi_lo = window[2] - 2.0 * delta;
  const double phi_hi = window[3] + 2.0 * delta;
  g.periodic = !(phi_lo > -PI && phi_hi < PI);
  g.n_phi = (uint32_t)fmax(3.0, g.periodic ? round(2.0 * PI / delta)
                                           : ceil((phi_hi - phi_lo) / delta));

  uint32_t top = 0;
  while (top + 1 < MAX_LEVELS && (g.n_mu >> (top + 1)) >= LEVEL_MIN_SIDE &&
         (g.n_phi >> (top + 1)) >= LEVEL_MIN_SIDE)
    top++;
  g.levels = top + 1;
  g.n_mu = round_up(g.n_mu, 1u << top);
  g.n_phi = round_up(g.n_phi, 1u << top);

  g.mu0 = mu_lo;
  g.dmu = (mu_hi - mu_lo) / g.n_mu;
  if (g.periodic) {
    g.phi0 = -PI;
    g.dphi = 2.0 * PI / g.n_phi;
  } else {
    g.phi0 = phi_lo;
    g.dphi = (phi_hi - phi_lo) / g.n_phi;
  }
  return g;
}

/* The grid of level l: the same window in cells 2^l times larger. */
static sky_grid_t sky_grid_level(const sky_grid_t* g, uint32_t l) {
  sky_grid_t c = *g;
  c.n_mu = g->n_mu >> l;
  c.n_phi = g->n_phi >> l;
  c.dmu = g->dmu * (double)(1u << l);
  c.dphi = g->dphi * (double)(1u << l);
  return c;
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
  double dir[3];    /* sum of unit vectors */
  double mom[3][3]; /* sum of their outer products */
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
    scan_block_t s = {{0.0, 0.0, 0.0}, {{0.0}}, INFINITY, 0.0, 0, 0};

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
      for (int a = 0; a < 3; a++) {
        s.dir[a] += d[a] / r;
        for (int b = 0; b < 3; b++)
          s.mom[a][b] += d[a] * d[b] / (r * r);
      }
      s.r_min = fmin(s.r_min, r);
      s.r_max = fmax(s.r_max, r);
    }
    blocks[b] = s;
  }

  scan_block_t t = {{0.0, 0.0, 0.0}, {{0.0}}, INFINITY, 0.0, 0, 0};
  for (int b = 0; b < REDUCE_BLOCKS; b++) {
    for (int a = 0; a < 3; a++) {
      t.dir[a] += blocks[b].dir[a];
      for (int e = 0; e < 3; e++)
        t.mom[a][e] += blocks[b].mom[a][e];
    }
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

/* The randoms each cell of level l holds: its 2^l x 2^l fine cells'. */
static void level_counts(
  const uint32_t* counts, const sky_grid_t* g, uint32_t l, uint32_t* out) {

  const uint32_t b = 1u << l;
  const sky_grid_t c = sky_grid_level(g, l);

#pragma omp parallel for schedule(static)
  for (uint32_t i = 0; i < c.n_mu; i++)
    for (uint32_t j = 0; j < c.n_phi; j++) {
      uint32_t s = 0;
      for (uint32_t a = 0; a < b; a++)
        for (uint32_t e = 0; e < b; e++)
          s += counts[((uint64_t)i * b + a) * g->n_phi + (uint64_t)j * b + e];
      out[(uint64_t)i * c.n_phi + j] = s;
    }
}

/* Marks level l's interior cells, and returns how many of them hold
 * randoms. */
static uint64_t level_interior(
  const uint32_t* counts, const sky_grid_t* c, uint8_t* interior) {

  const uint64_t n_cells = (uint64_t)c->n_mu * c->n_phi;
  uint64_t n_interior = 0;
#pragma omp parallel for schedule(static) reduction(+ : n_interior)
  for (uint64_t q = 0; q < n_cells; q++) {
    interior[q] = (uint8_t)sky_interior(
      counts, c, (uint32_t)(q / c->n_phi), (uint32_t)(q % c->n_phi));
    n_interior += interior[q] && counts[q] > 0;
  }
  return n_interior;
}

/* Cell q's neighbour di, dj in r; 0 if the grid ends first. */
static int neighbour(
  const sky_grid_t* c, uint64_t q, int di, int dj, uint64_t* r) {

  const uint32_t i = (uint32_t)(q / c->n_phi), j = (uint32_t)(q % c->n_phi);
  if ((di < 0 && i == 0) || (di > 0 && i + 1 >= c->n_mu))
    return 0;
  if (!c->periodic && ((dj < 0 && j == 0) || (dj > 0 && j + 1 >= c->n_phi)))
    return 0;
  const uint32_t jj = (uint32_t)(((int64_t)j + c->n_phi + dj) % c->n_phi);
  *r = (uint64_t)(i + di) * c->n_phi + jj;
  return 1;
}

/* The randoms and the count of the occupied interior cells among q and its
 * eight neighbours. */
static void near_interior(const uint32_t* counts, const uint8_t* interior,
  const sky_grid_t* c, uint64_t q, float* num, float* den) {

  *num = *den = 0.0f;
  for (int d = 0; d < 9; d++) {
    uint64_t r;
    if (neighbour(c, q, d / 3 - 1, d % 3 - 1, &r) && interior[r] &&
        counts[r] > 0) {
      *num += (float)counts[r];
      *den += 1.0f;
    }
  }
}

/*
 * Whether cell q, which holds randoms, is handed down to the level below.
 * near_num and near_den hold, for every cell, the randoms and the count of
 * the occupied interior cells in its 3 x 3 block. Around each neighbour of
 * q that density, with q itself left out, must say that a quarter holds
 * LEVEL_RANDOMS: the block's mean dilutes the few cells a hole or an edge
 * covers in part, so that they do not keep q whole, and it takes all of
 * them, so that a cell along a sharp fall in density stays whole and its
 * sparse quarters are never measured alone. q's own count is left out so
 * that what its quarters hold does not decide on which level they are
 * measured.
 */
static int level_dense(const uint32_t* counts, const uint8_t* interior,
  const float* near_num, const float* near_den, const sky_grid_t* c,
  uint64_t q) {

  const double own_num = interior[q] ? (double)counts[q] : 0.0;
  const double own_den = interior[q] ? 1.0 : 0.0;
  int seen = 0;
  for (int d = 0; d < 9; d++) {
    uint64_t r;
    if (d == 4 || !neighbour(c, q, d / 3 - 1, d % 3 - 1, &r))
      continue;
    const double n = (double)near_num[r] - own_num;
    const double m = (double)near_den[r] - own_den;
    if (m <= 0.0)
      continue;
    if (n / m < 4.0 * LEVEL_RANDOMS)
      return 0;
    seen = 1;
  }
  return seen;
}

/*
 * What a measured cell counts for, in cells, from the n randoms it holds
 * and its box: num randoms in den occupied interior cells.
 *
 * An interior cell lies wholly inside the mask and counts as one cell,
 * unless it is empty: the cells measured hold LEVEL_RANDOMS, so an empty one
 * is a hole. A cell inside the mask is occupied with chance 1 - e^-lambda,
 * so an occupied interior cell counts 1 / (1 - e^-lambda), and one counts
 * one on average. Any other cell counts as n / lambda, the share of it the
 * randoms reach. Whether a cell is interior depends on its neighbours alone,
 * so n is the same either way, and a cell inside the mask counts one on
 * average whichever it is judged to be. Only the cells not judged interior
 * -- along the edges, around holes -- depend on lambda beyond that small
 * correction, which keeps the result from depending on how the density
 * varies away from them.
 *
 * The box holds occupied cells only -- an empty interior neighbour is what
 * makes a cell inside the mask not interior, and would otherwise sit in its
 * box -- so num / den estimates m = lambda / (1 - e^-lambda), the mean of a
 * count that is not zero, and lambda is solved for. For the share of a cell
 * not interior, 1 / m is taken as den / (num + 1): for num Poisson of mean
 * M, E[1 / (num + 1)] = (1 - e^-M) / M, unbiased however small the box. The
 * same + 1 keeps m above one, and lambda away from zero, in a sparse box.
 */
static double cell_share(uint32_t n, int interior, float num, float den) {
  if (!(den > 0.0f)) /* no occupied interior cell on the level */
    return interior ? 1.0 : 0.0;

  /* lambda = m (1 - e^-lambda) by Newton's method: the difference is convex
   * and increasing to the right of its root, where lambda = m starts it. */
  const double m = ((double)num + 1.0) / (double)den;
  double lambda = m;
  for (int it = 0; it < 64; it++) {
    const double e = exp(-lambda);
    const double step = (lambda - m * (1.0 - e)) / (1.0 - m * e);
    lambda -= step;
    if (!(step > 1e-12 * lambda))
      break;
  }

  const double occupied = 1.0 - exp(-lambda);
  if (interior)
    return 1.0 / occupied;
  return (double)n * (double)den / ((double)num + 1.0) / occupied;
}

/*
 * The solid angle, over the quadtree of the grid's levels: each part of the
 * sky is measured on the finest level whose cells there hold LEVEL_RANDOMS,
 * so that sparse and dense parts are each judged interior reliably. From
 * the coarsest level down, a cell whose surroundings are dense enough is
 * handed to the level below as its four quarters; any other is measured
 * where it is, by cell_share(), with a box of the interior cells nearest it:
 * the smallest that holds SUPPORT_CELLS of them. The box is chosen by how
 * many interior cells it holds, never by what they count, so the choice
 * cannot favour boxes that happened to count high.
 */
static int solid_angle(
  const uint32_t* counts, const sky_grid_t* g, uint64_t n_dir, double* omega) {

  const uint64_t n_cells = (uint64_t)g->n_mu * g->n_phi;
  uint32_t* merged = malloc((n_cells / 4 + 1) * sizeof(uint32_t));
  uint8_t* interior = malloc(n_cells);
  uint8_t* active = malloc(n_cells);
  uint8_t* split = calloc(n_cells, 1);
  float* num = malloc(n_cells * sizeof(float));
  float* den = malloc(n_cells * sizeof(float));
  float* tmp = malloc(n_cells * sizeof(float));
  float* box_num = malloc(n_cells * sizeof(float));
  float* box_den = malloc(n_cells * sizeof(float));
  int status = SIF_OK;

  if (!merged || !interior || !active || !split || !num || !den || !tmp ||
      !box_num || !box_den) {
    SIF_LOG_ERROR(
      TAG, "out of memory: sky grid of %u x %u cells", g->n_mu, g->n_phi);
    status = SIF_ERR_ALLOC;
    goto out;
  }

  /* Which levels have the interior cells a density estimate rests on: only
   * those can measure the sky, and the coarsest of them is where the
   * quadtree starts. */
  uint64_t n_interior[MAX_LEVELS] = {0};
  uint32_t top = 0;
  for (uint32_t l = 0; l < g->levels; l++) {
    const sky_grid_t c = sky_grid_level(g, l);
    if (l > 0)
      level_counts(counts, g, l, merged);
    n_interior[l] = level_interior(l > 0 ? merged : counts, &c, interior);
    if ((double)n_interior[l] >= SUPPORT_CELLS)
      top = l;
  }

  if ((double)n_interior[top] < SUPPORT_CELLS) {
    const double cell_deg2 = g->dmu * g->dphi * pow(180.0 / PI, 2);
    SIF_LOG_ERROR(TAG,
      "randoms: %" PRIu64 " too few to resolve the mask: fewer than %g sky "
      "cells of %.3g deg^2 or more lie wholly inside it",
      n_dir, SUPPORT_CELLS, cell_deg2);
    status = SIF_ERR_INVALID;
    goto out;
  }

  double total = 0.0;
  for (uint32_t l = top + 1; l-- > 0;) {
    const sky_grid_t c = sky_grid_level(g, l);
    const uint64_t n_l = (uint64_t)c.n_mu * c.n_phi;
    const uint32_t* cl = l > 0 ? merged : counts;
    if (l > 0)
      level_counts(counts, g, l, merged);
    level_interior(cl, &c, interior);

    /* The cells the level above handed down that hold randoms; at the top,
     * all of them. An empty cell counts for nothing wherever it is: the
     * cells measured hold LEVEL_RANDOMS, and an empty one is a hole. */
#pragma omp parallel for schedule(static)
    for (uint64_t q = 0; q < n_l; q++) {
      const uint64_t parent =
        (q / c.n_phi / 2) * (c.n_phi / 2) + (q % c.n_phi) / 2;
      active[q] = (l == top || split[parent]) && cl[q] > 0;
    }

    /* 1. Which of them to hand down in turn. */
    const int may_split =
      l > 0 && (double)n_interior[l - 1] >= SUPPORT_CELLS;
    if (may_split) {
#pragma omp parallel for schedule(static)
      for (uint64_t q = 0; q < n_l; q++)
        near_interior(cl, interior, &c, q, &num[q], &den[q]);
    }
    uint64_t remaining = 0;
#pragma omp parallel for schedule(static) reduction(+ : remaining)
    for (uint64_t q = 0; q < n_l; q++) {
      split[q] = active[q] && may_split &&
                 level_dense(cl, interior, num, den, &c, q);
      box_num[q] = box_den[q] = 0.0f;
      remaining += active[q] && !split[q];
    }

    /* 2. Each measured cell's box. */
    const uint32_t k_cover = c.n_mu > c.n_phi ? c.n_mu : c.n_phi;
    for (uint32_t k = 1; remaining > 0; k *= 2) {
#pragma omp parallel for schedule(static)
      for (uint64_t q = 0; q < n_l; q++) {
        const int in = interior[q] && cl[q] > 0;
        num[q] = in ? (float)cl[q] : 0.0f;
        den[q] = in ? 1.0f : 0.0f;
      }
      box_sum(num, tmp, &c, k);
      box_sum(den, tmp, &c, k);

      const int last = k >= k_cover;
      uint64_t taken = 0;
#pragma omp parallel for schedule(static) reduction(+ : taken)
      for (uint64_t q = 0; q < n_l; q++) {
        if (!active[q] || split[q] || box_den[q] > 0.0f ||
            (double)den[q] < SUPPORT_CELLS)
          continue;
        box_num[q] = num[q];
        box_den[q] = den[q];
        taken++;
      }
      remaining -= taken;

      /* Once the box covers the grid every cell has the whole of the
       * interior behind it, which holds SUPPORT_CELLS on any level used. */
      SIF_ASSERT(remaining == 0 || !last);
      if (last)
        break;
    }

    /* 3. Measure. */
    double partial[REDUCE_BLOCKS];
#pragma omp parallel for schedule(static)
    for (int b = 0; b < REDUCE_BLOCKS; b++) {
      uint64_t lo, hi;
      block_bounds(n_l, b, &lo, &hi);
      double s = 0.0;
      for (uint64_t q = lo; q < hi; q++)
        if (active[q] && !split[q])
          s += cell_share(cl[q], interior[q], box_num[q], box_den[q]);
      partial[b] = s;
    }

    double level_total = 0.0;
    for (int b = 0; b < REDUCE_BLOCKS; b++)
      level_total += partial[b];
    total += level_total * c.dmu * c.dphi;
  }

  *omega = total;

out:
  free(merged);
  free(interior);
  free(active);
  free(split);
  free(num);
  free(den);
  free(tmp);
  free(box_num);
  free(box_den);
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

  /* 1. Distances, and the directions the sky grid is laid out along. */
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

  sky_rotation(scan.dir, scan.mom, (double)n_dir, frame.rot);

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
