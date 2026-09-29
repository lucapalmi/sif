/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/model/cosmology.h"

#include "sif/core/macros.h"
#include "sif/utils/logger.h"

#include "cosmology_internal.h"

#include <math.h>
#include <stdlib.h>

#define TAG "cosmology"

/* Width of the pieces the integral is split into. The integrand, 1/E(z), is
 * smooth, and eight Gauss-Legendre points integrate a piece this narrow to
 * about machine precision. */
#define PIECE_DZ 0.05

/* Node spacing of the distance table: Hermite error of order dz^4 D'''', some
 * 1e-9 of the distance at this spacing -- two orders below single precision.
 * Stretched for tables reaching very high redshift, to cap their size. */
#define TABLE_DZ       0.005
#define TABLE_MAX_NODE 1000000u

/* Eight-point Gauss-Legendre rule on [-1, 1]. */
static const double GL_X[8] = {-0.9602898564975363, -0.7966664774136267,
  -0.5255324099163290, -0.1834346424956498, 0.1834346424956498,
  0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
static const double GL_W[8] = {0.1012285362903763, 0.2223810344533745,
  0.3137066458778873, 0.3626837833783620, 0.3626837833783620,
  0.3137066458778873, 0.2223810344533745, 0.1012285362903763};

/* E(z)^2 = H(z)^2 / H0^2. */
static double e2(const sif_cosmology_t* c, double z) {
  const double a1 = 1.0 + z;
  const double a2 = a1 * a1;
  const double omega_k = 1.0 - c->omega_m - c->omega_de - c->omega_r;
  const double de = c->omega_de * pow(a1, 3.0 * (1.0 + c->w0 + c->wa)) *
                    exp(-3.0 * c->wa * z / a1);
  return c->omega_r * a2 * a2 + c->omega_m * a2 * a1 + omega_k * a2 + de;
}

static int check_parameters(const sif_cosmology_t* c) {
  if (!c) {
    SIF_LOG_ERROR(TAG, "no cosmology");
    return SIF_ERR_INVALID;
  }
  if (!isfinite(c->omega_m) || !isfinite(c->omega_de) ||
      !isfinite(c->omega_r) || !isfinite(c->w0) || !isfinite(c->wa)) {
    SIF_LOG_ERROR(TAG, "a cosmological parameter is not a finite number");
    return SIF_ERR_INVALID;
  }
  if (c->omega_m < 0.0 || c->omega_de < 0.0 || c->omega_r < 0.0) {
    SIF_LOG_ERROR(TAG,
      "densities cannot be negative: omega_m = %g, omega_de = %g, "
      "omega_r = %g",
      c->omega_m, c->omega_de, c->omega_r);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/* (c / H0) * integral of dz / E over [a, b], with *bad set if E^2 was not
 * positive at a point it looked at. */
static double integrate(
  const sif_cosmology_t* c, double a, double b, int* bad) {
  const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
  double sum = 0.0;
  for (int i = 0; i < 8; i++) {
    const double v = e2(c, mid + half * GL_X[i]);
    if (!(v > 0.0)) {
      *bad = 1;
      return 0.0;
    }
    sum += GL_W[i] / sqrt(v);
  }
  return SIF_HUBBLE_DISTANCE * half * sum;
}

static int report_no_expansion(double z) {
  SIF_LOG_ERROR(TAG,
    "E(z)^2 is not positive before z = %g: this cosmology has no expansion "
    "history out to there",
    z);
  return SIF_ERR_RANGE;
}

int sif_cosmology_comoving_distance(
  const sif_cosmology_t* cosmo, double z, double* distance) {

  if (!distance) {
    SIF_LOG_ERROR(TAG, "no output for the distance");
    return SIF_ERR_INVALID;
  }
  int status = check_parameters(cosmo);
  if (status != SIF_OK)
    return status;
  if (!(z >= 0.0) || !isfinite(z)) {
    SIF_LOG_ERROR(
      TAG, "the redshift has to be finite and at least 0, not %g", z);
    return SIF_ERR_INVALID;
  }

  const uint64_t pieces = (uint64_t)ceil(z / PIECE_DZ);
  double d = 0.0;
  int bad = 0;
  for (uint64_t k = 0; k < pieces && !bad; k++) {
    const double a = z * (double)k / (double)pieces;
    const double b = z * (double)(k + 1) / (double)pieces;
    d += integrate(cosmo, a, b, &bad);
  }
  if (bad)
    return report_no_expansion(z);

  *distance = d;
  return SIF_OK;
}

int sif__distance_table_build(
  const sif_cosmology_t* cosmo, double z_max, sif__distance_table_t* t) {

  t->distance = t->slope = NULL;
  t->n = 0;

  int status = check_parameters(cosmo);
  if (status != SIF_OK)
    return status;
  if (!(z_max >= 0.0) || !isfinite(z_max)) {
    SIF_LOG_ERROR(
      TAG, "the redshift has to be finite and at least 0, not %g", z_max);
    return SIF_ERR_INVALID;
  }

  double dz = TABLE_DZ;
  if (z_max / dz > TABLE_MAX_NODE - 2)
    dz = z_max / (TABLE_MAX_NODE - 2);
  const uint64_t n = (uint64_t)ceil(z_max / dz) + 2;

  double* distance = malloc(n * sizeof(double));
  double* slope = malloc(n * sizeof(double));
  if (!distance || !slope) {
    free(distance);
    free(slope);
    SIF_LOG_ERROR(TAG, "failed to allocate a distance table of %llu nodes",
      (unsigned long long)n);
    return SIF_ERR_ALLOC;
  }

  /* Each piece integrated on its own, in parallel, then summed in order: the
   * table does not depend on the thread count. */
  int bad = 0;
  distance[0] = 0.0;
#pragma omp parallel for schedule(static) reduction(| : bad)
  for (uint64_t i = 1; i < n; i++)
    distance[i] = integrate(cosmo, (double)(i - 1) * dz, (double)i * dz, &bad);
  for (uint64_t i = 1; i < n && !bad; i++)
    distance[i] += distance[i - 1];

#pragma omp parallel for schedule(static) reduction(| : bad)
  for (uint64_t i = 0; i < n; i++) {
    const double v = e2(cosmo, (double)i * dz);
    if (v > 0.0)
      slope[i] = SIF_HUBBLE_DISTANCE / sqrt(v);
    else
      bad = 1;
  }

  if (bad) {
    free(distance);
    free(slope);
    return report_no_expansion(z_max);
  }

  t->dz = dz;
  t->n = n;
  t->distance = distance;
  t->slope = slope;
  return SIF_OK;
}

double sif__distance_table_eval(const sif__distance_table_t* t, double z) {
  uint64_t i = (uint64_t)(z / t->dz);
  if (i > t->n - 2)
    i = t->n - 2;

  const double s = z / t->dz - (double)i;
  const double s2 = s * s, s3 = s2 * s;
  const double h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
  const double h10 = s3 - 2.0 * s2 + s;
  const double h01 = -2.0 * s3 + 3.0 * s2;
  const double h11 = s3 - s2;

  return h00 * t->distance[i] + h10 * t->dz * t->slope[i] +
         h01 * t->distance[i + 1] + h11 * t->dz * t->slope[i + 1];
}

/* How far out a table is built to find a distance: well past anything a
 * galaxy survey reaches, and short of where the table would stop resolving
 * the integrand. */
#define TABLE_Z_LIMIT 1e4

int sif__distance_table_build_to(
  const sif_cosmology_t* cosmo, double d_max, sif__distance_table_t* t) {

  t->distance = t->slope = NULL;
  t->n = 0;
  if (!(d_max >= 0.0) || !isfinite(d_max)) {
    SIF_LOG_ERROR(
      TAG, "the distance has to be finite and at least 0, not %g", d_max);
    return SIF_ERR_INVALID;
  }

  /* Doubled until the table reaches the distance: a few cheap rebuilds, and
   * no guess at the redshift a distance belongs to. */
  for (double z_max = 1.0; z_max <= TABLE_Z_LIMIT; z_max *= 2.0) {
    const int status = sif__distance_table_build(cosmo, z_max, t);
    if (status != SIF_OK)
      return status;
    if (t->distance[t->n - 1] >= d_max)
      return SIF_OK;
    sif__distance_table_free(t);
  }

  SIF_LOG_ERROR(TAG,
    "a comoving distance of %g Mpc/h is not reached by z = %g in this "
    "cosmology",
    d_max, TABLE_Z_LIMIT);
  return SIF_ERR_RANGE;
}

double sif__distance_table_invert(const sif__distance_table_t* t, double d) {
  if (d <= 0.0)
    return 0.0;

  /* The node interval holding d: the distance only grows with z. */
  uint64_t lo = 0, hi = t->n - 1;
  if (d >= t->distance[hi])
    return (double)hi * t->dz;
  while (hi - lo > 1) {
    const uint64_t mid = lo + (hi - lo) / 2;
    if (t->distance[mid] <= d)
      lo = mid;
    else
      hi = mid;
  }

  /* Newton on the Hermite cubic of that interval, from the linear guess. The
   * cubic is monotonic there -- its end slopes are c / H, positive -- so a
   * few steps reach machine precision. */
  const double d0 = t->distance[lo], d1 = t->distance[lo + 1];
  const double m0 = t->dz * t->slope[lo], m1 = t->dz * t->slope[lo + 1];
  double s = (d - d0) / (d1 - d0);
  for (int it = 0; it < 8; it++) {
    const double s2 = s * s, s3 = s2 * s;
    const double f = (2.0 * s3 - 3.0 * s2 + 1.0) * d0 +
                     (s3 - 2.0 * s2 + s) * m0 + (-2.0 * s3 + 3.0 * s2) * d1 +
                     (s3 - s2) * m1 - d;
    const double df = (6.0 * s2 - 6.0 * s) * d0 +
                      (3.0 * s2 - 4.0 * s + 1.0) * m0 +
                      (-6.0 * s2 + 6.0 * s) * d1 + (3.0 * s2 - 2.0 * s) * m1;
    if (!(df > 0.0))
      break;
    const double step = f / df;
    s -= step;
    if (s < 0.0)
      s = 0.0;
    else if (s > 1.0)
      s = 1.0;
    if (fabs(step) < 1e-15)
      break;
  }
  return ((double)lo + s) * t->dz;
}

void sif__distance_table_free(sif__distance_table_t* t) {
  if (!t)
    return;
  free(t->distance);
  free(t->slope);
  t->distance = t->slope = NULL;
  t->n = 0;
}
