/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Comoving distances, and the conversion of sky coordinates that uses them.
 *
 * The distance is checked against the three universes where it has a closed
 * form, and a w0waCDM one against a brute-force integral written here; the
 * conversion against directions whose positions are known exactly; and every
 * function that reads positions as lengths against a field still holding sky
 * coordinates, which it has to refuse. The way back -- void centres to the
 * sky -- is checked as the inverse of the way in.
 */

#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/io/catalog_io.h"
#include "sif/io/field_io.h"
#include "sif/measure/profiles.h"
#include "sif/model/cosmology.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/structures/octree.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define DH SIF_HUBBLE_DISTANCE

static double rel(double a, double b) {
  return fabs(a - b) / fmax(fabs(b), 1e-300);
}

/* --- distances ---------------------------------------------------------- */

static void check_closed_form(
  const char* label, const sif_cosmology_t* c, double (*exact)(double)) {

  static const double zs[] = {0.0, 0.01, 0.3, 1.0, 2.5, 7.0};
  for (size_t i = 0; i < sizeof(zs) / sizeof(zs[0]); i++) {
    double d = -1.0;
    const int s = sif_cosmology_comoving_distance(c, zs[i], &d);
    const double want = exact(zs[i]);
    CHECK(s == SIF_OK && (want == 0.0 ? d == 0.0 : rel(d, want) < 1e-12),
      "%s: D_C(%g) = %.15g, expected %.15g", label, zs[i], d, want);
  }
}

static double eds(double z) { return 2.0 * DH * (1.0 - 1.0 / sqrt(1.0 + z)); }
static double milne(double z) { return DH * log1p(z); }
static double de_sitter(double z) { return DH * z; }

/* The integral by brute force: Simpson with a million intervals. */
static double brute(const sif_cosmology_t* c, double z) {
  const int n = 1000000;
  const double h = z / n;
  double sum = 0.0;
  for (int i = 0; i <= n; i++) {
    const double x = i * h, a1 = 1.0 + x;
    const double ok = 1.0 - c->omega_m - c->omega_de - c->omega_r;
    const double e2 = c->omega_r * pow(a1, 4) + c->omega_m * pow(a1, 3) +
                      ok * a1 * a1 +
                      c->omega_de * pow(a1, 3.0 * (1.0 + c->w0 + c->wa)) *
                        exp(-3.0 * c->wa * x / a1);
    const double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
    sum += w / sqrt(e2);
  }
  return DH * sum * h / 3.0;
}

static void test_distances(void) {
  printf("comoving distances\n");

  check_closed_form(
    "Einstein-de Sitter", &(sif_cosmology_t){1.0, 0.0, 0.0, -1.0, 0.0}, eds);
  check_closed_form(
    "empty (Milne)", &(sif_cosmology_t){0.0, 0.0, 0.0, -1.0, 0.0}, milne);
  check_closed_form(
    "de Sitter", &(sif_cosmology_t){0.0, 1.0, 0.0, -1.0, 0.0}, de_sitter);

  /* w0waCDM, open, with radiation: nothing is special about it. */
  const sif_cosmology_t w0wa = {0.3, 0.65, 8.5e-5, -0.8, -0.6};
  static const double zs[] = {0.5, 1.2, 3.0};
  for (size_t i = 0; i < 3; i++) {
    double d;
    CHECK(sif_cosmology_comoving_distance(&w0wa, zs[i], &d) == SIF_OK &&
            rel(d, brute(&w0wa, zs[i])) < 1e-10,
      "w0waCDM: D_C(%g) = %.12g, brute force %.12g", zs[i], d,
      brute(&w0wa, zs[i]));
  }

  /* The flat LambdaCDM preset is what it says. */
  double a, b;
  CHECK(
    sif_cosmology_comoving_distance(SIF_COSMOLOGY_FLAT_LCDM(0.31), 1.0, &a) ==
        SIF_OK &&
      sif_cosmology_comoving_distance(
        &(sif_cosmology_t){0.31, 0.69, 0.0, -1.0, 0.0}, 1.0, &b) == SIF_OK &&
      a == b,
    "the flat LambdaCDM preset differs from the parameters it stands for");

  /* What it refuses. */
  double d = 0.0;
  CHECK(sif_cosmology_comoving_distance(NULL, 1.0, &d) == SIF_ERR_INVALID,
    "accepted no cosmology");
  CHECK(sif_cosmology_comoving_distance(&w0wa, -0.1, &d) == SIF_ERR_INVALID,
    "accepted a negative redshift");
  CHECK(sif_cosmology_comoving_distance(&w0wa, NAN, &d) == SIF_ERR_INVALID,
    "accepted a NaN redshift");
  CHECK(sif_cosmology_comoving_distance(
          &(sif_cosmology_t){-0.1, 1.1, 0.0, -1.0, 0.0}, 1.0, &d) ==
          SIF_ERR_INVALID,
    "accepted a negative density");
  /* No matter, omega_de = 3 and so omega_k = -2: E^2 = 3 - 2 (1+z)^2, which
   * turns negative past z = 0.22 -- a bounce, with no past beyond it. */
  CHECK(
    sif_cosmology_comoving_distance(
      &(sif_cosmology_t){0.0, 3.0, 0.0, -1.0, 0.0}, 3.0, &d) == SIF_ERR_RANGE,
    "accepted a cosmology with no expansion history out to z = 3");
}

/* --- conversion --------------------------------------------------------- */

static sif_field_t* sky_field(
  const sif_real* ra, const sif_real* dec, const sif_real* z, uint64_t n) {
  sif_field_t* f = sif_field_alloc(n);
  if (sif_field_assign_positions(f, ra, dec, z) != SIF_OK) {
    sif_field_free(f);
    return NULL;
  }
  f->units = SIF_COORDINATES_SKY;
  return f;
}

static void test_conversion(void) {
  printf("sky coordinates to positions\n");

  const sif_cosmology_t* c = SIF_COSMOLOGY_FLAT_LCDM(0.31);
  double d1;
  (void)sif_cosmology_comoving_distance(c, 1.0, &d1);

  /* Along x, along y, along z, and down -z; the last at z = 0, the origin. */
  const sif_real ra[] = {0.0f, 90.0f, 17.0f, 250.0f, 33.0f};
  const sif_real dec[] = {0.0f, 0.0f, 90.0f, -90.0f, 12.0f};
  const sif_real red[] = {1.0f, 1.0f, 1.0f, 1.0f, 0.0f};
  sif_field_t* fresh = sif_field_alloc(1);
  CHECK(fresh && fresh->units == SIF_COORDINATES_CARTESIAN,
    "a new field is not Cartesian");
  sif_field_free(fresh);

  sif_field_t* f = sky_field(ra, dec, red, 5);

  CHECK(
    sif_field_convert_sky_coordinates(f, c) == SIF_OK, "the conversion failed");
  CHECK(f->units == SIF_COORDINATES_CARTESIAN, "the field is still marked sky");

  const double tol = 1e-5 * d1; /* single-precision positions */
  const double want[5][3] = {
    {d1, 0, 0}, {0, d1, 0}, {0, 0, d1}, {0, 0, -d1}, {0, 0, 0}};
  for (int i = 0; i < 5; i++)
    CHECK(fabs(f->x[i] - want[i][0]) < tol &&
            fabs(f->y[i] - want[i][1]) < tol &&
            fabs(f->z[i] - want[i][2]) < tol,
      "tracer %d at (%g, %g, %g), expected (%g, %g, %g)", i, (double)f->x[i],
      (double)f->y[i], (double)f->z[i], want[i][0], want[i][1], want[i][2]);

  /* Converted twice would read x as a right ascension. */
  CHECK(sif_field_convert_sky_coordinates(f, c) == SIF_ERR_INVALID,
    "converted a field that was already Cartesian");
  sif_field_free(f);

  /* Many tracers through the table against the integral itself, and each at
   * its own distance from the observer. */
  const uint64_t n = 20000;
  sif_real* a = malloc(n * sizeof(sif_real));
  sif_real* b = malloc(n * sizeof(sif_real));
  sif_real* z = malloc(n * sizeof(sif_real));
  for (uint64_t i = 0; i < n; i++) {
    a[i] = (sif_real)(360.0 * i / n);
    b[i] = (sif_real)(-90.0 + 180.0 * ((i * 7919) % n) / n);
    z[i] = (sif_real)(3.0 * ((i * 104729) % n) / n);
  }
  f = sky_field(a, b, z, n);
  CHECK(sif_field_convert_sky_coordinates(f, c) == SIF_OK,
    "the conversion of %llu tracers failed", (unsigned long long)n);
  double worst = 0.0;
  for (uint64_t i = 0; i < n; i++) {
    double want_d;
    (void)sif_cosmology_comoving_distance(c, (double)z[i], &want_d);
    const double got =
      sqrt((double)f->x[i] * f->x[i] + (double)f->y[i] * f->y[i] +
           (double)f->z[i] * f->z[i]);
    worst = fmax(worst, fabs(got - want_d) / fmax(want_d, 1.0));
  }
  CHECK(worst < 1e-6, "a tracer's distance is off by %.2e of itself", worst);
  sif_field_free(f);

  /* Refused before anything changes. */
  const sif_real bad_dec[] = {0.0f, 95.0f};
  const sif_real ok_ra[] = {10.0f, 20.0f}, ok_z[] = {0.5f, 0.6f};
  f = sky_field(ok_ra, bad_dec, ok_z, 2);
  CHECK(sif_field_convert_sky_coordinates(f, c) == SIF_ERR_INVALID,
    "accepted a declination of 95");
  CHECK(f->units == SIF_COORDINATES_SKY && f->x[0] == 10.0f &&
          f->y[1] == 95.0f && f->z[0] == 0.5f,
    "a refused conversion changed the field");
  sif_field_free(f);

  const sif_real neg_z[] = {0.5f, -0.01f};
  const sif_real ok_dec[] = {0.0f, 1.0f};
  f = sky_field(ok_ra, ok_dec, neg_z, 2);
  CHECK(sif_field_convert_sky_coordinates(f, c) == SIF_ERR_INVALID,
    "accepted a negative redshift");
  sif_field_free(f);

  free(a);
  free(b);
  free(z);
}

/* --- void centres back to the sky ---------------------------------------- */

static void test_catalog_to_sky(void) {
  printf("void centres to the sky\n");

  /* Tracers on the sky, taken to positions and back through a catalogue:
   * the two conversions have to be each other's inverse. */
  const sif_cosmology_t cosmo = {0.3, 0.65, 8.5e-5, -0.9, -0.3};
  const uint64_t n = 5000;
  sif_real* a = malloc(n * sizeof(sif_real));
  sif_real* b = malloc(n * sizeof(sif_real));
  sif_real* z = malloc(n * sizeof(sif_real));
  for (uint64_t i = 0; i < n; i++) {
    a[i] = (sif_real)(360.0 * i / n);
    b[i] = (sif_real)(-89.0 + 178.0 * ((i * 7919) % n) / n);
    z[i] = (sif_real)(0.01 + 2.5 * ((i * 104729) % n) / n);
  }
  sif_field_t* f = sky_field(a, b, z, n);
  CHECK(sif_field_convert_sky_coordinates(f, &cosmo) == SIF_OK,
    "the conversion to positions failed");

  sif_catalog_t* cat = sif_catalog_alloc(n);
  CHECK(cat && cat->units == SIF_COORDINATES_CARTESIAN,
    "a new catalogue is not Cartesian");
  for (uint64_t i = 0; i < n; i++)
    (void)sif_catalog_append(cat, f->x[i], f->y[i], f->z[i], 10.0f);
  sif_field_free(f);

  CHECK(sif_catalog_to_sky(cat, &cosmo) == SIF_OK &&
          cat->units == SIF_COORDINATES_SKY,
    "the conversion to the sky failed");
  double worst_angle = 0.0, worst_z = 0.0;
  for (uint64_t i = 0; i < n; i++) {
    double dra = fabs((double)cat->cx[i] - a[i]);
    dra = fmin(dra, 360.0 - dra) * cos((double)b[i] * 3.14159265358979 / 180);
    worst_angle = fmax(worst_angle, fmax(dra, fabs((double)cat->cy[i] - b[i])));
    worst_z = fmax(worst_z, fabs((double)cat->cz[i] - z[i]) / z[i]);
  }
  /* Single-precision positions some 1e3 Mpc/h out: about 1e-7 of each. */
  CHECK(
    worst_angle < 1e-4, "an angle came back off by %.2e degrees", worst_angle);
  CHECK(worst_z < 1e-5, "a redshift came back off by %.2e of itself", worst_z);
  CHECK(cat->radii[0] == 10.0f, "the radii were changed");

  /* Once on the sky, what moves or measures around centres refuses it, and
   * so does a second conversion. */
  const sif_real offset[3] = {1, 1, 1};
  CHECK(sif_catalog_translate(cat, offset) == SIF_ERR_INVALID,
    "translate accepted a sky catalogue");
  CHECK(sif_catalog_to_sky(cat, &cosmo) == SIF_ERR_INVALID,
    "converted a catalogue already on the sky");
  /* A real mesh, so that it is the catalogue being refused. */
  const sif_real px[] = {10, 50, 90}, py[] = {20, 50, 80}, pz[] = {30, 50, 70};
  sif_field_t* tracers = sif_field_alloc(3);
  (void)sif_field_assign_positions(tracers, px, py, pz);
  sif_chain_mesh_t* mesh =
    sif_chain_mesh_alloc(4, 100.0f, tracers, SIF_DEFAULT);
  sif_density_profiles_t* dens = NULL;
  CHECK(mesh && sif_profiles(cat, mesh, 2.0f, 10, SIF_DEFAULT, &dens, NULL) ==
                  SIF_ERR_INVALID,
    "profiles accepted a sky catalogue");
  sif_chain_mesh_free(mesh);
  sif_field_free(tracers);

  /* ASCII keeps the flag. */
  CHECK(sif_catalog_write_ascii("test_cosmology_sky.txt", cat) == SIF_OK,
    "writing the sky catalogue failed");
  sif_catalog_t* back = sif_catalog_read_ascii("test_cosmology_sky.txt");
  CHECK(back && back->units == SIF_COORDINATES_SKY && back->n_voids == n &&
          back->cz[7] == cat->cz[7],
    "the sky catalogue did not read back on the sky");
  sif_catalog_free(back);
  remove("test_cosmology_sky.txt");
  sif_catalog_free(cat);

  /* The observer's own position, and refusals before anything changes. */
  cat = sif_catalog_alloc(2);
  (void)sif_catalog_append(cat, 0.0f, 0.0f, 0.0f, 1.0f);
  (void)sif_catalog_append(cat, NAN, 0.0f, 0.0f, 1.0f);
  CHECK(sif_catalog_to_sky(cat, &cosmo) == SIF_ERR_INVALID &&
          cat->units == SIF_COORDINATES_CARTESIAN && cat->cx[0] == 0.0f,
    "a NaN centre was not refused, or the refusal changed the catalogue");
  cat->cx[1] = 1e7f; /* past the horizon of any of these models */
  CHECK(sif_catalog_to_sky(cat, &cosmo) == SIF_ERR_RANGE,
    "a centre past the horizon was not refused as out of range");
  cat->cx[1] = 100.0f;
  CHECK(sif_catalog_to_sky(cat, &cosmo) == SIF_OK && cat->cz[0] == 0.0f &&
          cat->cx[1] == 0.0f && cat->cy[1] == 0.0f && cat->cz[1] > 0.0f,
    "the observer or a centre along +x did not land where expected");
  sif_catalog_free(cat);

  free(a);
  free(b);
  free(z);
}

/* --- refusals ----------------------------------------------------------- */

static void test_refusals(void) {
  printf("a sky field is refused where positions are lengths\n");

  const sif_real ra[] = {10.0f, 20.0f, 30.0f}, dec[] = {1.0f, 2.0f, 3.0f},
                 z[] = {0.1f, 0.2f, 0.3f};
  sif_field_t* f = sky_field(ra, dec, z, 3);
  const sif_real offset[3] = {1.0f, 1.0f, 1.0f};
  const sif_real radii[] = {5.0f};

  CHECK(sif_field_wrap_periodic(f, 100.0f, NULL, NULL) == SIF_ERR_INVALID,
    "wrap_periodic accepted it");
  CHECK(
    sif_field_translate(f, offset) == SIF_ERR_INVALID, "translate accepted it");
  CHECK(sif_field_refresh_bounds(f) == SIF_ERR_INVALID,
    "refresh_bounds accepted it");
  CHECK(sif_field_require_bounds(f) == SIF_ERR_INVALID,
    "require_bounds accepted it");
  CHECK(sif_field_sort_morton(f) == SIF_ERR_INVALID, "sort_morton accepted it");
  CHECK(sif_field_require_morton(f) == SIF_ERR_INVALID,
    "require_morton accepted it");
  CHECK(sif_field_write("/dev/null", f, 100.0) == SIF_ERR_INVALID,
    "write accepted it");

  sif_grid_t* g = sif_grid_alloc(16, 100.0f);
  CHECK(sif_grid_assign_cic(g, f) == SIF_ERR_INVALID, "assign_cic accepted it");
  sif_grid_free(g);

  sif_chain_mesh_t* m = sif_chain_mesh_alloc(4, 100.0f, f, SIF_DEFAULT);
  CHECK(!m, "chain_mesh_alloc accepted it");
  sif_chain_mesh_free(m);
  m = sif_chain_mesh_alloc_consume(4, 100.0f, f, SIF_DEFAULT);
  CHECK(!m, "chain_mesh_alloc_consume accepted it");
  CHECK(f->n_particles == 3 && f->x && f->x[0] == 10.0f,
    "a refused consuming build emptied the field");
  sif_chain_mesh_free(m);

  sif_octree_t* o = sif_octree_alloc(f, 8);
  CHECK(!o, "octree_alloc accepted it");
  sif_octree_free(o);

  sif_real off[3], box;
  CHECK(sif_finder_exodus_survey_box(f, radii, 1, 32, SIF_DEFAULT, off, &box) ==
          SIF_ERR_INVALID,
    "survey_box accepted it");

  /* And, once converted, all of that works. */
  CHECK(sif_field_convert_sky_coordinates(f, SIF_COSMOLOGY_FLAT_LCDM(0.3)) ==
            SIF_OK &&
          sif_field_refresh_bounds(f) == SIF_OK,
    "a converted field is still refused");
  sif_field_free(f);
}

int main(void) {
  if (sif_init(SIF_CONFIG_QUIET) != SIF_OK)
    return 1;

  test_distances();
  test_conversion();
  test_catalog_to_sky();
  test_refusals();

  sif_finalize();
  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
