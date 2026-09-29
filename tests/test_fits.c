/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * FITS tables, against a file this test writes with cfitsio itself: a
 * catalogue whose every column is a closed form of the row number, in the
 * mix of types a real one has -- doubles, floats, integers, a string and a
 * vector column, and one column with undefined values.
 *
 * Without cfitsio, the reader is checked to refuse as unsupported.
 */
#include "sif/core/system.h"
#include "sif/io/fits_io.h"
#include "sif/structures/catalog.h"
#include "sif/structures/size_function.h"

#include "measure/profiles_internal.h"
#include "sif/structures/field.h"
#include "structures/results_internal.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SIF_HAVE_FITS
#  include <fitsio.h>
#endif

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define FIXTURE "test_fits_fixture.fits"

#ifdef SIF_HAVE_FITS

/* Rows of the catalogue; not a multiple of the reader's chunk, and more than
 * one of them, so the chunk boundaries are crossed. */
#  define N_ROWS  300007L
#  define N_OTHER 5L

static double ra_of(long i) { return fmod(0.37 * (double)i, 360.0); }
static float dec_of(long i) { return (float)(-60.0 + fmod(0.11 * i, 120.0)); }
static double z_of(long i) { return 0.1 + 0.8 * (double)i / N_ROWS; }
static int vx_of(long i) { return (int)(i % 1000) - 500; }
static float w1_of(long i) { return 1.0f + (float)(i % 7) * 0.125f; }
static double w2_of(long i) { return 0.5 + (double)(i % 3); }
static bool zbad_is_null(long i) { return i % 1000 == 7; }

/* Rows [first, first + n) of the catalogue, as extension 1 (GALAXIES) after
 * an empty primary image, and a small second table as extension 2 (OTHER).
 * Every column is a closed form of the row's index in the whole catalogue, so
 * the catalogue split over several files reads back as the same rows. A name
 * ending in .gz makes cfitsio write the file compressed. */
static int write_catalog(const char* path, long first, long n) {
  fitsfile* f;
  int st = 0;
  remove(path);
  fits_create_file(&f, path, &st);
  fits_create_img(f, 8, 0, NULL, &st);

  char* ttype[] = {
    "RA", "DEC", "Z", "VX", "VY", "VZ", "W1", "W2", "ZBAD", "NAME", "VEC"};
  char* tform[] = {
    "1D", "1E", "1D", "1J", "1J", "1J", "1E", "1D", "1D", "8A", "3D"};
  char* tunit[] = {
    "deg", "deg", "", "km/s", "km/s", "km/s", "", "", "", "", "Mpc/h"};
  fits_create_tbl(f, BINARY_TBL, n, 11, ttype, tform, tunit, "GALAXIES", &st);

  double* d = malloc((size_t)n * 3 * sizeof(double));
  float* e = malloc((size_t)n * sizeof(float));
  int* j = malloc((size_t)n * sizeof(int));
  char** names = malloc((size_t)n * sizeof(char*));
  for (long i = 0; i < n; i++)
    names[i] = "galaxy";

  for (long i = 0; i < n; i++)
    d[i] = ra_of(first + i);
  fits_write_col(f, TDOUBLE, 1, 1, 1, n, d, &st);
  for (long i = 0; i < n; i++)
    e[i] = dec_of(first + i);
  fits_write_col(f, TFLOAT, 2, 1, 1, n, e, &st);
  for (long i = 0; i < n; i++)
    d[i] = z_of(first + i);
  fits_write_col(f, TDOUBLE, 3, 1, 1, n, d, &st);
  for (int c = 0; c < 3; c++) {
    for (long i = 0; i < n; i++)
      j[i] = vx_of(first + i) * (c + 1);
    fits_write_col(f, TINT, 4 + c, 1, 1, n, j, &st);
  }
  for (long i = 0; i < n; i++)
    e[i] = w1_of(first + i);
  fits_write_col(f, TFLOAT, 7, 1, 1, n, e, &st);
  for (long i = 0; i < n; i++)
    d[i] = w2_of(first + i);
  fits_write_col(f, TDOUBLE, 8, 1, 1, n, d, &st);
  for (long i = 0; i < n; i++)
    d[i] = zbad_is_null(first + i) ? NAN : z_of(first + i);
  fits_write_col(f, TDOUBLE, 9, 1, 1, n, d, &st);
  fits_write_col(f, TSTRING, 10, 1, 1, n, names, &st);
  for (long i = 0; i < 3 * n; i++)
    d[i] = (double)(3 * first + i);
  fits_write_col(f, TDOUBLE, 11, 1, 1, 3 * n, d, &st);

  char* ttype2[] = {"X"};
  char* tform2[] = {"1D"};
  fits_create_tbl(
    f, BINARY_TBL, N_OTHER, 1, ttype2, tform2, NULL, "OTHER", &st);
  for (long i = 0; i < N_OTHER; i++)
    d[i] = 1000.0 + (double)i;
  fits_write_col(f, TDOUBLE, 1, 1, 1, N_OTHER, d, &st);

  fits_close_file(f, &st);
  free(d);
  free(e);
  free(j);
  free(names);

  if (st) {
    fits_report_error(stdout, st);
    return 1;
  }
  return 0;
}

/* A GALAXIES table with positions but none of the other columns: a file that
 * does not belong with the rest of a catalogue. */
static int write_stranger(const char* path) {
  fitsfile* f;
  int st = 0;
  remove(path);
  fits_create_file(&f, path, &st);
  fits_create_img(f, 8, 0, NULL, &st);
  char* ttype[] = {"RA", "DEC"};
  char* tform[] = {"1D", "1D"};
  fits_create_tbl(f, BINARY_TBL, 3, 2, ttype, tform, NULL, "GALAXIES", &st);
  const double d[3] = {1, 2, 3};
  fits_write_col(f, TDOUBLE, 1, 1, 1, 3, (void*)d, &st);
  fits_write_col(f, TDOUBLE, 2, 1, 1, 3, (void*)d, &st);
  fits_close_file(f, &st);
  return st != 0;
}

#  define PART_A   "test_fits_part_a.fits"
#  define PART_B   "test_fits_part_b.fits"
#  define GZIPPED  "test_fits_fixture.fits.gz"
#  define STRANGER "test_fits_stranger.fits"
/* Where the catalogue is split: inside a chunk of the reader's, so the two
 * files' rows meet mid-chunk. */
#  define SPLIT 100003L

static int write_fixtures(void) {
  return write_catalog(FIXTURE, 0, N_ROWS) || write_catalog(PART_A, 0, SPLIT) ||
         write_catalog(PART_B, SPLIT, N_ROWS - SPLIT) ||
         write_catalog(GZIPPED, 0, N_ROWS) || write_stranger(STRANGER);
}

static void remove_fixtures(void) {
  remove(FIXTURE);
  remove(PART_A);
  remove(PART_B);
  remove(GZIPPED);
  remove(STRANGER);
}

/* One file. */
static sif_field_t* read1(const char* path, const char* hdu,
  const sif_field_columns_t* cols, const char* where, double fraction,
  uint64_t seed) {
  return sif_field_read_fits(&path, 1, hdu, cols, where, fraction, seed);
}

static const sif_field_columns_t SKY = {.ra = "RA", .dec = "DEC", .z = "Z"};
static const sif_field_columns_t CART = {.x = "RA", .y = "DEC", .z = "Z"};

/* Whether a read fails. A field that comes back anyway is freed. */
static bool refused(const char* path, const char* hdu,
  const sif_field_columns_t* cols, const char* where, double fraction) {
  sif_field_t* fl = read1(path, hdu, cols, where, fraction, 0);
  sif_field_free(fl);
  return !fl;
}

static void test_plain(void) {
  printf("plain columns\n");

  sif_field_t* fl = read1(FIXTURE, NULL, &CART, NULL, 1.0, 0);
  CHECK(fl, "read failed");
  if (!fl)
    return;

  CHECK(fl->n_particles == (uint64_t)N_ROWS, "%llu rows, expected %ld",
    (unsigned long long)fl->n_particles, N_ROWS);
  CHECK(fl->units == SIF_COORDINATES_CARTESIAN, "CARTESIAN came out as sky");
  CHECK(!fl->vx && !fl->weights, "velocities or weights nobody asked for");

  long bad = 0;
  for (long i = 0; i < N_ROWS; i++) {
    bad += fl->x[i] != (sif_real)ra_of(i);
    bad += fl->y[i] != (sif_real)dec_of(i);
    bad += fl->z[i] != (sif_real)z_of(i);
  }
  CHECK(bad == 0, "%ld values differ from the file", bad);
  sif_field_free(fl);

  /* Names match without regard to case, and SKY marks the field. */
  const sif_field_columns_t lower = {.ra = "ra", .dec = "Dec", .z = "z"};
  fl = read1(FIXTURE, "galaxies", &lower, NULL, 1.0, 0);
  CHECK(fl && fl->units == SIF_COORDINATES_SKY,
    "lower-case names, or SKY, not honoured");
  sif_field_free(fl);
}

static void test_hdus(void) {
  printf("HDUs\n");

  const sif_field_columns_t other = {.x = "X", .y = "X", .z = "X"};
  const char* ways[] = {"OTHER", "other", "2"};
  for (int k = 0; k < 3; k++) {
    sif_field_t* fl = read1(FIXTURE, ways[k], &other, NULL, 1.0, 0);
    CHECK(fl && fl->n_particles == (uint64_t)N_OTHER &&
            fl->x[3] == (sif_real)1003.0,
      "HDU \"%s\" not read as the second table", ways[k]);
    sif_field_free(fl);
  }

  sif_field_t* fl = read1(FIXTURE, "1", &CART, NULL, 1.0, 0);
  CHECK(fl && fl->n_particles == (uint64_t)N_ROWS,
    "HDU \"1\" is not the catalogue");
  sif_field_free(fl);

  CHECK(refused(FIXTURE, "0", &CART, NULL, 1.0),
    "the primary image read as a table");
  CHECK(refused(FIXTURE, "9", &CART, NULL, 1.0),
    "a missing extension number accepted");
  CHECK(refused(FIXTURE, "RANDOMS", &CART, NULL, 1.0),
    "a missing EXTNAME accepted");
}

static void test_velocities_and_weights(void) {
  printf("velocities and weights\n");

  /* Integer velocity columns, a weight that is a product of a float and a
   * double column, and one that is not a product at all. */
  const sif_field_columns_t cols = {.ra = "RA",
    .dec = "DEC",
    .z = "Z",
    .vx = "VX",
    .vy = "VY",
    .vz = "VZ",
    .w = "W1 * W2"};
  sif_field_t* fl = read1(FIXTURE, NULL, &cols, NULL, 1.0, 0);
  CHECK(fl && fl->vx && fl->weights, "read failed");
  if (fl && fl->vx && fl->weights) {
    long bad = 0;
    for (long i = 0; i < N_ROWS; i++) {
      bad += fl->vx[i] != (sif_real)vx_of(i);
      bad += fl->vz[i] != (sif_real)(3 * vx_of(i));
      bad += fl->weights[i] != (sif_real)((double)w1_of(i) * w2_of(i));
    }
    CHECK(bad == 0, "%ld velocities or weights differ", bad);
  }
  sif_field_free(fl);

  const sif_field_columns_t boss = {
    .ra = "RA", .dec = "DEC", .z = "Z", .w = "W1 * (W2 + 1) - 1"};
  fl = read1(FIXTURE, NULL, &boss, NULL, 1.0, 0);
  CHECK(fl, "a compound weight failed");
  if (fl) {
    long bad = 0;
    for (long i = 0; i < N_ROWS; i++)
      bad +=
        fl->weights[i] != (sif_real)((double)w1_of(i) * (w2_of(i) + 1) - 1);
    CHECK(bad == 0, "%ld compound weights differ", bad);
  }
  sif_field_free(fl);

  /* A constant is an expression too, repeated down the rows. */
  const sif_field_columns_t two = {
    .ra = "RA", .dec = "DEC", .z = "Z", .w = "2"};
  fl = read1(FIXTURE, NULL, &two, NULL, 1.0, 0);
  CHECK(fl && fl->weights[0] == 2 && fl->weights[N_ROWS - 1] == 2,
    "a constant weight was not 2 in every row");
  sif_field_free(fl);
}

static void test_filter_and_subsample(void) {
  printf("filter and subsample\n");

  const char* where = "Z > 0.3 && Z < 0.6";
  long n_pass = 0;
  for (long i = 0; i < N_ROWS; i++)
    n_pass += z_of(i) > 0.3 && z_of(i) < 0.6;

  sif_field_t* fl = read1(FIXTURE, NULL, &SKY, where, 1.0, 0);
  CHECK(fl && fl->n_particles == (uint64_t)n_pass,
    "filter kept %llu rows, expected %ld",
    fl ? (unsigned long long)fl->n_particles : 0ULL, n_pass);
  if (fl) {
    /* In file order: the rows passing are a contiguous run in z. */
    long first = 0;
    while (!(z_of(first) > 0.3))
      first++;
    long bad = 0;
    for (uint64_t k = 0; k < fl->n_particles; k++)
      bad += fl->x[k] != (sif_real)ra_of(first + (long)k);
    CHECK(bad == 0, "%ld filtered rows out of order", bad);
  }
  sif_field_free(fl);

  /* The subsample is drawn from the rows the filter keeps. */
  const long n_keep = lround(0.25 * (double)n_pass);
  sif_field_t* a = read1(FIXTURE, NULL, &SKY, where, 0.25, 42);
  CHECK(a && a->n_particles == (uint64_t)n_keep,
    "subsample of %llu, expected %ld",
    a ? (unsigned long long)a->n_particles : 0ULL, n_keep);
  if (a) {
    long outside = 0;
    for (uint64_t k = 0; k < a->n_particles; k++)
      outside += !(a->z[k] > (sif_real)0.3 && a->z[k] < (sif_real)0.6);
    CHECK(outside == 0, "%ld subsampled rows fail the filter", outside);
  }

  sif_field_t* b = read1(FIXTURE, NULL, &SKY, where, 0.25, 42);
  sif_field_t* c = read1(FIXTURE, NULL, &SKY, where, 0.25, 43);
  CHECK(a && b && c, "a subsample failed");
  if (a && b && c) {
    CHECK(memcmp(a->x, b->x, a->n_particles * sizeof(sif_real)) == 0,
      "the same seed gave different rows");
    CHECK(memcmp(a->x, c->x, a->n_particles * sizeof(sif_real)) != 0,
      "a different seed gave the same rows");
  }
  sif_field_free(a);
  sif_field_free(b);
  sif_field_free(c);

  CHECK(refused(FIXTURE, NULL, &SKY, "Z > 5", 1.0),
    "a filter that keeps nothing accepted");
  CHECK(refused(FIXTURE, NULL, &SKY, "Z * 2", 1.0),
    "a filter that is not boolean accepted");
  CHECK(refused(FIXTURE, NULL, &SKY, "Z >> > 1", 1.0),
    "a filter that does not parse accepted");
}

static void test_undefined(void) {
  printf("undefined values\n");

  const sif_field_columns_t cols = {.ra = "RA", .dec = "DEC", .z = "ZBAD"};
  CHECK(refused(FIXTURE, NULL, &cols, NULL, 1.0), "NaN redshifts read");

  /* Dropped by the filter, they are not there to refuse. */
  sif_field_t* fl = read1(FIXTURE, NULL, &cols, "!ISNULL(ZBAD)", 1.0, 0);
  const long n_null = (N_ROWS - 8) / 1000 + 1;
  CHECK(fl && fl->n_particles == (uint64_t)(N_ROWS - n_null),
    "!ISNULL did not drop the %ld NaN rows (%llu kept)", n_null,
    fl ? (unsigned long long)fl->n_particles : 0ULL);
  sif_field_free(fl);
}

static void test_refusals(void) {
  printf("refusals\n");

  const struct {
    const char* what;
    sif_field_columns_t cols;
  } bad[] = {
    {"a missing column", {.x = "RA", .y = "DEC", .z = "NOPE"}},
    {"a string column", {.x = "RA", .y = "DEC", .z = "NAME"}},
    {"a vector column", {.x = "RA", .y = "DEC", .z = "VEC"}},
    {"an expression that does not parse", {.x = "RA", .y = "DEC", .z = "Z +*"}},
    {"a boolean expression", {.x = "RA", .y = "DEC", .z = "Z > 1"}},
    {"a missing position", {.x = "RA", .y = "DEC"}},
    {"half the velocities",
      {.x = "RA", .y = "DEC", .z = "Z", .vx = "VX", .vy = "VY"}},
    {"an empty name", {.x = "RA", .y = "DEC", .z = "Z", .w = ""}},
  };
  for (size_t k = 0; k < sizeof(bad) / sizeof(bad[0]); k++)
    CHECK(refused(FIXTURE, NULL, &bad[k].cols, NULL, 1.0), "%s accepted",
      bad[k].what);

  const sif_field_columns_t mixed = {.x = "RA", .dec = "DEC", .z = "Z"};
  CHECK(refused(FIXTURE, NULL, &mixed, NULL, 1.0),
    "positions and sky coordinates mixed accepted");
  const sif_field_columns_t half = {.ra = "RA", .z = "Z"};
  CHECK(refused(FIXTURE, NULL, &half, NULL, 1.0), "ra without dec accepted");
  CHECK(refused(FIXTURE, NULL, &SKY, NULL, 0.0), "a fraction of 0 accepted");
  CHECK(refused(FIXTURE, NULL, &SKY, NULL, 1e-9),
    "a fraction that keeps nothing accepted");
  CHECK(refused("no_such_file.fits", NULL, &SKY, NULL, 1.0),
    "a missing file accepted");
  CHECK(refused(FIXTURE "[2]", NULL, &SKY, NULL, 1.0),
    "the extended filename syntax was interpreted");
  CHECK(refused(NULL, NULL, &SKY, NULL, 1.0), "a NULL path accepted");
  CHECK(refused(FIXTURE, NULL, NULL, NULL, 1.0), "NULL columns accepted");
}

static void test_several_files(void) {
  printf("several files\n");

  /* The two halves read as one are the whole catalogue: the same rows, in
   * the same order, with the same weights. */
  const char* halves[] = {PART_A, PART_B};
  const sif_field_columns_t cols = {
    .ra = "RA", .dec = "DEC", .z = "Z", .w = "W1 * W2"};
  sif_field_t* whole = read1(FIXTURE, NULL, &cols, NULL, 1.0, 0);
  sif_field_t* joined =
    sif_field_read_fits(halves, 2, NULL, &cols, NULL, 1.0, 0);
  CHECK(
    whole && joined && joined->n_particles == whole->n_particles &&
      memcmp(whole->x, joined->x, N_ROWS * sizeof(sif_real)) == 0 &&
      memcmp(whole->weights, joined->weights, N_ROWS * sizeof(sif_real)) == 0,
    "the two halves do not read back as the whole catalogue");
  sif_field_free(whole);
  sif_field_free(joined);

  /* The filter and the subsample run over the rows of both files together:
   * the same seed draws the same rows as from the single file. The cut
   * straddles the split. */
  const char* where = "Z > 0.3 && Z < 0.5";
  whole = read1(FIXTURE, NULL, &SKY, where, 0.3, 7);
  joined = sif_field_read_fits(halves, 2, NULL, &SKY, where, 0.3, 7);
  CHECK(
    whole && joined && joined->n_particles == whole->n_particles &&
      memcmp(whole->x, joined->x, whole->n_particles * sizeof(sif_real)) == 0,
    "a filtered subsample of the halves differs from the whole's");
  sif_field_free(whole);
  sif_field_free(joined);

  /* Every file is checked before anything is read. */
  const char* stranger[] = {FIXTURE, STRANGER};
  sif_field_t* fl = sif_field_read_fits(stranger, 2, NULL, &SKY, NULL, 1.0, 0);
  CHECK(!fl, "a second file without the Z column accepted");
  sif_field_free(fl);

  const char* missing[] = {FIXTURE, "test_fits_no_such_part.fits"};
  fl = sif_field_read_fits(missing, 2, NULL, &SKY, NULL, 1.0, 0);
  CHECK(!fl, "a missing second file accepted");
  sif_field_free(fl);

  const char* with_null[] = {FIXTURE, NULL};
  fl = sif_field_read_fits(with_null, 2, NULL, &SKY, NULL, 1.0, 0);
  CHECK(!fl, "a NULL path in the list accepted");
  sif_field_free(fl);

  fl = sif_field_read_fits(halves, 0, NULL, &SKY, NULL, 1.0, 0);
  CHECK(!fl, "no paths accepted");
  sif_field_free(fl);
}

static void test_vectors_and_gzip(void) {
  printf("vector columns and compressed files\n");

  /* VEC holds 3i, 3i + 1, 3i + 2 in row i. */
  const sif_field_columns_t vec = {.x = "VEC[1]", .y = "VEC[2]", .z = "VEC[3]"};
  sif_field_t* fl = read1(FIXTURE, NULL, &vec, NULL, 1.0, 0);
  CHECK(fl, "vector elements not read");
  if (fl) {
    long bad = 0;
    for (long i = 0; i < N_ROWS; i += 997) {
      bad += fl->x[i] != (sif_real)(3.0 * i);
      bad += fl->z[i] != (sif_real)(3.0 * i + 2);
    }
    CHECK(bad == 0, "%ld vector elements differ", bad);
  }
  sif_field_free(fl);

  sif_field_t* plain = read1(FIXTURE, NULL, &SKY, "Z < 0.5", 1.0, 0);
  sif_field_t* gz = read1(GZIPPED, NULL, &SKY, "Z < 0.5", 1.0, 0);
  CHECK(plain && gz && gz->n_particles == plain->n_particles &&
          memcmp(plain->y, gz->y, plain->n_particles * sizeof(sif_real)) == 0,
    "the gzipped file does not read as the plain one");
  sif_field_free(plain);
  sif_field_free(gz);
}

/* The summary, printed into a string. */
static bool summary_of(const char* path, char* buf, size_t len) {
  FILE* tmp = tmpfile();
  if (!tmp)
    return false;
  const int s = sif_fits_print_summary(path, tmp);
  rewind(tmp);
  const size_t n = fread(buf, 1, len - 1, tmp);
  buf[n] = '\0';
  fclose(tmp);
  return s == SIF_OK;
}

static void test_summary(void) {
  printf("summary\n");

  static char text[8192];
  CHECK(summary_of(FIXTURE, text, sizeof(text)), "summary failed");
  CHECK(strstr(text, "3 HDUs"), "HDU count missing:\n%s", text);
  CHECK(strstr(text, "[1] GALAXIES"), "extension name missing:\n%s", text);
  CHECK(
    strstr(text, "300007 rows, 11 columns"), "row count missing:\n%s", text);
  CHECK(strstr(text, "vector: VEC[1] ... VEC[3]"), "vector note missing:\n%s",
    text);
  CHECK(strstr(text, "not readable"), "string column not flagged:\n%s", text);
  CHECK(strstr(text, "km/s"), "units missing:\n%s", text);
  CHECK(strstr(text, "[2] OTHER"), "second table missing:\n%s", text);

  CHECK(sif_fits_inspect("test_fits_no_such_file.fits") == SIF_ERR_IO,
    "inspecting a missing file did not fail with SIF_ERR_IO");
  CHECK(sif_fits_print_summary(FIXTURE, NULL) == SIF_ERR_INVALID,
    "a NULL stream accepted");
}

#  define CAT_PATH "test_fits_catalog.fits"

static bool same_catalog(const sif_catalog_t* a, const sif_catalog_t* b) {
  if (!a || !b || a->n_voids != b->n_voids || a->units != b->units ||
      !a->footprint != !b->footprint)
    return false;
  const size_t bytes = a->n_voids * sizeof(sif_real);
  return memcmp(a->cx, b->cx, bytes) == 0 && memcmp(a->cy, b->cy, bytes) == 0 &&
         memcmp(a->cz, b->cz, bytes) == 0 &&
         memcmp(a->radii, b->radii, bytes) == 0 &&
         (!a->footprint ||
           (memcmp(a->footprint, b->footprint, bytes) == 0 &&
             memcmp(a->footprint_shell, b->footprint_shell, bytes) == 0));
}

static void test_catalogs(void) {
  printf("catalogues\n");

  sif_catalog_t* cat = sif_catalog_alloc(4);
  for (int i = 0; i < 1000; i++)
    (void)sif_catalog_append(cat, (sif_real)(0.5 * i), (sif_real)(-0.25 * i),
      (sif_real)(1000.0 - i), (sif_real)(5.0 + 0.01 * i));

  /* Cartesian, without a footprint. */
  CHECK(sif_catalog_write_fits(CAT_PATH, cat) == SIF_OK, "write failed");
  sif_catalog_t* back = sif_catalog_read_fits(CAT_PATH);
  CHECK(same_catalog(cat, back), "a Cartesian catalogue did not round-trip");
  sif_catalog_free(back);

  static char text[4096];
  CHECK(summary_of(CAT_PATH, text, sizeof text) && strstr(text, "[1] VOIDS") &&
          strstr(text, "      R") && strstr(text, "      CX "),
    "the Cartesian table is not laid out as documented:\n%s", text);

  /* On the sky, with a footprint: RA and DEC columns, and the flag back. */
  CHECK(sif_catalog_reserve_footprint(cat) == SIF_OK, "footprint failed");
  for (uint64_t i = 0; i < cat->n_voids; i++) {
    cat->footprint[i] = (sif_real)(i % 10) / 10;
    cat->footprint_shell[i] = (sif_real)(i % 7) / 7;
  }
  CHECK(sif_catalog_to_sky(cat, SIF_COSMOLOGY_FLAT_LCDM(0.31)) == SIF_OK,
    "the conversion failed");
  CHECK(sif_catalog_write_fits(CAT_PATH, cat) == SIF_OK, "sky write failed");
  back = sif_catalog_read_fits(CAT_PATH);
  CHECK(same_catalog(cat, back),
    "a sky catalogue with a footprint did not round-trip");
  sif_catalog_free(back);
  CHECK(summary_of(CAT_PATH, text, sizeof text) && strstr(text, "RA ") &&
          strstr(text, "FOOTPRINT_SHELL") && strstr(text, "deg"),
    "the sky table is not laid out as documented:\n%s", text);
  char coords[16];
  sif_fits_get_key_string(CAT_PATH, "VOIDS", "COORDS", coords, sizeof coords);
  CHECK(strcmp(coords, "sky") == 0, "COORDS says \"%s\"", coords);
  sif_catalog_free(cat);

  /* An empty catalogue is a catalogue. */
  cat = sif_catalog_alloc(1);
  CHECK(sif_catalog_write_fits(CAT_PATH, cat) == SIF_OK, "empty write failed");
  back = sif_catalog_read_fits(CAT_PATH);
  CHECK(back && back->n_voids == 0, "an empty catalogue did not round-trip");
  sif_catalog_free(back);
  sif_catalog_free(cat);

  /* A table that is not a catalogue. */
  CHECK(!sif_catalog_read_fits(FIXTURE), "the galaxy table read as voids");
  CHECK(!sif_catalog_read_fits("test_fits_no_such_file.fits"),
    "a missing file read as a catalogue");
}

static void test_keywords(void) {
  printf("header keywords\n");

  sif_catalog_t* cat = sif_catalog_alloc(1);
  (void)sif_catalog_append(cat, 1, 2, 3, 4);
  (void)sif_catalog_write_fits(CAT_PATH, cat);
  sif_catalog_free(cat);

  /* Each kind, in the primary header by default, a long name as HIERARCH,
   * and a string longer than a card. */
  const char* long_path =
    "a/very/long/path/to/some/file/that/goes/on/and/on/past/sixty/eight/"
    "characters/tracers.fits";
  CHECK(
    sif_fits_set_key_int(CAT_PATH, NULL, "NTRACER", 123456789012LL) == SIF_OK &&
      sif_fits_set_key_real(CAT_PATH, NULL, "search_factor", 1.5) == SIF_OK &&
      sif_fits_set_key_string(CAT_PATH, NULL, "INPUT", long_path) == SIF_OK &&
      sif_fits_set_key_real(CAT_PATH, "VOIDS", "THRESH", -0.7) == SIF_OK,
    "setting keywords failed");

  CHECK(sif_fits_get_key_real(CAT_PATH, NULL, "ntracer") == 123456789012.0,
    "an integer keyword read back as %g",
    sif_fits_get_key_real(CAT_PATH, NULL, "ntracer"));
  CHECK(sif_fits_get_key_real(CAT_PATH, NULL, "SEARCH_FACTOR") == 1.5,
    "a HIERARCH keyword did not read back");
  char buf[256];
  sif_fits_get_key_string(CAT_PATH, NULL, "input", buf, sizeof buf);
  CHECK(strcmp(buf, long_path) == 0, "a long string read back as \"%s\"", buf);
  sif_fits_get_key_string(CAT_PATH, NULL, "input", buf, 8);
  CHECK(
    strcmp(buf, "a/very/") == 0, "a string was not cut to fit: \"%s\"", buf);
  sif_fits_get_key_string(CAT_PATH, NULL, "SEARCH_FACTOR", buf, sizeof buf);
  CHECK(strcmp(buf, "1.5") == 0, "a number as text read \"%s\"", buf);

  /* SIMPLE is the primary header's own logical, T. */
  CHECK(sif_fits_key_kind(CAT_PATH, NULL, "SIMPLE") == SIF_FITS_KEY_LOGICAL &&
          sif_fits_get_key_real(CAT_PATH, NULL, "SIMPLE") == 1.0,
    "a logical keyword did not read as 1");
  CHECK(
    sif_fits_key_kind(CAT_PATH, NULL, "NTRACER") == SIF_FITS_KEY_INT &&
      sif_fits_key_kind(CAT_PATH, NULL, "SEARCH_FACTOR") == SIF_FITS_KEY_REAL &&
      sif_fits_key_kind(CAT_PATH, NULL, "INPUT") == SIF_FITS_KEY_STRING,
    "a keyword's kind is wrong");

  /* Where they live: the table's keyword is not in the primary header. */
  CHECK(sif_fits_get_key_real(CAT_PATH, "VOIDS", "THRESH") == -0.7,
    "a keyword in the table's header did not read back");
  CHECK(sif_fits_key_kind(CAT_PATH, NULL, "THRESH") == SIF_FITS_KEY_MISSING &&
          sif_fits_get_key_real(CAT_PATH, "1", "THRESH") == -0.7,
    "the primary header default, or an HDU by number, is wrong");

  /* Missing: the standard values. */
  sif_fits_get_key_string(CAT_PATH, NULL, "NOPE", buf, sizeof buf);
  CHECK(sif_fits_get_key_real(CAT_PATH, NULL, "NOPE") == 0.0 &&
          buf[0] == '\0' &&
          sif_fits_key_kind(CAT_PATH, NULL, "NOPE") == SIF_FITS_KEY_MISSING,
    "a missing keyword did not give 0 and \"\"");
  CHECK(
    sif_fits_get_key_real("test_fits_no_such_file.fits", NULL, "X") == 0.0 &&
      sif_fits_get_key_real(CAT_PATH, "NOHDU", "X") == 0.0,
    "a missing file or HDU did not give 0");

  /* What cannot be written. */
  CHECK(sif_fits_set_key_int("test_fits_no_such_file.fits", NULL, "X", 1) ==
          SIF_ERR_IO,
    "setting a keyword in a missing file did not fail with SIF_ERR_IO");
  CHECK(sif_fits_set_key_int(CAT_PATH, "NOHDU", "X", 1) == SIF_ERR_INVALID,
    "setting a keyword in a missing HDU did not fail with SIF_ERR_INVALID");
  CHECK(sif_fits_set_key_real(CAT_PATH, NULL, "X", NAN) == SIF_ERR_INVALID,
    "a NaN keyword was accepted");
  remove(CAT_PATH);
}

static void test_products(void) {
  printf("catalogue metadata and products\n");
  remove(CAT_PATH);

  enum { N = 37, NB = 12 };
  sif_catalog_t* cat = sif_catalog_alloc(N);
  for (int i = 0; i < N; i++)
    (void)sif_catalog_append(cat, (sif_real)(1234.5678901234 + 0.1 * i),
      (sif_real)(0.1234567890123 * (i + 1)), (sif_real)(9.87654321e-3 * i),
      (sif_real)(3.14159265 + i));
  sif_catalog_meta_string_set(cat, "finder", "exodus");
  sif_catalog_meta_real_set(cat, "search_factor", 1.5);
  sif_catalog_meta_int_set(cat, "n_tracers", 123456789012LL);
  sif_catalog_meta_string_set(cat, "input",
    "a/very/long/path/to/some/file/that/goes/on/and/on/past/sixty/eight/"
    "characters/tracers.fits");

  sif_density_profiles_t* d =
    sif__density_profiles_alloc(N, NB, (sif_real)3.0, true);
  sif_velocity_profiles_t* v =
    sif__velocity_profiles_alloc(N, NB, (sif_real)2.5);
  for (uint32_t b = 0; b <= NB; b++) {
    d->r_edges[b] = (sif_real)(3.0 * b / NB);
    v->r_edges[b] = (sif_real)(2.5 * b / NB);
  }
  for (uint64_t i = 0; i < (uint64_t)N * NB; i++) {
    d->profiles[i] = (sif_real)(-0.987654321 + 1e-3 * (double)i);
    v->v_rad[i] = (sif_real)(123.456789 - 0.37 * (double)i);
  }
  sif_size_function_t* f = sif__size_function_alloc(NB);
  f->options = SIF_VSF_BIN_LINEAR;
  f->r_min = 5;
  f->r_max = 65;
  for (uint32_t b = 0; b <= NB; b++)
    f->r_edges[b] = (sif_real)(5.0 + 5.0 * b);
  for (uint32_t b = 0; b < NB; b++) {
    f->r_centers[b] = (sif_real)(7.5 + 5.0 * b);
    f->counts[b] = (uint64_t)1 << (20 + b);
    f->vsf[b] = (sif_real)(1.23456789e-5 / (b + 1));
    f->err[b] = (sif_real)(2.3456789e-7 / (b + 1));
  }

  /* Every product into one file. */
  CHECK(sif_catalog_write_fits(CAT_PATH, cat) == SIF_OK &&
          sif_profiles_write_fits(CAT_PATH, d, v) == SIF_OK &&
          sif_size_function_write_fits(CAT_PATH, f) == SIF_OK,
    "writing the products failed");

  sif_catalog_t* cat2 = sif_catalog_read_fits(CAT_PATH);
  CHECK(cat2 && cat2->n_voids == N &&
          memcmp(cat2->radii, cat->radii, N * sizeof(sif_real)) == 0 &&
          sif_catalog_meta_count(cat2) == 4 &&
          strcmp(sif_catalog_meta_string_get(cat2, "finder"), "exodus") == 0 &&
          sif_catalog_meta_real_get(cat2, "search_factor") == 1.5 &&
          sif_catalog_meta_int_get(cat2, "n_tracers") == 123456789012LL &&
          strcmp(sif_catalog_meta_string_get(cat2, "input"),
            sif_catalog_meta_string_get(cat, "input")) == 0,
    "the catalogue or its metadata did not survive FITS");
  sif_catalog_free(cat2);

  int has_d = -1, has_v = -1;
  CHECK(sif_profiles_read_header_fits(CAT_PATH, &has_d, &has_v) == SIF_OK &&
          has_d == 1 && has_v == 1,
    "the header does not report both profile sets");
  sif_density_profiles_t* d2 = NULL;
  sif_velocity_profiles_t* v2 = NULL;
  CHECK(sif_profiles_read_fits(CAT_PATH, &d2, &v2) == SIF_OK && d2 && v2 &&
          d2->n_voids == N && d2->n_bins == NB && d2->ext == d->ext &&
          d2->differential &&
          memcmp(d2->r_edges, d->r_edges, (NB + 1) * sizeof(sif_real)) == 0 &&
          memcmp(d2->profiles, d->profiles, N * NB * sizeof(sif_real)) == 0 &&
          memcmp(v2->v_rad, v->v_rad, N * NB * sizeof(sif_real)) == 0 &&
          v2->ext == v->ext,
    "the profiles did not round-trip");
  sif_density_profiles_free(d2);
  sif_velocity_profiles_free(v2);

  sif_size_function_t* f2 = sif_size_function_read_fits(CAT_PATH);
  CHECK(f2 && f2->n_bins == NB && f2->options == f->options &&
          f2->r_min == f->r_min && f2->r_max == f->r_max &&
          memcmp(f2->r_edges, f->r_edges, (NB + 1) * sizeof(sif_real)) == 0 &&
          memcmp(f2->r_centers, f->r_centers, NB * sizeof(sif_real)) == 0 &&
          memcmp(f2->counts, f->counts, NB * sizeof(uint64_t)) == 0 &&
          memcmp(f2->vsf, f->vsf, NB * sizeof(sif_real)) == 0 &&
          memcmp(f2->err, f->err, NB * sizeof(sif_real)) == 0,
    "the size function did not round-trip");
  sif_size_function_free(f2);

  /* Rewriting the catalogue replaces it, and keeps what was measured. */
  sif_catalog_meta_remove(cat, "input");
  CHECK(sif_catalog_write_fits(CAT_PATH, cat) == SIF_OK,
    "rewriting the catalogue failed");
  cat2 = sif_catalog_read_fits(CAT_PATH);
  CHECK(cat2 && sif_catalog_meta_count(cat2) == 3 &&
          sif_catalog_meta_kind(cat2, "input") == SIF_CATALOG_META_MISSING,
    "the old catalogue's metadata outlived it");
  sif_catalog_free(cat2);
  d2 = NULL;
  CHECK(sif_profiles_read_fits(CAT_PATH, &d2, NULL) == SIF_OK && d2,
    "rewriting the catalogue lost the profiles");
  sif_density_profiles_free(d2);
  static char text[8192];
  CHECK(summary_of(CAT_PATH, text, sizeof text) && strstr(text, "5 HDUs"),
    "rewriting the catalogue left the file with the wrong HDUs:\n%s", text);

  /* Asked for what the file does not hold, or into a file that is not
   * sif's. */
  remove(CAT_PATH);
  CHECK(sif_catalog_write_fits(CAT_PATH, cat) == SIF_OK,
    "writing the catalogue alone failed");
  v2 = NULL;
  CHECK(sif_profiles_read_fits(CAT_PATH, NULL, &v2) == SIF_ERR_INVALID && !v2,
    "a set the file does not hold was not refused");
  CHECK(!sif_size_function_read_fits(CAT_PATH),
    "a size function the file does not hold was read");
  CHECK(sif_catalog_write_fits(FIXTURE, cat) == SIF_ERR_IO &&
          sif_profiles_write_fits(FIXTURE, d, NULL) == SIF_ERR_IO,
    "a FITS file sif did not write was written into");

  remove(CAT_PATH);
  sif_catalog_free(cat);
  sif_density_profiles_free(d);
  sif_velocity_profiles_free(v);
  sif_size_function_free(f);
}

#endif /* SIF_HAVE_FITS */

int main(void) {
  if (sif_init(SIF_CONFIG_QUIET) != SIF_OK)
    return 1;

#ifdef SIF_HAVE_FITS
  if (write_fixtures() != 0) {
    printf("could not write the fixtures\n");
    remove_fixtures();
    return 1;
  }
  test_plain();
  test_hdus();
  test_velocities_and_weights();
  test_filter_and_subsample();
  test_undefined();
  test_refusals();
  test_several_files();
  test_vectors_and_gzip();
  test_summary();
  test_catalogs();
  test_keywords();
  test_products();
  remove_fixtures();
#else
  printf("no cfitsio: the reader refuses\n");
  const sif_field_columns_t cols = {.ra = "RA", .dec = "DEC", .z = "Z"};
  const char* path = FIXTURE;
  sif_field_t* fl = sif_field_read_fits(&path, 1, NULL, &cols, NULL, 1.0, 0);
  CHECK(!fl, "the reader returned a field without cfitsio");
  sif_field_free(fl);
  CHECK(sif_fits_inspect(FIXTURE) == SIF_ERR_UNSUPPORTED,
    "inspecting did not refuse as unsupported");
  sif_catalog_t* cat = sif_catalog_alloc(1);
  CHECK(sif_catalog_write_fits("test_fits_catalog.fits", cat) ==
            SIF_ERR_UNSUPPORTED &&
          !sif_catalog_read_fits("test_fits_catalog.fits") &&
          sif_fits_get_key_real(FIXTURE, NULL, "X") == 0.0 &&
          sif_fits_set_key_int(FIXTURE, NULL, "X", 1) == SIF_ERR_UNSUPPORTED,
    "the catalogue and keyword functions did not refuse");
  sif_catalog_free(cat);
#endif

  sif_finalize();
  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
