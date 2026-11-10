/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * File checks: every format read whole. A sound file has to come out sound,
 * with every value of it counted; a damaged one -- cut short, a bit flipped,
 * a value out of bounds -- has to say so, as an error naming where. The
 * damage is done here, to files sif writes and to the fixtures in
 * tests/data.
 */

#include "sif/io/file_check.h"

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/io/catalogue_io.h"
#include "sif/io/field_io.h"
#include "sif/io/fits_io.h"
#include "sif/io/gadget_io.h"
#include "sif/io/grid_io.h"
#include "sif/io/hdf5_io.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: ");                                                      \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define GADGET SIF_TEST_DATA_DIR "/gadget/"

/* --- helpers --- */

/* Whether a problem of the check says @p needle. */
static bool said(const sif_file_check_t* c, const char* needle) {
  for (uint32_t i = 0; c && i < c->n_problems; i++)
    if (strstr(c->problems[i].message, needle))
      return true;
  return false;
}

/* The first problem, for a failure's message. */
static const char* first(const sif_file_check_t* c) {
  return c && c->n_problems ? c->problems[0].message : "(none)";
}

static const sif_file_stats_t* column_stats(
  const sif_file_check_t* c, const char* table, const char* column) {
  for (uint32_t t = 0; c && c->info && t < c->info->n_tables; t++) {
    const sif_file_table_t* tb = &c->info->tables[t];
    if (strcmp(tb->name, table) != 0)
      continue;
    for (uint32_t k = 0; k < tb->n_columns; k++)
      if (strcmp(tb->columns[k].name, column) == 0)
        return &c->column_stats[t][k];
  }
  return NULL;
}

/* A sound file: no errors, no warnings. */
static void expect_sound(const sif_file_check_t* c, const char* what) {
  CHECK(c != NULL, "%s: no check", what);
  CHECK(c && c->info, "%s: not described", what);
  CHECK(c && c->n_errors == 0 && c->n_warnings == 0,
    "%s: %llu errors, %llu warnings: %s", what,
    c ? (unsigned long long)c->n_errors : 0ull,
    c ? (unsigned long long)c->n_warnings : 0ull, first(c));
}

/* A file with an error that says @p needle. */
static void expect_error(
  const sif_file_check_t* c, const char* needle, const char* what) {
  CHECK(c != NULL, "%s: no check", what);
  CHECK(c && c->n_errors > 0, "%s: no error", what);
  CHECK(said(c, needle), "%s: no '%s' in: %s", what, needle, first(c));
}

static void write_text(const char* path, const char* text) {
  FILE* f = fopen(path, "w");
  if (f) {
    fputs(text, f);
    fclose(f);
  }
}

/* Copy @p from to @p to, cut at @p keep bytes (-1 for all of it). */
static void copy_file(const char* from, const char* to, long keep) {
  FILE* in = fopen(from, "rb");
  FILE* out = fopen(to, "wb");
  if (in && out) {
    char buf[8192];
    size_t n;
    long written = 0;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
      if (keep >= 0 && written + (long)n > keep)
        n = (size_t)(keep - written);
      fwrite(buf, 1, n, out);
      written += (long)n;
      if (keep >= 0 && written >= keep)
        break;
    }
  }
  if (in)
    fclose(in);
  if (out)
    fclose(out);
}

/* Flip one bit at @p offset. */
static void flip_bit(const char* path, long offset) {
  FILE* f = fopen(path, "r+b");
  if (!f)
    return;
  fseek(f, offset, SEEK_SET);
  const int c = fgetc(f);
  fseek(f, offset, SEEK_SET);
  fputc(c ^ 0x01, f);
  fclose(f);
}

static long file_size(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f)
    return 0;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fclose(f);
  return n;
}

/* --- a file that cannot be read at all --- */

static void test_unreadable(void) {
  printf("unreadable files\n");
  sif_file_check_t* c = sif_file_check("fc_missing.txt", 0);
  CHECK(c && !c->info, "a missing file: described");
  expect_error(
    c, "fc_missing.txt: No such file or directory", "a missing file");
  sif_file_check_free(c);

  write_text("fc_text.txt", "1 2 3\n");
  c = sif_file_check("fc_text.txt", SIF_FILE_FORMAT_XFIELD);
  CHECK(c && !c->info, "a text file as .xfield: described");
  expect_error(c, "not an .xfield file", "a text file as .xfield");
  sif_file_check_free(c);
  remove("fc_text.txt");
}

/* --- ASCII --- */

static void test_ascii(void) {
  printf("ASCII\n");
  write_text("fc_table.csv", "x,y,name\n1,2,a\n3,4,b\n");
  sif_file_check_t* c = sif_file_check("fc_table.csv", 0);
  expect_sound(c, "a CSV table");
  const sif_file_stats_t* s = column_stats(c, "", "y");
  CHECK(s && s->n == 2 && s->min == 2 && s->max == 4, "y: not ranged 2 to 4");
  s = column_stats(c, "", "name");
  CHECK(s && s->n == 2, "name: not counted");
  sif_file_check_free(c);

  /* Every bad row, each with its line. */
  write_text("fc_bad.txt", "1 2 3\n4 5\n7 8 abc\n1,2,3\n");
  c = sif_file_check("fc_bad.txt", 0);
  expect_error(c, "fc_bad.txt:2: 2 fields (expected 3", "a short row");
  expect_error(c, "fc_bad.txt:3: column 3 (3): 'abc'", "not a number");
  expect_error(c, "(comma-separated in places?)", "the delimiter hint");
  CHECK(c && c->n_errors == 4, "a bad table: %llu errors, expected 4",
    c ? (unsigned long long)c->n_errors : 0ull);
  sif_file_check_free(c);

  /* A catalogue sif wrote: checked as one. */
  sif_catalogue_t* cat = sif_catalogue_alloc(2);
  if (cat) {
    sif_catalogue_append(cat, 1, 2, 3, 4);
    sif_catalogue_append(cat, 5, 6, 7, 8);
    CHECK(sif_catalogue_write_ascii("fc_cat.txt", cat) == SIF_OK,
      "catalogue write failed");
    sif_catalogue_free(cat);
  }
  c = sif_file_check("fc_cat.txt", 0);
  expect_sound(c, "an ASCII catalogue");
  CHECK(c && strcmp(c->content, "sif catalogue") == 0,
    "an ASCII catalogue: checked as '%s'", c ? c->content : "");
  sif_file_check_free(c);

  /* Its contents wrong: a radius not positive, a count that disagrees. */
  write_text("fc_cat.txt", "#n=3\n#cx cy cz r\n1 2 3 4\n5 6 7 -1\n");
  c = sif_file_check("fc_cat.txt", 0);
  expect_error(c, "expected every radius > 0", "a negative radius");
  expect_error(c, "n = 3, the table has 2 rows", "a wrong count");
  sif_file_check_free(c);

  write_text("fc_cat.txt", "#n=1\n#ra dec z r\n10 95 0.1 4\n");
  c = sif_file_check("fc_cat.txt", 0);
  expect_error(c, "expected within [-90, 90]", "a declination past the pole");
  sif_file_check_free(c);

  remove("fc_table.csv");
  remove("fc_bad.txt");
  remove("fc_cat.txt");
}

/* --- .xfield and .xgrid --- */

static void test_xfield(void) {
  printf(".xfield\n");
  const char* path = "fc.xfield";
  sif_field_t* f = sif_field_alloc(100);
  if (!f || sif_field_reserve_positions(f) != SIF_OK ||
      sif_field_reserve_weights(f) != SIF_OK) {
    CHECK(false, "field allocation failed");
    sif_field_free(f);
    return;
  }
  for (int i = 0; i < 100; i++) {
    f->x[i] = (sif_real)(i % 10);
    f->y[i] = (sif_real)(i / 10);
    f->z[i] = 5;
    f->weights[i] = 1;
  }
  CHECK(sif_field_write(path, f, 10.0) == SIF_OK, "write failed");

  sif_file_check_t* c = sif_file_check(path, 0);
  expect_sound(c, "an .xfield");
  const sif_file_stats_t* s = column_stats(c, "particles", "x");
  CHECK(s && s->n == 100 && s->min == 0 && s->max == 9, "x: not ranged");
  s = column_stats(c, "particles", "w");
  CHECK(s && s->n == 100 && s->min == 1, "w: not ranged");
  sif_file_check_free(c);

  /* A flipped bit: the checksum. */
  flip_bit(path, 64 + 40);
  c = sif_file_check(path, 0);
  expect_error(c, "(expected", "a corrupt .xfield");
  expect_error(c, "corrupt)", "a corrupt .xfield");
  sif_file_check_free(c);

  /* Cut short. */
  CHECK(truncate(path, 1000) == 0, "truncate failed");
  c = sif_file_check(path, 0);
  expect_error(c, "1000 bytes (expected", "a truncated .xfield");
  expect_error(c, "truncated", "a truncated .xfield");
  sif_file_check_free(c);

  /* Intact, and a particle outside the box. */
  f->x[7] = 20;
  CHECK(sif_field_write(path, f, 10.0) == SIF_OK, "write failed");
  c = sif_file_check(path, 0);
  expect_error(c, "x: 1 of 100 outside [0, 10)", "a particle outside");
  sif_file_check_free(c);

  sif_field_free(f);
  remove(path);
}

static void test_xgrid(void) {
  printf(".xgrid\n");
  const char* path = "fc.xgrid";
  sif_grid_t* g = sif_grid_alloc(4, 20);
  if (!g) {
    CHECK(false, "grid allocation failed");
    return;
  }
  for (uint64_t i = 0; i < g->total_cells; i++)
    g->values[i] = (sif_real)i;
  g->content = SIF_GRID_DENSITY;
  CHECK(sif_grid_write(path, g) == SIF_OK, "write failed");

  sif_file_check_t* c = sif_file_check(path, 0);
  expect_sound(c, "an .xgrid");
  CHECK(c && c->info && c->info->n_arrays == 1 && c->array_stats[0].n == 64 &&
          c->array_stats[0].max == 63,
    "an .xgrid: the cells not ranged");
  sif_file_check_free(c);

  /* A negative density. */
  g->values[5] = -1;
  CHECK(sif_grid_write(path, g) == SIF_OK, "write failed");
  c = sif_file_check(path, 0);
  expect_error(c, "smallest density -1 (expected >= 0)", "a negative density");
  sif_file_check_free(c);

  flip_bit(path, 64 + 8);
  c = sif_file_check(path, 0);
  expect_error(c, "corrupt)", "a corrupt .xgrid");
  sif_file_check_free(c);

  sif_grid_free(g);
  remove(path);
}

/* --- GADGET --- */

static void test_gadget(void) {
  printf("GADGET\n");
  const char* sound[] = {GADGET "f1_legacy/snap_005", GADGET "f1_g4/snap_005",
    GADGET "f1_g4_u32/snap_005", GADGET "f2_swapped/snap_005",
    GADGET "hdf5_g4/snap_005", GADGET "hdf5_legacy/snap_005"};
  for (size_t i = 0; i < sizeof(sound) / sizeof(sound[0]); i++) {
#ifndef SIF_HAVE_HDF5
    if (strstr(sound[i], "hdf5"))
      continue;
#endif
    sif_file_check_t* c = sif_file_check(sound[i], 0);
    expect_sound(c, sound[i]);
    CHECK(c && strcmp(c->content, "GADGET snapshot") == 0,
      "%s: checked as '%s'", sound[i], c ? c->content : "");
    CHECK(sif_gadget_is_snapshot(sound[i]), "%s: not taken for a snapshot",
      sound[i]);
    sif_file_check_free(c);
  }
  CHECK(!sif_gadget_is_snapshot(GADGET "make_fixtures.py"),
    "a script taken for a snapshot");
  CHECK(!sif_gadget_is_snapshot(GADGET "nothing/snap_005"),
    "a missing path taken for a snapshot");

  /* Every block, IDs included. */
  sif_file_check_t* c = sif_file_check(GADGET "f1_legacy/snap_005", 0);
  const sif_file_stats_t* s = column_stats(c, "PartType1", "ID");
  CHECK(s && s->n == 100 && s->min == 1000000 && s->max == 1000099,
    "PartType1 ID: not ranged 1000000 to 1000099");
  s = column_stats(c, "PartType1", "POS");
  CHECK(s && s->n == 300, "PartType1 POS: %llu values, expected 300",
    s ? (unsigned long long)s->n : 0ull);
  sif_file_check_free(c);

  /* Three files, the middle one cut short and the last missing: both are
   * errors, and the first file is still read. */
  copy_file(GADGET "f1_legacy/snap_005.0", "fc_snap.0", -1);
  copy_file(GADGET "f1_legacy/snap_005.1", "fc_snap.1", 600);
  remove("fc_snap.2");
  c = sif_file_check("fc_snap", 0);
  expect_error(c, "fc_snap.1: record", "a snapshot cut short");
  expect_error(c, "fc_snap.2", "a snapshot missing a file");
  s = column_stats(c, "PartType0", "POS");
  CHECK(s && s->n > 0, "the first file was not read");
  sif_file_check_free(c);
  remove("fc_snap.0");
  remove("fc_snap.1");
}

/* --- FITS --- */

static void test_fits(void) {
  printf("FITS\n");
#if defined(SIF_HAVE_FITS)
  const char* fixture = SIF_TEST_DATA_DIR "/fits/catalogue.fits";
  sif_file_check_t* c = sif_file_check(fixture, 0);
  expect_sound(c, "the FITS fixture");
  const sif_file_stats_t* s = column_stats(c, "GALAXIES", "POS");
  CHECK(s && s->n == 300, "POS: %llu values, expected 300",
    s ? (unsigned long long)s->n : 0ull);
  s = column_stats(c, "GALAXIES", "ZNULL");
  CHECK(s && s->n_nonfinite > 0, "ZNULL: its undefined values not counted");
  s = column_stats(c, "GALAXIES", "NAME");
  CHECK(s && s->n == 100, "NAME: the strings not read");
  sif_file_check_free(c);

  /* Cut inside the last table's data: every header still reads. */
  copy_file(fixture, "fc_cut.fits", file_size(fixture) - 100);
  c = sif_file_check("fc_cut.fits", 0);
  expect_error(c, "not a multiple of 2880 (truncated)", "a truncated FITS");
  sif_file_check_free(c);
  remove("fc_cut.fits");

  /* A catalogue sif wrote. */
  sif_catalogue_t* cat = sif_catalogue_alloc(2);
  if (cat) {
    sif_catalogue_append(cat, 1, 2, 3, 4);
    CHECK(sif_catalogue_write_fits("fc_cat.fits", cat) == SIF_OK,
      "FITS write failed");
    sif_catalogue_free(cat);
  }
  c = sif_file_check("fc_cat.fits", 0);
  expect_sound(c, "a FITS catalogue");
  CHECK(c && strcmp(c->content, "sif catalogue") == 0,
    "a FITS catalogue: checked as '%s'", c ? c->content : "");
  sif_file_check_free(c);
  remove("fc_cat.fits");
#endif
}

/* --- HDF5 --- */

static void test_hdf5(void) {
  printf("HDF5\n");
#if defined(SIF_HAVE_HDF5)
  sif_catalogue_t* cat = sif_catalogue_alloc(4);
  if (!cat)
    return;
  for (int i = 0; i < 4; i++)
    sif_catalogue_append(cat, (sif_real)i, 1, 2, (sif_real)(i + 1));
  remove("fc_cat.h5");
  CHECK(
    sif_catalogue_write_hdf5("fc_cat.h5", cat) == SIF_OK, "HDF5 write failed");
  sif_catalogue_free(cat);

  sif_file_check_t* c = sif_file_check("fc_cat.h5", 0);
  expect_sound(c, "an HDF5 catalogue");
  CHECK(c && strcmp(c->content, "sif catalogue") == 0,
    "an HDF5 catalogue: checked as '%s'", c ? c->content : "");
  const sif_file_stats_t* s = column_stats(c, "catalogue", "radii");
  CHECK(
    s && s->n == 4 && s->min == 1 && s->max == 4, "radii: not ranged 1 to 4");
  sif_file_check_free(c);

  /* Cut short: HDF5's own diagnosis. */
  CHECK(truncate("fc_cat.h5", 1000) == 0, "truncate failed");
  c = sif_file_check("fc_cat.h5", 0);
  expect_error(c, "(truncated)", "a truncated HDF5 file");
  sif_file_check_free(c);
  remove("fc_cat.h5");
#endif
}

/* --- printing --- */

static void test_print(void) {
  printf("printing\n");
  write_text("fc_print.txt", "1 2\n3 x\n");
  sif_file_check_t* c = sif_file_check("fc_print.txt", 0);
  FILE* out = tmpfile();
  if (c && out) {
    CHECK(sif_file_check_print(c, out) == SIF_OK, "print failed");
    char text[4096] = {0};
    rewind(out);
    text[fread(text, 1, sizeof(text) - 1, out)] = '\0';
    CHECK(strstr(text, "non-finite"), "print: no statistics: %s", text);
    CHECK(strstr(text, "problems\n  error    fc_print.txt:2"),
      "print: no problem: %s", text);
    CHECK(strstr(text, "result    1 error, 0 warnings"), "print: %s", text);
  }
  if (out)
    fclose(out);
  sif_file_check_free(c);
  remove("fc_print.txt");
}

int main(void) {
  if (sif_system_init(SIF_CONFIG_QUIET) != SIF_OK) {
    printf("FAIL: sif_system_init\n");
    return 1;
  }

  test_unreadable();
  test_ascii();
  test_xfield();
  test_xgrid();
  test_gadget();
  test_fits();
  test_hdf5();
  test_print();

  sif_system_finalise();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
