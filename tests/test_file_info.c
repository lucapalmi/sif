/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * File descriptions: every format, described with the one model -- metadata,
 * tables of columns, arrays -- from files sif writes here and the GADGET
 * fixtures in tests/data. What matters is that each format lands in the same
 * slots: the table's name, its rows, the columns' names and types, and the
 * metadata under the file's own keys.
 */

#include "sif/io/file_info.h"

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/io/catalogue_io.h"
#include "sif/io/field_io.h"
#include "sif/io/fits_io.h"
#include "sif/io/grid_io.h"
#include "sif/io/hdf5_io.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static const char* const REAL_TYPE =
  sizeof(sif_real) == 8 ? "float64" : "float32";

/* --- lookups --- */

static const sif_file_meta_t* meta_find(
  const sif_file_meta_t* meta, uint32_t n, const char* key) {
  for (uint32_t i = 0; i < n; i++)
    if (strcmp(meta[i].key, key) == 0)
      return &meta[i];
  return NULL;
}

static const sif_file_table_t* table_find(
  const sif_file_info_t* info, const char* name) {
  for (uint32_t t = 0; t < info->n_tables; t++)
    if (strcmp(info->tables[t].name, name) == 0)
      return &info->tables[t];
  return NULL;
}

static const sif_file_column_t* column_find(
  const sif_file_table_t* tb, const char* name) {
  for (uint32_t c = 0; tb && c < tb->n_columns; c++)
    if (strcmp(tb->columns[c].name, name) == 0)
      return &tb->columns[c];
  return NULL;
}

/* A column has the type and width given. */
static void expect_column(const sif_file_table_t* tb, const char* name,
  const char* type, uint64_t width, const char* where) {
  const sif_file_column_t* c = column_find(tb, name);
  CHECK(c != NULL, "%s: no column %s", where, name);
  if (!c)
    return;
  CHECK(strcmp(c->type, type) == 0, "%s: %s is %s, expected %s", where, name,
    c->type, type);
  CHECK(c->width == width, "%s: %s has width %llu, expected %llu", where, name,
    (unsigned long long)c->width, (unsigned long long)width);
}

static void write_text(const char* path, const char* text) {
  FILE* f = fopen(path, "w");
  if (f) {
    fputs(text, f);
    fclose(f);
  }
}

/* --- formats --- */

static void test_format_names(void) {
  printf("format names\n");
  for (int k = SIF_FILE_FORMAT_ASCII; k <= SIF_FILE_FORMAT_CHECKPOINT; k++) {
    sif_file_format_t back = SIF_FILE_FORMAT_UNKNOWN;
    const char* name = sif_file_format_name((sif_file_format_t)k);
    CHECK(sif_file_format_parse(name, &back) == SIF_OK && (int)back == k,
      "%s does not parse back", name);
  }
  sif_file_format_t f;
  CHECK(sif_file_format_parse("unknown", &f) != SIF_OK, "'unknown' parses");
  CHECK(sif_file_format_parse("csv", &f) != SIF_OK, "'csv' parses");
}

/* --- ASCII --- */

static void test_ascii(void) {
  printf("ASCII\n");

  /* As sif writes a catalogue: key=value lines, then the column names. Blank
   * and comment lines among the rows are not rows. */
  write_text("fi_cat.txt", "# voids\n"
                           "#n=3\n"
                           "#box_length=1000.5\n"
                           "#variant=\"excursion-peak\"\n"
                           "#cx cy cz r\n"
                           "1 2 3 4\n"
                           "5 6 7 8\n"
                           "\n"
                           "# a comment\n"
                           "  9 10 11 12\n");
  CHECK(sif_file_format_detect("fi_cat.txt") == SIF_FILE_FORMAT_ASCII,
    "a text file is not detected as ASCII");
  sif_file_info_t* info = sif_file_info_read("fi_cat.txt", 0);
  CHECK(info != NULL, "a catalogue: not described");
  if (info) {
    CHECK(info->n_tables == 1 && info->tables[0].n_rows == 3,
      "a catalogue: %u tables, %llu rows", info->n_tables,
      info->n_tables ? (unsigned long long)info->tables[0].n_rows : 0ull);
    const sif_file_table_t* tb = table_find(info, "");
    expect_column(tb, "cx", "number", 1, "a catalogue");
    expect_column(tb, "r", "number", 1, "a catalogue");
    const sif_file_meta_t* m = meta_find(info->meta, info->n_meta, "n");
    CHECK(m && m->kind == SIF_FILE_META_INT && m->i == 3, "#n=3: not an int 3");
    m = meta_find(info->meta, info->n_meta, "box_length");
    CHECK(m && m->kind == SIF_FILE_META_REAL && m->r == 1000.5,
      "#box_length: not a real 1000.5");
    m = meta_find(info->meta, info->n_meta, "variant");
    CHECK(m && m->kind == SIF_FILE_META_STRING &&
            strcmp(m->s, "excursion-peak") == 0,
      "#variant: not the unquoted string");
    CHECK(info->size_bytes > 0, "a catalogue: no size");
    sif_file_info_free(info);
  }

  /* A header line names the columns; the first row says which are numbers. */
  write_text("fi_cat.csv", "ra,dec,z,name\n10.5,-3.2,0.1,a\n11,4,0.2,b\n");
  info = sif_file_info_read("fi_cat.csv", SIF_FILE_FORMAT_ASCII);
  CHECK(info != NULL, "a CSV file: not described");
  if (info) {
    CHECK(strstr(info->detail, "comma-separated") &&
            strstr(info->detail, "1 header line"),
      "a CSV file: detail '%s'", info->detail);
    const sif_file_table_t* tb = table_find(info, "");
    CHECK(tb && tb->n_rows == 2 && tb->n_columns == 4,
      "a CSV file: not 2 rows of 4 columns");
    expect_column(tb, "dec", "number", 1, "a CSV file");
    expect_column(tb, "name", "text", 1, "a CSV file");
    sif_file_info_free(info);
  }

  /* No names anywhere: numbered from 1, as the check numbers them. */
  write_text("fi_plain.txt", "1 2 3\n4 5 6\n");
  info = sif_file_info_read("fi_plain.txt", 0);
  if (info) {
    const sif_file_table_t* tb = table_find(info, "");
    expect_column(tb, "1", "number", 1, "unnamed columns");
    expect_column(tb, "3", "number", 1, "unnamed columns");
    sif_file_info_free(info);
  }

  /* No rows: a table of none. */
  write_text("fi_empty.txt", "# x y z\n");
  info = sif_file_info_read("fi_empty.txt", 0);
  CHECK(info && info->n_tables == 1 && info->tables[0].n_rows == 0,
    "a file without rows: not a table of none");
  sif_file_info_free(info);

  sif_error_clear();
  CHECK(sif_file_info_read("fi_missing.txt", 0) == NULL,
    "a missing file is described");
  CHECK(
    error_has("fi_missing.txt"), "a missing file: '%s'", sif_error_message());

  remove("fi_cat.txt");
  remove("fi_cat.csv");
  remove("fi_plain.txt");
  remove("fi_empty.txt");
}

/* --- .xfield and .xgrid --- */

static void test_xfield_xgrid(void) {
  printf(".xfield and .xgrid\n");

  sif_field_t* f = sif_field_alloc(5);
  CHECK(f && sif_field_reserve_positions(f) == SIF_OK &&
          sif_field_reserve_weights(f) == SIF_OK,
    "field allocation failed");
  if (f) {
    for (int i = 0; i < 5; i++) {
      f->x[i] = f->y[i] = f->z[i] = (sif_real)i;
      f->weights[i] = 1;
    }
    CHECK(sif_field_write("fi.xfield", f, 50.0) == SIF_OK, "write failed");
    sif_field_free(f);
  }

  CHECK(sif_file_format_detect("fi.xfield") == SIF_FILE_FORMAT_XFIELD,
    ".xfield: not detected");
  sif_file_info_t* info = sif_file_info_read("fi.xfield", 0);
  CHECK(info != NULL, ".xfield: not described");
  if (info) {
    const sif_file_table_t* tb = table_find(info, "particles");
    CHECK(tb && tb->n_rows == 5 && tb->n_columns == 4,
      ".xfield: not 5 particles with x y z w");
    expect_column(tb, "w", REAL_TYPE, 1, ".xfield");
    CHECK(!column_find(tb, "vx"), ".xfield: velocities it does not have");
    const sif_file_meta_t* m =
      meta_find(info->meta, info->n_meta, "box_length");
    CHECK(m && m->kind == SIF_FILE_META_REAL && m->r == 50.0,
      ".xfield: box_length");
    CHECK(info->size_bytes == 64 + 5 * 4 * sizeof(sif_real),
      ".xfield: %llu bytes", (unsigned long long)info->size_bytes);
    sif_file_info_free(info);
  }

  /* Forced to be what it is not, it is refused. */
  CHECK(sif_file_info_read("fi.xfield", SIF_FILE_FORMAT_XGRID) == NULL,
    ".xfield read as .xgrid");
  CHECK(sif_file_info_read("fi.xfield", SIF_FILE_FORMAT_CHECKPOINT) == NULL,
    ".xfield read as a checkpoint");

  sif_grid_t* g = sif_grid_alloc(4, 20);
  CHECK(g != NULL, "grid allocation failed");
  if (g) {
    CHECK(sif_grid_write("fi.xgrid", g) == SIF_OK, "grid write failed");
    sif_grid_free(g);
  }
  CHECK(sif_file_format_detect("fi.xgrid") == SIF_FILE_FORMAT_XGRID,
    ".xgrid: not detected");
  info = sif_file_info_read("fi.xgrid", 0);
  CHECK(info && info->n_tables == 0 && info->n_arrays == 1,
    ".xgrid: not one array");
  if (info && info->n_arrays == 1) {
    const sif_file_array_t* a = &info->arrays[0];
    CHECK(strcmp(a->name, "cells") == 0 && a->rank == 3 && a->dims[0] == 4 &&
            a->dims[2] == 4 && strcmp(a->type, REAL_TYPE) == 0,
      ".xgrid: the cells are %s, rank %u", a->name, a->rank);
    const sif_file_meta_t* m = meta_find(info->meta, info->n_meta, "content");
    CHECK(m && strcmp(m->s, "empty") == 0, ".xgrid: content");
  }
  sif_file_info_free(info);

  /* The same description, printed. */
  info = sif_file_info_read("fi.xfield", 0);
  FILE* out = tmpfile();
  if (info && out) {
    CHECK(sif_file_info_print(info, out) == SIF_OK, "print failed");
    char text[2048] = {0};
    rewind(out);
    const size_t n = fread(text, 1, sizeof(text) - 1, out);
    text[n] = '\0';
    CHECK(strstr(text, "format    .xfield, version 2"), "print: %s", text);
    CHECK(
      strstr(text, "table particles, 5 rows, 4 columns"), "print: %s", text);
    CHECK(strstr(text, "box_length"), "print: no metadata");
  }
  if (out)
    fclose(out);
  sif_file_info_free(info);

  remove("fi.xfield");
  remove("fi.xgrid");
}

/* --- GADGET --- */

static void test_gadget(void) {
  printf("GADGET\n");
  char path[1024];

  /* SnapFormat 1, three files, a MassTable entry for type 1 only. */
  snprintf(
    path, sizeof(path), "%s/gadget/f1_legacy/snap_005", SIF_TEST_DATA_DIR);
  CHECK(sif_file_format_detect(path) == SIF_FILE_FORMAT_GADGET,
    "a snapshot's base name: not detected");
  sif_file_info_t* info = sif_file_info_read(path, 0);
  CHECK(info != NULL, "SnapFormat 1: not described");
  if (info) {
    CHECK(info->n_files == 3, "SnapFormat 1: %u files", info->n_files);
    CHECK(info->n_tables == 3, "SnapFormat 1: %u tables", info->n_tables);
    const sif_file_table_t* t0 = table_find(info, "PartType0");
    const sif_file_table_t* t1 = table_find(info, "PartType1");
    CHECK(t0 && t0->n_rows == 40, "PartType0: not 40 rows");
    CHECK(t1 && t1->n_rows == 100, "PartType1: not 100 rows");
    expect_column(t1, "POS", "float32", 3, "PartType1");
    expect_column(t1, "ID", "uint32", 1, "PartType1");
    CHECK(!column_find(t1, "MASS"), "PartType1: a MASS column, with a "
                                    "MassTable entry");
    expect_column(t0, "MASS", "float32", 1, "PartType0");
    CHECK(t1 && meta_find(t1->meta, t1->n_meta, "MassTable"),
      "PartType1: no MassTable");
    const sif_file_meta_t* m = meta_find(info->meta, info->n_meta, "BoxSize");
    CHECK(m && m->r == 10000.0, "SnapFormat 1: BoxSize");
    sif_file_info_free(info);
  }

#if defined(SIF_HAVE_HDF5)
  /* The legacy HDF5 layout names its groups ParticleTypeN, and so does the
   * description. */
  snprintf(path, sizeof(path), "%s/gadget/hdf5_legacy/snap_005.hdf5",
    SIF_TEST_DATA_DIR);
  info = sif_file_info_read(path, 0);
  CHECK(info != NULL, "HDF5 snapshot: not described");
  if (info) {
    const sif_file_table_t* tb = table_find(info, "ParticleType0");
    CHECK(tb && tb->n_rows == 40, "HDF5 snapshot: no ParticleType0 of 40");
    expect_column(tb, "Coordinates", "float64", 3, "ParticleType0");
    sif_file_info_free(info);
  }

  /* Read as plain HDF5, the same file is its groups and attributes. */
  info = sif_file_info_read(path, SIF_FILE_FORMAT_HDF5);
  CHECK(info != NULL, "HDF5 snapshot as HDF5: not described");
  if (info) {
    const sif_file_meta_t* m =
      meta_find(info->meta, info->n_meta, "Header/NumPart_Total");
    CHECK(m && m->kind == SIF_FILE_META_STRING &&
            strncmp(m->s, "40, 100, 30", 11) == 0,
      "Header/NumPart_Total: %s", m && m->s ? m->s : "missing");
    CHECK(table_find(info, "ParticleType2") != NULL,
      "HDF5 snapshot as HDF5: no ParticleType2 table");
    sif_file_info_free(info);
  }

  /* Several files, the first without particles of type 2: their datasets
   * come from the next file that has them. */
  snprintf(path, sizeof(path), "%s/gadget/hdf5_g4/snap_005", SIF_TEST_DATA_DIR);
  info = sif_file_info_read(path, 0);
  CHECK(info && info->n_files == 3, "HDF5 multi-file: not 3 files");
  if (info) {
    const sif_file_table_t* tb = table_find(info, "PartType2");
    CHECK(tb && tb->n_columns > 0, "PartType2: no columns");
    sif_file_info_free(info);
  }
#endif
}

/* --- sif's own products --- */

static void test_catalogues(void) {
  printf("catalogues\n");
  sif_catalogue_t* cat = sif_catalogue_alloc(4);
  CHECK(cat != NULL, "catalogue allocation failed");
  if (!cat)
    return;
  sif_catalogue_append(cat, 1, 2, 3, 4);
  sif_catalogue_append(cat, 5, 6, 7, 8);
  CHECK(sif_catalogue_meta_string_set(cat, "finder", "exodus") == SIF_OK,
    "metadata failed");

#if defined(SIF_HAVE_HDF5)
  CHECK(
    sif_catalogue_write_hdf5("fi_cat.h5", cat) == SIF_OK, "HDF5 write failed");
  sif_file_info_t* info = sif_file_info_read("fi_cat.h5", 0);
  CHECK(info != NULL, "HDF5 catalogue: not described");
  if (info) {
    CHECK(info->format == SIF_FILE_FORMAT_HDF5, "HDF5 catalogue: format %d",
      info->format);
    const sif_file_table_t* tb = table_find(info, "catalogue");
    CHECK(tb && tb->n_rows == 2, "HDF5 catalogue: no catalogue of 2 rows");
    expect_column(tb, "centres", REAL_TYPE, 3, "HDF5 catalogue");
    const sif_file_meta_t* m =
      tb ? meta_find(tb->meta, tb->n_meta, "finder") : NULL;
    CHECK(m && strcmp(m->s, "exodus") == 0, "HDF5 catalogue: no finder");
    CHECK(meta_find(info->meta, info->n_meta, "sif_format") != NULL,
      "HDF5 catalogue: no sif_format");
    sif_file_info_free(info);
  }
  remove("fi_cat.h5");
#endif

#if defined(SIF_HAVE_FITS)
  CHECK(sif_catalogue_write_fits("fi_cat.fits", cat) == SIF_OK,
    "FITS write failed");
  sif_file_info_t* fits = sif_file_info_read("fi_cat.fits", 0);
  CHECK(fits != NULL, "FITS catalogue: not described");
  if (fits) {
    const sif_file_table_t* tb = table_find(fits, "VOIDS");
    CHECK(tb && tb->n_rows == 2, "FITS catalogue: no VOIDS of 2 rows");
    expect_column(tb, "CX", REAL_TYPE, 1, "FITS catalogue");
    /* The keywords that lay the table out are its columns already. */
    CHECK(tb && !meta_find(tb->meta, tb->n_meta, "TTYPE1") &&
            !meta_find(tb->meta, tb->n_meta, "NAXIS2"),
      "FITS catalogue: structural keywords as metadata");
    CHECK(tb && meta_find(tb->meta, tb->n_meta, "FINDER"),
      "FITS catalogue: no FINDER");
    sif_file_info_free(fits);
  }
  remove("fi_cat.fits");
#endif

  sif_catalogue_free(cat);
}

int main(void) {
  if (sif_system_init(SIF_CONFIG_QUIET) != SIF_OK) {
    printf("FAIL: sif_system_init\n");
    return 1;
  }

  test_format_names();
  test_ascii();
  test_xfield_xgrid();
  test_gadget();
  test_catalogues();

  sif_system_finalise();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
