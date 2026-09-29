/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The HDF5 catalogue file, in whichever build this is.
 *
 * With HDF5: every product round-trips bit for bit; a file can hold any
 * subset of them; each writer replaces its own group and leaves the rest;
 * row-count mismatches are written anyway; and files sif did not write --
 * text, someone else's HDF5, a newer layout -- are refused rather than
 * overwritten or guessed at.
 *
 * Without HDF5: every writer succeeds by dumping plain text beside the
 * requested path, never at it, and every reader refuses.
 */
#include "sif/core/system.h"
#include "sif/io/catalog_io.h"
#include "sif/io/hdf5_io.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalog.h"
#include "sif/structures/field.h"
#include "sif/structures/size_function.h"

#include "measure/profiles_internal.h"
#include "structures/results_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef SIF_HAVE_HDF5
#  include <hdf5.h>
#endif

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define N_VOIDS 37
#define N_BINS  12

static const char* PATH = "test_hdf5.h5";

static int file_exists(const char* p) {
  struct stat st;
  return stat(p, &st) == 0;
}

static void remove_all(void) {
  const char* suffixes[] = {"", ".catalog.txt", ".density_profiles.txt",
    ".velocity_profiles.txt", ".size_function.txt", ".attributes.txt"};
  char p[256];
  for (size_t i = 0; i < sizeof(suffixes) / sizeof(*suffixes); i++) {
    snprintf(p, sizeof(p), "%s%s", PATH, suffixes[i]);
    remove(p);
  }
}

/* --- fixtures: values that need most of the mantissa to survive --- */

static sif_catalog_t* make_catalog(uint64_t n, int footprint) {
  sif_catalog_t* cat = sif_catalog_alloc(n ? n : 1);
  for (uint64_t i = 0; i < n; i++) {
    sif_catalog_append(cat, (sif_real)(1234.5678901234 + 0.1 * (double)i),
      (sif_real)(0.1234567890123 * (double)(i + 1)),
      (sif_real)(9.87654321e-3 * (double)i), (sif_real)(3.14159265 + i));
  }
  if (footprint) {
    sif_catalog_reserve_footprint(cat);
    for (uint64_t i = 0; i < n; i++) {
      cat->footprint[i] = (sif_real)(1.0 / (double)(i + 1));
      cat->footprint_shell[i] = (sif_real)(0.5 / (double)(i + 1));
    }
  }
  return cat;
}

static sif_density_profiles_t* make_density(uint64_t n) {
  sif_density_profiles_t* d =
    sif__density_profiles_alloc(n, N_BINS, (sif_real)3.0, true);
  for (uint32_t b = 0; b <= N_BINS; b++)
    d->r_edges[b] = (sif_real)(3.0 * b / N_BINS);
  for (uint64_t i = 0; i < n * N_BINS; i++)
    d->profiles[i] = (sif_real)(-0.987654321 + 1e-3 * (double)i);
  return d;
}

static sif_velocity_profiles_t* make_velocity(uint64_t n) {
  sif_velocity_profiles_t* v =
    sif__velocity_profiles_alloc(n, N_BINS, (sif_real)2.5);
  for (uint32_t b = 0; b <= N_BINS; b++)
    v->r_edges[b] = (sif_real)(2.5 * b / N_BINS);
  for (uint64_t i = 0; i < n * N_BINS; i++)
    v->v_rad[i] = (sif_real)(123.456789 - 0.37 * (double)i);
  return v;
}

static sif_size_function_t* make_vsf(void) {
  sif_size_function_t* f = sif__size_function_alloc(N_BINS);
  f->options = SIF_VSF_BIN_LINEAR;
  f->r_min = (sif_real)5.0;
  f->r_max = (sif_real)65.0;
  for (uint32_t b = 0; b <= N_BINS; b++)
    f->r_edges[b] = (sif_real)(5.0 + 5.0 * b);
  for (uint32_t b = 0; b < N_BINS; b++) {
    f->r_centers[b] = (sif_real)(7.5 + 5.0 * b);
    f->counts[b] = (uint64_t)1 << (20 + b); /* past 32 bits by the end */
    f->vsf[b] = (sif_real)(1.23456789e-5 / (double)(b + 1));
    f->err[b] = (sif_real)(2.3456789e-7 / (double)(b + 1));
  }
  return f;
}

/* --- comparisons, bit for bit --- */

static int same_catalog(const sif_catalog_t* a, const sif_catalog_t* b) {
  if (!a || !b || a->n_voids != b->n_voids)
    return 0;
  if ((a->footprint == NULL) != (b->footprint == NULL))
    return 0;
  for (uint64_t i = 0; i < a->n_voids; i++) {
    if (a->cx[i] != b->cx[i] || a->cy[i] != b->cy[i] || a->cz[i] != b->cz[i] ||
        a->radii[i] != b->radii[i])
      return 0;
    if (a->footprint && (a->footprint[i] != b->footprint[i] ||
                          a->footprint_shell[i] != b->footprint_shell[i]))
      return 0;
  }
  return 1;
}

static int same_reals(const sif_real* a, const sif_real* b, uint64_t n) {
  return memcmp(a, b, n * sizeof(sif_real)) == 0;
}

#ifdef SIF_HAVE_HDF5

/* A plain HDF5 file, with nothing of sif's in it. */
static void write_foreign_hdf5(const char* path) {
  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t g = H5Gcreate2(f, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Gclose(g);
  H5Fclose(f);
}

/* A snapshot of someone else's, as IllustrisTNG writes one: particle i of
 * the whole snapshot at (i, 2i, 3i) + 0.5 kpc/h, mass 1 + i/1024, an integer
 * ID, rows [first, first + n) of it in this file. */
static void write_snapshot(const char* path, int first, int n) {
  hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t g = H5Gcreate2(f, "PartType1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  float* pos = malloc((size_t)n * 3 * sizeof(float));
  double* mass = malloc((size_t)n * sizeof(double));
  int* id = malloc((size_t)n * sizeof(int));
  for (int k = 0; k < n; k++) {
    const int i = first + k;
    pos[3 * k] = (float)i + 0.5f;
    pos[3 * k + 1] = (float)(2 * i) + 0.5f;
    pos[3 * k + 2] = (float)(3 * i) + 0.5f;
    mass[k] = 1.0 + i / 1024.0;
    id[k] = 1000 + i;
  }
  hsize_t d2[2] = {(hsize_t)n, 3}, d1[1] = {(hsize_t)n};
  hid_t s2 = H5Screate_simple(2, d2, NULL), s1 = H5Screate_simple(1, d1, NULL);
  hid_t c = H5Dcreate2(g, "Coordinates", H5T_IEEE_F32LE, s2, H5P_DEFAULT,
    H5P_DEFAULT, H5P_DEFAULT);
  hid_t m = H5Dcreate2(
    g, "Masses", H5T_IEEE_F64LE, s1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t d = H5Dcreate2(
    g, "ParticleIDs", H5T_STD_I32LE, s1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(c, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, pos);
  H5Dwrite(m, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mass);
  H5Dwrite(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, id);
  H5Dclose(c);
  H5Dclose(m);
  H5Dclose(d);
  H5Sclose(s1);
  H5Sclose(s2);
  H5Gclose(g);
  H5Fclose(f);
  free(pos);
  free(mass);
  free(id);
}

static void test_particles(void) {
  printf("particles from any HDF5 file\n");
  const char* whole = "test_hdf5_snap.hdf5";
  const char* parts[] = {"test_hdf5_snap.0.hdf5", "test_hdf5_snap.1.hdf5"};
  enum { N = 1000, SPLIT = 377 };
  write_snapshot(whole, 0, N);
  write_snapshot(parts[0], 0, SPLIT);
  write_snapshot(parts[1], SPLIT, N - SPLIT);

  const sif_field_columns_t cols = {.x = "PartType1/Coordinates[0]",
    .y = "/PartType1/Coordinates[1]",
    .z = "PartType1/Coordinates[2]",
    .w = "PartType1/Masses"};
  sif_field_t* f = sif_field_read_hdf5(&whole, 1, &cols, 1e-3, 1.0, 0);
  CHECK(f && f->n_particles == N && f->units == SIF_COORDINATES_CARTESIAN &&
          f->weights,
    "a snapshot of someone else's did not read");
  if (f) {
    long bad = 0;
    for (int i = 0; i < N; i++) {
      bad += f->x[i] != (sif_real)(((float)i + 0.5f) * 1e-3);
      bad += f->z[i] != (sif_real)(((float)(3 * i) + 0.5f) * 1e-3);
      bad += f->weights[i] != (sif_real)(1.0 + i / 1024.0);
    }
    CHECK(bad == 0, "%ld values differ from the file", bad);
  }

  /* Two files read as one, the subsample drawn over both. */
  sif_field_t* a = sif_field_read_hdf5(&whole, 1, &cols, 1.0, 0.3, 11);
  sif_field_t* b = sif_field_read_hdf5(parts, 2, &cols, 1.0, 0.3, 11);
  CHECK(a && b && a->n_particles == 300 && b->n_particles == 300 &&
          memcmp(a->x, b->x, 300 * sizeof(sif_real)) == 0,
    "the split snapshot did not read as the whole one");
  sif_field_free(a);
  sif_field_free(b);

  /* An integer dataset is a number like any other; ra dec z a sky field. */
  const sif_field_columns_t sky = {.ra = "PartType1/ParticleIDs",
    .dec = "PartType1/Masses",
    .z = "PartType1/Masses"};
  sif_field_t* g = sif_field_read_hdf5(&whole, 1, &sky, 1.0, 1.0, 0);
  CHECK(g && g->units == SIF_COORDINATES_SKY && g->x[5] == 1005,
    "an integer dataset, or sky names, did not read");
  sif_field_free(g);

  const struct {
    const char* what;
    sif_field_columns_t cols;
    double scale;
  } bad[] = {
    {"a missing dataset", {.x = "Nope", .y = "Nope", .z = "Nope"}, 1},
    {"a 2D dataset without its column",
      {.x = "PartType1/Coordinates",
        .y = "PartType1/Masses",
        .z = "PartType1/Masses"},
      1},
    {"a column the dataset does not have",
      {.x = "PartType1/Coordinates[3]",
        .y = "PartType1/Masses",
        .z = "PartType1/Masses"},
      1},
    {"a column of a 1D dataset",
      {.x = "PartType1/Masses[0]",
        .y = "PartType1/Masses",
        .z = "PartType1/Masses"},
      1},
    {"a group for a dataset",
      {.x = "PartType1", .y = "PartType1/Masses", .z = "PartType1/Masses"}, 1},
    {"positions mixed with sky",
      {.x = "PartType1/Masses",
        .dec = "PartType1/Masses",
        .z = "PartType1/Masses"},
      1},
    {"a scale on sky coordinates", sky, 2},
    {"a negative scale", cols, -1},
  };
  for (size_t k = 0; k < sizeof(bad) / sizeof(*bad); k++) {
    sif_field_t* h =
      sif_field_read_hdf5(&whole, 1, &bad[k].cols, bad[k].scale, 1.0, 0);
    CHECK(!h, "%s accepted", bad[k].what);
    sif_field_free(h);
  }
  const char* missing[] = {whole, "test_hdf5_no_such_file.hdf5"};
  sif_field_t* h = sif_field_read_hdf5(missing, 2, &cols, 1.0, 1.0, 0);
  CHECK(!h, "a missing second file accepted");
  sif_field_free(h);

  sif_field_free(f);
  remove(whole);
  remove(parts[0]);
  remove(parts[1]);
  printf("  ok\n");
}

/* Rewrites the layout version a sif file declares. */
static void set_format_version(const char* path, uint32_t v) {
  hid_t f = H5Fopen(path, H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t a = H5Aopen(f, "sif_format_version", H5P_DEFAULT);
  H5Awrite(a, H5T_NATIVE_UINT32, &v);
  H5Aclose(a);
  H5Fclose(f);
}

static void test_round_trips(void) {
  printf("round trips\n");
  remove_all();

  /* Every product in one file, then every one read back. */
  sif_catalog_t* cat = make_catalog(N_VOIDS, 1);
  sif_density_profiles_t* dens = make_density(N_VOIDS);
  sif_velocity_profiles_t* vel = make_velocity(N_VOIDS);
  sif_size_function_t* vsf = make_vsf();

  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_OK, "catalogue write failed");
  CHECK(sif_profiles_write_hdf5(PATH, dens, vel) == SIF_OK,
    "profiles write failed");
  CHECK(sif_size_function_write_hdf5(PATH, vsf) == SIF_OK,
    "size function write failed");

  sif_catalog_t* cat2 = sif_catalog_read_hdf5(PATH);
  CHECK(same_catalog(cat, cat2), "the catalogue did not round-trip");

  sif_density_profiles_t* dens2 = NULL;
  sif_velocity_profiles_t* vel2 = NULL;
  CHECK(sif_profiles_read_hdf5(PATH, &dens2, &vel2) == SIF_OK,
    "profiles read failed");
  CHECK(dens2 && dens2->n_voids == N_VOIDS && dens2->n_bins == N_BINS &&
          dens2->ext == dens->ext && dens2->differential &&
          same_reals(dens->r_edges, dens2->r_edges, N_BINS + 1) &&
          same_reals(dens->profiles, dens2->profiles, N_VOIDS * N_BINS),
    "the density profiles did not round-trip");
  CHECK(vel2 && vel2->n_voids == N_VOIDS && vel2->ext == vel->ext &&
          same_reals(vel->r_edges, vel2->r_edges, N_BINS + 1) &&
          same_reals(vel->v_rad, vel2->v_rad, N_VOIDS * N_BINS),
    "the velocity profiles did not round-trip");

  sif_size_function_t* vsf2 = sif_size_function_read_hdf5(PATH);
  CHECK(vsf2 && vsf2->n_bins == N_BINS && vsf2->options == vsf->options &&
          vsf2->r_min == vsf->r_min && vsf2->r_max == vsf->r_max &&
          same_reals(vsf->r_edges, vsf2->r_edges, N_BINS + 1) &&
          same_reals(vsf->r_centers, vsf2->r_centers, N_BINS) &&
          memcmp(vsf->counts, vsf2->counts, N_BINS * sizeof(uint64_t)) == 0 &&
          same_reals(vsf->vsf, vsf2->vsf, N_BINS) &&
          same_reals(vsf->err, vsf2->err, N_BINS),
    "the size function did not round-trip");

  int has_d = -1, has_v = -1;
  CHECK(sif_profiles_read_header_hdf5(PATH, &has_d, &has_v) == SIF_OK &&
          has_d == 1 && has_v == 1,
    "the header should report both profile sets");

  /* Only what was asked for is read. */
  sif_density_profiles_t* dens3 = NULL;
  CHECK(sif_profiles_read_hdf5(PATH, &dens3, NULL) == SIF_OK && dens3,
    "reading the densities alone failed");

  CHECK(cat2 && cat2->units == SIF_COORDINATES_CARTESIAN,
    "a Cartesian catalogue came back on the sky");
  sif_catalog_free(cat2);

  /* Metadata go with the catalogue as attributes of /catalog, and an
   * attribute anyone adds there afterwards comes back as metadata too. */
  sif_catalog_meta_string_set(cat, "finder", "exodus");
  sif_catalog_meta_real_set(cat, "threshold", -0.7);
  sif_catalog_meta_int_set(cat, "n_tracers", 123456789012LL);
  CHECK(
    sif_catalog_write_hdf5(PATH, cat) == SIF_OK &&
      sif_hdf5_set_attr_string(PATH, "catalog", "Note", "by hand") == SIF_OK,
    "writing a catalogue with metadata failed");
  cat2 = sif_catalog_read_hdf5(PATH);
  CHECK(cat2 && sif_catalog_meta_count(cat2) == 4 &&
          strcmp(sif_catalog_meta_string_get(cat2, "finder"), "exodus") == 0 &&
          sif_catalog_meta_real_get(cat2, "threshold") == -0.7 &&
          sif_catalog_meta_int_get(cat2, "n_tracers") == 123456789012LL &&
          strcmp(sif_catalog_meta_string_get(cat2, "note"), "by hand") == 0,
    "the metadata did not survive HDF5");
  sif_catalog_free(cat2);
  sif_catalog_meta_remove(cat, "finder");
  sif_catalog_meta_remove(cat, "threshold");
  sif_catalog_meta_remove(cat, "n_tracers");

  /* A sky catalogue says so, and reads back on the sky; the attribute that
   * says it is the library's. */
  cat->units = SIF_COORDINATES_SKY;
  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_OK,
    "writing a sky catalogue failed");
  cat2 = sif_catalog_read_hdf5(PATH);
  CHECK(cat2 && cat2->units == SIF_COORDINATES_SKY && same_catalog(cat, cat2),
    "a sky catalogue did not round-trip on the sky");
  CHECK(sif_hdf5_set_attr_string(PATH, "catalog", "coordinates", "cartesian") ==
          SIF_ERR_INVALID,
    "the coordinates attribute could be overwritten");
  cat->units = SIF_COORDINATES_CARTESIAN;
  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_OK,
    "rewriting the catalogue failed");

  sif_catalog_free(cat2);
  sif_density_profiles_free(dens2);
  sif_density_profiles_free(dens3);
  sif_velocity_profiles_free(vel2);
  sif_size_function_free(vsf2);

  /* An empty catalogue is a catalogue. */
  remove_all();
  sif_catalog_t* empty = make_catalog(0, 0);
  CHECK(sif_catalog_write_hdf5(PATH, empty) == SIF_OK,
    "writing an empty catalogue failed");
  sif_catalog_t* empty2 = sif_catalog_read_hdf5(PATH);
  CHECK(empty2 && empty2->n_voids == 0 && !empty2->footprint,
    "an empty catalogue did not round-trip");
  sif_catalog_free(empty);
  sif_catalog_free(empty2);

  sif_catalog_free(cat);
  sif_density_profiles_free(dens);
  sif_velocity_profiles_free(vel);
  sif_size_function_free(vsf);
  printf("  ok\n");
}

static void test_groups(void) {
  printf("groups: subsets, replacement, mismatches\n");
  remove_all();

  /* A product on its own, in a file that never had a catalogue. */
  sif_size_function_t* vsf = make_vsf();
  CHECK(sif_size_function_write_hdf5(PATH, vsf) == SIF_OK,
    "a size function should be writable on its own");
  sif_catalog_t* none = sif_catalog_read_hdf5(PATH);
  CHECK(none == NULL, "a file without /catalog should not read as one");

  int has_d = -1, has_v = -1;
  CHECK(sif_profiles_read_header_hdf5(PATH, &has_d, &has_v) == SIF_OK &&
          has_d == 0 && has_v == 0,
    "the header should report no profiles in a VSF-only file");

  sif_velocity_profiles_t* vel_none = NULL;
  CHECK(sif_profiles_read_hdf5(PATH, NULL, &vel_none) == SIF_ERR_INVALID &&
          vel_none == NULL,
    "asking for profiles the file does not hold should be SIF_ERR_INVALID");

  /* Products written against one catalogue survive its rewrite -- including
   * a rewrite with a different number of voids, which is warned about and
   * written anyway. */
  sif_catalog_t* cat = make_catalog(N_VOIDS, 1);
  sif_density_profiles_t* dens = make_density(N_VOIDS);
  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_OK, "catalogue write failed");
  CHECK(sif_profiles_write_hdf5(PATH, dens, NULL) == SIF_OK,
    "density write failed");

  sif_catalog_t* smaller = make_catalog(N_VOIDS - 5, 0);
  CHECK(sif_catalog_write_hdf5(PATH, smaller) == SIF_OK,
    "a catalogue disagreeing with the profiles should still be written");

  sif_catalog_t* back = sif_catalog_read_hdf5(PATH);
  CHECK(same_catalog(smaller, back),
    "the rewritten catalogue should replace the old one, footprint and all");
  sif_catalog_free(back);

  sif_density_profiles_t* dens2 = NULL;
  CHECK(sif_profiles_read_hdf5(PATH, &dens2, NULL) == SIF_OK && dens2 &&
          dens2->n_voids == N_VOIDS,
    "rewriting the catalogue should leave the profiles in place");
  sif_density_profiles_free(dens2);

  sif_size_function_t* vsf2 = sif_size_function_read_hdf5(PATH);
  CHECK(vsf2 != NULL, "rewriting the catalogue should leave the VSF in place");
  sif_size_function_free(vsf2);

  /* Writing only the velocities keeps the densities written before. */
  sif_velocity_profiles_t* vel = make_velocity(N_VOIDS);
  CHECK(sif_profiles_write_hdf5(PATH, NULL, vel) == SIF_OK,
    "velocity write failed");
  dens2 = NULL;
  sif_velocity_profiles_t* vel2 = NULL;
  CHECK(sif_profiles_read_hdf5(PATH, &dens2, &vel2) == SIF_OK && dens2 && vel2,
    "writing velocities alone should not remove the densities");
  sif_density_profiles_free(dens2);
  sif_velocity_profiles_free(vel2);

  CHECK(sif_profiles_write_hdf5(PATH, NULL, NULL) == SIF_ERR_INVALID,
    "writing no profiles at all should be SIF_ERR_INVALID");

  sif_catalog_free(cat);
  sif_catalog_free(smaller);
  sif_density_profiles_free(dens);
  sif_velocity_profiles_free(vel);
  sif_size_function_free(vsf);
  printf("  ok\n");
}

static void test_foreign_files(void) {
  printf("files sif did not write\n");
  sif_catalog_t* cat = make_catalog(4, 0);

  /* Text: never truncated to make room. */
  remove_all();
  FILE* f = fopen(PATH, "w");
  fputs("somebody's notes\n", f);
  fclose(f);
  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_ERR_IO,
    "a text file should be refused, not overwritten");
  char line[64] = {0};
  f = fopen(PATH, "r");
  CHECK(f && fgets(line, sizeof(line), f) &&
          strcmp(line, "somebody's notes\n") == 0,
    "the refused text file was modified");
  if (f)
    fclose(f);
  CHECK(sif_catalog_read_hdf5(PATH) == NULL, "text should not read");

  /* Someone else's HDF5. */
  remove_all();
  write_foreign_hdf5(PATH);
  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_ERR_IO,
    "an HDF5 file sif did not write should be refused");
  CHECK(sif_catalog_read_hdf5(PATH) == NULL,
    "an HDF5 file sif did not write should not read");

  /* A layout newer than this build: refused rather than half-understood. */
  remove_all();
  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_OK, "catalogue write failed");
  set_format_version(PATH, 99);
  CHECK(sif_catalog_read_hdf5(PATH) == NULL,
    "a newer format version should not read");
  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_ERR_IO,
    "a newer format version should not be written into");

  CHECK(sif_catalog_write_hdf5(NULL, cat) == SIF_ERR_INVALID &&
          sif_catalog_write_hdf5(PATH, NULL) == SIF_ERR_INVALID,
    "NULL arguments should be SIF_ERR_INVALID");

  sif_catalog_free(cat);
  remove_all();
  printf("  ok\n");
}

static void test_metadata(void) {
  printf("free-form metadata\n");
  remove_all();

  /* The root can be described before anything else is in the file -- which
   * creates it -- and every kind reads back as what it was. */
  CHECK(sif_hdf5_set_attr_string(PATH, NULL, "simulation", "Quijote fid 0") ==
            SIF_OK &&
          sif_hdf5_set_attr_int(PATH, "/", "snapshot", 4) == SIF_OK &&
          sif_hdf5_set_attr_real(PATH, "", "redshift", 0.5) == SIF_OK,
    "setting root metadata failed");

  char buf[64];
  int64_t i = 0;
  double d = 0.0;
  sif_hdf5_attr_kind_t kind = 0;
  CHECK(sif_hdf5_get_attr_string(PATH, NULL, "simulation", buf, sizeof(buf)) ==
            SIF_OK &&
          strcmp(buf, "Quijote fid 0") == 0,
    "string metadata did not round-trip");
  CHECK(sif_hdf5_get_attr_int(PATH, NULL, "snapshot", &i) == SIF_OK && i == 4,
    "integer metadata did not round-trip");
  CHECK(
    sif_hdf5_get_attr_real(PATH, NULL, "redshift", &d) == SIF_OK && d == 0.5,
    "real metadata did not round-trip");
  CHECK(
    sif_hdf5_get_attr_real(PATH, NULL, "snapshot", &d) == SIF_OK && d == 4.0,
    "an integer should also read as a real");
  CHECK(sif_hdf5_attr_kind(PATH, NULL, "simulation", &kind) == SIF_OK &&
          kind == SIF_HDF5_ATTR_STRING,
    "a string entry should report its kind");

  /* Wrong kind, missing entry, a string that does not fit. */
  CHECK(sif_hdf5_get_attr_int(PATH, NULL, "simulation", &i) == SIF_ERR_INVALID,
    "a string should not read as an integer");
  CHECK(sif_hdf5_get_attr_int(PATH, NULL, "nope", &i) == SIF_ERR_INVALID,
    "a missing entry should be SIF_ERR_INVALID");
  char small[6];
  CHECK(sif_hdf5_get_attr_string(
          PATH, NULL, "simulation", small, sizeof(small)) == SIF_ERR_RANGE &&
          strcmp(small, "Quijo") == 0,
    "a truncated string should be SIF_ERR_RANGE with what fitted");

  /* The whole header: sif's own three and the three set above, by name. */
  uint32_t n = 0;
  CHECK(sif_hdf5_attr_count(PATH, NULL, &n) == SIF_OK && n == 6,
    "the root should list 6 entries, not %u", n);
  CHECK(sif_hdf5_attr_name(PATH, NULL, 0, buf, sizeof(buf)) == SIF_OK &&
          strcmp(buf, "redshift") == 0,
    "entries should list in alphabetical order (got '%s')", buf);
  CHECK(sif_hdf5_attr_name(PATH, NULL, n, buf, sizeof(buf)) == SIF_ERR_INVALID,
    "an index past the end should be SIF_ERR_INVALID");

  /* sif's own names are not for setting. */
  CHECK(sif_hdf5_set_attr_int(PATH, NULL, "sif_format_version", 7) ==
          SIF_ERR_INVALID,
    "a root name beginning sif_ should be refused");
  CHECK(sif_hdf5_set_attr_string(PATH, NULL, "sif_anything", "x") ==
          SIF_ERR_INVALID,
    "every root name beginning sif_ should be refused");

  /* A product is described once it exists, and not before. */
  CHECK(sif_hdf5_set_attr_real(PATH, "catalog", "threshold", -0.7) ==
          SIF_ERR_INVALID,
    "describing a product that is not there should be refused");

  sif_catalog_t* cat = make_catalog(5, 0);
  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_OK, "catalogue write failed");
  CHECK(
    sif_hdf5_set_attr_real(PATH, "catalog", "threshold", -0.7) == SIF_OK &&
      sif_hdf5_set_attr_string(PATH, "/catalog", "finder", "exodus") == SIF_OK,
    "describing the catalogue failed");
  CHECK(
    sif_hdf5_set_attr_int(PATH, "catalog", "n_voids", 99) == SIF_ERR_INVALID,
    "a product's own attribute should be refused");
  CHECK(sif_hdf5_get_attr_real(PATH, "catalog", "threshold", &d) == SIF_OK &&
          d == -0.7,
    "catalogue metadata did not round-trip");

  /* Rewriting the catalogue drops what described the old one; the file's own
   * description stays. */
  CHECK(
    sif_catalog_write_hdf5(PATH, cat) == SIF_OK, "catalogue rewrite failed");
  CHECK(
    sif_hdf5_attr_kind(PATH, "catalog", "threshold", &kind) == SIF_ERR_INVALID,
    "a rewritten catalogue should not keep the old one's metadata");
  CHECK(sif_hdf5_get_attr_int(PATH, NULL, "snapshot", &i) == SIF_OK && i == 4,
    "root metadata should survive a product rewrite");

  /* Freely overwritten, even with another kind. */
  CHECK(sif_hdf5_set_attr_string(PATH, NULL, "snapshot", "four") == SIF_OK &&
          sif_hdf5_attr_kind(PATH, NULL, "snapshot", &kind) == SIF_OK &&
          kind == SIF_HDF5_ATTR_STRING,
    "an entry should be replaceable by one of another kind");

  sif_catalog_free(cat);
  remove_all();
  printf("  ok\n");
}

#else /* no HDF5 */

static long count_lines(const char* path) {
  FILE* f = fopen(path, "r");
  if (!f)
    return -1;
  long n = 0;
  for (int c; (c = fgetc(f)) != EOF;)
    n += c == '\n';
  fclose(f);
  return n;
}

static void test_fallback(void) {
  printf("no HDF5: writers dump, readers refuse\n");
  remove_all();

  sif_catalog_t* cat = make_catalog(N_VOIDS, 1);
  sif_density_profiles_t* dens = make_density(N_VOIDS);
  sif_velocity_profiles_t* vel = make_velocity(N_VOIDS);
  sif_size_function_t* vsf = make_vsf();

  CHECK(sif_catalog_write_hdf5(PATH, cat) == SIF_OK &&
          sif_profiles_write_hdf5(PATH, dens, vel) == SIF_OK &&
          sif_size_function_write_hdf5(PATH, vsf) == SIF_OK,
    "a writer should succeed by falling back to text");

  CHECK(!file_exists(PATH),
    "nothing should be written under the name meant for the HDF5 file");

  /* The catalogue dump is an ordinary sif ASCII catalogue. */
  char p[256];
  snprintf(p, sizeof(p), "%s.catalog.txt", PATH);
  sif_catalog_t* back = sif_catalog_read_ascii(p, NULL);
  CHECK(same_catalog(cat, back),
    "the catalogue dump should read back as the catalogue");
  sif_catalog_free(back);

  /* The others: a comment header, then one line per row. */
  snprintf(p, sizeof(p), "%s.density_profiles.txt", PATH);
  CHECK(count_lines(p) == 3 + N_VOIDS, "density dump has %ld lines",
    count_lines(p));
  snprintf(p, sizeof(p), "%s.velocity_profiles.txt", PATH);
  CHECK(count_lines(p) == 3 + N_VOIDS, "velocity dump has %ld lines",
    count_lines(p));
  snprintf(p, sizeof(p), "%s.size_function.txt", PATH);
  CHECK(count_lines(p) == 2 + N_BINS, "size function dump has %ld lines",
    count_lines(p));

  CHECK(sif_catalog_read_hdf5(PATH) == NULL, "a read should refuse");
  CHECK(sif_size_function_read_hdf5(PATH) == NULL, "a read should refuse");
  int has_d = -1;
  CHECK(
    sif_profiles_read_header_hdf5(PATH, &has_d, NULL) == SIF_ERR_UNSUPPORTED &&
      has_d == 0,
    "the profiles header should be SIF_ERR_UNSUPPORTED");

  sif_density_profiles_t* d = (sif_density_profiles_t*)1;
  CHECK(
    sif_profiles_read_hdf5(PATH, &d, NULL) == SIF_ERR_UNSUPPORTED && d == NULL,
    "a profiles read should be SIF_ERR_UNSUPPORTED and leave NULL");

  /* Metadata is kept in a dump of its own, with the same keys refused. */
  CHECK(
    sif_hdf5_set_attr_string(PATH, NULL, "simulation", "Quijote") == SIF_OK &&
      sif_hdf5_set_attr_real(PATH, "catalog", "threshold", -0.7) == SIF_OK &&
      sif_hdf5_set_attr_int(PATH, NULL, "snapshot", 4) == SIF_OK,
    "a metadata setter should succeed by falling back to text");
  CHECK(
    sif_hdf5_set_attr_int(PATH, "catalog", "n_voids", 3) == SIF_ERR_INVALID &&
      sif_hdf5_set_attr_int(PATH, NULL, "sif_format", 3) == SIF_ERR_INVALID,
    "the keys refused with HDF5 should be refused without it");
  snprintf(p, sizeof(p), "%s.attributes.txt", PATH);
  CHECK(count_lines(p) == 3, "attributes dump has %ld lines", count_lines(p));

  int64_t i = 0;
  CHECK(
    sif_hdf5_get_attr_int(PATH, NULL, "snapshot", &i) == SIF_ERR_UNSUPPORTED,
    "a metadata read should be SIF_ERR_UNSUPPORTED");

  sif_catalog_free(cat);
  sif_density_profiles_free(dens);
  sif_velocity_profiles_free(vel);
  sif_size_function_free(vsf);
  remove_all();
  printf("  ok\n");
}

#endif

int main(void) {
  if (sif_init(SIF_CONFIG_QUIET) != SIF_OK) {
    printf("FAIL: sif_init\n");
    return 1;
  }

#ifdef SIF_HAVE_HDF5
  test_round_trips();
  test_groups();
  test_foreign_files();
  test_metadata();
  test_particles();
#else
  test_fallback();
#endif

  (void)file_exists;
  sif_finalize();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
