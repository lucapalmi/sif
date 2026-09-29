/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The file formats: round trips, and the checks that are supposed to reject a
 * bad file.
 *
 * The interesting cases here are the negative ones. A reader that accepts a
 * corrupt file is worse than one that fails, because the data flows onward and
 * whatever it produces looks like a result -- so every rejection path is
 * exercised by actually damaging a file and requiring the reader to say so.
 */

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/io/catalog_io.h"
#include "sif/io/field_io.h"
#include "sif/io/grid_io.h"
#include "sif/io/profiles_io.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"

#include <math.h>
#include <stdint.h>
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

#define N_P     1000u
#define N_CELLS 8u
#define BOX     100.0f

static const char* FIELD_PATH = "test_io.xfield";
static const char* GRID_PATH = "test_io.xgrid";
static const char* CAT_PATH = "test_io.cat";
static const char* ASCII_PATH = "test_io.txt";
static const char* PROF_PATH = "test_io.prof";
static const char* PROF_HALF_PATH = "test_io.prof.dens";

/* Flip one bit inside the payload, past the 64-byte header. */
static void corrupt_payload(const char* path, long offset_in_payload) {
  FILE* f = fopen(path, "r+b");
  if (!f)
    return;
  fseek(f, 64 + offset_in_payload, SEEK_SET);
  int c = fgetc(f);
  fseek(f, 64 + offset_in_payload, SEEK_SET);
  fputc(c ^ 0x01, f);
  fclose(f);
}

static sif_field_t* make_field(void) {
  sif_field_t* f = sif_field_alloc(N_P);
  if (!f)
    return NULL;
  sif_field_reserve_positions(f);
  sif_field_reserve_velocities(f);
  sif_field_reserve_weights(f);
  for (uint64_t i = 0; i < N_P; i++) {
    f->x[i] = (sif_real)((double)i * 0.31379);
    f->y[i] = (sif_real)((double)i * 1.70021);
    f->z[i] = (sif_real)i;
    f->vx[i] = (sif_real)(-(double)i * 0.5);
    f->vy[i] = (sif_real)((double)i * 0.25);
    f->vz[i] = (sif_real)((double)i * 0.125);
    f->weights[i] = (sif_real)(1.0 + (double)i * 1e-3);
  }
  return f;
}

static void test_header_layout(void) {
  printf("header layout\n");
  /* The formats are read back byte for byte, so a header that changed size
   * would silently shift every payload offset. */
  CHECK(sizeof(sif_xfield_header_t) == 64, "xfield header is %zu bytes, not 64",
    sizeof(sif_xfield_header_t));
  CHECK(sizeof(sif_xgrid_header_t) == 64, "xgrid header is %zu bytes, not 64",
    sizeof(sif_xgrid_header_t));
}

static void test_field_roundtrip(void) {
  printf("xfield round trip\n");

  sif_field_t* f = make_field();
  CHECK(f != NULL, "field allocation failed");
  if (!f)
    return;

  CHECK(sif_field_write(FIELD_PATH, f, (double)BOX) == SIF_OK, "write failed");

  sif_field_t* g = sif_field_alloc(N_P);
  sif_field_reserve_positions(g);
  sif_field_reserve_velocities(g);
  sif_field_reserve_weights(g);

  CHECK(sif_field_read_into(g, FIELD_PATH) == SIF_OK, "read failed");

  int identical = 1;
  for (uint64_t i = 0; i < N_P; i++) {
    if (f->x[i] != g->x[i] || f->y[i] != g->y[i] || f->z[i] != g->z[i] ||
        f->vx[i] != g->vx[i] || f->vy[i] != g->vy[i] || f->vz[i] != g->vz[i] ||
        f->weights[i] != g->weights[i]) {
      identical = 0;
      break;
    }
  }
  CHECK(identical, "binary round trip is not bit-exact");

  /* Allocating form. */
  double box_out = 0.0;
  sif_field_t* h = sif_field_read(FIELD_PATH, &box_out);
  CHECK(h != NULL, "sif_field_read returned NULL");
  CHECK(box_out == (double)BOX, "box length %g survived as %g", (double)BOX,
    box_out);
  if (h)
    CHECK(h->n_particles == N_P, "read %llu particles, expected %u",
      (unsigned long long)h->n_particles, N_P);
  sif_field_free(h);

  CHECK(sif_field_read_into(g, NULL) == SIF_ERR_INVALID,
    "a NULL path should be SIF_ERR_INVALID");
  CHECK(sif_field_read_into(g, "test_io_does_not_exist.xfield") == SIF_ERR_IO,
    "a missing file should be SIF_ERR_IO");

  sif_field_free(g);
  sif_field_free(f);
}

static void test_field_corruption(void) {
  printf("xfield corruption is caught\n");

  sif_field_t* f = make_field();
  if (!f)
    return;
  CHECK(sif_field_write(FIELD_PATH, f, (double)BOX) == SIF_OK, "write failed");
  sif_field_free(f);

  corrupt_payload(FIELD_PATH, 40);

  sif_field_t* g = sif_field_alloc(N_P);
  sif_field_reserve_positions(g);
  sif_field_reserve_velocities(g);
  sif_field_reserve_weights(g);
  CHECK(sif_field_read_into(g, FIELD_PATH) == SIF_ERR_IO,
    "a flipped payload bit should fail the checksum");
  sif_field_free(g);
}

static void test_grid_roundtrip(void) {
  printf("xgrid round trip\n");

  sif_grid_t* grid = sif_grid_alloc(N_CELLS, BOX);
  CHECK(grid != NULL, "grid allocation failed");
  if (!grid)
    return;
  for (uint64_t i = 0; i < grid->total_cells; i++)
    grid->values[i] = (sif_real)((double)i * 0.017 - 1.0);
  grid->content = SIF_GRID_DENSITY_CONTRAST;

  CHECK(sif_grid_write(GRID_PATH, grid) == SIF_OK, "write failed");

  sif_grid_t* back = sif_grid_alloc(N_CELLS, BOX);
  CHECK(sif_grid_read_into(back, GRID_PATH) == SIF_OK, "read failed");

  int identical = 1;
  for (uint64_t i = 0; i < grid->total_cells; i++)
    if (grid->values[i] != back->values[i]) {
      identical = 0;
      break;
    }
  CHECK(identical, "binary round trip is not bit-exact");
  CHECK(back->content == grid->content,
    "content tag did not survive the round trip (%d vs %d)", (int)back->content,
    (int)grid->content);

  /* A grid of the wrong shape must be refused, not partially filled: this is
   * the path the CIC cache relies on to detect a stale entry. */
  sif_grid_t* wrong = sif_grid_alloc(N_CELLS * 2u, BOX);
  CHECK(sif_grid_read_into(wrong, GRID_PATH) == SIF_ERR_IO,
    "a geometry mismatch should be rejected");
  sif_grid_free(wrong);

  corrupt_payload(GRID_PATH, 16);
  CHECK(sif_grid_read_into(back, GRID_PATH) == SIF_ERR_IO,
    "a flipped payload bit should fail the checksum");

  sif_grid_free(back);
  sif_grid_free(grid);
}

static void test_catalog_roundtrip(void) {
  printf("catalogue round trip\n");

  sif_catalog_t* cat = sif_catalog_alloc(8);
  CHECK(cat != NULL, "catalog allocation failed");
  if (!cat)
    return;

  /* Values chosen to need most of the mantissa: a six-decimal writer loses
   * them, which is what SIF_PRI_REAL exists to prevent. */
  sif_catalog_append(cat, (sif_real)1234.5678901234, (sif_real)0.1234567890123,
    (sif_real)9.87654321e-7, (sif_real)3.14159265358979);
  sif_catalog_append(cat, (sif_real)-0.0009765625, (sif_real)65536.03125,
    (sif_real)1e-8, (sif_real)0.5);

  CHECK(sif_catalog_write_ascii(CAT_PATH, cat) == SIF_OK, "write failed");

  sif_catalog_t* back = sif_catalog_read_ascii(CAT_PATH);
  CHECK(back != NULL, "read returned NULL");
  if (back) {
    CHECK(back->n_voids == cat->n_voids, "read %llu voids, expected %llu",
      (unsigned long long)back->n_voids, (unsigned long long)cat->n_voids);

    int exact = 1;
    for (uint64_t i = 0; i < cat->n_voids && i < back->n_voids; i++) {
      if (cat->cx[i] != back->cx[i] || cat->cy[i] != back->cy[i] ||
          cat->cz[i] != back->cz[i] || cat->radii[i] != back->radii[i])
        exact = 0;
    }
    CHECK(exact, "text round trip lost precision");
    CHECK(back->footprint == NULL,
      "a catalogue written without a footprint should read back without one");
    sif_catalog_free(back);
  }

  CHECK(sif_catalog_write_ascii(CAT_PATH, NULL) == SIF_ERR_INVALID,
    "a NULL catalogue should be SIF_ERR_INVALID");

  /* With a footprint: two more columns, back bit for bit. */
  CHECK(sif_catalog_reserve_footprint(cat) == SIF_OK, "reserve failed");
  cat->footprint[0] = (sif_real)0.123456789;
  cat->footprint_shell[0] = (sif_real)1.0;
  cat->footprint[1] = (sif_real)0.0;
  cat->footprint_shell[1] = SIF_CATALOG_FOOTPRINT_UNKNOWN;

  CHECK(sif_catalog_write_ascii(CAT_PATH, cat) == SIF_OK,
    "write with footprint failed");
  back = sif_catalog_read_ascii(CAT_PATH);
  CHECK(back && back->footprint && back->footprint_shell,
    "the footprint columns did not come back");
  if (back && back->footprint) {
    int exact = back->n_voids == cat->n_voids;
    for (uint64_t i = 0; i < cat->n_voids && i < back->n_voids; i++) {
      exact &= cat->cx[i] == back->cx[i] && cat->radii[i] == back->radii[i] &&
               cat->footprint[i] == back->footprint[i] &&
               cat->footprint_shell[i] == back->footprint_shell[i];
    }
    CHECK(exact, "the footprint round trip lost precision");
  }
  sif_catalog_free(back);

  /* Rows that disagree about their columns are a malformed file, not a
   * catalogue with a hole in it. Blank lines, as the old reader allowed, are
   * still fine. */
  FILE* f = fopen(CAT_PATH, "w");
  if (f) {
    fputs("2\n1 2 3 4 0.5 0.25\n\n5 6 7 8\n", f);
    fclose(f);
  }
  back = sif_catalog_read_ascii(CAT_PATH);
  CHECK(back == NULL, "rows of four and six columns should be refused");
  sif_catalog_free(back);

  f = fopen(CAT_PATH, "w");
  if (f) {
    fputs("2\n\n1 2 3 4\n  \n5 6 7 8\n", f);
    fclose(f);
  }
  back = sif_catalog_read_ascii(CAT_PATH);
  CHECK(back && back->n_voids == 2 && back->footprint == NULL &&
          back->radii[1] == (sif_real)8,
    "a four-column file with blank lines should read as before");
  sif_catalog_free(back);

  f = fopen(CAT_PATH, "w");
  if (f) {
    fputs("3\n1 2 3 4\n5 6 7 8\n", f);
    fclose(f);
  }
  back = sif_catalog_read_ascii(CAT_PATH);
  CHECK(back == NULL, "a file shorter than its count should be refused");
  sif_catalog_free(back);

  sif_catalog_t* empty = sif_catalog_alloc(1);
  CHECK(sif_catalog_write_ascii(CAT_PATH, empty) == SIF_OK,
    "writing an empty catalogue failed");
  back = sif_catalog_read_ascii(CAT_PATH);
  CHECK(back && back->n_voids == 0, "an empty catalogue did not read back");
  sif_catalog_free(back);
  sif_catalog_free(empty);

  sif_catalog_free(cat);
}

static void test_profiles_roundtrip(void) {
  printf("profile round trip\n");

  /* The mesh rejects anything outside the box, so this field is wrapped into
   * it rather than reusing make_field()'s spread-out one. */
  sif_field_t* f = sif_field_alloc(N_P);
  if (!f) {
    CHECK(0, "field allocation failed");
    return;
  }
  sif_field_reserve_positions(f);
  sif_field_reserve_velocities(f);

  for (uint64_t i = 0; i < N_P; i++) {
    f->x[i] = (sif_real)fmod((double)i * 7.31379, (double)BOX);
    f->y[i] = (sif_real)fmod((double)i * 13.70021, (double)BOX);
    f->z[i] = (sif_real)fmod((double)i * 3.14159, (double)BOX);
    f->vx[i] = (sif_real)(-(double)i * 0.5);
    f->vy[i] = (sif_real)((double)i * 0.25);
    f->vz[i] = (sif_real)((double)i * 0.125);
  }

  sif_catalog_t* cat = sif_catalog_alloc(4);
  sif_catalog_append(
    cat, (sif_real)50.0, (sif_real)50.0, (sif_real)50.0, (sif_real)8.0);
  sif_catalog_append(
    cat, (sif_real)12.5, (sif_real)77.25, (sif_real)3.125, (sif_real)5.5);
  sif_catalog_append(cat, (sif_real)0.0009765625, (sif_real)65.03125,
    (sif_real)99.5, (sif_real)6.25);

  sif_chain_mesh_t* mesh =
    sif_chain_mesh_alloc(N_CELLS, BOX, f, SIF_MESH_DROP_INDICES);

  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;
  CHECK(mesh != NULL, "mesh construction failed");
  if (mesh)
    CHECK(sif_profiles(cat, mesh, (sif_real)4.0, 8, SIF_PBC_PERIODIC, &dens,
            &vel) == SIF_OK,
      "profile computation failed");

  if (dens && vel) {
    CHECK(sif_profiles_write_ascii(PROF_PATH, dens, vel, cat) == SIF_OK,
      "write failed");

    sif_catalog_t* cat_back = NULL;
    sif_density_profiles_t* dens_back = NULL;
    sif_velocity_profiles_t* vel_back = NULL;

    CHECK(sif_profiles_read_ascii(
            PROF_PATH, &cat_back, &dens_back, &vel_back) == SIF_OK,
      "read failed");

    if (cat_back && dens_back && vel_back) {
      CHECK(dens_back->n_voids == dens->n_voids &&
              dens_back->n_bins == dens->n_bins && dens_back->ext == dens->ext,
        "the shape did not survive the round trip");
      CHECK(cat_back->n_voids == cat->n_voids, "the catalogue lost voids");

      int exact = 1;
      for (uint32_t j = 0; j <= dens->n_bins; j++)
        if (dens->r_edges[j] != dens_back->r_edges[j] ||
            vel->r_edges[j] != vel_back->r_edges[j])
          exact = 0;

      for (uint64_t i = 0; i < cat->n_voids; i++) {
        if (cat->cx[i] != cat_back->cx[i] || cat->cy[i] != cat_back->cy[i] ||
            cat->cz[i] != cat_back->cz[i] ||
            cat->radii[i] != cat_back->radii[i])
          exact = 0;

        const sif_real* d0 = sif_density_profiles_get(dens, i);
        const sif_real* d1 = sif_density_profiles_get(dens_back, i);
        const sif_real* v0 = sif_velocity_profiles_get(vel, i);
        const sif_real* v1 = sif_velocity_profiles_get(vel_back, i);

        for (uint32_t j = 0; j < dens->n_bins; j++)
          if (d0[j] != d1[j] || v0[j] != v1[j])
            exact = 0;
      }
      CHECK(exact, "text round trip lost precision");
    }

    sif_catalog_free(cat_back);
    sif_density_profiles_free(dens_back);
    sif_velocity_profiles_free(vel_back);

    /* Half of a file holding both: the velocity columns are parsed to get
     * past them, not stored. */
    sif_density_profiles_t* only_dens = NULL;
    CHECK(sif_profiles_read_ascii(PROF_PATH, NULL, &only_dens, NULL) == SIF_OK,
      "reading only the densities failed");
    CHECK(only_dens != NULL && only_dens->n_voids == cat->n_voids,
      "the density-only read came back wrong");
    if (only_dens) {
      int exact = 1;
      for (uint64_t i = 0; i < cat->n_voids; i++) {
        const sif_real* d0 = sif_density_profiles_get(dens, i);
        const sif_real* d1 = sif_density_profiles_get(only_dens, i);
        for (uint32_t j = 0; j < dens->n_bins; j++)
          if (d0[j] != d1[j])
            exact = 0;
      }
      CHECK(exact, "the density-only read skipped the wrong columns");
    }
    sif_density_profiles_free(only_dens);

    /* Whether a bin holds its own shell or everything enclosed is not visible
     * in the values, so the file has to say, and the reader has to believe
     * it. */
    sif_density_profiles_t* shells = NULL;
    if (sif_profiles(cat, mesh, (sif_real)4.0, 8,
          SIF_PBC_PERIODIC | SIF_PROFILES_DIFFERENTIAL, &shells,
          NULL) == SIF_OK) {
      CHECK(
        sif_profiles_write_ascii(PROF_HALF_PATH, shells, NULL, cat) == SIF_OK,
        "differential write failed");

      int is_differential = 0;
      CHECK(sif_profiles_read_header_ascii(PROF_HALF_PATH, NULL, NULL, NULL,
              NULL, NULL, &is_differential) == SIF_OK,
        "reading the differential header failed");
      CHECK(is_differential, "the file does not record differential binning");

      sif_density_profiles_t* shells_back = NULL;
      CHECK(sif_profiles_read_ascii(PROF_HALF_PATH, NULL, &shells_back, NULL) ==
              SIF_OK,
        "differential read failed");
      CHECK(shells_back && shells_back->differential,
        "the differential flag did not survive the round trip");
      sif_density_profiles_free(shells_back);
      sif_density_profiles_free(shells);
    } else {
      CHECK(0, "differential profile computation failed");
    }

    /* Asking a file for what it does not carry has to fail rather than hand
     * back a set of zeros, which would look like a measurement. */
    CHECK(sif_profiles_write_ascii(PROF_HALF_PATH, dens, NULL, cat) == SIF_OK,
      "density-only write failed");

    sif_velocity_profiles_t* missing = (sif_velocity_profiles_t*)1;
    CHECK(sif_profiles_read_ascii(PROF_HALF_PATH, NULL, NULL, &missing) ==
            SIF_ERR_INVALID,
      "a file without velocities should refuse to supply them");
    CHECK(missing == NULL, "a failed read left its output non-NULL");

    /* A catalogue of a different length labels every row with the wrong
     * void. */
    sif_catalog_t* short_cat = sif_catalog_alloc(2);
    sif_catalog_append(short_cat, 1.0f, 2.0f, 3.0f, 4.0f);
    CHECK(sif_profiles_write_ascii(PROF_PATH, dens, vel, short_cat) ==
            SIF_ERR_INVALID,
      "a catalogue of the wrong length should be SIF_ERR_INVALID");
    sif_catalog_free(short_cat);

    CHECK(
      sif_profiles_write_ascii(PROF_PATH, NULL, NULL, cat) == SIF_ERR_INVALID,
      "writing no profile set at all should be SIF_ERR_INVALID");
  }

  sif_density_profiles_free(dens);
  sif_velocity_profiles_free(vel);
  sif_chain_mesh_free(mesh);
  sif_catalog_free(cat);
  sif_field_free(f);
}

/* Write an ASCII table verbatim, so a test can say exactly what the file
 * looks like -- including whether it ends in a newline. */
static void write_text(const char* body) {
  FILE* f = fopen(ASCII_PATH, "w");
  if (!f)
    return;
  fputs(body, f);
  fclose(f);
}

/* Read ASCII_PATH into a fresh field sized from the file itself. */
static sif_field_t* read_ascii(const char* fmt, char delim, uint32_t skip) {
  return sif_field_read_ascii(ASCII_PATH, fmt, delim, skip);
}

static void test_ascii_field(void) {
  printf("ASCII field input\n");

  /* A last line with no terminator is still a row. Counting newlines alone
   * drops it, and drops it silently, since the same count decides how many
   * rows the parser then goes looking for. */
  write_text("1 2 3\n4 5 6\n7 8 9");
  sif_field_t* f = read_ascii("xyz", ' ', 0);
  CHECK(f != NULL, "read of an unterminated file failed");
  if (f) {
    CHECK(f->n_particles == 3, "unterminated last line: got %llu rows, not 3",
      (unsigned long long)f->n_particles);
    CHECK(f->x[2] == 7 && f->y[2] == 8 && f->z[2] == 9,
      "unterminated last line read as (%g %g %g)", (double)f->x[2],
      (double)f->y[2], (double)f->z[2]);
    sif_field_free(f);
  }

  /* Blank lines, comments and a row that runs out of columns are not
   * particles. Counting one leaves an entry whose components were never
   * assigned, which reads as a particle at whatever the allocator handed
   * back -- the origin on a fresh page, arbitrary coordinates otherwise. */
  write_text("# a comment\n"
             "1 2 3\n"
             "\n"
             "; another comment\n"
             "4 5 6\n"
             "   \n"
             "7 8\n"
             "10 11 12\n"
             "\n");
  f = read_ascii("xyz", ' ', 0);
  CHECK(f != NULL, "read of a file with blanks and comments failed");
  if (f) {
    CHECK(f->n_particles == 3, "noise lines: got %llu rows, not 3",
      (unsigned long long)f->n_particles);
    CHECK(f->x[0] == 1 && f->x[1] == 4 && f->x[2] == 10,
      "noise lines shifted the data");
    CHECK(f->y[2] == 11 && f->z[2] == 12,
      "the short row was counted as a particle");
    sif_field_free(f);
  }

  /* Skipped header lines are not rows either, whatever they contain. */
  write_text("n_particles = 2\n1 2 3\n4 5 6\n");
  f = read_ascii("xyz", ' ', 1);
  CHECK(f != NULL, "read with a header failed");
  if (f) {
    CHECK(f->n_particles == 2 && f->x[0] == 1,
      "header line was counted as a particle");
    sif_field_free(f);
  }

  /* Velocities, an ignored column, and a non-space delimiter. */
  write_text("1,2,3,99,-1,-2,-3\n4,5,6,99,-4,-5,-6\n");
  f = read_ascii("x y z * vx vy vz", ',', 0);
  CHECK(f != NULL, "comma-delimited read failed");
  if (f) {
    CHECK(f->n_particles == 2, "comma-delimited: got %llu rows, not 2",
      (unsigned long long)f->n_particles);
    CHECK(f->vx != NULL && f->vx[1] == -4 && f->vz[1] == -6,
      "velocity columns did not land");
    CHECK(f->z[1] == 6, "the ignored column was not skipped");
    sif_field_free(f);
  }

  /* Reserving is a no-op on a block that is already there, so a second pass
   * with a weight-only format keeps the positions the first pass loaded. */
  write_text("1 2 3 100\n4 5 6 200\n");
  f = read_ascii("xyz*", ' ', 0);
  CHECK(f != NULL, "first pass failed");
  if (f) {
    CHECK(sif_field_read_ascii_into(f, ASCII_PATH, "***w", ' ', 0) == SIF_OK,
      "weight-only second pass failed");
    CHECK(f->x[1] == 4 && f->z[1] == 6, "the second pass lost the positions");
    CHECK(f->weights != NULL && f->weights[1] == 200, "weights did not land");
    sif_field_free(f);
  }

  /* A field that already has a count is filled to it and no further. */
  write_text("1 2 3\n4 5 6\n7 8 9\n");
  f = sif_field_alloc(2);
  CHECK(sif_field_read_ascii_into(f, ASCII_PATH, "xyz", ' ', 0) == SIF_OK,
    "read into a pre-sized field failed");
  CHECK(
    f->n_particles == 2 && f->x[1] == 4, "a pre-sized field was not respected");
  sif_field_free(f);
}

static void test_ascii_field_rejections(void) {
  printf("ASCII field input rejects bad input\n");

  write_text("1 2 3\n");
  sif_field_t* f = sif_field_alloc(4);
  if (!f)
    return;

  /* A format that names nothing would consume every line and write nothing,
   * leaving a field of uninitialized memory that reads as loaded. */
  CHECK(sif_field_read_ascii_into(f, ASCII_PATH, "", ' ', 0) == SIF_ERR_INVALID,
    "an empty format should be SIF_ERR_INVALID");
  CHECK(
    sif_field_read_ascii_into(f, ASCII_PATH, "?!", ' ', 0) == SIF_ERR_INVALID,
    "a format of unknown characters should be SIF_ERR_INVALID");

  /* Position and velocity are reserved as one block each, so naming two of
   * the three would allocate the third and never write it. */
  CHECK(
    sif_field_read_ascii_into(f, ASCII_PATH, "xy", ' ', 0) == SIF_ERR_INVALID,
    "a partial position should be SIF_ERR_INVALID");
  CHECK(sif_field_read_ascii_into(f, ASCII_PATH, "xyz vx vy", ' ', 0) ==
          SIF_ERR_INVALID,
    "a partial velocity should be SIF_ERR_INVALID");
  CHECK(
    sif_field_read_ascii_into(f, ASCII_PATH, "w", ' ', 0) == SIF_ERR_INVALID,
    "a weight-only format on a field with no positions should be rejected");
  CHECK(!sif_field_read_ascii(ASCII_PATH, "w", ' ', 0),
    "sif_field_read_ascii() returned a field with no positions");
  CHECK(!sif_field_read_ascii("test_io_does_not_exist.txt", "xyz", ' ', 0),
    "sif_field_read_ascii() returned a field for a missing file");

  CHECK(sif_field_read_ascii_into(NULL, ASCII_PATH, "xyz", ' ', 0) ==
          SIF_ERR_INVALID,
    "a NULL field should be SIF_ERR_INVALID");
  CHECK(sif_field_read_ascii_into(
          f, "test_io_does_not_exist.txt", "xyz", ' ', 0) == SIF_ERR_IO,
    "a missing file should be SIF_ERR_IO");
  sif_field_free(f);

  /* A file of nothing but noise yields no particles, which is a failure and
   * not an empty field: the caller would otherwise carry on with one. */
  write_text("# only\n\n; comments\n");
  f = sif_field_alloc(0);
  CHECK(sif_field_read_ascii_into(f, ASCII_PATH, "xyz", ' ', 0) == SIF_ERR_IO,
    "a file with no data rows should be SIF_ERR_IO");
  sif_field_free(f);
}

/* --- column formats --- */

/* The format language, through the ASCII reader: every spelling of the same
 * layout reads the same, and every malformed one is refused. */
static void test_column_formats(void) {
  printf("column formats\n");

  write_text("1 2 3 4 5 6 7\n8 9 10 11 12 13 14\n");
  const char* same[] = {"x y z vx vy vz w", "x,y,z,vx,vy,vz,w", "XYZVXVYVZW",
    "xyzvxvyvzw", "  x\ty z vX Vy vz  w  "};
  for (size_t i = 0; i < sizeof(same) / sizeof(*same); i++) {
    sif_field_t* f = read_ascii(same[i], ' ', 0);
    CHECK(f && f->n_particles == 2 && f->x[1] == 8 && f->z[1] == 10 && f->vx &&
            f->vx[1] == 11 && f->vz[1] == 13 && f->weights &&
            f->weights[1] == 14,
      "format '%s' did not read as x y z vx vy vz w", same[i]);
    sif_field_free(f);
  }

  /* w used to be vz and is now the weight. */
  write_text("1 2 3 4\n");
  sif_field_t* f = read_ascii("xyzw", ' ', 0);
  CHECK(f && f->weights && f->weights[0] == 4 && !f->vx,
    "'xyzw' should read the fourth column as the weight");
  sif_field_free(f);

  /* Old spellings, typos, repeats, dangling or misplaced pieces: all refused,
   * none shortened into something that reads. */
  const char* bad[] = {
    "xyzuvw",    /* the old velocity letters */
    "xyz*m",     /* the old weight letter */
    "x y z x",   /* a column named twice */
    "x y z w w", /* the weight named twice */
    "x y z v",   /* v with nothing after it */
    "x y z vw",  /* v with the wrong thing after it */
    "x y z q",   /* not a column name */
    "x y z *8",  /* a skip width, which only binary files take */
    "* * *",     /* skips alone */
    "x;y;z",     /* ; is not a separator */
  };
  write_text("1 2 3 4 5 6 7\n");
  for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
    f = sif_field_alloc(0);
    CHECK(sif_field_read_ascii_into(f, ASCII_PATH, bad[i], ' ', 0) ==
            SIF_ERR_INVALID,
      "format '%s' should be SIF_ERR_INVALID", bad[i]);
    sif_field_free(f);
  }

  /* A sorted field is no longer in file order, so a column added without the
   * positions would land on the wrong particles. Refused -- and reading the
   * positions again is allowed, and drops the permutation the sort kept. */
  write_text("3 0 0 30\n1 0 0 10\n2 0 0 20\n");
  f = read_ascii("x y z *", ' ', 0);
  if (f) {
    CHECK(sif_field_sort_morton(f) == SIF_OK, "sort failed");
    CHECK(sif_field_read_ascii_into(f, ASCII_PATH, "* * * w", ' ', 0) ==
            SIF_ERR_INVALID,
      "adding a column to a sorted field should be SIF_ERR_INVALID");
    CHECK(
      sif_field_read_ascii_into(f, ASCII_PATH, "x y z w", ' ', 0) == SIF_OK &&
        f->original_indices == NULL &&
        !(f->state_flags & SIF_FIELD_STATE_MORTON_SORTED) && f->x[0] == 3 &&
        f->weights[0] == 30,
      "re-reading the positions should reset the sort");
    sif_field_free(f);
  }
}

/* --- raw binary --- */

static const char* BIN_PATH = "test_io.bin";

#define BIN_N      37u
#define BIN_HEADER 40u

/* Column c of particle i. Every value is exact in single precision, so the
 * checks compare with ==. */
static double bin_value(int c, uint64_t i) {
  return 1000.0 * (c + 1) + 0.25 * (double)i;
}

/* Appends one value in the requested precision and byte order, built byte by
 * byte rather than through the reader's own swapping code. */
static void put_value(FILE* f, double v, int is_double, int big) {
  unsigned char b[8];
  uint64_t u;
  int size;
  if (is_double) {
    memcpy(&u, &v, 8);
    size = 8;
  } else {
    const float fv = (float)v;
    uint32_t u32;
    memcpy(&u32, &fv, 4);
    u = u32;
    size = 4;
  }
  for (int k = 0; k < size; k++) {
    const int shift = big ? 8 * (size - 1 - k) : 8 * k;
    b[k] = (unsigned char)(u >> shift);
  }
  fwrite(b, 1, (size_t)size, f);
}

/* A file of BIN_N particles in "x y z *8 vx vy vz w": a junk header, then
 * the values, with a 64-bit ID in the skipped column. */
static void write_binary(int blocks, int is_double, int big) {
  FILE* f = fopen(BIN_PATH, "wb");
  if (!f)
    return;
  for (unsigned k = 0; k < BIN_HEADER; k++)
    fputc(0xA5, f);

  /* Value columns 0..6 are x y z vx vy vz w; -1 is the ID. */
  const int cols[] = {0, 1, 2, -1, 3, 4, 5, 6};
  const int n_cols = 8;

  if (blocks) {
    for (int c = 0; c < n_cols; c++)
      for (uint64_t i = 0; i < BIN_N; i++) {
        if (cols[c] < 0) {
          const uint64_t id = 0xDEADBEEF00000000ull + i;
          fwrite(&id, 8, 1, f);
        } else {
          put_value(f, bin_value(cols[c], i), is_double, big);
        }
      }
  } else {
    for (uint64_t i = 0; i < BIN_N; i++)
      for (int c = 0; c < n_cols; c++) {
        if (cols[c] < 0) {
          const uint64_t id = 0xDEADBEEF00000000ull + i;
          fwrite(&id, 8, 1, f);
        } else {
          put_value(f, bin_value(cols[c], i), is_double, big);
        }
      }
  }
  fclose(f);
}

static uint64_t bin_mismatches(const sif_field_t* f, uint64_t n) {
  uint64_t bad = 0;
  const sif_real* arrays[] = {
    f->x, f->y, f->z, f->vx, f->vy, f->vz, f->weights};
  for (int c = 0; c < 7; c++) {
    if (!arrays[c])
      return UINT64_MAX;
    for (uint64_t i = 0; i < n; i++)
      bad += arrays[c][i] != (sif_real)bin_value(c, i);
  }
  return bad;
}

static void test_binary_field(void) {
  printf("raw binary field input\n");

  const int host_big = 0 == *(const unsigned char*)&(const uint16_t){1};
  const char* fmt = "x y z *8 vx vy vz w";

  /* Every layout, precision and byte order, sized from the file. */
  for (int blocks = 0; blocks < 2; blocks++)
    for (int dbl = 0; dbl < 2; dbl++)
      for (int e = 0; e < 3; e++) {
        const sif_binary_endian_t endian =
          (sif_binary_endian_t)(SIF_BINARY_NATIVE + e);
        const int big = e == 0 ? host_big : e == 2;
        write_binary(blocks, dbl, big);

        sif_field_t* f = sif_field_alloc(0);
        const int status = sif_field_read_binary_into(f, BIN_PATH, fmt,
          blocks ? SIF_BINARY_BLOCKS : SIF_BINARY_ROWS,
          dbl ? SIF_BINARY_FLOAT64 : SIF_BINARY_FLOAT32, endian, BIN_HEADER);
        CHECK(status == SIF_OK && f->n_particles == BIN_N,
          "%s, %s, endian %d: status %d, %llu particles",
          blocks ? "blocks" : "rows", dbl ? "double" : "float", e, status,
          (unsigned long long)f->n_particles);
        if (status == SIF_OK)
          CHECK(bin_mismatches(f, BIN_N) == 0,
            "%s, %s, endian %d: wrong values", blocks ? "blocks" : "rows",
            dbl ? "double" : "float", e);
        sif_field_free(f);
      }

  /* The allocating reader: the same particles, in a field of its own. */
  write_binary(1, 1, host_big);
  sif_field_t* g = sif_field_read_binary(BIN_PATH, fmt, SIF_BINARY_BLOCKS,
    SIF_BINARY_FLOAT64, SIF_BINARY_NATIVE, BIN_HEADER);
  CHECK(g && g->n_particles == BIN_N && bin_mismatches(g, BIN_N) == 0,
    "sif_field_read_binary() did not read the file");
  sif_field_free(g);
  CHECK(!sif_field_read_binary(BIN_PATH, "* * * w", SIF_BINARY_BLOCKS,
          SIF_BINARY_FLOAT64, SIF_BINARY_NATIVE, BIN_HEADER),
    "sif_field_read_binary() returned a field with no positions");
  CHECK(!sif_field_read_binary("test_io_missing.bin", fmt, SIF_BINARY_ROWS,
          SIF_BINARY_FLOAT32, SIF_BINARY_NATIVE, BIN_HEADER),
    "sif_field_read_binary() returned a field for a missing file");

  /* A pre-sized field reads that many; a count the file cannot hold is an
   * I/O error. */
  write_binary(0, 0, host_big);
  sif_field_t* f = sif_field_alloc(10);
  CHECK(sif_field_read_binary_into(f, BIN_PATH, fmt, SIF_BINARY_ROWS,
          SIF_BINARY_FLOAT32, SIF_BINARY_NATIVE, BIN_HEADER) == SIF_OK &&
          bin_mismatches(f, 10) == 0,
    "a pre-sized field was not filled with the first particles");
  sif_field_free(f);

  f = sif_field_alloc(BIN_N + 1);
  CHECK(sif_field_read_binary_into(f, BIN_PATH, fmt, SIF_BINARY_ROWS,
          SIF_BINARY_FLOAT32, SIF_BINARY_NATIVE, BIN_HEADER) == SIF_ERR_IO,
    "asking for more particles than the file holds should be SIF_ERR_IO");
  sif_field_free(f);

  /* The wrong header length, precision or format leaves a remainder, which
   * is refused rather than rounded down into misread particles. */
  const struct {
    const char* fmt;
    sif_binary_precision_t prec;
    uint64_t header;
  } off[] = {
    {fmt, SIF_BINARY_FLOAT32, BIN_HEADER - 4},
    {fmt, SIF_BINARY_FLOAT64, BIN_HEADER},
    {"x y z * vx vy vz w", SIF_BINARY_FLOAT32, BIN_HEADER},
  };
  for (size_t i = 0; i < sizeof(off) / sizeof(*off); i++) {
    f = sif_field_alloc(0);
    CHECK(sif_field_read_binary_into(f, BIN_PATH, off[i].fmt, SIF_BINARY_ROWS,
            off[i].prec, SIF_BINARY_NATIVE, off[i].header) == SIF_ERR_IO,
      "mismatched layout %zu should be SIF_ERR_IO", i);
    sif_field_free(f);
  }

  /* Only the velocities, into a field whose positions are already loaded --
   * and skipping a whole block in the block layout. */
  write_binary(1, 1, host_big);
  f = sif_field_alloc(0);
  CHECK(sif_field_read_binary_into(f, BIN_PATH, "x y z *8 * * * *",
          SIF_BINARY_BLOCKS, SIF_BINARY_FLOAT64, SIF_BINARY_NATIVE,
          BIN_HEADER) == SIF_OK &&
          !f->vx,
    "positions-only read failed");
  CHECK(sif_field_read_binary_into(f, BIN_PATH, "* * * *8 vx vy vz *",
          SIF_BINARY_BLOCKS, SIF_BINARY_FLOAT64, SIF_BINARY_NATIVE,
          BIN_HEADER) == SIF_OK,
    "velocities-only second read failed");
  CHECK(f->vx && !f->weights && f->x[5] == (sif_real)bin_value(0, 5) &&
          f->vx[5] == (sif_real)bin_value(3, 5) &&
          f->vz[BIN_N - 1] == (sif_real)bin_value(5, BIN_N - 1),
    "the second read did not add the velocities to the first");
  sif_field_free(f);

  /* Arguments in the wrong slot, or out of range. */
  f = sif_field_alloc(0);
  CHECK(sif_field_read_binary_into(f, BIN_PATH, fmt,
          (sif_binary_layout_t)SIF_BINARY_FLOAT64,
          (sif_binary_precision_t)SIF_BINARY_ROWS, SIF_BINARY_NATIVE,
          BIN_HEADER) == SIF_ERR_INVALID,
    "swapped layout and precision should be SIF_ERR_INVALID");
  CHECK(sif_field_read_binary_into(f, BIN_PATH, fmt, SIF_BINARY_ROWS,
          SIF_BINARY_FLOAT32, (sif_binary_endian_t)3,
          BIN_HEADER) == SIF_ERR_INVALID,
    "a bare int byte order should be SIF_ERR_INVALID");
  CHECK(sif_field_read_binary_into(f, BIN_PATH, "x y z *0 w", SIF_BINARY_ROWS,
          SIF_BINARY_FLOAT32, SIF_BINARY_NATIVE, 0) == SIF_ERR_INVALID,
    "a zero skip width should be SIF_ERR_INVALID");
  CHECK(
    sif_field_read_binary_into(f, "test_io_missing.bin", fmt, SIF_BINARY_ROWS,
      SIF_BINARY_FLOAT32, SIF_BINARY_NATIVE, 0) == SIF_ERR_IO,
    "a missing file should be SIF_ERR_IO");
  CHECK(sif_field_read_binary_into(f, BIN_PATH, fmt, SIF_BINARY_ROWS,
          SIF_BINARY_FLOAT32, SIF_BINARY_NATIVE, 1u << 20) == SIF_ERR_IO,
    "a header longer than the file should be SIF_ERR_IO");
  sif_field_free(f);
}

int main(void) {
  if (sif_init(SIF_CONFIG_QUIET) != SIF_OK) {
    printf("FAIL: sif_init\n");
    return 1;
  }

  test_header_layout();
  test_field_roundtrip();
  test_field_corruption();
  test_grid_roundtrip();
  test_catalog_roundtrip();
  test_profiles_roundtrip();
  test_ascii_field();
  test_ascii_field_rejections();
  test_column_formats();
  test_binary_field();

  remove(FIELD_PATH);
  remove(GRID_PATH);
  remove(CAT_PATH);
  remove(ASCII_PATH);
  remove(BIN_PATH);
  remove(PROF_PATH);
  remove(PROF_HALF_PATH);

  sif_finalize();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
