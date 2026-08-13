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
#include "sif/structures/catalog.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"

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

#define N_P 1000u
#define N_CELLS 8u
#define BOX 100.0f

static const char* FIELD_PATH = "test_io.xfield";
static const char* GRID_PATH = "test_io.xgrid";
static const char* CAT_PATH = "test_io.cat";

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
  sif_field_reserve_masses(f);
  for (uint64_t i = 0; i < N_P; i++) {
    f->x[i] = (sif_real)((double)i * 0.31379);
    f->y[i] = (sif_real)((double)i * 1.70021);
    f->z[i] = (sif_real)i;
    f->vx[i] = (sif_real)(-(double)i * 0.5);
    f->vy[i] = (sif_real)((double)i * 0.25);
    f->vz[i] = (sif_real)((double)i * 0.125);
    f->masses[i] = (sif_real)(1.0 + (double)i * 1e-3);
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
  sif_field_reserve_masses(g);

  CHECK(sif_field_read_into(FIELD_PATH, g) == SIF_OK, "read failed");

  int identical = 1;
  for (uint64_t i = 0; i < N_P; i++) {
    if (f->x[i] != g->x[i] || f->y[i] != g->y[i] || f->z[i] != g->z[i] ||
        f->vx[i] != g->vx[i] || f->vy[i] != g->vy[i] || f->vz[i] != g->vz[i] ||
        f->masses[i] != g->masses[i]) {
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

  CHECK(sif_field_read_into(NULL, g) == SIF_ERR_INVALID,
    "a NULL path should be SIF_ERR_INVALID");
  CHECK(sif_field_read_into("test_io_does_not_exist.xfield", g) == SIF_ERR_IO,
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
  sif_field_reserve_masses(g);
  CHECK(sif_field_read_into(FIELD_PATH, g) == SIF_ERR_IO,
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
  CHECK(sif_grid_read_into(GRID_PATH, back) == SIF_OK, "read failed");

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
  CHECK(sif_grid_read_into(GRID_PATH, wrong) == SIF_ERR_IO,
    "a geometry mismatch should be rejected");
  sif_grid_free(wrong);

  corrupt_payload(GRID_PATH, 16);
  CHECK(sif_grid_read_into(GRID_PATH, back) == SIF_ERR_IO,
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

  CHECK(sif_catalog_write_ascii(cat, CAT_PATH) == SIF_OK, "write failed");

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
    sif_catalog_free(back);
  }

  CHECK(sif_catalog_write_ascii(NULL, CAT_PATH) == SIF_ERR_INVALID,
    "a NULL catalogue should be SIF_ERR_INVALID");

  sif_catalog_free(cat);
}

int main(void) {
  sif_init(SIF_CONFIG_QUIET);

  test_header_layout();
  test_field_roundtrip();
  test_field_corruption();
  test_grid_roundtrip();
  test_catalog_roundtrip();

  remove(FIELD_PATH);
  remove(GRID_PATH);
  remove(CAT_PATH);

  sif_finalize();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
