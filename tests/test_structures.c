/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* Smoke test for the four reworked data structures. */
#include "sif/structures/bitmask.h"
#include "sif/structures/catalog.h"
#include "sif/structures/cell_linked_list.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"

#include <stdio.h>
#include <stdlib.h>

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: " msg "\n", ##__VA_ARGS__);                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

/* --- bitmask: the concurrent-marking regression --- */
static void test_bitmask_atomic(void) {
  printf("bitmask\n");

  const uint64_t n_bits = (uint64_t)SIF_TEST_SCALE(1 << 20);
  sif_bitmask_t* m = sif_bitmask_alloc(n_bits);
  CHECK(m != NULL, "alloc returned NULL");
  if (!m)
    return;

  CHECK(m->n_words == (n_bits + 63) / 64, "n_words = %llu, expected %llu",
    (unsigned long long)m->n_words, (unsigned long long)((n_bits + 63) / 64));
  CHECK(sif_bitmask_count_set(m) == 0, "fresh mask is not empty");

  /* Every bit set exactly once, from many threads. schedule(static, 1)
   * round-robins consecutive indices across threads, so every 64-bit word is
   * written by several threads at once: a non-atomic setter loses updates. */
#pragma omp parallel for schedule(static, 1)
  for (uint64_t i = 0; i < n_bits; i++) {
    sif_bitmask_set_atomic(m, i);
  }

  uint64_t set = sif_bitmask_count_set(m);
  CHECK(set == n_bits, "concurrent set lost updates: %llu of %llu bits",
    (unsigned long long)set, (unsigned long long)n_bits);

/* And the same for clearing. */
#pragma omp parallel for schedule(static, 1)
  for (uint64_t i = 0; i < n_bits; i += 2) {
    sif_bitmask_unset_atomic(m, i);
  }

  set = sif_bitmask_count_set(m);
  CHECK(set == n_bits / 2, "concurrent unset lost updates: %llu, expected %llu",
    (unsigned long long)set, (unsigned long long)(n_bits / 2));

  CHECK(sif_bitmask_get(m, 0) == 0, "bit 0 should be clear");
  CHECK(sif_bitmask_get(m, 1) == 1, "bit 1 should be set");

  sif_bitmask_free(m);
  sif_bitmask_free(NULL); /* must not crash */
  printf("  ok\n");
}

/* --- catalog: arena growth, trim, zero-capacity --- */
static void test_catalog(void) {
  printf("catalog\n");

  sif_catalog_t* cat = sif_catalog_alloc(4);
  CHECK(cat != NULL, "alloc returned NULL");
  if (!cat)
    return;

  /* Force several reallocations and check nothing is corrupted. */
  const uint64_t n = (uint64_t)SIF_TEST_SCALE(10000);
  for (uint64_t i = 0; i < n; i++) {
    int st = sif_catalog_append(cat, (sif_real)i, (sif_real)(2 * i),
      (sif_real)(3 * i), (sif_real)(i + 1));
    CHECK(
      st == SIF_OK, "append %llu failed with %d", (unsigned long long)i, st);
  }

  CHECK(cat->n_voids == n, "n_voids = %llu, expected %llu",
    (unsigned long long)cat->n_voids, (unsigned long long)n);

  int corrupt = 0;
  for (uint64_t i = 0; i < n; i++) {
    if (cat->cx[i] != (sif_real)i || cat->cy[i] != (sif_real)(2 * i) ||
        cat->cz[i] != (sif_real)(3 * i) || cat->radii[i] != (sif_real)(i + 1))
      corrupt++;
  }
  CHECK(corrupt == 0, "%d entries corrupted across regrowth", corrupt);

  /* Views must stay mutually disjoint inside the arena. */
  CHECK(cat->cy >= cat->cx + cat->capacity, "cx and cy views overlap");
  CHECK(cat->cz >= cat->cy + cat->capacity, "cy and cz views overlap");
  CHECK(cat->radii >= cat->cz + cat->capacity, "cz and radii views overlap");

  uint64_t cap_before = cat->capacity;
  CHECK(cap_before > n, "capacity %llu should exceed n_voids before trim",
    (unsigned long long)cap_before);

  CHECK(sif_catalog_trim(cat) == SIF_OK, "trim failed");
  CHECK(cat->capacity == n, "trim left capacity %llu, expected %llu",
    (unsigned long long)cat->capacity, (unsigned long long)n);
  CHECK(cat->n_voids == n, "trim changed n_voids");
  CHECK(cat->cx[n - 1] == (sif_real)(n - 1), "trim corrupted the last entry");
  CHECK(sif_catalog_trim(cat) == SIF_OK, "second trim should be a no-op");

  sif_catalog_free(cat);

  /* An empty catalog is a legitimate result: it must still allocate and trim.
   */
  sif_catalog_t* empty = sif_catalog_alloc(0);
  CHECK(empty != NULL, "alloc(0) returned NULL");
  if (empty) {
    CHECK(empty->capacity >= 1, "alloc(0) left an unusable capacity");
    CHECK(sif_catalog_trim(empty) == SIF_OK, "trim of an empty catalog failed");
    CHECK(sif_catalog_append(empty, 1, 2, 3, 4) == SIF_OK,
      "append to an alloc(0) catalog failed");
    sif_catalog_free(empty);
  }

  CHECK(sif_catalog_append(NULL, 0, 0, 0, 0) == SIF_ERR_INVALID,
    "append(NULL) should report SIF_ERR_INVALID");
  sif_catalog_free(NULL); /* must not crash */
  printf("  ok\n");
}

/* --- cell linked list: boundary policy and bounds rejection --- */
static void test_cll(void) {
  printf("cell_linked_list\n");

  const uint32_t n_cells = 8;
  const sif_real box = 80.0f;

  sif_cell_linked_list_t* pbc =
    sif_cell_linked_list_alloc(n_cells, box, 16, SIF_PBC_PERIODIC);
  CHECK(pbc != NULL, "periodic alloc returned NULL");
  if (!pbc)
    return;

  CHECK(pbc->periodic == 1, "periodic flag not set");

  /* A point one box to the right of cell 1 must land in cell 1. */
  CHECK(sif_cell_linked_list_insert(pbc, 0, 15.0f, 15.0f, 15.0f) == SIF_OK,
    "in-box insert failed");
  CHECK(
    sif_cell_linked_list_insert(pbc, 1, 15.0f + box, 15.0f, 15.0f) == SIF_OK,
    "wrapped insert failed");

  uint64_t cell_1 = 1ull * n_cells * n_cells + 1ull * n_cells + 1ull;
  CHECK(pbc->head[cell_1] == 1, "wrapped item did not land in cell 1 (head=%d)",
    pbc->head[cell_1]);
  CHECK(pbc->next[1] == 0, "chain not linked (next=%d)", pbc->next[1]);

  /* Negative coordinates wrap to the far side, not to cell 0. */
  CHECK(sif_cell_linked_list_insert(pbc, 2, -5.0f, 5.0f, 5.0f) == SIF_OK,
    "negative insert failed");
  uint64_t cell_last = 7ull * n_cells * n_cells + 0ull * n_cells + 0ull;
  CHECK(pbc->head[cell_last] == 2, "negative coord did not wrap to cell 7");

  /* Out-of-range item ids are rejected instead of corrupting the heap. */
  CHECK(sif_cell_linked_list_insert(pbc, 16, 5.0f, 5.0f, 5.0f) == SIF_ERR_RANGE,
    "insert past capacity was not rejected");
  CHECK(sif_cell_linked_list_ensure_capacity(pbc, 100) == SIF_OK,
    "ensure_capacity failed");
  CHECK(pbc->capacity >= 100, "capacity did not grow");
  CHECK(sif_cell_linked_list_insert(pbc, 99, 5.0f, 5.0f, 5.0f) == SIF_OK,
    "insert after growth failed");
  sif_cell_linked_list_free(pbc);

  /* Open boundaries clamp instead. */
  sif_cell_linked_list_t* open =
    sif_cell_linked_list_alloc(n_cells, box, 16, SIF_PBC_OPEN);
  CHECK(open != NULL, "open alloc returned NULL");
  if (open) {
    CHECK(open->periodic == 0, "open flag not clear");
    CHECK(
      sif_cell_linked_list_insert(open, 0, 15.0f + box, 5.0f, 5.0f) == SIF_OK,
      "open insert failed");
    uint64_t clamped = 7ull * n_cells * n_cells + 0ull * n_cells + 0ull;
    CHECK(open->head[clamped] == 0, "open boundary did not clamp to cell 7");
    sif_cell_linked_list_free(open);
  }

  CHECK(sif_cell_linked_list_alloc(0, box, 16, SIF_DEFAULT) == NULL,
    "alloc with n_cells=0 should fail");
  sif_cell_linked_list_free(NULL); /* must not crash */
  printf("  ok\n");
}

/* --- chain mesh: origin-0 contract, binning/query agreement --- */
static void test_chain_mesh(void) {
  printf("chain_mesh\n");

  const uint64_t n_p = 4096;
  const sif_real box = 100.0f;

  sif_real* x = malloc(n_p * sizeof(sif_real));
  sif_real* y = malloc(n_p * sizeof(sif_real));
  sif_real* z = malloc(n_p * sizeof(sif_real));

  /* 16^3 lattice spanning [0, box) */
  uint64_t idx = 0;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      for (int k = 0; k < 16; k++) {
        x[idx] = (sif_real)(i * 6.25);
        y[idx] = (sif_real)(j * 6.25);
        z[idx] = (sif_real)(k * 6.25);
        idx++;
      }

  sif_field_t* field = sif_field_alloc(n_p);
  sif_field_assign_positions(field, x, y, z);

  sif_chain_mesh_t* mesh = sif_chain_mesh_alloc(8, box, field);
  CHECK(mesh != NULL, "alloc returned NULL for a valid field");

  if (mesh) {
    CHECK(mesh->cell_offsets[mesh->total_cells] == n_p,
      "binned %llu particles, expected %llu",
      (unsigned long long)mesh->cell_offsets[mesh->total_cells],
      (unsigned long long)n_p);

    /* Querying a particle's own position must return that particle: this is
     * what breaks when binning and queries disagree about the origin. */
    int wrong = 0;
    for (uint64_t p = 0; p < n_p; p += 37) {
      uint64_t got = sif_chain_mesh_find_nearest_pbc(mesh, x[p], y[p], z[p]);
      if (got >= n_p || x[got] != x[p] || y[got] != y[p] || z[got] != z[p])
        wrong++;
      uint64_t got_o = sif_chain_mesh_find_nearest_open(mesh, x[p], y[p], z[p]);
      if (got_o >= n_p || x[got_o] != x[p] || y[got_o] != y[p] ||
          z[got_o] != z[p])
        wrong++;
    }
    CHECK(wrong == 0, "%d self-queries returned the wrong particle", wrong);

    /* A point just past the far face is nearest to a particle at the near
     * face under PBC. */
    uint64_t wrapped =
      sif_chain_mesh_find_nearest_pbc(mesh, box - 0.01f, 0.0f, 0.0f);
    CHECK(wrapped < n_p, "pbc query past the far face failed");
    CHECK(x[wrapped] == 0.0f || x[wrapped] == (sif_real)(15 * 6.25),
      "pbc query did not pick a boundary particle (x=%g)", (double)x[wrapped]);

    sif_chain_mesh_free(mesh);
  }

  /* The map back to the field is not optional: it is the only thing that makes
   * a query answer usable, so every mesh carries it and it must be a genuine
   * permutation of the field's indices. */
  sif_chain_mesh_t* idx_mesh = sif_chain_mesh_alloc(8, box, field);
  CHECK(idx_mesh != NULL, "mesh allocation failed");
  if (idx_mesh) {
    CHECK(idx_mesh->original_indices != NULL,
      "original_indices must always be allocated");

    uint8_t* seen = calloc(n_p, 1);
    int bad = 0;
    for (uint64_t i = 0; i < idx_mesh->n_particles; i++) {
      const uint64_t o = idx_mesh->original_indices[i];
      if (o >= n_p || seen[o])
        bad++;
      else
        seen[o] = 1;
    }
    CHECK(bad == 0, "original_indices is not a permutation (%d bad)", bad);
    free(seen);

    /* A query therefore returns something that indexes the field. */
    const uint64_t hit =
      sif_chain_mesh_find_nearest_open(idx_mesh, 1.0f, 1.0f, 1.0f);
    CHECK(hit != UINT64_MAX && hit < n_p,
      "find_nearest returned %llu, which does not index the field",
      (unsigned long long)hit);

    sif_chain_mesh_free(idx_mesh);
  }

  /* Out-of-range coordinates are rejected, not silently folded. The field owns
   * its positions, so the corruption has to go into its copy: writing to the
   * caller's x[] would not reach the mesh. */
  field->x[10] = box + 1.0f;
  CHECK(sif_chain_mesh_alloc(8, box, field) == NULL,
    "a particle outside the box should be rejected");
  field->x[10] = -1.0f;
  CHECK(sif_chain_mesh_alloc(8, box, field) == NULL,
    "a negative coordinate should be rejected");
  field->x[10] = 0.0f / 0.0f; /* NaN */
  CHECK(sif_chain_mesh_alloc(8, box, field) == NULL,
    "a NaN coordinate should be rejected");

  CHECK(sif_chain_mesh_alloc(8, box, NULL) == NULL,
    "NULL field should be rejected");
  sif_chain_mesh_free(NULL); /* must not crash */

  sif_field_free(field);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

int main(void) {
  test_bitmask_atomic();
  test_catalog();
  test_cll();
  test_chain_mesh();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
