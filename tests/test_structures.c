/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* Smoke test for the four reworked data structures. */
#include "sif/core/settings.h"
#include "sif/finder/spherical_finder.h"
#include "sif/measure/delta_moments.h"
#include "sif/structures/bitmask.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/cell_linked_list.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/utils/logger.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* --- catalogue: arena growth, trim, zero-capacity --- */
static void test_catalogue(void) {
  printf("catalogue\n");

  sif_catalogue_t* cat = sif_catalogue_alloc(4);
  CHECK(cat != NULL, "alloc returned NULL");
  if (!cat)
    return;

  /* Force several reallocations and check nothing is corrupted. */
  const uint64_t n = (uint64_t)SIF_TEST_SCALE(10000);
  for (uint64_t i = 0; i < n; i++) {
    int st = sif_catalogue_append(cat, (sif_real)i, (sif_real)(2 * i),
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

  CHECK(sif_catalogue_trim(cat) == SIF_OK, "trim failed");
  CHECK(cat->capacity == n, "trim left capacity %llu, expected %llu",
    (unsigned long long)cat->capacity, (unsigned long long)n);
  CHECK(cat->n_voids == n, "trim changed n_voids");
  CHECK(cat->cx[n - 1] == (sif_real)(n - 1), "trim corrupted the last entry");
  CHECK(sif_catalogue_trim(cat) == SIF_OK, "second trim should be a no-op");

  sif_catalogue_free(cat);

  /* An empty catalogue is a legitimate result: it must still allocate and trim.
   */
  sif_catalogue_t* empty = sif_catalogue_alloc(0);
  CHECK(empty != NULL, "alloc(0) returned NULL");
  if (empty) {
    CHECK(empty->capacity >= 1, "alloc(0) left an unusable capacity");
    CHECK(sif_catalogue_trim(empty) == SIF_OK, "trim of an empty catalogue failed");
    CHECK(sif_catalogue_append(empty, 1, 2, 3, 4) == SIF_OK,
      "append to an alloc(0) catalogue failed");
    sif_catalogue_free(empty);
  }

  /*
   * The footprint columns. Absent until reserved; once reserved they start as
   * unknown for the voids already there, and have to survive every regrowth
   * and a trim with the values they were given, beside the voids they belong
   * to.
   */
  sif_catalogue_t* fp = sif_catalogue_alloc(2);
  CHECK(fp && fp->footprint == NULL && fp->footprint_shell == NULL,
    "a new catalogue should carry no footprint columns");
  if (fp) {
    sif_catalogue_append(fp, 1, 1, 1, 1);
    CHECK(sif_catalogue_reserve_footprint(fp) == SIF_OK &&
            fp->footprint != NULL && fp->footprint_shell != NULL,
      "reserving the footprint columns failed");
    CHECK(sif_catalogue_reserve_footprint(fp) == SIF_OK,
      "reserving them twice should be a no-op");
    CHECK(fp->footprint[0] == SIF_CATALOGUE_FOOTPRINT_UNKNOWN &&
            fp->footprint_shell[0] == SIF_CATALOGUE_FOOTPRINT_UNKNOWN,
      "a void already present should start as unknown");

    const uint64_t m = 1000;
    for (uint64_t i = 1; i < m; i++) {
      sif_catalogue_append(fp, (sif_real)i, 0, 0, 1);
      fp->footprint[i] = (sif_real)i / (sif_real)m;
      fp->footprint_shell[i] = (sif_real)i / (sif_real)(2 * m);
    }
    CHECK(sif_catalogue_trim(fp) == SIF_OK, "trim with footprint columns failed");

    int bad = 0;
    for (uint64_t i = 1; i < m; i++) {
      if (fp->cx[i] != (sif_real)i ||
          fp->footprint[i] != (sif_real)i / (sif_real)m ||
          fp->footprint_shell[i] != (sif_real)i / (sif_real)(2 * m))
        bad++;
    }
    CHECK(
      bad == 0, "%d footprint entries lost their void across regrowth", bad);

    sif_catalogue_append(fp, 0, 0, 0, 1);
    CHECK(fp->footprint[m] == SIF_CATALOGUE_FOOTPRINT_UNKNOWN,
      "a void appended afterwards should read as unknown");
    sif_catalogue_free(fp);
  }
  CHECK(sif_catalogue_reserve_footprint(NULL) == SIF_ERR_INVALID,
    "reserve_footprint(NULL) should report SIF_ERR_INVALID");

  /* Translation moves centres, and nothing else. */
  sif_catalogue_t* tr = sif_catalogue_alloc(2);
  if (tr) {
    sif_catalogue_append(tr, 1.0f, 2.0f, 3.0f, 4.0f);
    sif_catalogue_append(tr, -1.0f, 0.5f, 10.0f, 2.0f);
    const sif_real off[3] = {100.0f, -0.5f, 0.25f};
    CHECK(sif_catalogue_translate(tr, off) == SIF_OK, "translate failed");
    CHECK(tr->cx[0] == 101.0f && tr->cy[0] == 1.5f && tr->cz[0] == 3.25f &&
            tr->radii[0] == 4.0f && tr->cx[1] == 99.0f && tr->cy[1] == 0.0f &&
            tr->cz[1] == 10.25f && tr->radii[1] == 2.0f,
      "translate moved the wrong things");
    CHECK(sif_catalogue_translate(NULL, off) == SIF_ERR_INVALID,
      "translate(NULL) should report SIF_ERR_INVALID");
    sif_catalogue_free(tr);
  }

  CHECK(sif_catalogue_append(NULL, 0, 0, 0, 0) == SIF_ERR_INVALID,
    "append(NULL) should report SIF_ERR_INVALID");
  sif_catalogue_free(NULL); /* must not crash */
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

  /* Removal, from the middle of a chain and from its head: cell 1 holds
   * 3 -> 1 -> 0, and is found from a wrapped point as insertion is. */
  CHECK(sif_cell_linked_list_insert(pbc, 3, 15.0f, 15.0f, 15.0f) == SIF_OK,
    "third insert failed");
  CHECK(
    sif_cell_linked_list_remove(pbc, 1, 15.0f + box, 15.0f, 15.0f) == SIF_OK &&
      pbc->head[cell_1] == 3 && pbc->next[3] == 0,
    "middle removal left head=%d next=%d", pbc->head[cell_1], pbc->next[3]);
  CHECK(sif_cell_linked_list_remove(pbc, 3, 15.0f, 15.0f, 15.0f) == SIF_OK &&
          pbc->head[cell_1] == 0 && pbc->next[0] == -1,
    "head removal left head=%d", pbc->head[cell_1]);
  CHECK(
    sif_cell_linked_list_remove(pbc, 3, 15.0f, 15.0f, 15.0f) == SIF_ERR_RANGE,
    "removing an item twice was not rejected");
  CHECK(
    sif_cell_linked_list_remove(pbc, 2, 15.0f, 15.0f, 15.0f) == SIF_ERR_RANGE &&
      pbc->head[cell_last] == 2,
    "removing from the wrong cell was not rejected");
  CHECK(
    sif_cell_linked_list_remove(NULL, 0, 0.0f, 0.0f, 0.0f) == SIF_ERR_INVALID,
    "removing from NULL was not rejected");

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
  CHECK(sif_field_assign_positions(field, x, y, z) == SIF_OK,
    "sif_field_assign_positions failed");

  sif_chain_mesh_t* mesh = sif_chain_mesh_alloc(8, box, field, SIF_DEFAULT);
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

    /* A point just short of the far face is 0.01 from the lattice point at the
     * origin through the face, and 6.24 from the last row on its own side. */
    uint64_t wrapped =
      sif_chain_mesh_find_nearest_pbc(mesh, box - 0.01f, 0.0f, 0.0f);
    CHECK(wrapped < n_p, "pbc query past the far face failed");
    CHECK(wrapped < n_p && x[wrapped] == 0.0f,
      "pbc query did not reach through the face (x=%g)",
      wrapped < n_p ? (double)x[wrapped] : -1.0);

    sif_chain_mesh_free(mesh);
  }

  /* Off-lattice queries against a brute force, on sparse random tracers so the
   * nearest one is often several cells away and across a face. The cell
   * pruning has to be a lower bound through the periodic images; measuring a
   * cell across the face to its far side skipped the cell holding the answer
   * for ~5-20% of queries here. */
  {
    const uint64_t n_r = 200;
    sif_real* rx = malloc(n_r * sizeof(sif_real));
    sif_real* ry = malloc(n_r * sizeof(sif_real));
    sif_real* rz = malloc(n_r * sizeof(sif_real));
    uint64_t st = 0x9E3779B97F4A7C15ULL;

#define UNI()                                                                  \
  (st = st * 6364136223846793005ULL + 1442695040888963407ULL,                  \
    (sif_real)((double)(st >> 11) / 9007199254740992.0) * box)

    for (uint64_t i = 0; i < n_r; i++) {
      rx[i] = UNI();
      ry[i] = UNI();
      rz[i] = UNI();
    }

    sif_field_t* rf = sif_field_alloc(n_r);
    CHECK(sif_field_assign_positions(rf, rx, ry, rz) == SIF_OK,
      "sif_field_assign_positions failed");

    const uint32_t res[3] = {3, 8, 16};
    for (int r = 0; r < 3; r++) {
      sif_chain_mesh_t* rm = sif_chain_mesh_alloc(res[r], box, rf, SIF_DEFAULT);
      CHECK(rm != NULL, "alloc returned NULL for a valid field");
      if (!rm)
        continue;

      int bad_pbc = 0, bad_open = 0;
      for (int q = 0; q < 2000; q++) {
        const double px = UNI(), py = UNI(), pz = UNI();
        double best_p = INFINITY, best_o = INFINITY;

        for (uint64_t i = 0; i < n_r; i++) {
          double dx = rx[i] - px, dy = ry[i] - py, dz = rz[i] - pz;
          const double d2o = dx * dx + dy * dy + dz * dz;
          dx -= box * round(dx / box);
          dy -= box * round(dy / box);
          dz -= box * round(dz / box);
          const double d2p = dx * dx + dy * dy + dz * dz;
          best_p = d2p < best_p ? d2p : best_p;
          best_o = d2o < best_o ? d2o : best_o;
        }

        const uint64_t gp = sif_chain_mesh_find_nearest_pbc(
          rm, (sif_real)px, (sif_real)py, (sif_real)pz);
        const uint64_t go = sif_chain_mesh_find_nearest_open(
          rm, (sif_real)px, (sif_real)py, (sif_real)pz);

        double dx = rx[gp] - px, dy = ry[gp] - py, dz = rz[gp] - pz;
        dx -= box * round(dx / box);
        dy -= box * round(dy / box);
        dz -= box * round(dz / box);
        if (dx * dx + dy * dy + dz * dz > best_p * (1.0 + 1e-5) + 1e-9)
          bad_pbc++;

        dx = rx[go] - px, dy = ry[go] - py, dz = rz[go] - pz;
        if (dx * dx + dy * dy + dz * dz > best_o * (1.0 + 1e-5) + 1e-9)
          bad_open++;
      }
#undef UNI

      CHECK(bad_pbc == 0, "%u^3 mesh: %d of 2000 pbc queries missed the nearest",
        res[r], bad_pbc);
      CHECK(bad_open == 0,
        "%u^3 mesh: %d of 2000 open queries missed the nearest", res[r],
        bad_open);
      sif_chain_mesh_free(rm);
    }

    sif_field_free(rf);
    free(rx);
    free(ry);
    free(rz);
  }

  /* The map back to the field is not optional: it is the only thing that makes
   * a query answer usable, so every mesh carries it and it must be a genuine
   * permutation of the field's indices. */
  sif_chain_mesh_t* idx_mesh = sif_chain_mesh_alloc(8, box, field, SIF_DEFAULT);
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

  /*
   * SIF_MESH_DROP_INDICES releases the map *after* it has been sorted on, so
   * the claim worth testing is not that the array is gone -- it is that the
   * mesh underneath is the same one. If the flag ever turned into "skip
   * building it", the canonicalisation would lose its key and these two
   * orderings would drift apart.
   */
  sif_chain_mesh_t* kept = sif_chain_mesh_alloc(8, box, field, SIF_DEFAULT);
  sif_chain_mesh_t* dropped =
    sif_chain_mesh_alloc(8, box, field, SIF_MESH_DROP_INDICES);

  CHECK(kept != NULL && dropped != NULL, "mesh allocation failed");

  if (kept && dropped) {
    CHECK(dropped->original_indices == NULL,
      "SIF_MESH_DROP_INDICES should have released the index map");
    CHECK(kept->original_indices != NULL,
      "SIF_DEFAULT should have kept the index map");

    int differs = 0;
    for (uint64_t i = 0; i < kept->n_particles; i++) {
      if (kept->x[i] != dropped->x[i] || kept->y[i] != dropped->y[i] ||
          kept->z[i] != dropped->z[i])
        differs++;
    }
    CHECK(differs == 0,
      "dropping the index map changed the particle order in %d slots", differs);

    for (uint64_t c = 0; c <= kept->total_cells; c++) {
      if (kept->cell_offsets[c] != dropped->cell_offsets[c])
        differs++;
    }
    CHECK(differs == 0, "dropping the index map changed the cell offsets");

    /* And the queries that need the map refuse instead of answering with a
     * mesh offset. The ERROR line this prints is the point, not a failure. */
    CHECK(
      sif_chain_mesh_find_nearest_pbc(dropped, 1.0f, 1.0f, 1.0f) == UINT64_MAX,
      "find_nearest_pbc should refuse a mesh with no index map");
    CHECK(
      sif_chain_mesh_find_nearest_open(dropped, 1.0f, 1.0f, 1.0f) == UINT64_MAX,
      "find_nearest_open should refuse a mesh with no index map");
  }

  /*
   * The consuming constructor must be indistinguishable from the copying one.
   * It bins the same particles into the same cells and canonicalises on the
   * same key, so the only thing that may differ is where the storage came
   * from -- and if the deferred permutation is wrong, the positions land in
   * the wrong slots and this catches it immediately.
   */
  sif_field_t* victim = sif_field_alloc(n_p);
  CHECK(sif_field_assign_positions(victim, x, y, z) == SIF_OK,
    "sif_field_assign_positions failed");

  sif_chain_mesh_t* consumed =
    sif_chain_mesh_alloc_consume(8, box, victim, SIF_DEFAULT);
  CHECK(consumed != NULL, "consuming mesh allocation failed");

  if (kept && consumed) {
    CHECK(consumed->n_particles == kept->n_particles,
      "consuming mesh holds %llu particles, expected %llu",
      (unsigned long long)consumed->n_particles,
      (unsigned long long)kept->n_particles);

    int differs = 0;
    for (uint64_t i = 0; i < kept->n_particles; i++) {
      if (kept->x[i] != consumed->x[i] || kept->y[i] != consumed->y[i] ||
          kept->z[i] != consumed->z[i] ||
          kept->original_indices[i] != consumed->original_indices[i])
        differs++;
    }
    CHECK(differs == 0,
      "consuming and copying constructors disagree in %d slots", differs);

    for (uint64_t c = 0; c <= kept->total_cells; c++) {
      if (kept->cell_offsets[c] != consumed->cell_offsets[c])
        differs++;
    }
    CHECK(differs == 0, "consuming constructor produced different cells");

    /* Queries still work, so the payload really did move rather than being
     * left in field order and merely relabelled. */
    int wrong = 0;
    for (uint64_t p = 0; p < n_p; p += 37) {
      uint64_t got =
        sif_chain_mesh_find_nearest_pbc(consumed, x[p], y[p], z[p]);
      if (got >= n_p || x[got] != x[p] || y[got] != y[p] || z[got] != z[p])
        wrong++;
    }
    CHECK(
      wrong == 0, "%d self-queries on the consuming mesh were wrong", wrong);
  }

  /* The field it ate must be left empty and still safe to release. */
  CHECK(
    victim->n_particles == 0, "a consumed field should report no particles");
  CHECK(victim->x == NULL && victim->y == NULL && victim->z == NULL,
    "a consumed field should hold no position arrays");
  sif_field_free(victim); /* must not double-free the mesh's storage */

  sif_chain_mesh_free(consumed);
  sif_chain_mesh_free(kept);
  sif_chain_mesh_free(dropped);

  /*
   * The per-cell weight table. An unweighted mesh must not pay for one, and a
   * weighted one -- built either way -- has to hold exactly the sum of each
   * cell's own weights, with the cells adding up to the total. The weights are
   * multiples of a quarter so every sum here is exact and the comparison can
   * be too.
   */
  {
    sif_real* w = malloc(n_p * sizeof(sif_real));
    double w_total = 0.0;
    for (uint64_t i = 0; i < n_p; i++) {
      w[i] = 1.0f + 0.25f * (sif_real)(i % 7);
      w_total += (double)w[i];
    }

    sif_chain_mesh_t* plain = sif_chain_mesh_alloc(8, box, field, SIF_DEFAULT);
    CHECK(plain && plain->cell_weights == NULL,
      "an unweighted mesh should carry no per-cell weight table");
    CHECK(plain && plain->total_weight == (double)n_p,
      "an unweighted mesh should report its tracer count as its weight");
    sif_chain_mesh_free(plain);

    sif_field_t* wf = sif_field_alloc(n_p);
    CHECK(sif_field_assign_positions(wf, x, y, z) == SIF_OK,
      "sif_field_assign_positions failed");
    CHECK(sif_field_assign_weights(wf, w) == SIF_OK,
      "sif_field_assign_weights failed");

    sif_field_t* wf_victim = sif_field_alloc(n_p);
    CHECK(sif_field_assign_positions(wf_victim, x, y, z) == SIF_OK,
      "sif_field_assign_positions failed");
    CHECK(sif_field_assign_weights(wf_victim, w) == SIF_OK,
      "sif_field_assign_weights failed");

    sif_chain_mesh_t* built[2] = {
      sif_chain_mesh_alloc(8, box, wf, SIF_DEFAULT),
      sif_chain_mesh_alloc_consume(8, box, wf_victim, SIF_DEFAULT)};
    const char* how[2] = {"copying", "consuming"};

    for (int m = 0; m < 2; m++) {
      const sif_chain_mesh_t* wm = built[m];
      CHECK(wm && wm->cell_weights != NULL,
        "the %s constructor built no per-cell weight table", how[m]);
      if (!wm || !wm->cell_weights)
        continue;

      int bad = 0;
      double cells = 0.0;
      for (uint64_t c = 0; c < wm->total_cells; c++) {
        double sum = 0.0;
        for (uint64_t p = wm->cell_offsets[c]; p < wm->cell_offsets[c + 1]; p++)
          sum += (double)wm->weights[p];
        if ((double)wm->cell_weights[c] != sum)
          bad++;
        cells += (double)wm->cell_weights[c];
      }

      CHECK(bad == 0, "%s constructor: %d cells hold the wrong weight", how[m],
        bad);
      CHECK(cells == w_total && wm->total_weight == w_total,
        "%s constructor: cells sum to %.2f, total_weight is %.2f, expected "
        "%.2f",
        how[m], cells, wm->total_weight, w_total);
    }

    sif_chain_mesh_free(built[0]);
    sif_chain_mesh_free(built[1]);
    sif_field_free(wf);
    sif_field_free(wf_victim);
    free(w);
  }

  /* Out-of-range coordinates are rejected, not silently folded. The field owns
   * its positions, so the corruption has to go into its copy: writing to the
   * caller's x[] would not reach the mesh. */
  field->x[10] = box + 1.0f;
  CHECK(sif_chain_mesh_alloc(8, box, field, SIF_DEFAULT) == NULL,
    "a particle outside the box should be rejected");
  field->x[10] = -1.0f;
  CHECK(sif_chain_mesh_alloc(8, box, field, SIF_DEFAULT) == NULL,
    "a negative coordinate should be rejected");
  field->x[10] = 0.0f / 0.0f; /* NaN */
  CHECK(sif_chain_mesh_alloc(8, box, field, SIF_DEFAULT) == NULL,
    "a NaN coordinate should be rejected");

  CHECK(sif_chain_mesh_alloc(8, box, NULL, SIF_DEFAULT) == NULL,
    "NULL field should be rejected");
  sif_chain_mesh_free(NULL); /* must not crash */

  sif_field_free(field);
  free(x);
  free(y);
  free(z);
  printf("  ok\n");
}

/*
 * This suite never calls sif_system_init(), which makes it the place to check
 * what an entry point that needs the system state does without it: fail with
 * SIF_ERR_UNINITIALISED, returned or recorded for sif_error_status(). It used
 * to end the process, taking a Python interpreter with it.
 */
static void test_uninitialised(void) {
  printf("entry points before sif_system_init\n");

  sif_grid_t* g = sif_grid_alloc(16, 100.0f);
  CHECK(g != NULL, "grid allocation failed");
  if (g) {
    for (uint64_t i = 0; i < g->total_cells; i++)
      g->values[i] = (sif_real)(i % 7) * 0.1f - 0.3f;
    g->content = SIF_GRID_DENSITY_CONTRAST;

    const sif_real radii[] = {20.0f, 15.0f};
    sif_error_clear();
    sif_catalogue_t* cat = sif_finder_spherical(g, radii, 2, -0.2f, 0.0f, 0);
    CHECK(cat == NULL,
      "a finder run before sif_system_init() should fail, not return a "
      "catalogue");
    CHECK(sif_error_status() == SIF_ERR_UNINITIALISED,
      "a finder run before sif_system_init() should record "
      "SIF_ERR_UNINITIALISED");
    sif_catalogue_free(cat);

    sif_error_clear();
    sif_delta_moments_t* m = sif_delta_moments_grid(g, radii, 2, 2, 0, 1, 0);
    CHECK(m == NULL, "delta moments before sif_system_init() should fail");
    CHECK(sif_error_status() == SIF_ERR_UNINITIALISED,
      "delta moments before sif_system_init() should record "
      "SIF_ERR_UNINITIALISED");
    sif_delta_moments_free(m);

    sif_grid_free(g);
  }

  CHECK(sif_setting_set("key", "value") == SIF_ERR_UNINITIALISED,
    "sif_setting_set() before sif_system_init() should return "
    "SIF_ERR_UNINITIALISED");
  CHECK(strcmp(sif_setting_get("key", "fallback"), "fallback") == 0,
    "sif_setting_get() before sif_system_init() should return its fallback");
  printf("  ok\n");
}

/*
 * The constructors that return a pointer record why they failed, so a caller
 * -- the Python bindings, most of all -- can tell a refusal from running out
 * of memory, and the message names the part that was wrong.
 */
static void test_error_reasons(void) {
  printf("error reasons\n");

  sif_error_clear();
  CHECK(sif_grid_alloc(0, 100.0f) == NULL, "a zero-cell grid should fail");
  CHECK(sif_error_status() == SIF_ERR_INVALID && error_has("n_cells: 0"),
    "a zero-cell grid: status %d, message '%s'", sif_error_status(),
    sif_error_message());

  /* Particle 1 is outside the box; the message has to name it. */
  sif_real x[3] = {1.0f, 150.0f, 3.0f}, y[3] = {1, 2, 3}, z[3] = {1, 2, 3};
  sif_field_t* f = sif_field_alloc(3);
  CHECK(f && sif_field_assign_positions(f, x, y, z) == SIF_OK,
    "field setup failed");

  sif_error_clear();
  CHECK(sif_chain_mesh_alloc(4, 100.0f, f, SIF_DEFAULT) == NULL,
    "a mesh over a particle outside the box should fail");
  CHECK(sif_error_status() == SIF_ERR_RANGE &&
          error_has("1 of 3 particles outside the box") &&
          error_has("first: particle 1 at (150, 2, 2)"),
    "out of box: status %d, message '%s'", sif_error_status(),
    sif_error_message());

  /* Refused before the handover: the consuming build leaves the field. */
  sif_error_clear();
  CHECK(sif_chain_mesh_alloc_consume(4, 100.0f, f, SIF_DEFAULT) == NULL,
    "a consuming mesh over a particle outside the box should fail");
  CHECK(f->n_particles == 3 && f->x != NULL,
    "a refused consuming build should leave the field as it was");

  sif_grid_t* g = sif_grid_alloc(4, 100.0f);
  sif_error_clear();
  CHECK(g && sif_grid_assign_cic(g, f) == SIF_ERR_RANGE &&
          error_has("first: particle 1"),
    "deposit out of box: message '%s'", sif_error_message());

  /* An empty field, which is what a consuming build leaves behind. */
  sif_field_t* empty = sif_field_alloc(0);
  sif_error_clear();
  CHECK(g && sif_grid_assign_cic(g, empty) == SIF_ERR_INVALID &&
          error_has("field: 0 particles"),
    "deposit of an empty field: message '%s'", sif_error_message());

  sif_error_clear();
  CHECK(sif_chain_mesh_alloc(4, 100.0f, empty, SIF_DEFAULT) == NULL &&
          sif_error_status() == SIF_ERR_INVALID &&
          error_has("field: 0 particles"),
    "a mesh over an empty field: status %d, message '%s'",
    sif_error_status(), sif_error_message());

  sif_error_clear();
  CHECK(g && sif_grid_to_density_contrast(g) == SIF_ERR_RANGE &&
          error_has("nothing has been deposited"),
    "contrast of an empty grid: message '%s'", sif_error_message());

  sif_grid_free(g);
  sif_field_free(empty);
  sif_field_free(f);

  /* A metadata key is refused with the reason, and a failure to grow the
   * table is no longer reported as a bad key. */
  sif_catalogue_t* cat = sif_catalogue_alloc(1);
  sif_error_clear();
  CHECK(cat && sif_catalogue_meta_int_set(cat, "my-key", 1) == SIF_ERR_INVALID &&
          error_has("key has '-' at character 3"),
    "bad metadata key: message '%s'", sif_error_message());
  sif_catalogue_free(cat);

  printf("  ok\n");
}

/* Two ULPs of `v` up or down: within the binning's tolerance at any box. */
static sif_real ulps_off(sif_real v, int up) {
  const sif_real to = up ? SIF_REAL_MAX_VAL : -SIF_REAL_MAX_VAL;
  return SIF_REAL_NEXT_AFTER(SIF_REAL_NEXT_AFTER(v, to), to);
}

/*
 * Coordinates within rounding of a box face are binned as their periodic
 * fold, without a wrap; anything further out is still refused.
 */
static void test_box_edge_rounding(void) {
  printf("box edge rounding\n");

  const sif_real box = 100.0f;

  /* Particle 0 on the upper face, 1 just past it, 2 just below 0. */
  sif_real ex[3] = {box, ulps_off(box, 1), ulps_off(0.0f, 0)};
  sif_real ey[3] = {10.0f, box, 20.0f}, ez[3] = {30.0f, 40.0f, 0.0f};
  /* Their folds, which is what a wrap would have produced. */
  sif_real fx[3], fy[3], fz[3];
  for (int i = 0; i < 3; i++) {
    fx[i] = ex[i] >= box ? ex[i] - box : ex[i] < 0 ? ex[i] + box : ex[i];
    fy[i] = ey[i] >= box ? ey[i] - box : ey[i];
    fz[i] = ez[i];
    if (fx[i] >= box)
      fx[i] = 0.0f;
  }

  sif_field_t* edge = sif_field_alloc(3);
  sif_field_t* folded = sif_field_alloc(3);
  CHECK(edge && folded &&
          sif_field_assign_positions(edge, ex, ey, ez) == SIF_OK &&
          sif_field_assign_positions(folded, fx, fy, fz) == SIF_OK,
    "field setup failed");

  sif_grid_t* ge = sif_grid_alloc(8, box);
  sif_grid_t* gf = sif_grid_alloc(8, box);
  CHECK(ge && sif_grid_assign_cic(ge, edge) == SIF_OK,
    "a deposit within rounding of the faces was refused: %s",
    sif_error_message());
  CHECK(gf && sif_grid_assign_cic(gf, folded) == SIF_OK, "folded deposit");
  CHECK(ge && gf &&
          memcmp(ge->values, gf->values, ge->total_cells * sizeof(sif_real)) ==
            0,
    "the edge particles should deposit exactly as their folds do");
  CHECK(edge->x[0] == box, "the deposit should leave the field unchanged");

  /* The copying mesh folds its copy; the field keeps its values. */
  sif_chain_mesh_t* m = sif_chain_mesh_alloc(4, box, edge, SIF_DEFAULT);
  CHECK(m != NULL, "a mesh within rounding of the faces was refused: %s",
    sif_error_message());
  for (uint64_t i = 0; m && i < m->n_particles; i++)
    CHECK(m->x[i] >= 0.0f && m->x[i] < box && m->y[i] >= 0.0f &&
            m->y[i] < box && m->z[i] >= 0.0f && m->z[i] < box,
      "mesh slot %llu not folded into the box", (unsigned long long)i);
  sif_chain_mesh_free(m);

  /* The consuming one folds the columns it takes. */
  m = sif_chain_mesh_alloc_consume(4, box, edge, SIF_DEFAULT);
  CHECK(m != NULL, "a consuming mesh within rounding was refused");
  for (uint64_t i = 0; m && i < m->n_particles; i++)
    CHECK(m->x[i] >= 0.0f && m->x[i] < box && m->y[i] >= 0.0f &&
            m->y[i] < box,
      "consumed mesh slot %llu not folded", (unsigned long long)i);
  sif_chain_mesh_free(m);

  /* Past the tolerance is a wrong box, not rounding. */
  sif_real px[1] = {box * 1.001f}, py[1] = {1.0f}, pz[1] = {1.0f};
  sif_field_t* past = sif_field_alloc(1);
  CHECK(past && sif_field_assign_positions(past, px, py, pz) == SIF_OK,
    "field setup failed");
  sif_error_clear();
  CHECK(gf && sif_grid_assign_cic(gf, past) == SIF_ERR_RANGE &&
          error_has("outside the box"),
    "a particle 0.1%% past the box should still be refused");

  sif_field_free(past);
  sif_grid_free(ge);
  sif_grid_free(gf);
  sif_field_free(edge);
  sif_field_free(folded);
  printf("  ok\n");
}

int main(void) {
  test_uninitialised();
  test_box_edge_rounding();
  test_error_reasons();
  test_bitmask_atomic();
  test_catalogue();
  test_cll();
  test_chain_mesh();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures != 0;
}
