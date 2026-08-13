/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/catalog.h"

#include <stdlib.h>
#include <string.h>

#include "sif/utils/align.h"
#include "sif/utils/logger.h"

/* Number of sif_real views packed into the arena: cx, cy, cz, radii */
#define CATALOG_N_VIEWS 4

/* Each view starts on a cache-line boundary, so the per-view stride is the
 * capacity rounded up to a whole number of sif_real per cache line. */
static inline uint64_t catalog_stride(uint64_t capacity) {
  const uint64_t align_elements = SIF_CACHE_LINE / sizeof(sif_real);
  return (capacity + align_elements - 1) & ~(align_elements - 1);
}

/*
 * Allocates one arena for `capacity` voids and hands out the four views.
 * Returns NULL on failure without touching the outputs.
 */
static sif_real* catalog_block_alloc(uint64_t capacity, sif_real** out_cx,
  sif_real** out_cy, sif_real** out_cz, sif_real** out_radii) {

  const uint64_t stride = catalog_stride(capacity);

  sif_real* block =
    sif_malloc_aligned(CATALOG_N_VIEWS * stride * sizeof(sif_real));
  if (!block)
    return NULL;

  *out_cx = block;
  *out_cy = block + stride;
  *out_cz = block + (2 * stride);
  *out_radii = block + (3 * stride);

  return block;
}

/*
 * Moves the stored voids into a freshly allocated arena of `new_capacity` and
 * swaps it in. On failure the catalog is left exactly as it was.
 */
static int catalog_resize(sif_catalog_t* catalog, uint64_t new_capacity) {
  sif_real *new_cx, *new_cy, *new_cz, *new_radii;

  sif_real* new_block =
    catalog_block_alloc(new_capacity, &new_cx, &new_cy, &new_cz, &new_radii);

  if (!new_block) {
    SIF_LOG_ERROR("void_catalog",
      "failed to resize from %" PRIu64 " to %" PRIu64 " voids",
      catalog->capacity, new_capacity);
    return SIF_ERR_ALLOC;
  }

  const uint64_t n =
    (catalog->n_voids < new_capacity) ? catalog->n_voids : new_capacity;

  if (n > 0) {
    memcpy(new_cx, catalog->cx, n * sizeof(sif_real));
    memcpy(new_cy, catalog->cy, n * sizeof(sif_real));
    memcpy(new_cz, catalog->cz, n * sizeof(sif_real));
    memcpy(new_radii, catalog->radii, n * sizeof(sif_real));
  }

  sif_free_aligned(catalog->_block);

  catalog->_block = new_block;
  catalog->cx = new_cx;
  catalog->cy = new_cy;
  catalog->cz = new_cz;
  catalog->radii = new_radii;
  catalog->capacity = new_capacity;
  catalog->n_voids = n;

  return SIF_OK;
}

sif_catalog_t* sif_catalog_alloc(uint64_t initial_capacity) {
  if (initial_capacity == 0)
    initial_capacity = 1;

  sif_catalog_t* cat = malloc(sizeof(sif_catalog_t));
  if (!cat) {
    SIF_LOG_ERROR("void_catalog", "failed to allocate void catalog (%zu bytes)",
      sizeof(sif_catalog_t));
    return NULL;
  }

  cat->n_voids = 0;
  cat->capacity = initial_capacity;

  cat->_block = catalog_block_alloc(
    initial_capacity, &cat->cx, &cat->cy, &cat->cz, &cat->radii);

  if (!cat->_block) {
    SIF_LOG_ERROR("void_catalog",
      "failed to allocate void catalog arena for %" PRIu64 " voids",
      initial_capacity);
    free(cat);
    return NULL;
  }

  SIF_LOG_TRACE("void_catalog", "void catalog initialized");
  return cat;
}

void sif_catalog_free(sif_catalog_t* catalog) {
  if (!catalog)
    return;

  sif_free_aligned(catalog->_block);
  free(catalog);
}

int sif_catalog_append(
  sif_catalog_t* catalog, sif_real x, sif_real y, sif_real z, sif_real r) {

  if (!catalog)
    return SIF_ERR_INVALID;

  if (catalog->n_voids >= catalog->capacity) {
    int status = catalog_resize(catalog, catalog->capacity << 1);
    if (status != SIF_OK)
      return status;

    SIF_LOG_TRACE(
      "void_catalog", "resized catalog to %" PRIu64, catalog->capacity);
  }

  const uint64_t n = catalog->n_voids;
  catalog->cx[n] = x;
  catalog->cy[n] = y;
  catalog->cz[n] = z;
  catalog->radii[n] = r;
  catalog->n_voids++;

  return SIF_OK;
}

int sif_catalog_trim(sif_catalog_t* catalog) {
  if (!catalog)
    return SIF_ERR_INVALID;

  /* An empty catalog is a legitimate result, not an error. Trim it to the
   * minimum usable allocation rather than warning about it. */
  const uint64_t target = (catalog->n_voids > 0) ? catalog->n_voids : 1;

  if (target == catalog->capacity) {
    SIF_LOG_TRACE(
      "void_catalog", "catalog already exact, no trimming required");
    return SIF_OK;
  }

  int status = catalog_resize(catalog, target);
  if (status != SIF_OK) {
    /* Not fatal: the catalog is still correct, just larger than necessary. */
    SIF_LOG_WARNING(
      "void_catalog", "could not trim catalog, keeping current allocation");
    return status;
  }

  SIF_LOG_TRACE(
    "void_catalog", "trimmed catalog to %" PRIu64 " voids", catalog->n_voids);
  return SIF_OK;
}
