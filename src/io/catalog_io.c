/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/catalog_io.h"

#include "sif/utils/logger.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int sif_catalog_write_ascii(
  const sif_catalog_t* catalog, const char* filepath) {
  if (!catalog || !filepath) {
    SIF_LOG_ERROR("io", "invalid arguments for write_catalog_ascii");
    return SIF_ERR_INVALID;
  }

  FILE* file = fopen(filepath, "w");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for writing", filepath);
    return SIF_ERR_IO;
  }

  /* The count goes first so the reader can allocate the catalogue once,
   * instead of growing it a void at a time or scanning the file twice. */
  fprintf(file, "%" PRIu64 "\n", catalog->n_voids);

  /* SIF_PRI_REAL round-trips: a catalogue written and read back gives the same
   * radii bit for bit, which is what lets a size function computed from the
   * file match one computed in memory. */
  for (uint64_t i = 0; i < catalog->n_voids; i++) {
    fprintf(file,
      SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL "\n",
      catalog->cx[i], catalog->cy[i], catalog->cz[i], catalog->radii[i]);
  }

  /* fprintf() reports nothing useful per call, so the stream's error flag and
   * the flush inside fclose() are what say whether the file actually reached
   * the disk. Without this a full quota reads back as a short catalogue. */
  const bool ok = (ferror(file) == 0);
  if (fclose(file) != 0 || !ok) {
    SIF_LOG_ERROR("io", "failed to flush %s to disk", filepath);
    return SIF_ERR_IO;
  }

  SIF_LOG_INFO(
    "io", "saved %" PRIu64 " voids to %s (ASCII)", catalog->n_voids, filepath);
  return SIF_OK;
}

sif_catalog_t* sif_catalog_read_ascii(const char* filepath) {
  if (!filepath) {
    SIF_LOG_ERROR("io", "invalid filepath for read_catalog_ascii");
    return NULL;
  }

  FILE* file = fopen(filepath, "r");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for reading", filepath);
    return NULL;
  }

  uint64_t n_voids;
  if (fscanf(file, "%" SCNu64, &n_voids) != 1) {
    SIF_LOG_ERROR("io", "failed to read void count from %s", filepath);
    fclose(file);
    return NULL;
  }

  sif_catalog_t* catalog = sif_catalog_alloc(n_voids);
  if (!catalog) {
    SIF_LOG_ERROR("io", "failed to allocate catalog for loading");
    fclose(file);
    return NULL;
  }

  /* Rows are required to be there: the header said how many, and a file that
   * stops short is truncated rather than merely small. Reading fewer would
   * hand back a catalogue whose tail is uninitialized memory. */
  for (uint64_t i = 0; i < n_voids; i++) {
    sif_real x, y, z, r;
    if (fscanf(file,
          SIF_SCN_REAL " " SIF_SCN_REAL " " SIF_SCN_REAL " " SIF_SCN_REAL, &x,
          &y, &z, &r) != 4) {
      SIF_LOG_ERROR(
        "io", "failed reading void %" PRIu64 " from %s", i, filepath);
      sif_catalog_free(catalog);
      fclose(file);
      return NULL;
    }

    catalog->cx[i] = x;
    catalog->cy[i] = y;
    catalog->cz[i] = z;
    catalog->radii[i] = r;
  }

  /* Filled in directly rather than through sif_catalog_append(), so the size
   * has to be set by hand. */
  catalog->n_voids = n_voids;

  fclose(file);
  SIF_LOG_INFO(
    "io", "loaded %" PRIu64 " voids from %s (ASCII)", n_voids, filepath);

  return catalog;
}
