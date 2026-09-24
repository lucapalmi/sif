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
#include <string.h>

/*
 * Longest row the reader accepts. Six values at the widest SIF_PRI_REAL are
 * about 150 characters, so this is room to spare, not a limit a written file
 * can reach; a longer line is refused as malformed rather than split.
 */
#define CATALOG_LINE_MAX 1024

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
      SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL,
      catalog->cx[i], catalog->cy[i], catalog->cz[i], catalog->radii[i]);

    /* The footprint rides along as two more columns, only when there is one:
     * a box catalogue keeps the four-column layout it always had. */
    if (catalog->footprint)
      fprintf(file, " " SIF_PRI_REAL " " SIF_PRI_REAL, catalog->footprint[i],
        catalog->footprint_shell[i]);

    fputc('\n', file);
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

/*
 * The next line that holds anything, into `line`. Blank lines are skipped, as
 * the whitespace-driven reader this replaced skipped them.
 *
 * @return 1 with a line, 0 at the end of the file, -1 on a line too long to
 * be a row.
 */
static int catalog_next_line(FILE* file, char* line) {
  while (fgets(line, CATALOG_LINE_MAX, file)) {
    const size_t len = strlen(line);
    if (len == CATALOG_LINE_MAX - 1 && line[len - 1] != '\n' && !feof(file))
      return -1;

    if (strspn(line, " \t\r\n") != len)
      return 1;
  }
  return 0;
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

  char line[CATALOG_LINE_MAX];
  uint64_t n_voids;
  if (catalog_next_line(file, line) != 1 ||
      sscanf(line, "%" SCNu64, &n_voids) != 1) {
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

  /*
   * Rows are required to be there: the header said how many, and a file that
   * stops short is truncated rather than merely small. Reading fewer would
   * hand back a catalogue whose tail is uninitialized memory.
   *
   * The first row decides the layout -- four columns, or six with the
   * footprint -- and every row after it has to agree. Read line by line
   * rather than value by value for exactly that: a row is what has a column
   * count.
   */
  int n_columns = 0;

  for (uint64_t i = 0; i < n_voids; i++) {
    sif_real v[6];
    int got = -1;

    if (catalog_next_line(file, line) == 1) {
      got = sscanf(line,
        SIF_SCN_REAL " " SIF_SCN_REAL " " SIF_SCN_REAL " " SIF_SCN_REAL
                     " " SIF_SCN_REAL " " SIF_SCN_REAL,
        &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]);
    }

    if (i == 0 && (got == 4 || got == 6)) {
      n_columns = got;
      if (n_columns == 6 && sif_catalog_reserve_footprint(catalog) != SIF_OK) {
        sif_catalog_free(catalog);
        fclose(file);
        return NULL;
      }
    }

    if (got != n_columns) {
      SIF_LOG_ERROR("io",
        "void %" PRIu64 " of %s has %d readable columns, expected %d", i,
        filepath, got < 0 ? 0 : got, n_columns ? n_columns : 4);
      sif_catalog_free(catalog);
      fclose(file);
      return NULL;
    }

    catalog->cx[i] = v[0];
    catalog->cy[i] = v[1];
    catalog->cz[i] = v[2];
    catalog->radii[i] = v[3];
    if (n_columns == 6) {
      catalog->footprint[i] = v[4];
      catalog->footprint_shell[i] = v[5];
    }
  }

  /* Filled in directly rather than through sif_catalog_append(), so the size
   * has to be set by hand. */
  catalog->n_voids = n_voids;

  fclose(file);
  SIF_LOG_INFO("io", "loaded %" PRIu64 " voids from %s (ASCII%s)", n_voids,
    filepath, n_columns == 6 ? ", with footprint" : "");

  return catalog;
}
