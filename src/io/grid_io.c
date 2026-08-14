/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/grid_io.h"

#include "internal.h"
#include "sif/utils/crc32.h"
#include "sif/utils/logger.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Wrapped in do/while so an unbraced `if` around one of these cannot swallow
 * the failure path. */
#define CHECK_WRITE(ptr, size, count, stream, msg)                             \
  do {                                                                         \
    if (fwrite((ptr), (size), (count), (stream)) != (count)) {                 \
      SIF_LOG_ERROR("xgrid", "failed to write %s", (msg));                     \
      fclose(file);                                                            \
      return SIF_ERR_IO;                                                       \
    }                                                                          \
  } while (0)

#define CHECK_PREAD(fd, dest, bytes, offset, msg, file_ptr)                    \
  do {                                                                         \
    if (sif__io_pread_parallel((fd), (dest), (bytes), (offset)) != SIF_OK) {   \
      SIF_LOG_ERROR("xgrid", "parallel read error for %s", (msg));             \
      fclose(file_ptr);                                                        \
      return SIF_ERR_IO;                                                       \
    }                                                                          \
  } while (0)

int sif_grid_write(const char* filepath, const sif_grid_t* grid) {
  /* No key: a grid written through the public entry point records no
   * provenance, because its caller has none to record. */
  return sif__grid_write_keyed(filepath, grid, NULL);
}

int sif__grid_write_keyed(
  const char* filepath, const sif_grid_t* grid, const uint64_t source_key[2]) {
  if (!filepath || !grid)
    return SIF_ERR_INVALID;

  if (!grid->values || grid->total_cells == 0) {
    SIF_LOG_ERROR("xgrid", "grid carries no cell values to write");
    return SIF_ERR_INVALID;
  }

  FILE* file = fopen(filepath, "wb");
  if (!file) {
    SIF_LOG_ERROR("xgrid", "could not open %s for writing", filepath);
    return SIF_ERR_IO;
  }

  sif_xgrid_header_t header;
  memset(&header, 0, sizeof(sif_xgrid_header_t));

  strncpy(header.magic, SIF_XGRID_MAGIC, 4);
  header.version = SIF_XGRID_VERSION;
  header.n_cells = grid->n_cells;
  header.is_double = (sizeof(sif_real) == 8) ? 1 : 0;
  header.total_cells = grid->total_cells;
  header.box_length = (double)grid->box_length;
  /* What the cells hold travels with them: weights and density contrast are
   * the same bytes, and a reader cannot tell them apart by looking. */
  header.content = (uint32_t)grid->content;

  /* Left at the zero memset put there when there is no key, which is how a
   * reader tells "no provenance recorded" from "provenance that disagrees". */
  if (source_key) {
    header.source_key[0] = source_key[0];
    header.source_key[1] = source_key[1];
  }

  const size_t array_bytes = (size_t)grid->total_cells * sizeof(sif_real);
  header.crc32 = sif_crc32(grid->values, array_bytes);

  CHECK_WRITE(&header, sizeof(sif_xgrid_header_t), 1, file, "header");
  CHECK_WRITE(
    grid->values, sizeof(sif_real), grid->total_cells, file, "cell values");

  /* fwrite() succeeding only means the bytes reached the stdio buffer; a full
   * disk surfaces at the flush inside fclose(). This one matters beyond the
   * caller: sif_grid_assign_cic() renames a successful write into the shared
   * cache, so an unchecked truncation would be published to every later run. */
  const bool ok = (ferror(file) == 0);
  if (fclose(file) != 0 || !ok) {
    SIF_LOG_ERROR("xgrid", "failed to flush %s to disk", filepath);
    return SIF_ERR_IO;
  }

  return SIF_OK;
}

/*
 * Rejects a header this build cannot read into any grid.
 *
 * Both readers need it before trusting a single field: sif_grid_read() sizes
 * an allocation from n_cells, and n_cells^3 cells of a file that is not an
 * xgrid is an arbitrary number of gigabytes.
 */
static int header_validate(
  const sif_xgrid_header_t* header, const char* filepath) {
  if (strncmp(header->magic, SIF_XGRID_MAGIC, 4) != 0) {
    SIF_LOG_ERROR("xgrid", "%s is not a valid xgrid binary", filepath);
    return SIF_ERR_IO;
  }

  if (header->version > SIF_XGRID_VERSION) {
    SIF_LOG_ERROR("xgrid",
      "%s is version %u, but this build reads up to version %u", filepath,
      header->version, (unsigned)SIF_XGRID_VERSION);
    return SIF_ERR_IO;
  }

  /* The payload is raw sif_real, so a precision mismatch is a
   * reinterpretation rather than a conversion. */
  const uint32_t current_is_double = (sizeof(sif_real) == 8) ? 1u : 0u;
  if (header->is_double != current_is_double) {
    SIF_LOG_ERROR("xgrid", "%s is %s, this build is %s", filepath,
      header->is_double ? "FP64" : "FP32", current_is_double ? "FP64" : "FP32");
    return SIF_ERR_IO;
  }

  return SIF_OK;
}

int sif_grid_read_into(const char* filepath, sif_grid_t* grid) {
  return sif__grid_read_into_keyed(filepath, grid, NULL);
}

int sif__grid_read_into_keyed(
  const char* filepath, sif_grid_t* grid, const uint64_t expect_key[2]) {
  if (!filepath || !grid)
    return SIF_ERR_INVALID;

  FILE* file = fopen(filepath, "rb");
  if (!file) {
    /* Quiet on purpose: the CIC cache calls this to find out whether a grid
     * has been computed before, and a miss is the ordinary case. */
    return SIF_ERR_IO;
  }

  sif_xgrid_header_t header;
  if (fread(&header, sizeof(sif_xgrid_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xgrid", "failed to read header from %s", filepath);
    fclose(file);
    return SIF_ERR_IO;
  }

  if (header_validate(&header, filepath) != SIF_OK) {
    fclose(file);
    return SIF_ERR_IO;
  }

  /* Both halves of the geometry are compared, not just the cell count: this
   * doubles as the cache's hit test, and a grid of the right resolution over
   * the wrong box is a different field entirely. The box lengths compare
   * exactly because the writer stored a widened sif_real and any grid built
   * from this file gets the same value back, bit for bit. */
  if (header.n_cells != grid->n_cells ||
      header.box_length != (double)grid->box_length) {
    SIF_LOG_WARNING("xgrid",
      "%s does not match the target grid geometry, not loading", filepath);
    fclose(file);
    return SIF_ERR_IO;
  }

  /*
   * Provenance, when the caller has a key to check.
   *
   * The geometry test above says the file has the right shape and the checksum
   * below says its bytes are intact; neither says it was built from the data
   * in hand. For the cache that is the whole question, and the answer cannot
   * come from the filename -- which is a hash, and so is exactly what a
   * collision would agree on.
   */
  if (expect_key) {
    if (header.source_key[0] != expect_key[0] ||
        header.source_key[1] != expect_key[1]) {
      SIF_LOG_WARNING("xgrid",
        "%s records source key %016llx%016llx, not the %016llx%016llx asked "
        "for; treating it as a miss",
        filepath, (unsigned long long)header.source_key[1],
        (unsigned long long)header.source_key[0],
        (unsigned long long)expect_key[1], (unsigned long long)expect_key[0]);
      fclose(file);
      return SIF_ERR_IO;
    }
  }

  /* Hand the descriptor to the parallel reader. pread() carries its own offset
   * and ignores the stream position, so the header bytes stdio has already
   * buffered are of no consequence. */
  int fd = fileno(file);

  const size_t array_bytes = (size_t)grid->total_cells * sizeof(sif_real);
  const off_t current_offset = sizeof(sif_xgrid_header_t);

  CHECK_PREAD(
    fd, grid->values, array_bytes, current_offset, "cell values", file);

  fclose(file);

  grid->content = (sif_grid_content_t)header.content;

  /* A version 1 file predates the checksum and carries a zero, which is why
   * the version rather than the value decides whether to check. */
  if (header.version >= 2) {
    const uint32_t got = sif_crc32(grid->values, array_bytes);
    if (got != header.crc32) {
      SIF_LOG_ERROR("xgrid", "%s is corrupt: checksum %08x, expected %08x",
        filepath, got, header.crc32);
      return SIF_ERR_IO;
    }
  } else {
    SIF_LOG_WARNING("xgrid",
      "%s is a version %u file and carries no checksum; contents cannot be "
      "validated",
      filepath, header.version);
  }

  return SIF_OK;
}

sif_grid_t* sif_grid_read(const char* filepath) {
  if (!filepath)
    return NULL;

  FILE* file = fopen(filepath, "rb");
  if (!file) {
    SIF_LOG_ERROR("xgrid", "could not open %s for reading", filepath);
    return NULL;
  }

  sif_xgrid_header_t header;
  if (fread(&header, sizeof(sif_xgrid_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xgrid", "failed to read header from %s", filepath);
    fclose(file);
    return NULL;
  }
  fclose(file); /* the reader below opens it again for the payload */

  /* Validated before anything is sized from it -- see header_validate(). */
  if (header_validate(&header, filepath) != SIF_OK)
    return NULL;

  if (header.n_cells == 0) {
    SIF_LOG_ERROR("xgrid", "%s describes an empty grid", filepath);
    return NULL;
  }

  sif_grid_t* grid =
    sif_grid_alloc(header.n_cells, (sif_real)header.box_length);
  if (!grid) {
    SIF_LOG_ERROR("xgrid", "OOM allocating grid");
    return NULL;
  }

  if (sif_grid_read_into(filepath, grid) != SIF_OK) {
    sif_grid_free(grid);
    return NULL;
  }

  SIF_LOG_INFO("xgrid", "loaded %u^3 grid from %s", header.n_cells, filepath);
  return grid;
}
