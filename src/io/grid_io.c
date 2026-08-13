/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/grid_io.h"

#include "internal.h"
#include "sif/utils/crc32.h"
#include "sif/utils/logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
  if (!filepath || !grid)
    return SIF_ERR_INVALID;

  FILE* file = fopen(filepath, "wb");
  if (!file) {
    SIF_LOG_ERROR("xgrid", "Could not open %s for writing", filepath);
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
  header.content = (uint32_t)grid->content;
  header.crc32 =
    sif_crc32(grid->values, (size_t)grid->total_cells * sizeof(sif_real));

  CHECK_WRITE(&header, sizeof(sif_xgrid_header_t), 1, file, "header");
  CHECK_WRITE(
    grid->values, sizeof(sif_real), grid->total_cells, file, "cell values");

  fclose(file);
  return SIF_OK;
}

int sif_grid_read_into(const char* filepath, sif_grid_t* grid) {
  if (!filepath || !grid)
    return SIF_ERR_INVALID;

  FILE* file = fopen(filepath, "rb");
  if (!file)
    return SIF_ERR_IO; /* Silent failure allows cache logic to smoothly fall
                 back to compute */

  sif_xgrid_header_t header;
  if (fread(&header, sizeof(sif_xgrid_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xgrid", "Failed to read header from %s", filepath);
    fclose(file);
    return SIF_ERR_IO;
  }

  if (strncmp(header.magic, SIF_XGRID_MAGIC, 4) != 0) {
    SIF_LOG_ERROR("xgrid", "File %s is not a valid xgrid binary", filepath);
    fclose(file);
    return SIF_ERR_IO;
  }

  if (header.version > SIF_XGRID_VERSION) {
    SIF_LOG_ERROR("xgrid",
      "%s is version %u, but this build reads up to version %u", filepath,
      header.version, (unsigned)SIF_XGRID_VERSION);
    fclose(file);
    return SIF_ERR_IO;
  }

  uint32_t current_is_double = (sizeof(sif_real) == 8) ? 1u : 0u;
  if (header.is_double != current_is_double) {
    SIF_LOG_ERROR("xgrid",
      "xgrid precision mismatch. File is %s, Library compiled as %s.",
      header.is_double ? "FP64" : "FP32", current_is_double ? "FP64" : "FP32");
    fclose(file);
    return SIF_ERR_IO;
  }

  /* Cache-hit integrity check: make sure the file matches the target grid shape
   */
  if (header.n_cells != grid->n_cells ||
      header.box_length != (double)grid->box_length) {
    SIF_LOG_WARNING("xgrid",
      "xgrid geometry mismatch. Cannot load into the target grid buffer.");
    fclose(file);
    return SIF_ERR_IO;
  }

  /* Transition to parallel POSIX I/O */
  int fd = fileno(file);
  fflush(file);

  size_t array_bytes = grid->total_cells * sizeof(sif_real);
  off_t current_offset = sizeof(sif_xgrid_header_t);

  CHECK_PREAD(
    fd, grid->values, array_bytes, current_offset, "cell values", file);

  fclose(file);

  grid->content = (sif_grid_content_t)header.content;

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
    SIF_LOG_ERROR("xgrid", "Could not open %s for reading", filepath);
    return NULL;
  }

  sif_xgrid_header_t header;
  if (fread(&header, sizeof(sif_xgrid_header_t), 1, file) != 1) {
    SIF_LOG_ERROR("xgrid", "Failed to read header from %s", filepath);
    fclose(file);
    return NULL;
  }
  fclose(file); /* Close immediately, _into will reopen and validate */

  sif_grid_t* grid =
    sif_grid_alloc(header.n_cells, (sif_real)header.box_length);
  if (!grid) {
    SIF_LOG_ERROR("xgrid", "OOM allocating grid");
    return NULL;
  }

  if (sif_grid_read_into(filepath, grid) != 0) {
    SIF_LOG_ERROR("xgrid", "Failed to read grid data into struct");
    sif_grid_free(grid);
    return NULL;
  }

  SIF_LOG_INFO("xgrid", "Loaded %u^3 grid from %s", header.n_cells, filepath);
  return grid;
}