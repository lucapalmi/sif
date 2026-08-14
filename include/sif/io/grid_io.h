/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file grid_io.h
 * @brief Reading and writing grids: the .xgrid binary format.
 *
 * Same shape as .xfield -- a fixed 64-byte header followed by the cell values
 * in host byte order -- and the same trade: fast and layout-compatible with
 * memory, portable only between machines that agree on endianness and on the
 * precision sif was built with.
 *
 * Since version 2 the header carries a CRC32 of the cell data, which the
 * reader verifies. Version 1 files are still accepted, unvalidated.
 *
 * This is also the format the optional CIC cache uses; see
 * sif_grid_assign_cic().
 */

#ifndef SIF_IO_GRID_IO_H
#define SIF_IO_GRID_IO_H

#include "sif/core/macros.h"
#include "sif/structures/grid.h"

#include <stdint.h>

/** @brief Magic number at the start of every .xgrid file. */
#define SIF_XGRID_MAGIC "XGRD"
/** @brief Version of the .xgrid layout this build reads and writes. */
#define SIF_XGRID_VERSION 2

/**
 * @brief The 64-byte header of an .xgrid file.
 */
typedef struct {
  char magic[4];        /**< #SIF_XGRID_MAGIC. */
  uint32_t version;     /**< #SIF_XGRID_VERSION. */
  uint32_t n_cells;     /**< Cells per side. */
  uint32_t is_double;   /**< 1 if written with a 64-bit sif_real. */
  uint64_t total_cells; /**< n_cells^3. */
  double box_length;    /**< Simulation box size. */
  /** CRC32 of the cell data that follows. Zero in a version 1 file, which
   *  carried no checksum. */
  uint32_t crc32;
  /** What the cells hold, as a #sif_grid_content_t. Written since version 2;
   *  a version 1 file has 0 here, which reads as SIF_GRID_EMPTY -- correct,
   *  since such a file does not say. */
  uint32_t content;
  /**
   * Key of the input this grid was derived from, or both words zero when the
   * file does not record one.
   *
   * Written only by the CIC cache, which needs to prove that a file it found
   * by name really came from the field in hand: the geometry check and the
   * CRC together establish that the file is intact and the right shape, and
   * neither says anything about which field produced it. A grid written
   * through sif_grid_write() leaves this zero, which reads as "no provenance
   * recorded" rather than as a key that failed to match.
   *
   * Zero is also what every file written before this field existed carries,
   * since it sat inside the reserved padding -- so no version bump: an old
   * file simply declines to prove anything, which is what it could always do.
   */
  uint64_t source_key[2];
  char padding[8]; /**< Reserved, to hold the header at 64 bytes. */
} sif_xgrid_header_t;

/**
 * @brief Read an .xgrid file into a newly allocated grid.
 *
 * @param filepath Path to the input file.
 * @return The grid, owned by the caller and released with sif_grid_free().
 * NULL on failure, including a precision mismatch.
 */
SIF_NODISCARD sif_grid_t* sif_grid_read(const char* filepath);

/**
 * @brief Read an .xgrid file into a grid that already exists.
 *
 * @param filepath Path to the input file.
 * @param grid Grid to fill. Its geometry must match the file's.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL argument, or SIF_ERR_IO on any
 * file error -- including a geometry or precision mismatch, or a failed
 * checksum, which are rejected rather than adapted to.
 */
int sif_grid_read_into(const char* filepath, sif_grid_t* grid);

/**
 * @brief Write a grid to an .xgrid file.
 *
 * @param filepath Path to the output file.
 * @param grid Grid to write. Must carry cell values.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL argument or a grid without cell
 * values, or SIF_ERR_IO if the file could not be written -- including a
 * failure that only surfaces when the last buffered bytes are flushed.
 *
 * @note Writes whatever the cells currently hold, weights or density contrast,
 * and records which in the header so a reader gets it back.
 */
int sif_grid_write(const char* filepath, const sif_grid_t* grid);

#endif /* SIF_IO_GRID_IO_H */
