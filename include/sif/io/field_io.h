/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file field_io.h
 * @brief Reading and writing particle fields: the .xfield binary format, and
 * ASCII input.
 *
 * The binary format is a fixed 64-byte header followed by the coordinate
 * arrays back to back, uncompressed and in host byte order. It exists so that
 * a field lands in memory in the layout the library already uses: the reader
 * pread()s each block straight into its final aligned buffer, with no parsing
 * and no copy.
 *
 * The price is that a file is only portable between machines that agree on
 * endianness and on the precision sif was built with. The header records the
 * precision and the reader refuses a mismatch rather than reinterpreting the
 * bytes; endianness is not recorded and not checked.
 *
 * Since version 2 the header also carries a CRC32 of the payload, which the
 * reader verifies. Version 1 files carry no checksum and are still accepted,
 * with a warning that they cannot be validated.
 *
 * ASCII input is the portable path, and far slower.
 */

#ifndef SIF_IO_FIELD_IO_H
#define SIF_IO_FIELD_IO_H

#include "sif/core/macros.h"
#include "sif/structures/field.h"

#include <stdint.h>

/** @brief Magic number at the start of every .xfield file. */
#define SIF_XFIELD_MAGIC "XFLD"
/** @brief Version of the .xfield layout this build reads and writes. */
#define SIF_XFIELD_VERSION 2

/**
 * @brief The 64-byte header of a .xfield file.
 *
 * Fixed width, with explicit padding, so the layout does not move with the
 * compiler. `box_length` is a `double` regardless of how sif_real is
 * configured, for the same reason.
 */
typedef struct {
  char magic[4];           /**< #SIF_XFIELD_MAGIC. */
  uint32_t version;        /**< #SIF_XFIELD_VERSION. */
  uint64_t n_particles;    /**< Particles in the file. */
  double box_length;       /**< Simulation box size. */
  uint32_t has_weights;    /**< 1 if a weight block follows. */
  uint32_t has_velocities; /**< 1 if vx, vy, vz blocks follow. */
  uint32_t is_double;      /**< 1 if written with a 64-bit sif_real. */
  /** CRC32 of the payload that follows, in the order it is written. Zero in a
   *  version 1 file, which carried no checksum. */
  uint32_t crc32;
  char padding[24]; /**< Reserved, to hold the header at 64 bytes. */
} sif_xfield_header_t;

/**
 * @brief Read a .xfield file into a newly allocated field.
 *
 * @param filepath Path to the input file.
 * @param out_box_length Optional; written with the box length from the header.
 * @return The field, owned by the caller and released with sif_field_free().
 * NULL on failure, including a precision mismatch.
 */
SIF_NODISCARD sif_field_t* sif_field_read(
  const char* filepath, double* out_box_length);

/**
 * @brief Read a .xfield file into a field that already exists.
 *
 * @param filepath Path to the input file.
 * @param field Field to fill. Its `n_particles` must equal the file's, and its
 * arrays must already be reserved.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL argument, or SIF_ERR_IO on any
 * file error -- including a bad magic number, an unknown version, a
 * particle-count or precision mismatch, or a failed checksum, all of which
 * are rejected rather than adapted to.
 */
int sif_field_read_into(const char* filepath, sif_field_t* field);

/**
 * @brief Write a field to a .xfield file.
 *
 * Weight and velocity blocks are written only if the field carries them, and
 * the header records which.
 *
 * @param filepath Path to the output file.
 * @param field Field to write. Must carry positions; they are the one block
 * every .xfield has.
 * @param box_length Box length to record in the header. Not otherwise used by
 * the field, so it has to be supplied here.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL argument or a field without
 * positions, or SIF_ERR_IO if the file could not be written -- including a
 * failure that only surfaces when the last buffered bytes are flushed.
 */
int sif_field_write(
  const char* filepath, const sif_field_t* field, double box_length);

/**
 * @brief Read an ASCII table into an existing field.
 *
 * Blank lines, and lines whose first non-blank character is `#` or `;`, are
 * skipped wherever they appear. So is a row that runs out of columns before
 * the format is satisfied -- a partial row is not a particle, and counting one
 * would leave an entry whose remaining components were never assigned.
 *
 * A field whose `n_particles` is 0 is sized from the file. Since the sizing
 * pass counts lines rather than parsing them, that count is an upper bound,
 * and `n_particles` is corrected down to what actually loaded. A field that
 * already has a count is filled to that count and no further; anything left in
 * the file is reported.
 *
 * Reserving is a no-op on a block the field already has, so a format naming
 * only the weight column can be read into a field whose positions are already
 * loaded, and they survive.
 *
 * @param field Field to fill. Either already sized, or with `n_particles` 0 to
 * take the count from the file.
 * @param filepath Path to the input file.
 * @param fmt Column layout, one character per column; see
 * sif_str_decode_format() for the alphabet. For example `"xyz*m"`. Position and
 * velocity must be named in full or not at all: each is reserved as one block,
 * so `"xy"` would allocate a z array and never write it.
 * @param delimiter Character separating columns. Use `' '` for whitespace.
 * @param skip_header Lines to skip before the data starts.
 * @return SIF_OK, SIF_ERR_INVALID for a NULL argument or a format that names
 * no usable columns, only part of the position or velocity, or no positions
 * for a field that has none; SIF_ERR_ALLOC if a block could not be reserved;
 * SIF_ERR_IO if the file could not be read or held no parsable row.
 *
 * @warning A column that is present but does not parse as a number reads as
 * 0.0 rather than failing; see sif_str_extract_next_real(). Only a *missing*
 * column causes a row to be skipped.
 */
int sif_field_read_ascii(sif_field_t* field, const char* filepath,
  const char* fmt, char delimiter, uint32_t skip_header);

#endif /* SIF_IO_FIELD_IO_H */
