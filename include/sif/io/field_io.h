/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file field_io.h
 * @brief Reading and writing particle fields: the .xfield binary format, and
 * ASCII and raw binary input.
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
 * ASCII input is the portable path, and far slower. Raw binary input reads
 * files written by anything else, given their layout.
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
  /** #SIF_XFIELD_MAGIC. */
  char magic[4];
  /** #SIF_XFIELD_VERSION. */
  uint32_t version;
  /** Particles in the file. */
  uint64_t n_particles;
  /** Simulation box size. */
  double box_length;
  /** 1 if a weight block follows. */
  uint32_t has_weights;
  /** 1 if vx, vy, vz blocks follow. */
  uint32_t has_velocities;
  /** 1 if written with a 64-bit sif_real. */
  uint32_t is_double;
  /** CRC32 of the payload that follows, in the order it is written. Zero in a
   * version 1 file, which carried no checksum.
   */
  uint32_t crc32;
  /** Reserved, to hold the header at 64 bytes. */
  char padding[24];
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
 * @param field Field to fill. Its `n_particles` must equal the file's, and its
 * arrays must already be reserved.
 * @param filepath Path to the input file.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL argument, or SIF_ERR_IO on any
 * file error -- including a bad magic number, an unknown version, a
 * particle-count or precision mismatch, or a failed checksum, all of which
 * are rejected rather than adapted to.
 */
int sif_field_read_into(sif_field_t* field, const char* filepath);

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
 * @brief Which of a file's named columns fill which part of a field, for the
 * readers of files whose columns have names: sif_field_read_fits() and
 * sif_field_read_hdf5().
 *
 * What an entry says is up to the reader -- a FITS column or expression, an
 * HDF5 dataset -- and is documented there. Built with designated
 * initialisers, so the part each one fills is written next to it and an
 * entry left out is NULL:
 *
 * @code
 * const sif_field_columns_t cols = {.ra = "RA", .dec = "DEC", .z = "Z"};
 * @endcode
 *
 * The positions are named one of two ways, and the names say which: x, y and
 * z for Cartesian positions; or ra, dec and z for sky coordinates -- right
 * ascension and declination in degrees, and redshift -- which give a field
 * that is #SIF_COORDINATES_SKY, for sif_field_convert_sky_coordinates().
 * Naming some of each is refused.
 */
typedef struct {
  /** Cartesian positions; with z, all three or none. */
  const char* x;
  const char* y;
  /** Sky coordinates, in place of x and y; with z, all three or none. */
  const char* ra;
  const char* dec;
  /** The third position, or the redshift on the sky; always required. */
  const char* z;
  /** Velocities: all three, or none (NULL) to read none. */
  const char* vx;
  const char* vy;
  const char* vz;
  /** Weight, or NULL for an unweighted field. */
  const char* w;
} sif_field_columns_t;

/**
 * @defgroup field_format Column formats
 * @brief How a file's columns map onto a field, for the ASCII and the binary
 * readers alike.
 *
 * .. _field-format:
 *
 * .. rubric:: Column formats
 *
 * A format names the columns in order:
 *
 * `x` `y` `z`
 *   position components
 *
 * `ra` `dec` `z`
 *   sky coordinates instead: right ascension and declination in degrees,
 *   and redshift. They load into x, y and z, and the field comes out as
 *   #SIF_COORDINATES_SKY, for sif_field_convert_sky_coordinates(). A format
 *   names positions or sky coordinates, never some of each.
 *
 * `vx` `vy` `vz`
 *   velocity components
 *
 * `w`
 *   per-particle weight (a mass, a luminosity, a selection weight)
 *
 * `*`
 *   a column that is present but not read. In a binary file it may carry a
 *   width in bytes, `*8`, for a column of another type -- a 64-bit ID among
 *   single-precision values, say. Without one it is one value at the file's
 *   precision.
 *
 * Case does not matter, and spaces, tabs or commas between names are
 * optional: `"x y z vx vy vz w"`, `"x,y,z,*,w"` and `"xyzvxvyvzw"` are all
 * valid. Anything else is an error, and so is a name given twice, or a
 * position or velocity named only in part -- each is stored as one block of
 * three, so `"x y"` would leave z unwritten.
 *
 * A format need not name the positions: `"* * * vx vy vz"` reads only the
 * velocities of a file that has both, into a field whose positions are
 * already loaded. Its particles have to be in the file's order, so a field
 * that has been Morton-sorted is refused.
 * @{
 */

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
 * @param fmt Column layout; see the column formats above. For example `"x y z *
 * w"`. Skip widths (`*8`) are for binary files and are refused here.
 * @param delimiter Character separating columns. Use `' '` for whitespace.
 * @param skip_header Lines to skip before the data starts.
 * @return SIF_OK, SIF_ERR_INVALID for a NULL argument, a malformed format, one
 * that loads no positions into a field that has none, or one that adds columns
 * to a Morton-sorted field; SIF_ERR_ALLOC if a block could not be reserved;
 * SIF_ERR_IO if the file could not be read or held no parsable row.
 *
 * @warning A column that is present but does not parse as a number reads as
 * 0.0 rather than failing. Only a *missing* column causes a row to be skipped.
 */
int sif_field_read_ascii_into(sif_field_t* field, const char* filepath,
  const char* fmt, char delimiter, uint32_t skip_header);

/**
 * @brief Read an ASCII table into a new field.
 *
 * sif_field_read_ascii_into() on a field sized from the file, for the common
 * case of one file holding everything: rows are parsed and skipped exactly as
 * there.
 *
 * @param fmt Column layout; see the column formats above. Must name the
 * positions, since the new field has none.
 * @return The field, owned by the caller and released with sif_field_free().
 * NULL on any failure sif_field_read_ascii_into() reports, with the reason in
 * the log.
 */
SIF_NODISCARD sif_field_t* sif_field_read_ascii(
  const char* filepath, const char* fmt, char delimiter, uint32_t skip_header);

/**
 * @brief How the values of a binary file are arranged.
 *
 * ROWS is one record per particle, every column of it together -- `x0 y0 z0
 * x1 y1 z1 ...`, what writing an array of C structs produces. BLOCKS is one
 * column at a time, every particle of it together -- `x0 x1 ... y0 y1 ...`,
 * what writing one array after another produces.
 */
typedef enum { SIF_BINARY_ROWS = 0x100, SIF_BINARY_BLOCKS } sif_binary_layout_t;

/** @brief The precision the values were written in. */
typedef enum {
  SIF_BINARY_FLOAT32 = 0x200,
  SIF_BINARY_FLOAT64
} sif_binary_precision_t;

/** @brief The byte order the values were written in. */
typedef enum {
  /** This machine's own: a file written here, and read here. */
  SIF_BINARY_NATIVE = 0x300,
  SIF_BINARY_LITTLE,
  SIF_BINARY_BIG
} sif_binary_endian_t;

/**
 * @brief Read a raw binary file -- a header of known length, then the values
 * -- into an existing field.
 *
 * For files no other reader knows: the header is skipped without being
 * interpreted, and everything about the data is given by the arguments. Each
 * enumeration has its own range of values, so an argument passed in the wrong
 * place is refused rather than read as another.
 *
 * A field whose `n_particles` is 0 is sized from the file: what follows the
 * header must be a whole number of particles, or the file is refused -- a
 * leftover means the format, the precision or the header length is wrong. A
 * field that already has a count reads that many, and a file holding more is
 * reported.
 *
 * @param field Field to fill, sized or with `n_particles` 0. As for
 * sif_field_read_ascii_into(), blocks it already has are kept, so a second
 * file can add columns to the first.
 * @param filepath Path to the input file.
 * @param fmt Column layout; see the column formats above. For example `"x y z
 * *8 w"`.
 * @param layout Rows (one record per particle) or blocks (one column at a
 * time).
 * @param precision Float or double, for every value column and every `*`
 * without a width.
 * @param endian Byte order of the values.
 * @param header_bytes Bytes to skip before the data.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument, an argument out of
 * range or in the wrong place, or a format sif_field_read_ascii_into() would
 * refuse; SIF_ERR_ALLOC; SIF_ERR_IO if the file could not be read, is shorter
 * than the particles asked for, or does not divide into whole particles.
 */
SIF_NODISCARD int sif_field_read_binary_into(sif_field_t* field,
  const char* filepath, const char* fmt, sif_binary_layout_t layout,
  sif_binary_precision_t precision, sif_binary_endian_t endian,
  uint64_t header_bytes);

/**
 * @brief Read a raw binary file into a new field.
 *
 * sif_field_read_binary_into() on a field sized from the file, for the common
 * case of one file holding everything: what follows the header must then be a
 * whole number of particles.
 *
 * @param fmt Column layout; see the column formats above. Must name the
 * positions, since the new field has none.
 * @return The field, owned by the caller and released with sif_field_free().
 * NULL on any failure sif_field_read_binary_into() reports, with the reason
 * in the log.
 */
SIF_NODISCARD sif_field_t* sif_field_read_binary(const char* filepath,
  const char* fmt, sif_binary_layout_t layout, sif_binary_precision_t precision,
  sif_binary_endian_t endian, uint64_t header_bytes);

/** @} */

#endif /* SIF_IO_FIELD_IO_H */
