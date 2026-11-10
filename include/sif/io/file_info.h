/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file file_info.h
 * @brief What a file holds, described the same way whatever its format.
 *
 * Every file sif reads or writes is described with one model: a file has a
 * format and a size, carries **metadata** (key = value), and holds
 * **tables** -- a number of rows and named, typed columns -- and
 * **arrays**, the datasets that are not columns of a table, such as a grid
 * of cells. Each format maps onto it:
 *
 * | format     | tables                                | metadata             |
 * |------------|---------------------------------------|----------------------|
 * | ASCII      | one, columns named by a header line   | `#key=value` lines   |
 * | FITS       | one per table HDU; images are arrays  | header keywords      |
 * | HDF5       | one per group, of its datasets of one length | attributes    |
 * | GADGET     | one per particle type present         | the snapshot header  |
 * | .xfield    | one: the particles                    | the header           |
 * | .xgrid     | none; the cells are an array          | the header           |
 * | checkpoint | one: the voids so far                 | the run's parameters |
 *
 * A description reads what a file says about itself -- its headers, its
 * structure -- and no values, except that an ASCII file has no header and
 * is scanned for its row count. It checks nothing either: a file is
 * described as far as it can be read, and only one that cannot be read at
 * all is refused. sif_file_check() (file_check.h) is what says whether a
 * file is sound.
 */

#ifndef SIF_IO_FILE_INFO_H
#define SIF_IO_FILE_INFO_H

#include "sif/core/macros.h"

#include <stdint.h>
#include <stdio.h>

/** @brief The formats sif reads and writes. */
typedef enum {
  /** Not one sif knows; detection found nothing. */
  SIF_FILE_FORMAT_UNKNOWN = 0,
  SIF_FILE_FORMAT_ASCII,
  SIF_FILE_FORMAT_FITS,
  SIF_FILE_FORMAT_HDF5,
  /** A GADGET snapshot, in any SnapFormat, HDF5 included. */
  SIF_FILE_FORMAT_GADGET,
  SIF_FILE_FORMAT_XFIELD,
  SIF_FILE_FORMAT_XGRID,
  /** An exodus checkpoint. */
  SIF_FILE_FORMAT_CHECKPOINT
} sif_file_format_t;

/**
 * @brief The format of @p path, told from its first bytes.
 *
 * A guess, logging nothing, for choosing how to read a file: FITS by its
 * `SIMPLE` card (or gzip and a `.fits` name), .xfield, .xgrid and checkpoints
 * by their magic, a GADGET snapshot -- also a multi-file snapshot's base name
 * -- before plain HDF5, and text as ASCII.
 *
 * @return The format; SIF_FILE_FORMAT_ASCII for a path that does not exist
 * and is not a snapshot's base name, so that reading it says what is wrong;
 * SIF_FILE_FORMAT_UNKNOWN for binary data in no format sif knows.
 */
sif_file_format_t sif_file_format_detect(const char* path);

/** @brief "ascii", "fits", "hdf5", "gadget", "xfield", "xgrid",
 * "checkpoint", or "unknown". */
const char* sif_file_format_name(sif_file_format_t format);

/**
 * @brief The format a name from sif_file_format_name() stands for.
 * @return SIF_OK; SIF_ERR_INVALID for a name it does not know, "unknown"
 * among them.
 */
SIF_NODISCARD int sif_file_format_parse(
  const char* name, sif_file_format_t* out);

/** @brief What a metadata value holds. */
typedef enum {
  SIF_FILE_META_INT,
  SIF_FILE_META_REAL,
  SIF_FILE_META_STRING
} sif_file_meta_kind_t;

/**
 * @brief One key = value of a file's, table's or array's metadata, under the
 * file's own name for it (`BoxSize`, `OMEGA_M`), or under sif's for a header
 * the format lays out in binary (`box_length`).
 *
 * A value that is a short list -- an HDF5 attribute of three numbers -- is
 * held as a string, "1, 2, 3".
 */
typedef struct {
  const char* key;
  sif_file_meta_kind_t kind;
  /** The value, in the member #kind names. */
  int64_t i;
  double r;
  const char* s;
} sif_file_meta_t;

/**
 * @brief A column of a table.
 *
 * #type is one name for all formats: "int8" ... "int64", "uint8" ...
 * "uint64", "float32", "float64", "complex64", "complex128", "bool",
 * "string"; "number" and "text" in ASCII, whose fields have no width; "other"
 * for anything else.
 */
typedef struct {
  const char* name;
  const char* type;
  /** Values per row: 1, or 3 for a column of positions. */
  uint64_t width;
  /** The unit the file gives it, or NULL. */
  const char* unit;
} sif_file_column_t;

/** @brief A table: rows of named columns. */
typedef struct {
  /** "PartType1", an HDU's EXTNAME, an HDF5 group's path; "" for the one
   * table of a format that names none. */
  const char* name;
  uint64_t n_rows;
  uint32_t n_columns;
  sif_file_column_t* columns;
  uint32_t n_meta;
  sif_file_meta_t* meta;
} sif_file_table_t;

/** @brief Most dimensions an array is described with. */
#define SIF_FILE_ARRAY_MAX_RANK 8

/** @brief A dataset that is not a column of a table: a grid, an image, an
 * HDF5 dataset of a length no other shares. */
typedef struct {
  const char* name;
  /** As sif_file_column_t::type. */
  const char* type;
  /** 0 for a scalar. */
  uint32_t rank;
  uint64_t dims[SIF_FILE_ARRAY_MAX_RANK];
  uint32_t n_meta;
  sif_file_meta_t* meta;
} sif_file_array_t;

/**
 * @brief A file, described.
 *
 * Everything it points to belongs to it, and goes with sif_file_info_free().
 */
typedef struct {
  /** The path as given. */
  const char* path;
  sif_file_format_t format;
  /** The variant within the format, e.g. "SnapFormat 2, single precision",
   * "comma-separated, 1 header line"; "" when there is nothing to say. */
  const char* detail;
  /** Files the path stands for: more than 1 for a multi-file snapshot. */
  uint32_t n_files;
  /** Their total size. */
  uint64_t size_bytes;

  uint32_t n_meta;
  sif_file_meta_t* meta;
  uint32_t n_tables;
  sif_file_table_t* tables;
  uint32_t n_arrays;
  sif_file_array_t* arrays;
} sif_file_info_t;

/**
 * @brief Describe a file.
 *
 * @param path The file, or a multi-file snapshot's base name.
 * @param format What it is, or SIF_FILE_FORMAT_UNKNOWN to detect it with
 * sif_file_format_detect().
 * @return The description, owned by the caller and released with
 * sif_file_info_free(); NULL, logged, for a file that cannot be read at all,
 * is not in @p format, or is in no format sif knows, and for HDF5 and FITS
 * in a build without them.
 */
SIF_NODISCARD sif_file_info_t* sif_file_info_read(
  const char* path, sif_file_format_t format);

/** @brief Release a description. NULL is accepted. */
void sif_file_info_free(sif_file_info_t* info);

/**
 * @brief Print a description: the file, its metadata, then each table and
 * array, laid out the same for every format.
 *
 * Written to a stream rather than through the logger: it is the answer to a
 * question, not a diagnostic.
 *
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument; SIF_ERR_IO if the
 * stream fails.
 */
int sif_file_info_print(const sif_file_info_t* info, FILE* stream);

#endif /* SIF_IO_FILE_INFO_H */
