/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file file_info_internal.h
 * @brief Building a sif_file_info_t, for the describers of each format.
 * Private.
 *
 * file_info.c holds the description and describes the formats that need
 * nothing but the C library (ASCII, .xfield, .xgrid); each of the others is
 * described beside its reader, where its internals are -- fits_io.c,
 * hdf5.c, gadget_io.c, exodus_checkpoint.c -- through the functions below.
 *
 * The adders never fail: a failed allocation is remembered, the rest of the
 * description goes on, and sif_file_info_read() reports SIF_ERR_ALLOC at the
 * end. A describer can call them without checking each one.
 */

#ifndef SIF__IO_FILE_INFO_INTERNAL_H
#define SIF__IO_FILE_INFO_INTERNAL_H

#include "sif/io/file_info.h"

#include "sif/core/macros.h"

#include <stdint.h>

/** @brief What a piece of metadata belongs to. */
typedef enum {
  SIF__FILE_INFO_FILE,
  SIF__FILE_INFO_TABLE,
  SIF__FILE_INFO_ARRAY
} sif_file_info_owner_t;

/** @brief Set the format's variant, sif_file_info_t::detail. */
void sif__file_info_detail(sif_file_info_t* info, const char* fmt, ...)
  SIF_PRINTF_LIKE(2, 3);

/** @brief Add a table; returns its index, for the columns and metadata. */
uint32_t sif__file_info_table(
  sif_file_info_t* info, const char* name, uint64_t n_rows);

/** @brief Rename table @p table. */
void sif__file_info_table_rename(
  sif_file_info_t* info, uint32_t table, const char* name);

/** @brief Add a column to table @p table. @p unit may be NULL or "". */
void sif__file_info_column(sif_file_info_t* info, uint32_t table,
  const char* name, const char* type, uint64_t width, const char* unit);

/** @brief Add an array of @p rank dimensions (at most
 * SIF_FILE_ARRAY_MAX_RANK are kept); returns its index. */
uint32_t sif__file_info_array(sif_file_info_t* info, const char* name,
  const char* type, uint32_t rank, const uint64_t* dims);

/** @brief Add metadata to the file (@p index ignored), or to the table or
 * array @p index. */
void sif__file_info_meta_int(sif_file_info_t* info, sif_file_info_owner_t owner,
  uint32_t index, const char* key, int64_t value);
void sif__file_info_meta_real(sif_file_info_t* info,
  sif_file_info_owner_t owner, uint32_t index, const char* key, double value);
void sif__file_info_meta_string(sif_file_info_t* info,
  sif_file_info_owner_t owner, uint32_t index, const char* key,
  const char* value);

/**
 * @brief A value written as text -- a FITS card's, an ASCII file's
 * `#key=value` -- as the metadata it reads as: an integer, a real, or else a
 * string, unquoted.
 */
void sif__file_info_meta_parse(sif_file_info_t* info,
  sif_file_info_owner_t owner, uint32_t index, const char* key,
  const char* text);

/* --- the describers --- */

/*
 * Each fills @p info for the file at info->path, which the dispatcher has
 * already sized, and returns SIF_OK, or the error, logged, for a file it
 * cannot read at all. Those for HDF5 and FITS refuse with
 * SIF_ERR_UNSUPPORTED in a build without them.
 */
int sif__file_info_fits(sif_file_info_t* info);
int sif__file_info_hdf5(sif_file_info_t* info);
int sif__file_info_gadget(sif_file_info_t* info);
int sif__file_info_checkpoint(sif_file_info_t* info);

/**
 * @brief The datasets of one group of an HDF5 file, as the columns of table
 * @p table: a GADGET snapshot's PartTypeN.
 * @return How many were added: 0 for a group the file does not have.
 */
uint32_t sif__file_info_hdf5_columns(
  sif_file_info_t* info, uint32_t table, const char* path, const char* group);

#endif /* SIF__IO_FILE_INFO_INTERNAL_H */
