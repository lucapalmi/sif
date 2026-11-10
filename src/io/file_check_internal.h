/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file file_check_internal.h
 * @brief Filling a sif_file_check_t, for the checkers of each format.
 * Private.
 *
 * file_check.c runs a check: it describes the file, then hands it to the
 * format's checker, then checks the contents the formats share (a sif
 * catalogue is one in ASCII, HDF5 or FITS). The checkers live beside the
 * readers, as the describers do.
 *
 * While a check runs, every WARNING and ERROR logged on its thread becomes
 * one of its problems (sif__log_capture_begin()), so a checker reports a
 * failed read by letting the reader log it, and its own findings with
 * sif__file_check_problem(). A checker returns SIF_OK, or SIF_ERR_ALLOC
 * when it could not go on; everything else it finds is a problem.
 */

#ifndef SIF__IO_FILE_CHECK_INTERNAL_H
#define SIF__IO_FILE_CHECK_INTERNAL_H

#include "sif/io/file_check.h"

#include "sif/core/macros.h"

#include <math.h>
#include <stdint.h>

/** @brief Add a problem. Only the first SIF_FILE_CHECK_PROBLEMS_KEPT are
 * formatted and kept; every one is counted. */
void sif__file_check_problem(sif_file_check_t* check,
  sif_file_severity_t severity, const char* fmt, ...) SIF_PRINTF_LIKE(3, 4);

/** @brief Say what the contents were checked as. */
void sif__file_check_content(sif_file_check_t* check, const char* content);

/** @brief The index of column @p name of table @p table, or UINT32_MAX. */
uint32_t sif__file_check_column_index(
  const sif_file_check_t* check, uint32_t table, const char* name);

/** @brief The statistics of a column, or NULL if there is no such column. */
sif_file_stats_t* sif__file_check_column(
  sif_file_check_t* check, uint32_t table, uint32_t column);

/** @brief The statistics of an array, or NULL. */
sif_file_stats_t* sif__file_check_array(
  sif_file_check_t* check, uint32_t array);

/** @brief Count a value read. */
static inline void sif__file_stats_add(sif_file_stats_t* s, double v) {
  s->n++;
  if (!isfinite(v)) {
    s->n_nonfinite++;
    return;
  }
  if (isnan(s->min) || v < s->min)
    s->min = v;
  if (isnan(s->max) || v > s->max)
    s->max = v;
}

/* --- the checkers --- */

/* Each reads the file its check describes, check->info->path, and fills its
 * statistics and problems; check->info is never NULL here. */
int sif__file_check_ascii(sif_file_check_t* check);
int sif__file_check_xfield(sif_file_check_t* check);
int sif__file_check_xgrid(sif_file_check_t* check);
int sif__file_check_fits(sif_file_check_t* check);
int sif__file_check_hdf5(sif_file_check_t* check);
int sif__file_check_gadget(sif_file_check_t* check);
int sif__file_check_checkpoint(sif_file_check_t* check);

/**
 * @brief sif_file_info_print(), with the statistics of a check beside each
 * column and array when they are given.
 */
int sif__file_info_print_stats(const sif_file_info_t* info,
  sif_file_stats_t* const* column_stats, const sif_file_stats_t* array_stats,
  FILE* stream);

#endif /* SIF__IO_FILE_CHECK_INTERNAL_H */
