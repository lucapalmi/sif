/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file file_check.h
 * @brief Whether a file is sound: read whole, every value, against what its
 * format and its contents require.
 *
 * A check starts from the file's description (file_info.h) and reads
 * everything it describes, at three levels:
 *
 * 1. **The file is intact**: every byte its structure points to is there,
 *    record framing is whole, sizes agree with the headers, checksums match
 *    where the format keeps them, a multi-file snapshot's files are all
 *    present and add up, every object of an HDF5 file opens.
 * 2. **Every value reads**: every numeric column and array is read through,
 *    its range taken and its non-finite values counted; every field of an
 *    ASCII table has the type its column has in the first row.
 * 3. **The contents make sense** for what the file holds: positions inside
 *    the box of a snapshot or an .xfield, masses and weights not negative, a
 *    sif catalogue's radii positive and its count what its header says, a
 *    density grid not negative, a checkpoint a run could resume from.
 *
 * What it finds is returned as data, the same for every format: the
 * description, a range for every column and array, and a list of problems,
 * each an error -- the file is not usable as it is -- or a warning --
 * readable, but not as it should be. Nothing goes to the log.
 *
 * Reading everything takes as long as reading the file: a check is asked
 * for, not done in passing.
 */

#ifndef SIF_IO_FILE_CHECK_H
#define SIF_IO_FILE_CHECK_H

#include "sif/core/macros.h"
#include "sif/io/file_info.h"

#include <stdint.h>
#include <stdio.h>

/** @brief How bad a problem is. */
typedef enum {
  /** Readable, but not as it should be: a NaN in a catalogue column, say. */
  SIF_FILE_PROBLEM_WARNING,
  /** The file is not usable as it is. */
  SIF_FILE_PROBLEM_ERROR
} sif_file_severity_t;

/** @brief One problem, as "<path>[:where]: what: detail". */
typedef struct {
  sif_file_severity_t severity;
  const char* message;
} sif_file_problem_t;

/** @brief Problems kept one by one; past these they are only counted. */
#define SIF_FILE_CHECK_PROBLEMS_KEPT 100

/** @brief What reading a column or array found. */
typedef struct {
  /** Values read: rows times the width of a column. 0 for one that was not
   * read, such as a string column, or one that could not be. */
  uint64_t n;
  uint64_t n_nonfinite;
  /** Over the finite values; NaN when there were none. */
  double min, max;
} sif_file_stats_t;

/** @brief A file, checked. */
typedef struct {
  /** What the file says it holds; NULL for a file that could not be read at
   * all, whose problems say why. */
  sif_file_info_t* info;
  /** Per column: column_stats[t][c] for info->tables[t].columns[c]. */
  sif_file_stats_t** column_stats;
  /** Per array, in the order of info->arrays. */
  sif_file_stats_t* array_stats;
  /** What the contents were checked as: "sif catalogue", "GADGET
   * snapshot", ...; "" for a file checked only as its format. */
  const char* content;

  uint64_t n_errors;
  uint64_t n_warnings;
  /** The first SIF_FILE_CHECK_PROBLEMS_KEPT problems, in the order found. */
  uint32_t n_problems;
  sif_file_problem_t* problems;
} sif_file_check_t;

/**
 * @brief Check a file, reading all of it.
 *
 * @param path The file, or a multi-file snapshot's base name.
 * @param format What it is, or SIF_FILE_FORMAT_UNKNOWN to detect it.
 * @return The check, owned by the caller and released with
 * sif_file_check_free(); the file is sound when its n_errors is 0. NULL,
 * logged, only for a NULL path or a failed allocation: a file that cannot
 * be read is a check that says so.
 */
SIF_NODISCARD sif_file_check_t* sif_file_check(
  const char* path, sif_file_format_t format);

/** @brief Release a check. NULL is accepted. */
void sif_file_check_free(sif_file_check_t* check);

/**
 * @brief Print a check: the description as sif_file_info_print() lays it
 * out, with each column's range, then the problems and their count.
 *
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument; SIF_ERR_IO if the
 * stream fails.
 */
int sif_file_check_print(const sif_file_check_t* check, FILE* stream);

#endif /* SIF_IO_FILE_CHECK_H */
