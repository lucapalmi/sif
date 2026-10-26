/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file internal.h
 * @brief Low-level file reading shared by the binary and ASCII loaders.
 * Private to the library.
 *
 * Private because it deals in raw file descriptors and POSIX types: keeping it
 * out of `include/` is what lets the public API stay free of `<unistd.h>` and
 * `off_t`.
 */

#ifndef SIF__IO_INTERNAL_H
#define SIF__IO_INTERNAL_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "sif/core/macros.h"
#include "sif/structures/grid.h"

/**
 * @brief Write a grid, recording the key of the input it was derived from.
 *
 * The keyed form of sif_grid_write(), for the CIC cache. The key goes into
 * sif_xgrid_header_t::source_key and is what
 * sif__grid_read_into_keyed() checks against, so that a cache hit has to prove
 * its provenance rather than being trusted because its filename matched.
 *
 * @param source_key Two words identifying the input, or NULL to record none,
 * which is what sif_grid_write() passes.
 * @return As sif_grid_write().
 */
int sif__grid_write_keyed(
  const char* filepath, const sif_grid_t* grid, const uint64_t source_key[2]);

/**
 * @brief Read a grid, requiring it to carry a particular source key.
 *
 * @param expect_key Key the file must record, or NULL to accept any file --
 * which is what the public sif_grid_read_into() passes, since a caller reading
 * a grid it was handed has no key to check against.
 * @return As sif_grid_read_into(), with SIF_ERR_IO for a file whose recorded
 * key differs from @p expect_key or which records none at all.
 */
int sif__grid_read_into_keyed(
  sif_grid_t* grid, const char* filepath, const uint64_t expect_key[2]);

/**
 * @brief Read a large block from a file descriptor using several threads.
 *
 * Splits the range into per-thread chunks and pread()s each one, which is what
 * keeps a multi-gigabyte field load from being bound by a single sequential
 * stream. Short reads are looped over, so a chunk is either filled completely
 * or reported as a failure.
 *
 * @param fd Open file descriptor, positioned anywhere: pread() does not use or
 * disturb the file offset.
 * @param dest Destination buffer, at least @p total_bytes long.
 * @param total_bytes Bytes to read. Zero succeeds and does nothing.
 * @param base_offset Byte offset in the file to start from.
 * @return SIF_OK, or SIF_ERR_IO if any chunk could not be filled.
 */
int sif__io_pread_parallel(
  int fd, void* dest, size_t total_bytes, off_t base_offset);

/**
 * @brief Count the lines in an ASCII file, after the header.
 *
 * Reads the file in large blocks and counts newlines, rather than parsing
 * lines, so the cost is a single streaming pass. Used to size a field before
 * the real parse runs.
 *
 * @param tag Log tag of the caller.
 * @param filepath Path to the file.
 * @param skip_header Header lines to exclude from the count.
 * @param[out] out_rows Lines after the header, at least 1 on success.
 * @return SIF_OK; SIF_ERR_IO, logged, if the file cannot be read, is empty,
 * or holds nothing past the header.
 *
 * @note This is an upper bound on the particle count, not the count itself:
 * blank and comment lines are counted here and dropped by the parser. Sizing a
 * field from it therefore over-allocates slightly, which is the intended
 * trade -- the alternative is parsing the file twice.
 */
SIF_NODISCARD int sif__io_ascii_row_count(const char* tag, const char* filepath,
  uint32_t skip_header, uint64_t* out_rows);

/**
 * @brief Log an operating-system failure as "<path>: <strerror>", or
 * "<path>: <what>: <strerror>", and record @p err with the error.
 *
 * @param what What was being done, or NULL when opening the file is all it
 * was.
 * @return SIF_ERR_IO, for `return sif__io_os_error(...)`.
 */
int sif__io_os_error(
  const char* tag, const char* path, const char* what, int err);

/**
 * @brief Check that @p path names something a reader can open: that it
 * exists, is not a directory, and is readable.
 *
 * Run before a library (HDF5, cfitsio) is handed the path, since what they
 * report for a missing file is their own failure, not the system's.
 *
 * @param[out] out_bytes The file's size, or NULL.
 * @return SIF_OK, or SIF_ERR_IO with the reason logged and its errno
 * recorded.
 */
SIF_NODISCARD int sif__io_check_readable(
  const char* tag, const char* path, uint64_t* out_bytes);

/** @brief strerror(), safe to call from several threads. */
const char* sif__io_strerror(int err);

/* --- file checks --- */

/**
 * @brief How many problems a check logs one by one; past this it counts them,
 * and says how many more there were at the end.
 */
#define SIF__IO_PROBLEMS_SHOWN 10

/** @brief The problems a check has found so far. */
typedef struct {
  const char* tag;
  uint64_t n;
} sif__io_problems_t;

/**
 * @brief Log one problem a check found, as an ERROR, while fewer than
 * SIF__IO_PROBLEMS_SHOWN have been; count it either way.
 */
void sif__io_problem(sif__io_problems_t* p, const char* fmt, ...)
  SIF_PRINTF_LIKE(2, 3);

/**
 * @brief Close a check: log how many problems went unlisted, and write the
 * total to @p report.
 *
 * @return SIF_OK for none, SIF_ERR_IO otherwise.
 */
int sif__io_problems_end(const sif__io_problems_t* p, FILE* report);

/** @brief The range of a column of values, as a deep check reports it. */
typedef struct {
  uint64_t n;
  uint64_t n_nonfinite;
  double min, max;
} sif__io_stats_t;

static inline void sif__io_stats_add(sif__io_stats_t* s, double v) {
  s->n++;
  if (!isfinite(v)) {
    s->n_nonfinite++;
    return;
  }
  if (s->n == s->n_nonfinite + 1 || v < s->min)
    s->min = v;
  if (s->n == s->n_nonfinite + 1 || v > s->max)
    s->max = v;
}

/** @brief The header line for a table of sif__io_stats_print() rows. */
void sif__io_stats_header(FILE* report, int name_width);

/** @brief One row of the table: the column's name, min, max and how many
 * values were not finite. */
void sif__io_stats_print(
  FILE* report, const char* name, int name_width, const sif__io_stats_t* s);

/**
 * @brief Reverse the bytes of a 32- or 64-bit word: the conversion between
 * little- and big-endian, for the readers of files from other machines.
 */
static inline uint32_t sif__io_bswap32(uint32_t v) {
  return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);
}

/** @copydoc sif__io_bswap32 */
static inline uint64_t sif__io_bswap64(uint64_t v) {
  return ((uint64_t)sif__io_bswap32((uint32_t)v) << 32) |
         sif__io_bswap32((uint32_t)(v >> 32));
}

/** @brief Whether this machine stores the least significant byte first. */
static inline int sif__io_host_is_little(void) {
  const uint16_t one = 1;
  unsigned char first;
  memcpy(&first, &one, 1);
  return first == 1;
}

#endif /* SIF__IO_INTERNAL_H */
