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

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "sif/core/macros.h"

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
 * @brief Count the data rows in an ASCII file.
 *
 * Reads the file in large blocks and counts newlines, rather than parsing
 * lines, so the cost is a single streaming pass. Used to size a field before
 * the real parse runs.
 *
 * @param filepath Path to the file.
 * @param skip_header Header lines to exclude from the count.
 * @return Rows after the header, or 0 if the file could not be opened.
 */
uint64_t sif__io_ascii_row_count(const char* filepath, uint32_t skip_header);

#endif /* SIF__IO_INTERNAL_H */
