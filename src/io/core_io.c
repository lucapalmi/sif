/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "internal.h"

#include "core/system_internal.h"
#include "sif/utils/logger.h"

#include <unistd.h>

/* Big enough that the newline scan is bound by the disk rather than by the
 * loop around fread(). */
#define COUNT_BUFFER_BYTES 16384

int sif__io_pread_parallel(
  int fd, void* dest, size_t total_bytes, off_t base_offset) {
  if (total_bytes == 0)
    return SIF_OK;

  /* Written by any thread that gives up, never read inside the loop: a thread
   * only needs to abandon its own chunk, so there is no shared state to poll
   * and no race between the write here and a bare read in a loop condition. */
  int failed = 0;

#pragma omp parallel
  {
    const int tid = sif__system_thread_num();
    const int num_t = sif__system_num_threads();

    /* Equal chunks, with the remainder going to the last thread. pread() takes
     * its offset as an argument and leaves the file position alone, which is
     * the whole reason several threads can share one descriptor here. */
    const size_t chunk = total_bytes / (size_t)num_t;
    const size_t local_size =
      (tid == num_t - 1) ? (total_bytes - (size_t)tid * chunk) : chunk;
    const off_t local_offset = base_offset + (off_t)((size_t)tid * chunk);
    char* local_dest = (char*)dest + (size_t)tid * chunk;

    size_t bytes_read = 0;
    while (bytes_read < local_size) {
      /* A short read is normal and says nothing about failure -- pread() is
       * free to return less than asked for. Only 0 (early EOF, so the file is
       * shorter than the header claims) or -1 ends the chunk. */
      ssize_t res = pread(fd, local_dest + bytes_read, local_size - bytes_read,
        local_offset + (off_t)bytes_read);
      if (res <= 0) {
#pragma omp atomic write
        failed = 1;
        break;
      }
      bytes_read += (size_t)res;
    }
  }

  return failed ? SIF_ERR_IO : SIF_OK;
}

uint64_t sif__io_ascii_row_count(const char* filepath, uint32_t skip_header) {
  FILE* f = fopen(filepath, "r");
  if (!f) {
    SIF_LOG_ERROR("io", "failed to open %s", filepath);
    return 0;
  }

  char buffer[COUNT_BUFFER_BYTES];
  size_t bytes_read;
  uint64_t total_lines = 0;
  char last = '\n';

  while ((bytes_read = fread(buffer, 1, sizeof(buffer), f)) > 0) {
    for (size_t i = 0; i < bytes_read; i++) {
      if (buffer[i] == '\n')
        total_lines++;
    }
    last = buffer[bytes_read - 1];
  }

  /* A file whose last line has no terminator still holds a row, and counting
   * newlines alone would drop it -- silently, since the caller sizes the field
   * from this number and then reads exactly that many rows. */
  if (last != '\n')
    total_lines++;

  /* A read error leaves a count that is merely plausible, which is worse than
   * none: the caller would size a field from it and truncate the data without
   * ever knowing the file was not fully read. */
  const int failed = ferror(f);
  fclose(f);
  if (failed) {
    SIF_LOG_ERROR("io", "error while reading %s", filepath);
    return 0;
  }

  if (total_lines <= skip_header)
    return 0;
  return total_lines - skip_header;
}

#undef COUNT_BUFFER_BYTES
