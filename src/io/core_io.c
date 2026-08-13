/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "internal.h"

#include "core/system_internal.h"
#include "sif/utils/logger.h"

#include <unistd.h>

int sif__io_pread_parallel(
  int fd, void* dest, size_t total_bytes, off_t base_offset) {
  if (total_bytes == 0)
    return SIF_OK;
  int success = 1;

#pragma omp parallel
  {
    int tid = sif__system_thread_num();
    int num_t = sif__system_num_threads();

    /* Distribute the byte load across threads */
    size_t chunk = total_bytes / num_t;
    size_t local_size =
      (tid == num_t - 1) ? (total_bytes - (tid * chunk)) : chunk;
    off_t local_offset = base_offset + (off_t)(tid * chunk);
    char* local_dest = (char*)dest + (tid * chunk);

    size_t bytes_read = 0;

    /* Linux pread can return 'short reads'. We loop until the chunk is full. */
    while (bytes_read < local_size && success) {
      ssize_t res = pread(fd, local_dest + bytes_read, local_size - bytes_read,
        local_offset + bytes_read);

      if (res <= 0) {
#pragma omp atomic write
        success = 0;
        break;
      }
      bytes_read += res;
    }
  }
  return success ? SIF_OK : SIF_ERR_IO;
}

uint64_t sif__io_ascii_row_count(const char* filepath, uint32_t skip_header) {
  FILE* f = fopen(filepath, "r");
  if (!f) {
    SIF_LOG_ERROR("io", "failed to open %s", filepath);
    return 0;
  }
  char buffer[16384];
  size_t bytes_read;
  uint64_t total_lines = 0;

  while ((bytes_read = fread(buffer, 1, sizeof(buffer), f)) > 0) {
    for (size_t i = 0; i < bytes_read; i++) {
      if (buffer[i] == '\n') {
        total_lines++;
      }
    }
  }
  fclose(f);

  if (total_lines <= skip_header)
    return 0;
  return total_lines - skip_header;
}