/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* The XSI strerror_r(), which returns an int. glibc swaps in its own, which
 * returns a char*, when _GNU_SOURCE is defined. */
#undef _GNU_SOURCE

#include "internal.h"

#include "core/system_internal.h"
#include "sif/utils/logger.h"
#include "utils/logger_internal.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <sys/stat.h>
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

const char* sif__io_strerror(int err) {
  static SIF_THREAD_LOCAL char buf[128];
  if (strerror_r(err, buf, sizeof(buf)) != 0)
    snprintf(buf, sizeof(buf), "error %d", err);
  return buf;
}

int sif__io_os_error(
  const char* tag, const char* path, const char* what, int err) {
  if (what)
    SIF_LOG_ERROR(tag, "%s: %s: %s", path, what, sif__io_strerror(err));
  else
    SIF_LOG_ERROR(tag, "%s: %s", path, sif__io_strerror(err));
  sif__error_os_set(err);
  return SIF_ERR_IO;
}

void sif__io_problem(sif__io_problems_t* p, const char* fmt, ...) {
  if (p->n++ >= SIF__IO_PROBLEMS_SHOWN)
    return;
  char msg[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);
  SIF_LOG_ERROR(p->tag, "%s", msg);
}

int sif__io_problems_end(const sif__io_problems_t* p, FILE* report) {
  if (p->n > SIF__IO_PROBLEMS_SHOWN)
    SIF_LOG_ERROR(p->tag, "%" PRIu64 " more problems not listed",
      p->n - SIF__IO_PROBLEMS_SHOWN);
  if (report) {
    if (p->n)
      fprintf(report, "problems      %" PRIu64 "\n", p->n);
    else
      fprintf(report, "problems      none\n");
    fflush(report);
  }
  return p->n ? SIF_ERR_IO : SIF_OK;
}

void sif__io_stats_merge(sif__io_stats_t* a, const sif__io_stats_t* b) {
  const bool a_has = a->n > a->n_nonfinite, b_has = b->n > b->n_nonfinite;
  if (b_has && (!a_has || b->min < a->min))
    a->min = b->min;
  if (b_has && (!a_has || b->max > a->max))
    a->max = b->max;
  a->n += b->n;
  a->n_nonfinite += b->n_nonfinite;
}

void sif__io_stats_header(FILE* report, int name_width) {
  fprintf(report, "%-*s  %15s  %15s  %10s\n", name_width, "column", "min",
    "max", "non-finite");
}

void sif__io_stats_print(
  FILE* report, const char* name, int name_width, const sif__io_stats_t* s) {
  if (s->n == s->n_nonfinite)
    fprintf(report, "%-*s  %15s  %15s  %10" PRIu64 "\n", name_width, name, "-",
      "-", s->n_nonfinite);
  else
    fprintf(report, "%-*s  %15.8g  %15.8g  %10" PRIu64 "\n", name_width, name,
      s->min, s->max, s->n_nonfinite);
}

int sif__io_check_readable(
  const char* tag, const char* path, uint64_t* out_bytes) {
  struct stat st;
  if (stat(path, &st) != 0)
    return sif__io_os_error(tag, path, NULL, errno);
  if (S_ISDIR(st.st_mode))
    return sif__io_os_error(tag, path, NULL, EISDIR);
  if (access(path, R_OK) != 0)
    return sif__io_os_error(tag, path, NULL, errno);

  if (out_bytes)
    *out_bytes = (uint64_t)st.st_size;
  return SIF_OK;
}

int sif__io_ascii_row_count(const char* tag, const char* filepath,
  uint32_t skip_header, uint64_t* out_rows) {
  *out_rows = 0;

  FILE* f = fopen(filepath, "r");
  if (!f)
    return sif__io_os_error(tag, filepath, NULL, errno);

  char buffer[COUNT_BUFFER_BYTES];
  size_t bytes_read;
  uint64_t total_lines = 0;
  uint64_t total_bytes = 0;
  char last = '\n';

  while ((bytes_read = fread(buffer, 1, sizeof(buffer), f)) > 0) {
    for (size_t i = 0; i < bytes_read; i++) {
      if (buffer[i] == '\n')
        total_lines++;
    }
    last = buffer[bytes_read - 1];
    total_bytes += bytes_read;
  }

  /* A file whose last line has no terminator still holds a row, and counting
   * newlines alone would drop it -- silently, since the caller sizes the field
   * from this number and then reads exactly that many rows. */
  if (last != '\n')
    total_lines++;

  /* A read error leaves a count that is merely plausible, which is worse than
   * none: the caller would size a field from it and truncate the data without
   * ever knowing the file was not fully read. */
  const int err = ferror(f) ? errno : 0;
  fclose(f);
  if (err)
    return sif__io_os_error(tag, filepath, "read", err);

  if (total_bytes == 0) {
    SIF_LOG_ERROR(tag, "%s: empty file", filepath);
    return SIF_ERR_IO;
  }
  if (total_lines <= skip_header) {
    SIF_LOG_ERROR(tag,
      "%s: no lines after the header (%" PRIu64 " lines, skip_header %u)",
      filepath, total_lines, skip_header);
    return SIF_ERR_IO;
  }

  *out_rows = total_lines - skip_header;
  return SIF_OK;
}

#undef COUNT_BUFFER_BYTES
