/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* sif-inspect: a file described, or checked, by the library. */

#ifndef SIF_INSPECT_INSPECT_H
#define SIF_INSPECT_INSPECT_H

#include "sif/io/file_info.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
  /* As given with --format, or SIF_FILE_FORMAT_UNKNOWN to detect it. */
  sif_file_format_t format;
  /* --check: the format's check, instead of the description. */
  bool check;
  /* With --check. ASCII only: NULL, '\0' and 0 when not given. */
  const char* columns;
  char delimiter;
  uint32_t skip_header;
  bool deep;
} inspect_opts_t;

/* Describe @p path with sif_file_info_read(), to stdout. The library is
 * initialised. Returns the exit status: 0, or 2 for a file that cannot be
 * read. */
int inspect_describe(const char* path, sif_file_format_t format);

/* Check @p path, a file of @p format, writing the report to stdout and the
 * problems to the log. Returns the exit status: 0 for a sound file, 2 for
 * one with problems. */
int inspect_check(
  const char* path, sif_file_format_t format, const inspect_opts_t* o);

#endif /* SIF_INSPECT_INSPECT_H */
