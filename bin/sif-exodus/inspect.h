/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* `sif-exodus --check FILE`: what kind of file FILE is, and the library's
 * check of it. */

#ifndef SIF_EXODUS_INSPECT_H
#define SIF_EXODUS_INSPECT_H

#include <stdbool.h>
#include <stdint.h>

typedef enum {
  FILE_KIND_AUTO = 0,
  FILE_KIND_LUA,
  FILE_KIND_ASCII,
  FILE_KIND_FITS,
  FILE_KIND_HDF5,
  FILE_KIND_GADGET,
  FILE_KIND_XFIELD,
  /* Detection found nothing it knows. */
  FILE_KIND_UNKNOWN
} file_kind_t;

typedef struct {
  /* As given with --format, or FILE_KIND_AUTO. */
  file_kind_t format;
  /* ASCII only; NULL, '\0' and 0 when not given. */
  const char* columns;
  char delimiter;
  uint32_t skip_header;
  bool deep;
} inspect_opts_t;

/* "lua", "ascii", ...: 0 with *out set, -1 for a name it does not know. */
int inspect_kind_parse(const char* name, file_kind_t* out);

const char* inspect_kind_name(file_kind_t kind);

/* The kind of @p path, from its name for a configuration and from its first
 * bytes for the rest. A path that does not exist is FILE_KIND_ASCII, whose
 * check then says so, unless it is a multi-file snapshot's base name. */
file_kind_t inspect_detect(const char* path);

/* Check a data file of @p kind (not a configuration), writing the report to
 * stdout and the problems to the log. The library is initialised. Returns
 * the exit status: 0 for a sound file, 2 for one with problems. */
int inspect_file(const char* path, file_kind_t kind, const inspect_opts_t* o);

#endif /* SIF_EXODUS_INSPECT_H */
