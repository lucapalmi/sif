/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "inspect.h"

#include "sif/core/macros.h"
#include "sif/finder/exodus_finder.h"
#include "sif/io/field_io.h"
#include "sif/io/fits_io.h"
#include "sif/io/gadget_io.h"
#include "sif/io/hdf5_io.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static const char* const KIND_NAMES[] = {"auto", "lua", "ascii", "fits", "hdf5",
  "gadget", "xfield", "checkpoint", "unknown"};

int inspect_kind_parse(const char* name, file_kind_t* out) {
  for (int k = FILE_KIND_AUTO; k < FILE_KIND_UNKNOWN; k++) {
    if (strcmp(name, KIND_NAMES[k]) == 0) {
      *out = (file_kind_t)k;
      return 0;
    }
  }
  return -1;
}

const char* inspect_kind_name(file_kind_t kind) { return KIND_NAMES[kind]; }

static bool ends_with(const char* s, const char* tail) {
  const size_t n = strlen(s), m = strlen(tail);
  return n >= m && strcmp(s + n - m, tail) == 0;
}

/* Whether a file starts as text does: no NUL, and few control characters
 * besides the blanks, in its first block. */
static bool looks_like_text(const unsigned char* head, size_t n) {
  size_t odd = 0;
  for (size_t i = 0; i < n; i++) {
    if (head[i] == 0)
      return false;
    if (head[i] < 32 && head[i] != '\n' && head[i] != '\r' && head[i] != '\t')
      odd++;
  }
  return odd * 100 <= n;
}

file_kind_t inspect_detect(const char* path) {
  if (ends_with(path, ".lua"))
    return FILE_KIND_LUA;

  struct stat st;
  if (stat(path, &st) != 0 || !S_ISREG(st.st_mode))
    return sif_gadget_is_snapshot(path) ? FILE_KIND_GADGET : FILE_KIND_ASCII;

  unsigned char head[4096];
  size_t n = 0;
  FILE* f = fopen(path, "rb");
  if (f) {
    n = fread(head, 1, sizeof(head), f);
    fclose(f);
  }
  if (!f || n == 0)
    return FILE_KIND_ASCII; /* whose check says what is wrong */

  static const unsigned char h5sig[8] = {
    0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
  if (n >= 6 && memcmp(head, "SIMPLE", 6) == 0)
    return FILE_KIND_FITS;
  if (n >= 2 && head[0] == 0x1f && head[1] == 0x8b && strstr(path, ".fits"))
    return FILE_KIND_FITS;
  if (n >= 4 && memcmp(head, SIF_XFIELD_MAGIC, 4) == 0)
    return FILE_KIND_XFIELD;
  if (n >= 4 && memcmp(head, SIF_FINDER_CHECKPOINT_MAGIC, 4) == 0)
    return FILE_KIND_CHECKPOINT;
  if (sif_gadget_is_snapshot(path))
    return FILE_KIND_GADGET;
  if (n >= 8 && memcmp(head, h5sig, 8) == 0)
    return FILE_KIND_HDF5;
  if (looks_like_text(head, n))
    return FILE_KIND_ASCII;
  return FILE_KIND_UNKNOWN;
}

int inspect_file(const char* path, file_kind_t kind, const inspect_opts_t* o) {
  int status;
  switch (kind) {
  case FILE_KIND_ASCII:
    status = sif_field_check_ascii(
      path, o->columns, o->delimiter, o->skip_header, stdout);
    break;
  case FILE_KIND_FITS:
    status = sif_fits_check(path, o->deep, stdout);
    break;
  case FILE_KIND_HDF5:
    status = sif_hdf5_check(path, o->deep, stdout);
    break;
  case FILE_KIND_GADGET:
    status = sif_gadget_check(path, SIF_GADGET_FORMAT_AUTO, o->deep, stdout);
    break;
  case FILE_KIND_XFIELD:
    status = sif_field_check_xfield(path, o->deep, stdout);
    break;
  case FILE_KIND_CHECKPOINT:
    status = sif_finder_exodus_checkpoint_check(path, stdout);
    break;
  default:
    fprintf(stderr,
      "%s: not a kind of file --check recognises (FITS, HDF5, GADGET, "
      ".xfield, ASCII, exodus checkpoint); say which with --format\n",
      path);
    return 2;
  }
  fflush(stdout);

  if (status != SIF_OK)
    return 2;
  fprintf(stderr, "%s: OK\n", path);
  return 0;
}
