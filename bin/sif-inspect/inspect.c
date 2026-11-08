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

int inspect_describe(const char* path, sif_file_format_t format) {
  sif_file_info_t* info = sif_file_info_read(path, format);
  if (!info)
    return 2;
  const int status = sif_file_info_print(info, stdout);
  sif_file_info_free(info);
  return status == SIF_OK ? 0 : 1;
}

int inspect_check(
  const char* path, sif_file_format_t format, const inspect_opts_t* o) {
  int status;
  switch (format) {
  case SIF_FILE_FORMAT_ASCII:
    status = sif_field_check_ascii(
      path, o->columns, o->delimiter, o->skip_header, stdout);
    break;
  case SIF_FILE_FORMAT_FITS:
    status = sif_fits_check(path, o->deep, stdout);
    break;
  case SIF_FILE_FORMAT_HDF5:
    status = sif_hdf5_check(path, o->deep, stdout);
    break;
  case SIF_FILE_FORMAT_GADGET:
    status = sif_gadget_check(path, SIF_GADGET_FORMAT_AUTO, o->deep, stdout);
    break;
  case SIF_FILE_FORMAT_XFIELD:
    status = sif_field_check_xfield(path, o->deep, stdout);
    break;
  case SIF_FILE_FORMAT_CHECKPOINT:
    status = sif_finder_exodus_checkpoint_check(path, stdout);
    break;
  case SIF_FILE_FORMAT_XGRID:
    fprintf(stderr, "%s: no check for .xgrid files yet\n", path);
    return 2;
  default:
    fprintf(stderr,
      "%s: not a kind of file sif-inspect recognises (FITS, HDF5, GADGET, "
      ".xfield, .xgrid, ASCII, exodus checkpoint); say which with --format\n",
      path);
    return 2;
  }
  fflush(stdout);

  if (status != SIF_OK)
    return 2;
  fprintf(stderr, "%s: OK\n", path);
  return 0;
}
