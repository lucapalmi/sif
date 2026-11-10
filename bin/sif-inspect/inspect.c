/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "inspect.h"

#include "sif/core/macros.h"
#include "sif/io/file_check.h"

#include <stdio.h>

int inspect_describe(const char* path, sif_file_format_t format) {
  sif_file_info_t* info = sif_file_info_read(path, format);
  if (!info)
    return 2;
  const int status = sif_file_info_print(info, stdout);
  sif_file_info_free(info);
  return status == SIF_OK ? 0 : 1;
}

int inspect_check(const char* path, sif_file_format_t format) {
  sif_file_check_t* check = sif_file_check(path, format);
  if (!check)
    return 1;
  const int status = sif_file_check_print(check, stdout);
  const int code = status != SIF_OK ? 1 : check->n_errors ? 2 : 0;
  sif_file_check_free(check);
  return code;
}
