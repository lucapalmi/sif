/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* sif-inspect: a file described, or checked, by the library. */

#ifndef SIF_INSPECT_INSPECT_H
#define SIF_INSPECT_INSPECT_H

#include "sif/io/file_info.h"

/* Describe @p path with sif_file_info_read(), to stdout. The library is
 * initialised. Returns the exit status: 0, or 2 for a file that cannot be
 * read. */
int inspect_describe(const char* path, sif_file_format_t format);

/* Check @p path with sif_file_check(), to stdout. Returns the exit status:
 * 0 for a file without errors, warnings or not; 2 for one with any. */
int inspect_check(const char* path, sif_file_format_t format);

#endif /* SIF_INSPECT_INSPECT_H */
