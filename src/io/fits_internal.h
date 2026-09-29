/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file fits_internal.h
 * @brief What the FITS reader tells the rest of the library. Private.
 *
 * Exactly one of fits_io.c and fits_off.c is compiled in, chosen by
 * SIF_FITS_SUPPORT, and both implement what is declared here. Nothing outside
 * fits_io.c includes fitsio.h.
 */

#ifndef SIF__IO_FITS_INTERNAL_H
#define SIF__IO_FITS_INTERNAL_H

#include <stddef.h>

/**
 * @brief Describe the cfitsio this build carries, if any.
 *
 * @param buf Receives "cfitsio <version>" -- the version of the library
 * actually loaded -- or "no cfitsio".
 * @param len Size of @p buf.
 * @return 1 if this build has cfitsio, 0 if not.
 */
int sif__fits_describe(char* buf, size_t len);

#endif /* SIF__IO_FITS_INTERNAL_H */
