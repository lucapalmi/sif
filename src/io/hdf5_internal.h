/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file hdf5_internal.h
 * @brief What the HDF5 half of the I/O layer tells the rest of the library.
 * Private.
 *
 * Exactly one of hdf5.c and hdf5_off.c is compiled in, chosen by
 * SIF_HDF5_SUPPORT, and both implement what is declared here. Nothing outside
 * those two files includes hdf5.h, and no public header ever will: a build
 * without HDF5 has no trace of it, and one with it does not make its users
 * need HDF5's headers.
 */

#ifndef SIF__IO_HDF5_INTERNAL_H
#define SIF__IO_HDF5_INTERNAL_H

#include <stddef.h>

/**
 * @brief Describe the HDF5 this build carries, if any.
 *
 * @param buf Receives "HDF5 <major>.<minor>.<release>" -- the version of the
 * library actually loaded, which can differ from the one compiled against --
 * or "no HDF5".
 * @param len Size of @p buf.
 * @return 1 if this build has HDF5, 0 if not.
 */
int sif__hdf5_describe(char* buf, size_t len);

/**
 * @brief A metadata call's group, normalized: NULL for the root (NULL, "" or
 * any run of "/"), otherwise the name without its leading slashes.
 */
const char* sif__hdf5_group(const char* group);

/**
 * @brief Whether @p key names an attribute sif keeps for itself in @p group
 * (already normalized), which the metadata setters refuse. hdf5_common.c,
 * compiled into both builds so both refuse the same keys.
 */
int sif__hdf5_key_reserved(const char* group, const char* key);

#endif /* SIF__IO_HDF5_INTERNAL_H */
