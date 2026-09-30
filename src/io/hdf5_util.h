/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file hdf5_util.h
 * @brief HDF5 plumbing shared by hdf5.c and gadget_hdf5.c. Private, and only
 * compiled into a build with HDF5.
 */

#ifndef SIF__IO_HDF5_UTIL_H
#define SIF__IO_HDF5_UTIL_H

#include <hdf5.h>

/**
 * @brief HDF5's automatic error printing, as it was before an entry point
 * turned it off.
 *
 * HDF5 prints its whole error stack to stderr on every failed call, including
 * the ones that are merely questions ("is this attribute here?"). Each entry
 * point turns that off for its own duration and says what went wrong through
 * the sif logger instead -- and puts back whatever handler was there, since
 * the process may be using HDF5 for other things.
 */
typedef struct {
  H5E_auto2_t func;
  void* data;
} sif__h5_quiet_t;

void sif__h5_quiet_begin(sif__h5_quiet_t* q);
void sif__h5_quiet_end(const sif__h5_quiet_t* q);

/**
 * @brief Why the last HDF5 call on this thread failed: the innermost message
 * on its error stack, which is the specific one ("file signature not found",
 * "object 'x' doesn't exist"); the outer ones only say which call it was.
 *
 * Call it straight after the failing call: the next HDF5 call, cleanup
 * included, replaces the stack.
 *
 * @return A per-thread buffer, valid until the next call.
 */
const char* sif__h5_cause(void);

/**
 * @brief Save sif__h5_cause() for a caller that reports the failure later.
 *
 * For helpers that fail deep inside a larger operation: HDF5 clears its stack
 * on entry to every call, so by the time the caller has closed what it opened
 * the cause is gone. The helper calls this at the point of failure, and the
 * caller logs sif__h5_kept().
 *
 * @return SIF_ERR_IO, for `return sif__h5_keep();`.
 */
int sif__h5_keep(void);

/** @brief The cause saved by the last sif__h5_keep() on this thread. */
const char* sif__h5_kept(void);

/**
 * @brief Open a file for reading, logging "<path>: <reason>" on failure.
 *
 * The path is checked with sif__io_check_readable() first, so a missing or
 * unreadable file is reported as the system's error rather than HDF5's.
 *
 * @return The file, or H5I_INVALID_HID.
 */
hid_t sif__h5_open_read(const char* tag, const char* path);

#endif /* SIF__IO_HDF5_UTIL_H */
