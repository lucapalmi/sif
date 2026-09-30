/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file logger_internal.h
 * @brief The parts of the error record only the library writes. Private to
 * the library.
 */

#ifndef SIF__UTILS_LOGGER_INTERNAL_H
#define SIF__UTILS_LOGGER_INTERNAL_H

/**
 * @brief Attach an errno value to the error just logged.
 *
 * Called straight after the SIF_LOG_ERROR() that reports an operating-system
 * failure. Does nothing if that error was not the one recorded -- an earlier
 * error is the root cause, and keeps its own errno, or none.
 */
void sif__error_os_set(int os_error);

/**
 * @brief Attach the status a pointer-returning function failed with, for
 * sif_error_status(). The innermost such function sets it; an outer one
 * leaves it be.
 */
void sif__error_status_set(int status);

#endif /* SIF__UTILS_LOGGER_INTERNAL_H */
