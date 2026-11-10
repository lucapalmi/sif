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

/**
 * @brief Bytes as GiB, for the sizes an allocation failure reports: a size is
 * what tells a caller whether it asked for too much or the machine has too
 * little.
 */
#define SIF__GIB(bytes) ((double)(bytes) / (1024.0 * 1024.0 * 1024.0))

/**
 * @brief Where captured messages go: the level, the tag, and the message
 * without either.
 */
typedef void (*sif__log_sink_t)(
  void* ctx, int level, const char* tag, const char* message);

/** @brief A capture in force, saved by sif__log_capture_begin(). */
typedef struct {
  sif__log_sink_t sink;
  void* ctx;
} sif__log_capture_t;

/**
 * @brief Send this thread's WARNINGs and ERRORs to @p sink instead of the
 * log, until sif__log_capture_end().
 *
 * For a file check, whose findings are the readers' own messages: they go
 * into its list of problems, not to stderr. An ERROR is still recorded for
 * sif_error_message(). Captures nest: the one in force is saved in
 * @p saved and restored at the end. Per thread, as the error record is.
 */
void sif__log_capture_begin(
  sif__log_sink_t sink, void* ctx, sif__log_capture_t* saved);

/** @brief Restore the capture sif__log_capture_begin() replaced. */
void sif__log_capture_end(const sif__log_capture_t* saved);

#endif /* SIF__UTILS_LOGGER_INTERNAL_H */
