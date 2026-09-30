/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file logger.h
 * @brief Levelled, tagged logging.
 *
 * Messages pass two independent filters. #SIF_LOG_LEVEL is a compile-time
 * floor: calls below it expand to nothing and cost not even a branch, which is
 * what makes SIF_LOG_TRACE() acceptable inside a hot loop. The level in
 * sif_config_t is the runtime floor, applied to whatever survived compilation.
 *
 * Output goes to stdout, or to stderr from WARNING upwards, and is coloured
 * when the destination is a terminal.
 *
 * @note Messages are written with several calls to fprintf(), so lines logged
 * concurrently from an OpenMP region can interleave with each other. Logging
 * from inside a parallel region is fine for diagnostics but should not be
 * relied on to produce parseable output.
 */

#ifndef SIF_UTILS_LOGGER_H
#define SIF_UTILS_LOGGER_H

#include <stdint.h>
#include <stdio.h>

#include "sif/core/macros.h"

/**
 * @brief Compile-time verbosity floor; one of the SIF_LOG_LEVEL_* constants.
 *
 * Define it when building to compile whole levels out of the library. Defaults
 * to TRACE, which keeps every call site and leaves the decision to the runtime
 * level in sif_config_t.
 */
#ifndef SIF_LOG_LEVEL
#  define SIF_LOG_LEVEL SIF_LOG_LEVEL_TRACE
#endif

/**
 * @brief Backs the SIF_LOG_* macros. Not part of the API -- call the macros.
 *
 * @param level One of the SIF_LOG_LEVEL_* constants, TRACE through ERROR.
 * @param tag Short subsystem name, printed in brackets.
 * @param fmt printf-style format string, followed by its arguments.
 */
void sif__log_impl(uint8_t level, const char* tag, const char* fmt, ...);

/** @brief Backs SIF_LOG_FLUSH(). Not part of the API. */
void sif__log_flush(void);

/**
 * @defgroup logging Logging macros
 * @brief Emit a message at a given level, if both the compile-time and runtime
 * floors allow it.
 *
 * Each takes a subsystem @p tag and a printf-style format:
 *
 * @code
 * SIF_LOG_INFO("finder", "found %" PRIu64 " voids", n);
 * @endcode
 *
 * @note These rely on the `, ##__VA_ARGS__` extension, which swallows the
 * comma when a call passes no arguments beyond the format string. It is not
 * C99, but GCC, Clang and MSVC all implement it, and the strictly conforming
 * alternatives all cost either a mandatory dummy argument at every call site
 * or a second macro per level.
 * @{
 */

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_TRACE
#  define SIF_LOG_TRACE(tag, fmt, ...)                                         \
    sif__log_impl(SIF_LOG_LEVEL_TRACE, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_TRACE(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_DEBUG
#  define SIF_LOG_DEBUG(tag, fmt, ...)                                         \
    sif__log_impl(SIF_LOG_LEVEL_DEBUG, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_DEBUG(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_INFO
#  define SIF_LOG_INFO(tag, fmt, ...)                                          \
    sif__log_impl(SIF_LOG_LEVEL_INFO, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_INFO(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_WARNING
#  define SIF_LOG_WARNING(tag, fmt, ...)                                       \
    sif__log_impl(SIF_LOG_LEVEL_WARNING, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_WARNING(tag, fmt, ...) ((void)0)
#endif

/* Compiled out, an error is still recorded for sif_error_message(): the
 * record is how a caller finds out why something failed, and a build that
 * prints nothing must not also lose that. */
#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_ERROR
#  define SIF_LOG_ERROR(tag, fmt, ...)                                         \
    sif__log_impl(SIF_LOG_LEVEL_ERROR, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_ERROR(tag, fmt, ...) sif__error_record(fmt, ##__VA_ARGS__)
#endif

/** @brief Flush both output streams. */
#define SIF_LOG_FLUSH() sif__log_flush()

/** @} */

/** @brief Backs a compiled-out SIF_LOG_ERROR(). Not part of the API. */
void sif__error_record(const char* fmt, ...);

/**
 * @defgroup error_record Error record
 * @brief Why the last failed call failed, for callers that cannot read the
 * log.
 *
 * Every error logged is also kept, per thread, until sif_error_clear(). Only
 * the first one since the clear is kept: that is the root cause, and anything
 * logged after it is a consequence. The intended use is
 *
 * @code
 * sif_error_clear();
 * if (sif_catalogue_read_hdf5(path) == NULL)
 *   fprintf(stderr, "%s\n", sif_error_message());
 * @endcode
 *
 * The record does not depend on the runtime log level: an error suppressed
 * from the output is still recorded.
 *
 * @note Per thread. An error logged by an OpenMP worker is recorded on that
 * worker, not on the thread that made the call; the library logs I/O errors
 * outside parallel regions for this reason.
 * @{
 */

/**
 * @brief The first error logged on this thread since the last
 * sif_error_clear(), without the level and tag; "" if there was none.
 *
 * @return A buffer owned by the library, valid until the next error or clear
 * on this thread.
 */
const char* sif_error_message(void);

/**
 * @brief The operating-system error (an errno value) behind the recorded
 * error, or 0 if it was not one -- a malformed file rather than a missing
 * one, say.
 */
int sif_error_errno(void);

/**
 * @brief The status behind the recorded error, for the readers that return a
 * pointer and so cannot return one: SIF_ERR_INVALID for a request the file
 * cannot satisfy (a column it does not have), SIF_ERR_IO for a file that is
 * not what it should be. SIF_OK if the failing call did not say, which the
 * caller should read as SIF_ERR_IO.
 */
int sif_error_status(void);

/** @brief Forget the recorded error on this thread. */
void sif_error_clear(void);

/** @} */

#endif /* SIF_UTILS_LOGGER_H */
