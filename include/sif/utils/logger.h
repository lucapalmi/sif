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

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_ERROR
#  define SIF_LOG_ERROR(tag, fmt, ...)                                         \
    sif__log_impl(SIF_LOG_LEVEL_ERROR, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_ERROR(tag, fmt, ...) ((void)0)
#endif

/** @brief Flush both output streams. */
#define SIF_LOG_FLUSH() sif__log_flush()

/** @} */

#endif /* SIF_UTILS_LOGGER_H */
