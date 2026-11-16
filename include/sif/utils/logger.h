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
 * floor: calls below it expand to nothing. The runtime level, set from
 * sif_config_t at init and by sif_log_level_set(), is checked inside the
 * macro, so a call it filters out costs a load and a branch: its arguments
 * are not evaluated and nothing is formatted, which is what makes
 * SIF_LOG_TRACE() acceptable inside a hot loop.
 *
 * A message that passes both is formatted once and handed to the sink in
 * force (sif_log_sink_set()). The default sink writes to stdout, or to stderr
 * from WARNING upwards, coloured when the destination is a terminal;
 * sif_log_sink_stream_set() and sif_log_sink_file_set() send everything to
 * one stream instead. The built-in sinks write each message with a single
 * call, so lines logged concurrently from an OpenMP region do not interleave.
 *
 * An ERROR is also kept in the error record (sif_error_message()) whatever
 * the level and the sink.
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
 * @param tag The subsystem, as `area.domain` (see the logging macros).
 * @param fmt printf-style format string, followed by its arguments.
 */
void sif__log_impl(uint8_t level, const char* tag, const char* fmt, ...);

/** @brief Backs SIF_LOG_FLUSH(). Not part of the API. */
void sif__log_flush(void);

/**
 * @brief The runtime level, read by the SIF_LOG_* macros. Not part of the
 * API: use sif_log_level_get() and sif_log_level_set().
 */
extern uint8_t sif__log_level;

/**
 * @brief How many threads have a capture in force. Not part of the API: it
 * lets a WARNING below the runtime level through the macro to
 * sif__log_impl(), which delivers it to the capture of its thread.
 */
extern int sif__log_captures;

/**
 * @defgroup log_level Runtime level
 * @brief The floor a message must reach to be logged, after the compile-time
 * one.
 *
 * sif_system_init() sets it from sif_config_t::log_level, and
 * sif_system_finalise() puts it back to SIF_LOG_LEVEL_INFO, the level in
 * force before init.
 *
 * @note Not synchronised: set it between library calls, not while one may be
 * logging from another thread.
 * @{
 */

/** @brief The runtime level; one of the SIF_LOG_LEVEL_* constants. */
uint8_t sif_log_level_get(void);

/**
 * @brief Set the runtime level, TRACE through NONE. A larger value is taken
 * as SIF_LOG_LEVEL_NONE.
 */
void sif_log_level_set(uint8_t level);

/** @} */

/**
 * @defgroup log_sink Sinks
 * @brief Where messages go once they pass both levels.
 *
 * A sink receives each message formatted, without the level and tag, which
 * come as fields of the record:
 *
 * @code
 * static void to_buffer(void* ctx, const sif_log_record_t* rec) {
 *   my_buffer_append(ctx, rec->level, rec->tag, rec->message, rec->length);
 * }
 *
 * sif_log_sink_t sink = {.write = to_buffer, .flush = NULL, .ctx = buf};
 * sif_log_sink_set(&sink);
 * @endcode
 *
 * The library calls the sink from whichever thread logged, OpenMP workers
 * included, so a sink must be safe to call concurrently; the built-in ones
 * are. Sinks are process-wide and independent of sif_system_init(): one set
 * before init receives init's own messages.
 *
 * @note Not synchronised: set the sink between library calls, not while one
 * may be logging from another thread.
 * @{
 */

/** @brief One message, as a sink receives it. Valid only during the call. */
typedef struct {
  /** One of the SIF_LOG_LEVEL_* constants, TRACE through ERROR. */
  uint8_t level;
  /** Subsystem name, as passed to the macro. */
  const char* tag;
  /** The formatted message: no level, no tag, no trailing newline. */
  const char* message;
  /** strlen(message). */
  size_t length;
  /** OpenMP thread number of the caller; 0 outside a parallel region. */
  int thread;
} sif_log_record_t;

/** @brief A log backend: what to call for each message, and to flush. */
typedef struct {
  /** Called once per message. Required. */
  void (*write)(void* ctx, const sif_log_record_t* record);
  /** Called by SIF_LOG_FLUSH(), or NULL if there is nothing to flush. */
  void (*flush)(void* ctx);
  /** Passed back to both, untouched. */
  void* ctx;
} sif_log_sink_t;

/**
 * @brief Send messages to @p sink, or back to the default sink if @p sink is
 * NULL.
 *
 * The sink is copied; @p ctx must stay valid until it is replaced. A file
 * opened by sif_log_sink_file_set() is closed when replaced.
 *
 * @return SIF_OK, or SIF_ERR_INVALID if @p sink has no write function (the
 * sink in force is kept).
 */
SIF_NODISCARD int sif_log_sink_set(const sif_log_sink_t* sink);

/**
 * @brief The sink in force, for saving it and putting it back with
 * sif_log_sink_set(). Writes the default sink when no other is set.
 *
 * @note Restoring a saved sif_log_sink_file_set() sink after replacing it
 * does not reopen the file, which was closed: save and restore only sinks
 * whose context you own.
 */
void sif_log_sink_get(sif_log_sink_t* out);

/**
 * @brief Write every message to @p stream, uncoloured, as
 * `[LEVEL] tag: message`.
 *
 * The stream stays the caller's: it is not closed, and must stay open until
 * the sink is replaced. An in-memory stream (open_memstream(), fmemopen())
 * makes a buffer sink.
 *
 * @return SIF_OK, or SIF_ERR_INVALID if @p stream is NULL.
 */
SIF_NODISCARD int sif_log_sink_stream_set(FILE* stream);

/**
 * @brief Write every message to the file at @p path, uncoloured, as
 * `[LEVEL] tag: message`, appending if it exists.
 *
 * The library owns the file and closes it when the sink is replaced. Each
 * message is written with one call to the stream, so lines from concurrent
 * threads stay whole; the stream is line-buffered, so a run that crashes
 * loses at most the line in progress.
 *
 * @return SIF_OK, or SIF_ERR_INVALID for a NULL @p path, or SIF_ERR_IO if
 * the file cannot be opened (the sink in force is kept).
 */
SIF_NODISCARD int sif_log_sink_file_set(const char* path);

/** @} */

/**
 * @defgroup logging Logging macros
 * @brief Emit a message at a given level, if both the compile-time and runtime
 * floors allow it.
 *
 * Each takes a subsystem @p tag and a printf-style format:
 *
 * @code
 * SIF_LOG_INFO("finder.exodus", "found %" PRIu64 " voids", n);
 * @endcode
 *
 * A tag is `area.domain`: the directory of the public header that declares
 * the function at work, and that header's domain, without a suffix repeating
 * the area (`_io`, `_finder`). Filtering on a prefix therefore selects an area
 * -- `io` is every reader and writer. Helpers shared between domains log under
 * their caller's tag. The library's tags are:
 *
 * - `core.settings`, `core.system`
 * - `utils.array`, `utils.logger`, `utils.timer`
 * - `math.fft`, `math.nn`
 * - `structures.bitmask`, `structures.catalogue`,
 *   `structures.cell_linked_list`, `structures.chain_mesh`, `structures.delta`,
 *   `structures.field`, `structures.grid`, `structures.size_function`,
 *   `structures.tessellation`
 * - `io.catalogue`, `io.field`, `io.file`, `io.fits`, `io.gadget`, `io.grid`,
 *   `io.hdf5`, `io.profiles`, `io.scratch`
 * - `measure.delta`, `measure.profiles`, `measure.size_function`,
 *   `measure.survey_volume`
 * - `model.bbks`, `model.cosmology`, `model.delta`, `model.ep`,
 *   `model.size_function`, `model.variance_kernel`
 * - `finder.exodus`, `finder.spherical`
 *
 * The programs in bin/ log under their own name (`sif-exodus`).
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
    (sif__log_level <= SIF_LOG_LEVEL_TRACE                                     \
        ? sif__log_impl(SIF_LOG_LEVEL_TRACE, tag, fmt, ##__VA_ARGS__)          \
        : (void)0)
#else
#  define SIF_LOG_TRACE(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_DEBUG
#  define SIF_LOG_DEBUG(tag, fmt, ...)                                         \
    (sif__log_level <= SIF_LOG_LEVEL_DEBUG                                     \
        ? sif__log_impl(SIF_LOG_LEVEL_DEBUG, tag, fmt, ##__VA_ARGS__)          \
        : (void)0)
#else
#  define SIF_LOG_DEBUG(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_INFO
#  define SIF_LOG_INFO(tag, fmt, ...)                                          \
    (sif__log_level <= SIF_LOG_LEVEL_INFO                                      \
        ? sif__log_impl(SIF_LOG_LEVEL_INFO, tag, fmt, ##__VA_ARGS__)           \
        : (void)0)
#else
#  define SIF_LOG_INFO(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_WARNING
#  define SIF_LOG_WARNING(tag, fmt, ...)                                       \
    (sif__log_level <= SIF_LOG_LEVEL_WARNING || sif__log_captures > 0          \
        ? sif__log_impl(SIF_LOG_LEVEL_WARNING, tag, fmt, ##__VA_ARGS__)        \
        : (void)0)
#else
#  define SIF_LOG_WARNING(tag, fmt, ...) ((void)0)
#endif

/* Not gated on the runtime level: an error is recorded for
 * sif_error_message() even when it is not logged. Compiled out, it is still
 * recorded: the record is how a caller finds out why something failed, and a
 * build that prints nothing must not also lose that. */
#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_ERROR
#  define SIF_LOG_ERROR(tag, fmt, ...)                                         \
    sif__log_impl(SIF_LOG_LEVEL_ERROR, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_ERROR(tag, fmt, ...) sif__error_record(fmt, ##__VA_ARGS__)
#endif

/** @brief Flush the sink: stdout and stderr for the default one. */
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
