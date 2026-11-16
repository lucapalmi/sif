/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/utils/logger.h"

#include "core/system_internal.h"
#include "io/internal.h"
#include "utils/logger_internal.h"

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TAG "utils.logger"

/* ANSI codes */
#define ANSI_RESET      "\x1b[0m"
#define ANSI_BOLD_WHITE "\x1b[1;37m"
#define ANSI_GREY       "\x1b[0;90m"
#define ANSI_MAGENTA    "\x1b[0;35m"
#define ANSI_BLUE       "\x1b[0;34m"
#define ANSI_YELLOW     "\x1b[0;33m"
#define ANSI_RED        "\x1b[0;31m"

/* Indexed by level, TRACE through ERROR. SIF_LOG_LEVEL_NONE is a threshold
 * rather than a level and has no entry, which is why sif__log_impl clamps
 * before indexing: it is reachable from user code through the ABI. */
static const char* level_strings[] = {
  "TRACE", "DEBUG", "INFO", "WARNING", "ERROR"};

static const char* level_colours[] = {
  ANSI_GREY,    /* TRACE   */
  ANSI_MAGENTA, /* DEBUG   */
  ANSI_BLUE,    /* INFO    */
  ANSI_YELLOW,  /* WARNING */
  ANSI_RED      /* ERROR   */
};

#define N_LEVELS ((uint8_t)(sizeof(level_strings) / sizeof(level_strings[0])))

/*
 * Whether a stream is a terminal, asked once per stream.
 *
 * isatty() is a syscall, and a TRACE line inside a loop that passes the
 * runtime gate would otherwise pay for one per message. The answer cannot
 * change under us: nothing in the library reopens stdout or stderr.
 *
 * -1 means "not yet asked". Two threads racing here both compute the same
 * answer and store the same value, so the race is benign.
 */
static int stdout_is_tty = -1;
static int stderr_is_tty = -1;

static int use_colour_for(FILE* out) {
  int* cached = (out == stderr) ? &stderr_is_tty : &stdout_is_tty;
  if (*cached < 0)
    *cached = isatty(fileno(out)) ? 1 : 0;

  return *cached;
}

/*
 * The error record: the first error since the last clear, per thread.
 *
 * `latest` says whether the most recent error logged is the recorded one, so
 * that sif__error_os_set() attaches an errno to the error it follows and not
 * to an earlier, unrelated one.
 */
#define RECORD_CAP 1024

typedef struct {
  bool set;
  bool latest;
  int os_error;
  int status;
  char text[RECORD_CAP];
} error_record_t;

static SIF_THREAD_LOCAL error_record_t record;

/* The capture in force on this thread, if any. */
static SIF_THREAD_LOCAL sif__log_capture_t capture;

int sif__log_captures = 0;

/* sif__log_demote_begin() calls in force on this thread. */
static SIF_THREAD_LOCAL int demoted = 0;

void sif__log_demote_begin(void) { demoted++; }

void sif__log_demote_end(void) {
  if (demoted > 0)
    demoted--;
}

void sif__log_capture_begin(
  sif__log_sink_t sink, void* ctx, sif__log_capture_t* saved) {
  *saved = capture;
  capture.sink = sink;
  capture.ctx = ctx;
  SIF_ATOMIC_ADD_INT(&sif__log_captures, 1);
}

void sif__log_capture_end(const sif__log_capture_t* saved) {
  capture = *saved;
  SIF_ATOMIC_ADD_INT(&sif__log_captures, -1);
}

static void record_v(const char* fmt, va_list args) {
  if (record.set) {
    record.latest = false;
    return;
  }
  vsnprintf(record.text, sizeof(record.text), fmt, args);
  record.set = true;
  record.latest = true;
  record.os_error = 0;
  record.status = SIF_OK;
}

void sif__error_record(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  record_v(fmt, args);
  va_end(args);
}

void sif__error_os_set(int os_error) {
  if (record.set && record.latest)
    record.os_error = os_error;
}

void sif__error_status_set(int status) {
  if (record.set && record.status == SIF_OK)
    record.status = status;
}

int sif_error_status(void) { return record.set ? record.status : SIF_OK; }

const char* sif_error_message(void) { return record.set ? record.text : ""; }

int sif_error_errno(void) { return record.set ? record.os_error : 0; }

void sif_error_clear(void) {
  record.set = false;
  record.latest = false;
  record.os_error = 0;
  record.status = SIF_OK;
  record.text[0] = '\0';
}

/* --- level --- */

/* INFO until sif_system_init() sets the configured one, and again after
 * sif_system_finalise(). */
uint8_t sif__log_level = SIF_LOG_LEVEL_INFO;

uint8_t sif_log_level_get(void) { return sif__log_level; }

void sif_log_level_set(uint8_t level) {
  sif__log_level = level > SIF_LOG_LEVEL_NONE ? SIF_LOG_LEVEL_NONE : level;
}

/* --- built-in sinks --- */

/* One fprintf per message: stdio locks the stream for the length of a call,
 * so a whole line goes out between two other threads' lines, never inside
 * one. */
static void write_default(void* ctx, const sif_log_record_t* rec) {
  (void)ctx;
  /* Diagnostics go to stderr so a run whose stdout is a data pipe stays
   * parseable, and so warnings survive being redirected away. */
  FILE* out = (rec->level >= SIF_LOG_LEVEL_WARNING) ? stderr : stdout;
  const int use_colour = use_colour_for(out);

  fprintf(out, "%s[%s]%s %s%s:%s %s\n",
    use_colour ? level_colours[rec->level] : "", level_strings[rec->level],
    use_colour ? ANSI_RESET : "", use_colour ? ANSI_BOLD_WHITE : "", rec->tag,
    use_colour ? ANSI_RESET : "", rec->message);
}

static void flush_default(void* ctx) {
  (void)ctx;
  fflush(stdout);
  fflush(stderr);
}

static void write_stream(void* ctx, const sif_log_record_t* rec) {
  fprintf((FILE*)ctx, "[%s] %s: %s\n", level_strings[rec->level], rec->tag,
    rec->message);
}

static void flush_stream(void* ctx) { fflush((FILE*)ctx); }

/* As write_stream(), with the wall-clock time first: a log file is read after
 * the fact, often beside a scheduler's own records, and the time is what
 * lines them up. To the millisecond, which separates the steps of a rung. */
static void write_file(void* ctx, const sif_log_record_t* rec) {
  struct timespec now;
  struct tm local;
  char stamp[32] = "";
  if (clock_gettime(CLOCK_REALTIME, &now) == 0 &&
      localtime_r(&now.tv_sec, &local))
    strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &local);
  else
    now.tv_nsec = 0;

  fprintf((FILE*)ctx, "%s.%03ld [%s] %s: %s\n", stamp, now.tv_nsec / 1000000L,
    level_strings[rec->level], rec->tag, rec->message);
}

static const sif_log_sink_t default_sink = {
  .write = write_default, .flush = flush_default, .ctx = NULL};

/* The sink in force, and the file sif_log_sink_file_set() opened for it, if
 * that is what it is. Process-wide; written only between library calls. */
static sif_log_sink_t sink = {
  .write = write_default, .flush = flush_default, .ctx = NULL};
static FILE* owned_file = NULL;

/* Install @p next, then close the file the old sink owned, if any: in that
 * order, so that no message is ever written to a closed stream. */
static void sink_install(const sif_log_sink_t* next, FILE* next_owned) {
  FILE* old_owned = owned_file;
  sink = *next;
  owned_file = next_owned;
  if (old_owned && old_owned != next_owned)
    fclose(old_owned);
}

int sif_log_sink_set(const sif_log_sink_t* next) {
  if (!next) {
    sink_install(&default_sink, NULL);
    return SIF_OK;
  }
  if (!next->write) {
    SIF_LOG_ERROR(TAG, "sink: no write function");
    return SIF_ERR_INVALID;
  }
  /* A saved file sink handed back keeps its file, rather than closing the
   * stream it is about to write to. */
  sink_install(
    next, (owned_file && next->ctx == owned_file) ? owned_file : NULL);
  return SIF_OK;
}

void sif_log_sink_get(sif_log_sink_t* out) { *out = sink; }

int sif_log_sink_stream_set(FILE* stream) {
  if (!stream) {
    SIF_LOG_ERROR(TAG, "stream sink: stream is NULL");
    return SIF_ERR_INVALID;
  }
  const sif_log_sink_t next = {
    .write = write_stream, .flush = flush_stream, .ctx = stream};
  sink_install(&next, NULL);
  return SIF_OK;
}

int sif_log_sink_file_set(const char* path) {
  if (!path) {
    SIF_LOG_ERROR(TAG, "file sink: path is NULL");
    return SIF_ERR_INVALID;
  }
  FILE* f = fopen(path, "a");
  if (!f)
    return sif__io_os_error(TAG, path, NULL, errno);
  /* Line-buffered: a crash loses at most the line in progress, and the file
   * can be followed with tail -f while the run goes on. */
  setvbuf(f, NULL, _IOLBF, 0);

  const sif_log_sink_t next = {
    .write = write_file, .flush = flush_stream, .ctx = f};
  sink_install(&next, f);
  return SIF_OK;
}

/* --- dispatch --- */

void sif__log_impl(uint8_t level, const char* tag, const char* fmt, ...) {
  /* The library's own rule, checked where it can be: a message from a
   * parallel region comes out in scheduling order, once per thread that hit
   * it. Count inside the region and log once after it. */
  SIF_ASSERT(!sif__system_in_parallel() && "logged from a parallel region");

  /* Clamped, not asserted: this function backs the SIF_LOG_* macros and is
   * therefore part of the ABI, so the level can be anything a caller passes,
   * while the tables above have one entry per real level. */
  if (level >= N_LEVELS)
    level = N_LEVELS - 1;
  if (demoted && level == SIF_LOG_LEVEL_ERROR)
    level = SIF_LOG_LEVEL_WARNING;

  const bool to_capture = capture.sink && level >= SIF_LOG_LEVEL_WARNING;
  const bool to_sink = level >= sif__log_level;

  /* The macros check the level already; this is for ERROR, which they do not
   * gate, and for callers of this function itself. */
  if (level != SIF_LOG_LEVEL_ERROR && !to_capture && !to_sink)
    return;

  /* Formatted once, for the record, the capture and the sink alike. A message
   * longer than the buffer goes to the heap rather than being cut short; only
   * the error record, which has a fixed size, truncates it. */
  char stack_buf[RECORD_CAP];
  char* message = stack_buf;
  va_list args;
  va_start(args, fmt);
  va_list again;
  va_copy(again, args);
  int length = vsnprintf(stack_buf, sizeof(stack_buf), fmt, args);
  va_end(args);
  if (length < 0) {
    stack_buf[0] = '\0';
    length = 0;
  } else if ((size_t)length >= sizeof(stack_buf)) {
    char* heap_buf = malloc((size_t)length + 1);
    if (heap_buf) {
      vsnprintf(heap_buf, (size_t)length + 1, fmt, again);
      message = heap_buf;
    } else {
      length = (int)sizeof(stack_buf) - 1;
    }
  }
  va_end(again);

  /* Recorded before the level is applied: an error nobody sees in the log is
   * still the reason the call failed. */
  if (level == SIF_LOG_LEVEL_ERROR)
    sif__error_record("%s", message);

  if (to_capture) {
    capture.sink(capture.ctx, level, tag, message);
  } else if (to_sink) {
    const sif_log_record_t rec = {.level = level,
      .tag = tag ? tag : "",
      .message = message,
      .length = (size_t)length,
      .thread = sif__system_thread_num()};
    sink.write(sink.ctx, &rec);
  }

  if (message != stack_buf)
    free(message);
}

void sif__log_flush(void) {
  if (sink.flush)
    sink.flush(sink.ctx);
}

#undef RECORD_CAP
