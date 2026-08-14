/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/utils/logger.h"

#include "core/system_internal.h"

#include <stdarg.h>
#include <unistd.h>

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

static const char* level_colors[] = {
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

static int use_color_for(FILE* out) {
  int* cached = (out == stderr) ? &stderr_is_tty : &stdout_is_tty;
  if (*cached < 0)
    *cached = isatty(fileno(out)) ? 1 : 0;

  return *cached;
}

void sif__log_impl(uint8_t level, const char* tag, const char* fmt, ...) {
  /* Clamped, not asserted: this function backs the SIF_LOG_* macros and is
   * therefore part of the ABI, so the level can be anything a caller passes,
   * while the tables above have one entry per real level. */
  if (level >= N_LEVELS)
    level = N_LEVELS - 1;

  /* The safe accessor, because logging before sif_init() has to work -- it is
   * how the failures during init itself get reported. INFO is the assumed
   * floor until the configured one exists. */
  const sif_system_state_t* state = sif__system_state_safe();
  const uint8_t runtime_level =
    state ? state->logger.level : SIF_LOG_LEVEL_INFO;

  if (level < runtime_level)
    return;

  /* Diagnostics go to stderr so a run whose stdout is a data pipe stays
   * parseable, and so warnings survive being redirected away. */
  FILE* out = (level >= SIF_LOG_LEVEL_WARNING) ? stderr : stdout;
  const int use_color = use_color_for(out);

  fprintf(out, "%s[%s]%s %s%s:%s ", use_color ? level_colors[level] : "",
    level_strings[level], use_color ? ANSI_RESET : "",
    use_color ? ANSI_BOLD_WHITE : "", tag, use_color ? ANSI_RESET : "");

  va_list args;
  va_start(args, fmt);
  vfprintf(out, fmt, args);
  va_end(args);

  fputc('\n', out);
}

void sif__log_flush(void) {
  fflush(stdout);
  fflush(stderr);
}
