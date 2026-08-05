#include "sif/utils/logger.h"

#include "core/get_system.h"
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

static const char* level_strings[] = {
  "TRACE", "DEBUG", "INFO", "WARNING", "ERROR"};

static const char* level_colors[] = {
  ANSI_GREY,    /* TRACE   */
  ANSI_MAGENTA, /* DEBUG   */
  ANSI_BLUE,    /* INFO    */
  ANSI_YELLOW,  /* WARNING */
  ANSI_RED      /* ERROR   */
};

void __sif_log_impl(uint8_t level, const char* tag, const char* fmt, ...) {
  system_state_t* state = get_system_state_safe();

  /* 1. Determine Runtime Level */
  /* Fallback to INFO if the user logs something before calling sif_init() */
  uint8_t runtime_level = SIF_LOG_LEVEL_INFO;
  if (state) {
    runtime_level = state->logger.level;
  }

  /* 2. The Gatekeeper: Check against the dynamic runtime level */
  if (level < runtime_level) {
    return;
  }

  /* 3. Output Logic */
  FILE* out = (level >= SIF_LOG_LEVEL_WARNING) ? stderr : stdout;
  int use_color = isatty(fileno(out));

  fprintf(out, "%s[%s]%s %s%s:%s ", use_color ? level_colors[level] : "",
    level_strings[level], use_color ? ANSI_RESET : "",
    use_color ? ANSI_BOLD_WHITE : "", tag, use_color ? ANSI_RESET : "");

  va_list args;
  va_start(args, fmt);
  vfprintf(out, fmt, args);
  va_end(args);

  fputc('\n', out);
}

void __sif_log_flush(void) {
  fflush(stdout);
  fflush(stderr);
}
