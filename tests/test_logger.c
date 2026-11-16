/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Logging: the runtime level, checked in the macros, and the sinks a message
 * goes to once it passes. A custom sink has to see each message whole and
 * once, with its level, tag and thread; the stream and file sinks have to
 * write one complete line per message, from any number of threads; the error
 * record and the per-thread capture have to work whatever sink and level are
 * in force.
 */

#include "sif/utils/logger.h"

#include "sif/core/macros.h"
#include "sif/core/system.h"

#include "utils/logger_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: ");                                                      \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

/* --- a sink that keeps what it is given --- */

#define KEPT_CAP 16

typedef struct {
  int n;
  int n_flushes;
  uint8_t level[KEPT_CAP];
  char tag[KEPT_CAP][32];
  char* message[KEPT_CAP];
  size_t length[KEPT_CAP];
} kept_t;

static void keep_write(void* ctx, const sif_log_record_t* rec) {
  kept_t* k = ctx;
  if (k->n >= KEPT_CAP)
    return;
  k->level[k->n] = rec->level;
  snprintf(k->tag[k->n], sizeof(k->tag[k->n]), "%s", rec->tag);
  k->message[k->n] = strdup(rec->message);
  k->length[k->n] = rec->length;
  k->n++;
}

static void keep_flush(void* ctx) { ((kept_t*)ctx)->n_flushes++; }

static void kept_clear(kept_t* k) {
  for (int i = 0; i < k->n; i++)
    free(k->message[i]);
  memset(k, 0, sizeof(*k));
}

static int side_effects = 0;
static int bump(void) { return ++side_effects; }

/* --- tests --- */

static void test_custom_sink(void) {
  printf("custom sink\n");
  kept_t k = {0};
  const sif_log_sink_t sink = {
    .write = keep_write, .flush = keep_flush, .ctx = &k};
  CHECK(sif_log_sink_set(&sink) == SIF_OK, "sink not set");
  sif_log_level_set(SIF_LOG_LEVEL_TRACE);

  SIF_LOG_INFO("finder", "found %d voids", 42);
  SIF_LOG_WARNING("io", "plain");
  CHECK(k.n == 2, "%d messages, expected 2", k.n);
  CHECK(k.level[0] == SIF_LOG_LEVEL_INFO, "level %u", k.level[0]);
  CHECK(strcmp(k.tag[0], "finder") == 0, "tag '%s'", k.tag[0]);
  CHECK(k.message[0] && strcmp(k.message[0], "found 42 voids") == 0,
    "message '%s'", k.message[0]);
  CHECK(k.length[0] == strlen("found 42 voids"), "length %zu", k.length[0]);
  CHECK(k.level[1] == SIF_LOG_LEVEL_WARNING, "level %u", k.level[1]);

  SIF_LOG_FLUSH();
  CHECK(k.n_flushes == 1, "%d flushes, expected 1", k.n_flushes);

  /* Longer than the formatting buffer: delivered whole, not cut short. */
  char long_text[3000];
  memset(long_text, 'x', sizeof(long_text) - 1);
  long_text[sizeof(long_text) - 1] = '\0';
  SIF_LOG_INFO("long", "<%s>", long_text);
  CHECK(k.n == 3 && k.length[2] == sizeof(long_text) + 1,
    "long message: length %zu, expected %zu", k.n == 3 ? k.length[2] : 0,
    sizeof(long_text) + 1);

  sif_log_sink_t got;
  sif_log_sink_get(&got);
  CHECK(got.write == keep_write && got.ctx == &k, "sink_get: not the sink");

  sif_error_clear();
  const sif_log_sink_t no_write = {.write = NULL, .flush = NULL, .ctx = NULL};
  CHECK(sif_log_sink_set(&no_write) == SIF_ERR_INVALID, "no write accepted");
  CHECK(error_has("no write function"), "error: '%s'", sif_error_message());
  sif_log_sink_get(&got);
  CHECK(got.write == keep_write, "refused sink replaced the one in force");

  CHECK(sif_log_sink_set(NULL) == SIF_OK, "default not restored");
  sif_log_sink_get(&got);
  CHECK(got.write != keep_write, "default not restored");
  kept_clear(&k);
}

static void test_level(void) {
  printf("runtime level\n");
  kept_t k = {0};
  const sif_log_sink_t sink = {.write = keep_write, .flush = NULL, .ctx = &k};
  CHECK(sif_log_sink_set(&sink) == SIF_OK, "sink not set");

  sif_log_level_set(SIF_LOG_LEVEL_WARNING);
  CHECK(sif_log_level_get() == SIF_LOG_LEVEL_WARNING, "level not set");

  /* Filtered in the macro: the arguments are not evaluated. */
  side_effects = 0;
  SIF_LOG_TRACE("t", "%d", bump());
  SIF_LOG_DEBUG("t", "%d", bump());
  SIF_LOG_INFO("t", "%d", bump());
  CHECK(side_effects == 0, "%d filtered arguments evaluated", side_effects);
  CHECK(k.n == 0, "%d filtered messages delivered", k.n);

  SIF_LOG_WARNING("t", "%d", bump());
  CHECK(side_effects == 1 && k.n == 1, "warning not delivered");

  /* An error below the floor is not logged, and still recorded. */
  sif_log_level_set(SIF_LOG_LEVEL_NONE);
  sif_error_clear();
  SIF_LOG_ERROR("t", "the cause: %d", 7);
  CHECK(k.n == 1, "error delivered at level NONE");
  CHECK(error_has("the cause: 7"), "error record: '%s'", sif_error_message());

  sif_log_level_set(200);
  CHECK(sif_log_level_get() == SIF_LOG_LEVEL_NONE, "level not clamped");

  /* Errors are delivered as well as recorded. */
  sif_log_level_set(SIF_LOG_LEVEL_TRACE);
  SIF_LOG_ERROR("t", "visible");
  CHECK(k.n == 2 && k.level[1] == SIF_LOG_LEVEL_ERROR, "error not delivered");

  CHECK(sif_log_sink_set(NULL) == SIF_OK, "default not restored");
  kept_clear(&k);
}

/* --- capture --- */

typedef struct {
  int n;
  char last[256];
} captured_t;

static void capture_sink(
  void* ctx, int level, const char* tag, const char* message) {
  (void)level;
  (void)tag;
  captured_t* c = ctx;
  c->n++;
  snprintf(c->last, sizeof(c->last), "%s", message);
}

static void test_capture(void) {
  printf("capture over the sink\n");
  kept_t k = {0};
  const sif_log_sink_t sink = {.write = keep_write, .flush = NULL, .ctx = &k};
  CHECK(sif_log_sink_set(&sink) == SIF_OK, "sink not set");

  /* A capture takes warnings even when the level hides them, as a file check
   * does for readers run with a quiet log. */
  sif_log_level_set(SIF_LOG_LEVEL_ERROR);
  captured_t c = {0};
  sif__log_capture_t saved;
  sif__log_capture_begin(capture_sink, &c, &saved);
  SIF_LOG_WARNING("io", "row %d skipped", 3);
  SIF_LOG_INFO("io", "not captured");
  sif__log_capture_end(&saved);

  CHECK(c.n == 1 && strcmp(c.last, "row 3 skipped") == 0,
    "capture: %d messages, last '%s'", c.n, c.last);
  CHECK(k.n == 0, "captured message also reached the sink");

  SIF_LOG_WARNING("io", "after");
  CHECK(c.n == 1 && k.n == 0, "warning below the level after the capture");

  sif_log_level_set(SIF_LOG_LEVEL_TRACE);
  CHECK(sif_log_sink_set(NULL) == SIF_OK, "default not restored");
  kept_clear(&k);
}

/* --- stream and file sinks --- */

/* Every line of @p text is "[LEVEL] tag: message"; returns the count, or -1
 * at the first that is not. */
static int count_lines(const char* text, const char* prefix) {
  int n = 0;
  const char* p = text;
  while (*p) {
    const char* nl = strchr(p, '\n');
    if (!nl)
      return -1;
    if (strncmp(p, prefix, strlen(prefix)) != 0)
      return -1;
    n++;
    p = nl + 1;
  }
  return n;
}

static void test_stream_sink(void) {
  printf("stream sink\n");
  char* buf = NULL;
  size_t size = 0;
  FILE* mem = open_memstream(&buf, &size);
  CHECK(mem != NULL, "open_memstream");
  if (!mem)
    return;

  CHECK(sif_log_sink_stream_set(mem) == SIF_OK, "stream sink not set");
  sif_log_level_set(SIF_LOG_LEVEL_TRACE);
  SIF_LOG_INFO("finder", "found %d voids", 5);
  SIF_LOG_WARNING("io", "short read");
  SIF_LOG_FLUSH();
  CHECK(
    buf && strcmp(buf,
             "[INFO] finder: found 5 voids\n[WARNING] io: short read\n") == 0,
    "stream output:\n%s", buf ? buf : "(null)");

  CHECK(sif_log_sink_stream_set(NULL) == SIF_ERR_INVALID, "NULL accepted");
  CHECK(sif_log_sink_set(NULL) == SIF_OK, "default not restored");
  fclose(mem);
  free(buf);
}

static void test_threads(void) {
  printf("stream sink, from threads\n");
  char* buf = NULL;
  size_t size = 0;
  FILE* mem = open_memstream(&buf, &size);
  if (!mem) {
    CHECK(0, "open_memstream");
    return;
  }
  CHECK(sif_log_sink_stream_set(mem) == SIF_OK, "stream sink not set");

  const int n = 4000;
#pragma omp parallel for num_threads(8)
  for (int i = 0; i < n; i++)
    SIF_LOG_INFO("worker",
      "message %d of a line long enough to interleave if "
      "it were written in pieces: %d %d %d",
      i, i, i, i);
  SIF_LOG_FLUSH();

  CHECK(sif_log_sink_set(NULL) == SIF_OK, "default not restored");
  fclose(mem);
  const int lines = buf ? count_lines(buf, "[INFO] worker: message ") : -1;
  CHECK(lines == n, "%d whole lines, expected %d", lines, n);
  free(buf);
}

static void test_file_sink(void) {
  printf("file sink\n");
  char path[] = "test_logger_XXXXXX";
  const int fd = mkstemp(path);
  CHECK(fd >= 0, "mkstemp");
  if (fd < 0)
    return;
  close(fd);

  sif_log_level_set(SIF_LOG_LEVEL_TRACE);
  CHECK(sif_log_sink_file_set(path) == SIF_OK, "file sink not set");
  SIF_LOG_INFO("a", "first");
  /* Replacing it closes the file; a second one on the same path appends. */
  CHECK(sif_log_sink_file_set(path) == SIF_OK, "file sink not set again");
  SIF_LOG_ERROR("b", "second");
  CHECK(sif_log_sink_set(NULL) == SIF_OK, "default not restored");

  char text[256] = {0};
  FILE* f = fopen(path, "r");
  CHECK(f != NULL, "%s: not readable", path);
  if (f) {
    const size_t got = fread(text, 1, sizeof(text) - 1, f);
    text[got] = '\0';
    fclose(f);
  }
  CHECK(strcmp(text, "[INFO] a: first\n[ERROR] b: second\n") == 0,
    "file output:\n%s", text);
  unlink(path);

  /* Unopenable: refused with the reason, and the sink in force kept. */
  kept_t k = {0};
  const sif_log_sink_t sink = {.write = keep_write, .flush = NULL, .ctx = &k};
  CHECK(sif_log_sink_set(&sink) == SIF_OK, "sink not set");
  sif_error_clear();
  CHECK(sif_log_sink_file_set("/nonexistent-dir/x.log") == SIF_ERR_IO,
    "unopenable path accepted");
  CHECK(
    error_has("/nonexistent-dir/x.log"), "error: '%s'", sif_error_message());
  sif_log_sink_t got;
  sif_log_sink_get(&got);
  CHECK(got.write == keep_write, "failed file sink replaced the one in force");
  CHECK(sif_log_sink_set(NULL) == SIF_OK, "default not restored");
  kept_clear(&k);
}

static void test_init_level(void) {
  printf("level across init and finalise\n");
  sif_system_finalise();
  CHECK(sif_log_level_get() == SIF_LOG_LEVEL_INFO, "finalise: level %u",
    sif_log_level_get());
  CHECK(sif_system_init(SIF_CONFIG_QUIET) == SIF_OK, "re-init");
  CHECK(sif_log_level_get() == SIF_LOG_LEVEL_WARNING, "init: level %u",
    sif_log_level_get());
}

int main(void) {
  if (sif_system_init(SIF_CONFIG_QUIET) != SIF_OK) {
    printf("FAIL: sif_system_init\n");
    return 1;
  }

  test_custom_sink();
  test_level();
  test_capture();
  test_stream_sink();
  test_threads();
  test_file_sink();
  test_init_level();

  sif_system_finalise();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
