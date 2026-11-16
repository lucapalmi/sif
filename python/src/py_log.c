/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The bridge from sif's log to Python's logging module: a sif sink that hands
 * every message to pysif._log.emit(), which logs it under pysif.<tag>.
 *
 * Calling into Python takes the GIL, and which thread logged decides whether
 * that is safe:
 *
 * - A Python thread -- the one that called into sif, which is also OpenMP
 *   thread 0 of any parallel region inside the call -- takes the GIL itself.
 *   That is safe whether or not the binding released it: PyGILState_Ensure()
 *   re-enters a GIL the thread already holds.
 * - Any other thread is an OpenMP worker, and never touches Python. If the
 *   binding kept the GIL, a worker waiting for it would never get it: the
 *   thread holding it is waiting for the worker at the end of the region.
 *   So a worker's message goes into a bounded queue instead, and a Python
 *   thread emits it later: before its own next message, and when a binding
 *   takes the GIL back after a call (PY_SIF_END_ALLOW_THREADS).
 *
 * A Python thread is one with a thread state; OpenMP's workers have none,
 * since this file is the only place that could give them one and it does
 * not.
 *
 * An ERROR from a Python thread is not emitted at all. It is the reason the
 * call fails, and the binding raises it as the exception's message (see
 * py_sif_raise()); a log line saying the same would only repeat it. One from
 * any other thread is emitted: it is recorded on that thread, where no
 * exception will ever read it.
 */

#include "py_common.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* pysif._log.emit and pysif._log.c_level, set by py_sif_log_install(). */
static PyObject* emit_fn = NULL;
static PyObject* level_fn = NULL;

/* The sink in force before the bridge, for messages logged while the
 * interpreter is shutting down, when Python can no longer be called. */
static sif_log_sink_t fallback;

/* --- the workers' queue --- */

/* Enough for a burst; past it, messages are counted and dropped rather than
 * held without bound. Logging from a parallel region is against the
 * library's own rules, so in practice it stays empty. */
#define QUEUE_CAP 1024
#define TAG_CAP   48

typedef struct {
  uint8_t level;
  char tag[TAG_CAP];
  char* message; /* malloc'd; NULL if the copy failed */
  size_t length;
} queued_t;

static pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;
static queued_t queue[QUEUE_CAP];
static int n_queued = 0;
static uint64_t n_dropped = 0;

/* Emptied into here under the lock, then emitted without it: a worker never
 * waits on Python. Only a thread holding the GIL drains, and the GIL lets
 * one do so at a time, so one buffer serves. */
static queued_t draining[QUEUE_CAP];

/* Set while this thread is inside emit(), so that a handler that logs
 * through sif again emits directly instead of draining from within a
 * drain. */
static SIF_THREAD_LOCAL int emitting = 0;

static void enqueue(const sif_log_record_t* rec) {
  char* copy = malloc(rec->length + 1);
  if (copy)
    memcpy(copy, rec->message, rec->length + 1);

  pthread_mutex_lock(&queue_lock);
  if (n_queued < QUEUE_CAP) {
    queued_t* q = &queue[n_queued++];
    q->level = rec->level;
    snprintf(q->tag, sizeof(q->tag), "%s", rec->tag);
    q->message = copy;
    q->length = copy ? rec->length : 0;
    copy = NULL;
  } else {
    n_dropped++;
  }
  pthread_mutex_unlock(&queue_lock);

  free(copy);
}

/* --- emitting, with the GIL held --- */

/* Never lets an exception out, and leaves one that was already set -- a
 * binding may log on its way to raising -- as it found it. */
static void emit(
  uint8_t level, const char* tag, const char* message, size_t length) {
  if (!emit_fn)
    return;

  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);

  /* Paths need not be UTF-8; an undecodable byte shows as an escape rather
   * than losing the message. */
  PyObject* text =
    PyUnicode_DecodeUTF8(message, (Py_ssize_t)length, "backslashreplace");
  PyObject* result =
    text ? PyObject_CallFunction(emit_fn, "isO", (int)level, tag, text) : NULL;
  if (!result)
    PyErr_WriteUnraisable(emit_fn);
  Py_XDECREF(result);
  Py_XDECREF(text);

  PyErr_Restore(type, value, traceback);
}

static void drain(void) {
  pthread_mutex_lock(&queue_lock);
  const int n = n_queued;
  const uint64_t dropped = n_dropped;
  memcpy(draining, queue, (size_t)n * sizeof(queued_t));
  n_queued = 0;
  n_dropped = 0;
  pthread_mutex_unlock(&queue_lock);

  for (int i = 0; i < n; i++) {
    queued_t* q = &draining[i];
    if (q->message)
      emit(q->level, q->tag, q->message, q->length);
    else
      emit(q->level, q->tag, "(message lost: out of memory)", 29);
    free(q->message);
  }

  if (dropped > 0) {
    char text[160];
    const int len = snprintf(text, sizeof(text),
      "%" PRIu64 " messages from worker threads dropped (the queue holds %d)",
      dropped, QUEUE_CAP);
    emit(SIF_LOG_LEVEL_WARNING, "utils.logger", text, (size_t)len);
  }
}

static int python_is_finalizing(void) {
#if PY_VERSION_HEX >= 0x030D0000
  return Py_IsFinalizing();
#else
  return _Py_IsFinalizing();
#endif
}

/* --- the sink --- */

static void sink_write(void* ctx, const sif_log_record_t* rec) {
  (void)ctx;
  if (python_is_finalizing()) {
    fallback.write(fallback.ctx, rec);
    return;
  }
  if (!PyGILState_GetThisThreadState()) {
    enqueue(rec);
    return;
  }
  if (rec->level >= SIF_LOG_LEVEL_ERROR)
    return;

  const PyGILState_STATE gil = PyGILState_Ensure();
  if (emitting) {
    emit(rec->level, rec->tag, rec->message, rec->length);
  } else {
    emitting = 1;
    drain();
    emit(rec->level, rec->tag, rec->message, rec->length);
    emitting = 0;
  }
  PyGILState_Release(gil);
}

static void sink_flush(void* ctx) {
  (void)ctx;
  if (python_is_finalizing() || !PyGILState_GetThisThreadState() || emitting)
    return;
  const PyGILState_STATE gil = PyGILState_Ensure();
  emitting = 1;
  drain();
  emitting = 0;
  PyGILState_Release(gil);
}

/* --- what the bindings call, with the GIL held --- */

void py_sif_log_sync(void) {
  if (!level_fn)
    return;
  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);

  PyObject* result = PyObject_CallNoArgs(level_fn);
  const long level = result ? PyLong_AsLong(result) : -1;
  if (level < 0 || PyErr_Occurred())
    PyErr_WriteUnraisable(level_fn);
  else
    sif_log_level_set((uint8_t)level);
  Py_XDECREF(result);

  PyErr_Restore(type, value, traceback);
}

void py_sif_log_drain(void) {
  if (!emit_fn || emitting)
    return;
  emitting = 1;
  drain();
  emitting = 0;
}

PyObject* py_sif_log_sync_py(PyObject* module, PyObject* args) {
  (void)module;
  (void)args;
  py_sif_log_sync();
  Py_RETURN_NONE;
}

PyObject* py_sif_log_level_py(PyObject* module, PyObject* args) {
  (void)module;
  (void)args;
  return PyLong_FromLong(sif_log_level_get());
}

PyObject* py_sif_log_install(PyObject* module, PyObject* args) {
  (void)module;
  PyObject* emit_arg;
  PyObject* level_arg;
  if (!PyArg_ParseTuple(args, "OO", &emit_arg, &level_arg))
    return NULL;
  if (!PyCallable_Check(emit_arg) || !PyCallable_Check(level_arg)) {
    PyErr_SetString(PyExc_TypeError, "_log_install: expected two callables");
    return NULL;
  }

  /* Installed once; a second call, from a reload, swaps the functions. */
  if (!emit_fn)
    sif_log_sink_get(&fallback);
  Py_INCREF(emit_arg);
  Py_INCREF(level_arg);
  Py_XSETREF(emit_fn, emit_arg);
  Py_XSETREF(level_fn, level_arg);

  const sif_log_sink_t sink = {
    .write = sink_write, .flush = sink_flush, .ctx = NULL};
  if (sif_log_sink_set(&sink) != SIF_OK)
    return py_sif_raise(SIF_ERR_INVALID, "_log_install");

  py_sif_log_sync();
  Py_RETURN_NONE;
}

/* --- a test hook --- */

/* Nothing in the library logs from a parallel region, by its own rules, so
 * the queue cannot be reached through it. This logs from threads the
 * interpreter does not know -- what OpenMP's workers are -- while the caller
 * either holds the GIL, as a binding that does not release it would, or has
 * released it. Holding it is the case that would deadlock if a worker took
 * the GIL itself. */

typedef struct {
  const char* message;
  int count;
  int error;
} worker_job_t;

static void* worker_log(void* arg) {
  const worker_job_t* job = arg;
  for (int i = 0; i < job->count; i++) {
    if (job->error)
      SIF_LOG_ERROR("utils.logger", "%s", job->message);
    else
      SIF_LOG_WARNING("utils.logger", "%s", job->message);
  }
  return NULL;
}

PyObject* py_sif_log_from_threads(PyObject* module, PyObject* args) {
  (void)module;
  const char* message;
  int n_threads, per_thread, release_gil, error = 0;
  if (!PyArg_ParseTuple(args, "siip|p", &message, &n_threads, &per_thread,
        &release_gil, &error))
    return NULL;
  if (n_threads < 1 || n_threads > 64 || per_thread < 0) {
    PyErr_SetString(PyExc_ValueError, "_log_from_threads: bad counts");
    return NULL;
  }

  pthread_t threads[64];
  worker_job_t job = {.message = message, .count = per_thread, .error = error};
  int started = 0;

  PyThreadState* saved = NULL;
  if (release_gil) {
    py_sif_log_sync();
    saved = PyEval_SaveThread();
  }
  for (; started < n_threads; started++)
    if (pthread_create(&threads[started], NULL, worker_log, &job) != 0)
      break;
  for (int i = 0; i < started; i++)
    pthread_join(threads[i], NULL);
  if (release_gil) {
    PyEval_RestoreThread(saved);
    py_sif_log_drain();
  }

  if (started < n_threads) {
    PyErr_SetString(PyExc_RuntimeError, "_log_from_threads: pthread_create");
    return NULL;
  }
  Py_RETURN_NONE;
}

#undef QUEUE_CAP
#undef TAG_CAP
