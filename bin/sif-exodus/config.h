/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* The configuration file: a Lua script, run in a sandbox, whose sections --
 * input, grid, mesh, finder, output, run -- become an exodus_params_t.
 *
 * The script can compute anything Lua can (math, string, table, utf8, and
 * os.getenv/date/time/clock), but it cannot read or write files, run
 * commands or load code: a configuration describes a run, it does not do
 * one. On top of Lua it gets ladder(), linspace() and geomspace() for the
 * radii, and a global per -D name=value on the command line.
 *
 * Reading is strict. An unknown key, a value of the wrong type or out of
 * range, a missing required setting: each is reported with its full path --
 * "exodus.lua: finder.treshold: unknown key (did you mean threshold?)" --
 * and all of them are reported before giving up, so one pass fixes the file.
 *
 * Loading needs nothing initialized and writes nothing through the library's
 * logger: the configuration is what says how the logger is set up, so its
 * own messages go to stderr directly.
 */

#ifndef SIF_EXODUS_CONFIG_H
#define SIF_EXODUS_CONFIG_H

#include "pipeline.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* A -D name=value from the command line: a global string in the script. */
typedef struct {
  const char* name;
  const char* value;
} config_define_t;

typedef struct {
  /* What the pipeline runs on. Its pointers point into the storage below. */
  exodus_params_t params;

  /* What main() hands sif_init(). */
  struct {
    /* 0 leaves the OpenMP default. */
    uint32_t n_threads;
    /* One of the SIF_LOG_LEVEL_* constants. */
    uint8_t log_level;
    bool tune_fft;
  } run;

  /* The configuration as it resolved: every setting, defaults included, as
   * a Lua script that runs as a configuration of its own. */
  char* resolved;

  /* Owned storage behind the pointers in `params`. Not for callers. */
  sif_real* _radii;
  char** _strings;
  size_t _n_strings;
  const char** _paths;
} exodus_config_t;

/* Run the script at `path` and read it into `config`. Every problem found is
 * printed to stderr. Returns 0, or -1 if the script failed or the
 * configuration is invalid; `config` is then left empty, and safe to pass to
 * config_free(). */
int config_load(const char* path, const config_define_t* defines,
  size_t n_defines, exodus_config_t* config);

/* Check what a run would trip over, before it reads anything: that the
 * input is there and says what the configuration expects of it (for an
 * .xfield or a GADGET snapshot its header is read, for a raw binary file its
 * size), that the largest radius fits the box, that the output's directory
 * takes files. From what the header tells it also works out the run's shape --
 * tracers, grid, mesh, radii in the box's units, peak memory -- and warns
 * about what will cost a run without stopping it: a smallest radius under two
 * grid cells, more memory than the machine has, an output about to be
 * replaced.
 *
 * With `summary` (for --check) everything goes to stderr under the name
 * `file`, the run's shape included; without it (before a run, with the
 * library up) errors and warnings go to the log. Returns 0, or -1 if the run
 * could not succeed. */
int config_check(const exodus_config_t* config, const char* file, bool summary);

/* Release everything config_load() allocated. */
void config_free(exodus_config_t* config);

/* A commented configuration with every setting at its default, for
 * `sif-exodus --template`: template.lua, compiled in by CMakeLists.txt, so
 * that the documentation can show the same file. */
extern const char config_template[];

#endif /* SIF_EXODUS_CONFIG_H */
