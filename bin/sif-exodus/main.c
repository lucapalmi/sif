/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* sif-exodus: the exodus void finder as a stand-alone program, run on a
 * configuration file.
 *
 *   sif-exodus [-D name=value ...] CONFIG     run
 *   sif-exodus --check [-D ...] CONFIG        resolve and check, do not run
 *   sif-exodus --template box|survey          a configuration to start from
 *
 * Data files and checkpoints are sif-inspect's to check.
 *
 * Exit status: 0 on success, 1 when the run fails, 2 when the command line
 * or the configuration is wrong -- which --check says without running.
 */

#include "config.h"
#include "pipeline.h"

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/utils/logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Enough for any command line anyone types. */
#define MAX_DEFINES 64

static void print_usage(FILE* out) {
  fputs(
    "usage: sif-exodus [options] CONFIG\n"
    "\n"
    "Run the exodus void finder as CONFIG, a Lua script, describes, or\n"
    "check CONFIG without running. sif-inspect checks data files and\n"
    "checkpoints.\n"
    "\n"
    "options:\n"
    "  -D NAME=VALUE  set NAME to the string VALUE in CONFIG; repeatable\n"
    "  --checkpoint PATH\n"
    "                 save the run's progress to PATH after every rung,\n"
    "                 and resume from it if it exists (default: the\n"
    "                 output path + .ckpt; run.checkpoint in CONFIG)\n"
    "  --no-checkpoint\n"
    "                 save no progress\n"
    "  -c, --check    resolve, check and print CONFIG, and stop\n"
    "  --deep         with --check, read CONFIG's inputs as the run would\n"
    "  -t, --template MODE\n"
    "                 print a configuration to start from, for MODE box\n"
    "                 or survey, and stop\n"
    "  -h, --help     show this message and exit\n"
    "  -V, --version  show the version and exit\n",
    out);
}

/* The value of an option that takes one, or NULL with a message. */
static const char* option_value(int argc, char** argv, int* i) {
  if (*i + 1 < argc)
    return argv[++*i];
  fprintf(stderr, "sif-exodus: %s takes a value\n", argv[*i]);
  return NULL;
}

int main(int argc, char** argv) {
  const char* config_path = NULL;
  config_define_t defines[MAX_DEFINES];
  size_t n_defines = 0;
  bool check_only = false;
  bool deep = false;
  config_overrides_t overrides = {NULL, false};

  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i];

    if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      print_usage(stdout);
      return 0;
    } else if (strcmp(arg, "-V") == 0 || strcmp(arg, "--version") == 0) {
      /* What the build can read and write: .xfield files are tied to the
       * precision, HDF5 outputs to HDF5, and FITS inputs to cfitsio. */
      printf("sif-exodus %s (%s precision, %s HDF5, %s FITS)\n",
        SIF_VERSION_STRING, sizeof(sif_real) == 8 ? "double" : "single",
#if defined(SIF_HAVE_HDF5)
        "with",
#else
        "without",
#endif
#if defined(SIF_HAVE_FITS)
        "with"
#else
        "without"
#endif
      );
      return 0;
    } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--template") == 0) {
      const char* mode = i + 1 < argc ? argv[i + 1] : NULL;
      if (mode && strcmp(mode, "box") == 0) {
        fputs(config_template_box, stdout);
      } else if (mode && strcmp(mode, "survey") == 0) {
        fputs(config_template_survey, stdout);
      } else {
        fprintf(stderr,
          "sif-exodus: --template takes the mode: --template box or "
          "--template survey\n");
        return 2;
      }
      return 0;
    } else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--check") == 0) {
      check_only = true;
    } else if (strcmp(arg, "--checkpoint") == 0) {
      if (!(overrides.checkpoint = option_value(argc, argv, &i)))
        return 2;
    } else if (strcmp(arg, "--no-checkpoint") == 0) {
      overrides.no_checkpoint = true;
    } else if (strcmp(arg, "--deep") == 0) {
      deep = true;
    } else if (strcmp(arg, "--format") == 0 ||
               strcmp(arg, "--columns") == 0 ||
               strcmp(arg, "--delimiter") == 0 ||
               strcmp(arg, "--skip-header") == 0) {
      /* sif-exodus --check took data files until 0.5. */
      fprintf(stderr,
        "sif-exodus: %s: data files are checked by sif-inspect now: "
        "sif-inspect %s ...\n",
        arg, arg);
      return 2;
    } else if (strncmp(arg, "-D", 2) == 0) {
      /* -D NAME=VALUE or -DNAME=VALUE. The name is cut out of argv in place:
       * nothing else reads it. */
      char* def = arg[2] ? argv[i] + 2 : (i + 1 < argc ? argv[++i] : NULL);
      char* eq = def ? strchr(def, '=') : NULL;
      if (!eq || eq == def) {
        fprintf(stderr, "sif-exodus: -D takes NAME=VALUE\n");
        return 2;
      }
      if (n_defines == MAX_DEFINES) {
        fprintf(stderr, "sif-exodus: more than %d -D\n", MAX_DEFINES);
        return 2;
      }
      *eq = '\0';
      for (size_t k = 0; k < n_defines; k++) {
        if (strcmp(defines[k].name, def) == 0) {
          fprintf(stderr, "sif-exodus: -D %s given twice\n", def);
          return 2;
        }
      }
      defines[n_defines].name = def;
      defines[n_defines].value = eq + 1;
      n_defines++;
    } else if (arg[0] == '-' && arg[1] != '\0') {
      fprintf(stderr, "sif-exodus: unknown option '%s'\n\n", arg);
      print_usage(stderr);
      return 2;
    } else if (config_path) {
      fprintf(stderr, "sif-exodus: one CONFIG only, got '%s' and '%s'\n",
        config_path, arg);
      return 2;
    } else {
      config_path = arg;
    }
  }

  if (!config_path) {
    print_usage(stderr);
    return 2;
  }

  if (overrides.checkpoint && overrides.no_checkpoint) {
    fprintf(stderr,
      "sif-exodus: --checkpoint and --no-checkpoint: one or the other\n");
    return 2;
  }

  if (!check_only && deep) {
    fprintf(stderr, "sif-exodus: --deep goes with --check\n");
    return 2;
  }

  exodus_config_t config;
  if (config_load(config_path, defines, n_defines, &overrides, &config) != 0) {
    /* sif-exodus --check took data files until 0.5. */
    const size_t n = strlen(config_path);
    if (check_only && (n < 4 || strcmp(config_path + n - 4, ".lua") != 0))
      fprintf(stderr,
        "sif-exodus: %s: not a configuration? sif-inspect checks data files "
        "and checkpoints\n",
        config_path);
    return 2;
  }

  if (check_only) {
    /* A survey's shape is only known once its files are read, which takes
     * the library, as does reading a box's input for --deep; its own log
     * stays quiet but for what goes wrong. */
    const bool needs_lib = config.params.mode == EXODUS_MODE_SURVEY || deep;
    sif_omp_config_t check_omp = {.n_threads = config.run.n_threads};
    sif_config_t check_lib = {.omp_config = &check_omp,
      .log_level = config.run.log_level > SIF_LOG_LEVEL_WARNING
                     ? config.run.log_level
                     : SIF_LOG_LEVEL_WARNING,
      .settings = config.settings,
      .n_settings = config.n_settings};
    if (needs_lib && sif_system_init(&check_lib) != SIF_OK) {
      fputs("sif-exodus: could not initialise the library\n", stderr);
      config_free(&config);
      return 1;
    }
    const int ok = config_check(&config, config_path, true, deep);
    if (needs_lib)
      sif_system_finalise();
    fputs(config.resolved, stdout);
    fflush(stdout);
    config_free(&config);
    if (ok != 0)
      return 2;
    fprintf(stderr, "%s: OK\n", config_path);
    return 0;
  }

  sif_fft_config_t fft = {.skip_tuning = !config.run.tune_fft};
  sif_omp_config_t omp = {.n_threads = config.run.n_threads};
  sif_config_t lib = {.fft_config = &fft,
    .omp_config = &omp,
    .log_level = config.run.log_level,
    .settings = config.settings,
    .n_settings = config.n_settings};
  if (sif_system_init(&lib) != SIF_OK) {
    fputs("sif-exodus: could not initialise the library\n", stderr);
    config_free(&config);
    return 1;
  }

  /* The same check as --check, before anything is read: what it finds wrong
   * would stop the run later anyway, after the expensive part. */
  if (config_check(&config, config_path, false, false) != 0) {
    sif_system_finalise();
    config_free(&config);
    return 2;
  }

  const int status = config.params.mode == EXODUS_MODE_SURVEY
                       ? pipeline_survey(&config.params)
                       : pipeline_box(&config.params);

  sif_system_finalise();
  config_free(&config);
  return status == SIF_OK ? 0 : 1;
}
