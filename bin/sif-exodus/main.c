/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* sif-exodus: the exodus void finder as a stand-alone program, run on a
 * parameter file.
 *
 * So far only the shell: the command line, and a library brought up and shut
 * down around a pipeline that does not exist yet.
 *
 * Exit status: 0 on success, 1 when the run fails, 2 when the command line is
 * wrong.
 */

#include "sif/core/macros.h"
#include "sif/core/system.h"

#include <stdio.h>
#include <string.h>

static void print_usage(FILE* out) {
  fputs("usage: sif-exodus [options] CONFIG\n"
        "\n"
        "Run the exodus void finder with the parameters in CONFIG.\n"
        "\n"
        "options:\n"
        "  -h, --help     show this message and exit\n"
        "  -V, --version  show the version and exit\n",
    out);
}

int main(int argc, char** argv) {
  const char* config_path = NULL;

  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i];

    if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      print_usage(stdout);
      return 0;
    } else if (strcmp(arg, "-V") == 0 || strcmp(arg, "--version") == 0) {
      printf("sif-exodus %s\n", SIF_VERSION_STRING);
      return 0;
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

  if (sif_init(SIF_CONFIG_STANDARD) != SIF_OK) {
    fputs("sif-exodus: could not initialize the library\n", stderr);
    return 1;
  }

  fprintf(
    stderr, "sif-exodus: running '%s' is not implemented yet\n", config_path);

  sif_finalize();
  return 1;
}
