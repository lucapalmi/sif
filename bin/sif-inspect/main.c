/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* sif-inspect: a look at a file sif reads or writes, without loading it.
 *
 *   sif-inspect [options] FILE    say what FILE is, check it and summarise it
 *
 * Exit status: 0 for a sound file, 1 when the library cannot start, 2 when
 * the command line is wrong or the file has problems.
 */

#include "inspect.h"

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/utils/logger.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(FILE* out) {
  fputs(
    "usage: sif-inspect [options] FILE\n"
    "\n"
    "Check FILE without loading it, and summarise it: a data file (ASCII,\n"
    "FITS, HDF5, GADGET, .xfield) or an exodus checkpoint, its kind told\n"
    "from its contents.\n"
    "\n"
    "options:\n"
    "  --deep         read every value as well, for each column's range\n"
    "  --format KIND  the kind of FILE, instead of telling it from its\n"
    "                 contents: ascii, fits, hdf5, gadget, xfield or\n"
    "                 checkpoint\n"
    "  --columns FMT  on an ASCII file, the column format to check it\n"
    "                 against, as in a configuration: \"x y z w\"\n"
    "  --delimiter C  on an ASCII file, the column delimiter (default a\n"
    "                 blank; \"tab\" for a tab)\n"
    "  --skip-header N\n"
    "                 on an ASCII file, lines to skip first\n"
    "  -h, --help     show this message and exit\n"
    "  -V, --version  show the version and exit\n",
    out);
}

/* The value of an option that takes one, or NULL with a message. */
static const char* option_value(int argc, char** argv, int* i) {
  if (*i + 1 < argc)
    return argv[++*i];
  fprintf(stderr, "sif-inspect: %s takes a value\n", argv[*i]);
  return NULL;
}

int main(int argc, char** argv) {
  const char* path = NULL;
  inspect_opts_t io = {FILE_KIND_AUTO, NULL, '\0', 0, false};
  bool ascii_options = false;

  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i];

    if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      print_usage(stdout);
      return 0;
    } else if (strcmp(arg, "-V") == 0 || strcmp(arg, "--version") == 0) {
      /* What the build can read: .xfield files are tied to the precision,
       * HDF5 files to HDF5, and FITS files to cfitsio. */
      printf("sif-inspect %s (%s precision, %s HDF5, %s FITS)\n",
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
    } else if (strcmp(arg, "--deep") == 0) {
      io.deep = true;
    } else if (strcmp(arg, "--format") == 0) {
      const char* v = option_value(argc, argv, &i);
      if (!v)
        return 2;
      if (inspect_kind_parse(v, &io.format) != 0 ||
          io.format == FILE_KIND_LUA) {
        fprintf(stderr,
          "sif-inspect: --format %s: expected ascii, fits, hdf5, gadget, "
          "xfield or checkpoint\n",
          v);
        return 2;
      }
    } else if (strcmp(arg, "--columns") == 0) {
      if (!(io.columns = option_value(argc, argv, &i)))
        return 2;
      ascii_options = true;
    } else if (strcmp(arg, "--delimiter") == 0) {
      const char* v = option_value(argc, argv, &i);
      if (!v)
        return 2;
      if (strcmp(v, "tab") == 0 || strcmp(v, "\\t") == 0)
        io.delimiter = '\t';
      else if (strcmp(v, "space") == 0)
        io.delimiter = ' ';
      else if (strlen(v) == 1)
        io.delimiter = v[0];
      else {
        fprintf(stderr,
          "sif-inspect: --delimiter %s: one character, or \"tab\"\n", v);
        return 2;
      }
      ascii_options = true;
    } else if (strcmp(arg, "--skip-header") == 0) {
      const char* v = option_value(argc, argv, &i);
      if (!v)
        return 2;
      char* end;
      const unsigned long n = strtoul(v, &end, 10);
      if (end == v || *end != '\0' || v[0] == '-' || n > UINT32_MAX) {
        fprintf(
          stderr, "sif-inspect: --skip-header %s: a number of lines\n", v);
        return 2;
      }
      io.skip_header = (uint32_t)n;
      ascii_options = true;
    } else if (arg[0] == '-' && arg[1] != '\0') {
      fprintf(stderr, "sif-inspect: unknown option '%s'\n\n", arg);
      print_usage(stderr);
      return 2;
    } else if (path) {
      fprintf(
        stderr, "sif-inspect: one FILE only, got '%s' and '%s'\n", path, arg);
      return 2;
    } else {
      path = arg;
    }
  }

  if (!path) {
    print_usage(stderr);
    return 2;
  }

  const file_kind_t kind =
    io.format != FILE_KIND_AUTO ? io.format : inspect_detect(path);
  if (kind == FILE_KIND_LUA) {
    fprintf(stderr,
      "sif-inspect: %s is a configuration: sif-exodus --check checks it\n",
      path);
    return 2;
  }
  if (kind != FILE_KIND_ASCII && ascii_options) {
    fprintf(stderr,
      "sif-inspect: --columns, --delimiter and --skip-header go with an "
      "ASCII file, and %s is %s\n",
      path, inspect_kind_name(kind));
    return 2;
  }

  /* The library's own log carries the problems; its chatter stays out of the
   * report. */
  sif_config_t lib = {.log_level = SIF_LOG_LEVEL_WARNING};
  if (sif_system_init(&lib) != SIF_OK) {
    fputs("sif-inspect: could not initialise the library\n", stderr);
    return 1;
  }
  const int code = inspect_file(path, kind, &io);
  sif_system_finalise();
  return code;
}
