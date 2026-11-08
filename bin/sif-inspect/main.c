/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/* sif-inspect: a look at a file sif reads or writes, without loading it.
 *
 *   sif-inspect [options] FILE           what FILE holds
 *   sif-inspect --check [options] FILE   whether FILE is sound
 *
 * Exit status: 0 on success, 1 when the library cannot start, 2 when the
 * command line is wrong, or the file cannot be read or has problems.
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
    "Describe FILE: its format, its metadata, and its tables and arrays,\n"
    "the same way for every format sif reads -- ASCII, FITS, HDF5, GADGET,\n"
    ".xfield, .xgrid and exodus checkpoints. Only what the file says about\n"
    "itself is read, and nothing is checked.\n"
    "\n"
    "options:\n"
    "  --format KIND  the format of FILE, instead of telling it from its\n"
    "                 contents: ascii, fits, hdf5, gadget, xfield, xgrid\n"
    "                 or checkpoint\n"
    "  --check        check FILE instead, for problems, without loading it\n"
    "  --deep         with --check, read every value as well, for each\n"
    "                 column's range\n"
    "  --columns FMT  with --check on an ASCII file, the column format to\n"
    "                 check it against, as in a configuration: \"x y z w\"\n"
    "  --delimiter C  with --check on an ASCII file, the column delimiter\n"
    "                 (default a blank; \"tab\" for a tab)\n"
    "  --skip-header N\n"
    "                 with --check on an ASCII file, lines to skip first\n"
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
  inspect_opts_t io = {SIF_FILE_FORMAT_UNKNOWN, false, NULL, '\0', 0, false};
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
    } else if (strcmp(arg, "--check") == 0) {
      io.check = true;
    } else if (strcmp(arg, "--deep") == 0) {
      io.deep = true;
    } else if (strcmp(arg, "--format") == 0) {
      const char* v = option_value(argc, argv, &i);
      if (!v)
        return 2;
      if (sif_file_format_parse(v, &io.format) != SIF_OK) {
        fprintf(stderr,
          "sif-inspect: --format %s: expected ascii, fits, hdf5, gadget, "
          "xfield, xgrid or checkpoint\n",
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

  if (!io.check && (io.deep || ascii_options)) {
    fprintf(stderr, "sif-inspect: --deep, --columns, --delimiter and "
                    "--skip-header go with --check\n");
    return 2;
  }

  /* A configuration is sif-exodus's: the library knows nothing of it. */
  const size_t len = strlen(path);
  if (io.format == SIF_FILE_FORMAT_UNKNOWN && len >= 4 &&
      strcmp(path + len - 4, ".lua") == 0) {
    fprintf(stderr,
      "sif-inspect: %s is a configuration: sif-exodus --check checks it\n",
      path);
    return 2;
  }

  const sif_file_format_t format = io.format != SIF_FILE_FORMAT_UNKNOWN
                                     ? io.format
                                     : sif_file_format_detect(path);
  if (format != SIF_FILE_FORMAT_ASCII && ascii_options) {
    fprintf(stderr,
      "sif-inspect: --columns, --delimiter and --skip-header go with an "
      "ASCII file, and %s is %s\n",
      path, sif_file_format_name(format));
    return 2;
  }

  /* The library's own log carries the problems; its chatter stays out of the
   * report. */
  sif_config_t lib = {.log_level = SIF_LOG_LEVEL_WARNING};
  if (sif_system_init(&lib) != SIF_OK) {
    fputs("sif-inspect: could not initialise the library\n", stderr);
    return 1;
  }
  const int code = io.check ? inspect_check(path, format, &io)
                            : inspect_describe(path, format);
  sif_system_finalise();
  return code;
}
