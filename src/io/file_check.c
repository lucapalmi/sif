/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/file_check.h"

#include "sif/utils/logger.h"

#include "io/file_check_internal.h"
#include "utils/logger_internal.h"

#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define TAG "io"

/* The check and what running it needs: the public struct first, so the two
 * share an address. */
typedef struct {
  sif_file_check_t check;
  /* While the file is described: what the describer could not read is an
   * error of the check, though only a warning of the description. */
  bool describing;
} check_box_t;

/* ------------------------------------------------------------------------ */
/* building                                                                  */
/* ------------------------------------------------------------------------ */

static void problem_add(
  sif_file_check_t* c, sif_file_severity_t severity, const char* message) {
  if (severity == SIF_FILE_PROBLEM_ERROR)
    c->n_errors++;
  else
    c->n_warnings++;
  if (c->n_problems >= SIF_FILE_CHECK_PROBLEMS_KEPT)
    return;
  if (!c->problems) {
    c->problems =
      calloc(SIF_FILE_CHECK_PROBLEMS_KEPT, sizeof(sif_file_problem_t));
    if (!c->problems)
      return;
  }
  char* text = strdup(message);
  if (!text)
    return;
  c->problems[c->n_problems].severity = severity;
  c->problems[c->n_problems].message = text;
  c->n_problems++;
}

void sif__file_check_problem(
  sif_file_check_t* c, sif_file_severity_t severity, const char* fmt, ...) {
  /* Formatted only while kept: a file with a million bad rows is counted,
   * not written out a million times. */
  if (c->n_problems >= SIF_FILE_CHECK_PROBLEMS_KEPT) {
    if (severity == SIF_FILE_PROBLEM_ERROR)
      c->n_errors++;
    else
      c->n_warnings++;
    return;
  }
  char message[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(message, sizeof(message), fmt, ap);
  va_end(ap);
  problem_add(c, severity, message);
}

/* What the readers log while the check runs. */
static void sink(void* ctx, int level, const char* tag, const char* message) {
  (void)tag;
  check_box_t* box = ctx;
  const bool error = level >= SIF_LOG_LEVEL_ERROR || box->describing;
  problem_add(&box->check,
    error ? SIF_FILE_PROBLEM_ERROR : SIF_FILE_PROBLEM_WARNING, message);
}

void sif__file_check_content(sif_file_check_t* c, const char* content) {
  char* text = strdup(content);
  if (!text)
    return;
  free((void*)c->content);
  c->content = text;
}

uint32_t sif__file_check_column_index(
  const sif_file_check_t* c, uint32_t table, const char* name) {
  if (!c->info || table >= c->info->n_tables)
    return UINT32_MAX;
  const sif_file_table_t* tb = &c->info->tables[table];
  for (uint32_t k = 0; k < tb->n_columns; k++)
    if (strcmp(tb->columns[k].name, name) == 0)
      return k;
  return UINT32_MAX;
}

sif_file_stats_t* sif__file_check_column(
  sif_file_check_t* c, uint32_t table, uint32_t column) {
  if (!c->info || table >= c->info->n_tables ||
      column >= c->info->tables[table].n_columns)
    return NULL;
  return &c->column_stats[table][column];
}

sif_file_stats_t* sif__file_check_array(sif_file_check_t* c, uint32_t array) {
  if (!c->info || array >= c->info->n_arrays)
    return NULL;
  return &c->array_stats[array];
}

static void stats_init(sif_file_stats_t* s, size_t n) {
  for (size_t i = 0; i < n; i++) {
    s[i].n = s[i].n_nonfinite = 0;
    s[i].min = s[i].max = NAN;
  }
}

/* One statistics slot per column and per array of the description. */
static bool stats_alloc(sif_file_check_t* c) {
  const sif_file_info_t* info = c->info;
  c->column_stats = calloc(info->n_tables + 1, sizeof(sif_file_stats_t*));
  c->array_stats = calloc(info->n_arrays + 1, sizeof(sif_file_stats_t));
  if (!c->column_stats || !c->array_stats)
    return false;
  stats_init(c->array_stats, info->n_arrays);
  for (uint32_t t = 0; t < info->n_tables; t++) {
    const uint32_t n = info->tables[t].n_columns;
    c->column_stats[t] = calloc(n + 1, sizeof(sif_file_stats_t));
    if (!c->column_stats[t])
      return false;
    stats_init(c->column_stats[t], n);
  }
  return true;
}

void sif_file_check_free(sif_file_check_t* c) {
  if (!c)
    return;
  if (c->column_stats && c->info)
    for (uint32_t t = 0; t < c->info->n_tables; t++)
      free(c->column_stats[t]);
  free(c->column_stats);
  free(c->array_stats);
  for (uint32_t i = 0; i < c->n_problems; i++)
    free((void*)c->problems[i].message);
  free(c->problems);
  free((void*)c->content);
  sif_file_info_free(c->info);
  free(c);
}

/* ------------------------------------------------------------------------ */
/* the contents the formats share                                            */
/* ------------------------------------------------------------------------ */

/* A metadata value of the table or of the file, its key in either case: a
 * FITS file spells sif's keys in capitals. */
static const sif_file_meta_t* meta_of(
  const sif_file_info_t* info, uint32_t table, const char* key) {
  const sif_file_table_t* tb = &info->tables[table];
  for (uint32_t i = 0; i < tb->n_meta; i++)
    if (strcasecmp(tb->meta[i].key, key) == 0)
      return &tb->meta[i];
  for (uint32_t i = 0; i < info->n_meta; i++)
    if (strcasecmp(info->meta[i].key, key) == 0)
      return &info->meta[i];
  return NULL;
}

static bool meta_number(const sif_file_meta_t* m, double* out) {
  if (m && m->kind == SIF_FILE_META_INT) {
    *out = (double)m->i;
    return true;
  }
  if (m && m->kind == SIF_FILE_META_REAL) {
    *out = m->r;
    return true;
  }
  return false;
}

static bool meta_is(const sif_file_meta_t* m, const char* value) {
  return m && m->kind == SIF_FILE_META_STRING && strcasecmp(m->s, value) == 0;
}

/* The columns of a void catalogue as each format names them. */
typedef struct {
  uint32_t table;
  uint32_t centres[3]; /* one column of width 3 in HDF5, in [0] */
  uint32_t radius;
  bool sky;
  /* The key that holds the count, where the format writes one. */
  const char* count_key;
} catalogue_t;

static bool columns_are(const sif_file_check_t* c, uint32_t t,
  const char* const names[4], catalogue_t* cat) {
  for (int k = 0; k < 3; k++)
    if ((cat->centres[k] = sif__file_check_column_index(c, t, names[k])) ==
        UINT32_MAX)
      return false;
  return (cat->radius = sif__file_check_column_index(c, t, names[3])) !=
         UINT32_MAX;
}

/* Whether the file is a sif void catalogue, and where its columns are. */
static bool find_catalogue(const sif_file_check_t* c, catalogue_t* cat) {
  const sif_file_info_t* info = c->info;
  static const char* const ascii_box[4] = {"cx", "cy", "cz", "r"};
  static const char* const ascii_sky[4] = {"ra", "dec", "z", "r"};
  static const char* const fits_box[4] = {"CX", "CY", "CZ", "R"};
  static const char* const fits_sky[4] = {"RA", "DEC", "Z", "R"};

  memset(cat, 0, sizeof(*cat));
  for (uint32_t t = 0; t < info->n_tables; t++) {
    cat->table = t;
    const char* name = info->tables[t].name;
    switch (info->format) {
    case SIF_FILE_FORMAT_ASCII:
      /* Every catalogue sif writes in text starts with its count. */
      if (!meta_of(info, t, "n"))
        return false;
      cat->count_key = "n";
      cat->sky = columns_are(c, t, ascii_sky, cat);
      return cat->sky || columns_are(c, t, ascii_box, cat);
    case SIF_FILE_FORMAT_FITS:
      if (strcmp(name, "VOIDS") != 0)
        continue;
      cat->sky = columns_are(c, t, fits_sky, cat);
      return cat->sky || columns_are(c, t, fits_box, cat);
    case SIF_FILE_FORMAT_HDF5:
      if (strcmp(name, "catalogue") != 0)
        continue;
      cat->centres[0] = sif__file_check_column_index(c, t, "centres");
      cat->radius = sif__file_check_column_index(c, t, "radii");
      cat->centres[1] = cat->centres[2] = cat->centres[0];
      cat->sky = meta_is(meta_of(info, t, "coordinates"), "sky");
      cat->count_key = "n_voids";
      return cat->centres[0] != UINT32_MAX && cat->radius != UINT32_MAX;
    default:
      return false;
    }
  }
  return false;
}

/*
 * A void catalogue, in whichever format: radii positive, centres finite, the
 * count its header gives, a sky position on the sphere, and a box's voids
 * inside the box.
 */
static void check_catalogue(sif_file_check_t* c) {
  catalogue_t cat;
  if (!find_catalogue(c, &cat))
    return;
  sif__file_check_content(c, "sif catalogue");

  const sif_file_info_t* info = c->info;
  const sif_file_table_t* tb = &info->tables[cat.table];
  const char* path = info->path;
  const sif_file_stats_t* r = &c->column_stats[cat.table][cat.radius];
  const char* r_name = tb->columns[cat.radius].name;

  if (r->n_nonfinite)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: %s: %" PRIu64 " radii not finite", path, r_name, r->n_nonfinite);
  if (r->n > r->n_nonfinite && r->min <= 0.0)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: %s: smallest radius %g (expected every radius > 0)", path, r_name,
      r->min);

  const int n_centres = info->format == SIF_FILE_FORMAT_HDF5 ? 1 : 3;
  for (int k = 0; k < n_centres; k++) {
    const sif_file_stats_t* s = &c->column_stats[cat.table][cat.centres[k]];
    if (s->n_nonfinite)
      sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
        "%s: %s: %" PRIu64 " values not finite", path,
        tb->columns[cat.centres[k]].name, s->n_nonfinite);
  }

  double count;
  if (cat.count_key &&
      meta_number(meta_of(info, cat.table, cat.count_key), &count) &&
      count != (double)tb->n_rows)
    sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
      "%s: %s = %.0f, the table has %" PRIu64 " rows", path, cat.count_key,
      count, tb->n_rows);

  /* One column per coordinate: on the sky, the declination is on the
   * sphere and the right ascension within a turn. */
  if (cat.sky && n_centres == 3) {
    const sif_file_stats_t* ra = &c->column_stats[cat.table][cat.centres[0]];
    const sif_file_stats_t* dec = &c->column_stats[cat.table][cat.centres[1]];
    if (ra->n > ra->n_nonfinite && (ra->min < 0.0 || ra->max > 360.0))
      sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
        "%s: %s: range [%g, %g] (expected within [0, 360])", path,
        tb->columns[cat.centres[0]].name, ra->min, ra->max);
    if (dec->n > dec->n_nonfinite && (dec->min < -90.0 || dec->max > 90.0))
      sif__file_check_problem(c, SIF_FILE_PROBLEM_ERROR,
        "%s: %s: range [%g, %g] (expected within [-90, 90])", path,
        tb->columns[cat.centres[1]].name, dec->min, dec->max);
  }

  /* The voids of a periodic box are found inside it; a centre outside says
   * the catalogue is not of the box its header names. */
  double box;
  if (!cat.sky && meta_is(meta_of(info, cat.table, "mode"), "box") &&
      meta_number(meta_of(info, cat.table, "box_length"), &box) && box > 0) {
    const double tol = 1e-5 * box;
    for (int k = 0; k < n_centres; k++) {
      const sif_file_stats_t* s = &c->column_stats[cat.table][cat.centres[k]];
      if (s->n > s->n_nonfinite && (s->min < -tol || s->max > box + tol))
        sif__file_check_problem(c, SIF_FILE_PROBLEM_WARNING,
          "%s: %s: range [%g, %g], outside the box [0, %g)", path,
          tb->columns[cat.centres[k]].name, s->min, s->max, box);
    }
  }
}

/* ------------------------------------------------------------------------ */
/* the check                                                                 */
/* ------------------------------------------------------------------------ */

sif_file_check_t* sif_file_check(const char* path, sif_file_format_t format) {
  if (!path) {
    SIF_LOG_ERROR(TAG, "sif_file_check: NULL path");
    return NULL;
  }
  check_box_t* box = calloc(1, sizeof(check_box_t));
  if (!box) {
    SIF_LOG_ERROR(TAG, "out of memory");
    return NULL;
  }
  sif_file_check_t* c = &box->check;

  sif__log_capture_t saved;
  sif__log_capture_begin(sink, box, &saved);

  box->describing = true;
  c->info = sif_file_info_read(path, format);
  box->describing = false;

  int status = SIF_OK;
  if (c->info && !stats_alloc(c))
    status = SIF_ERR_ALLOC;

  if (c->info && status == SIF_OK) {
    switch (c->info->format) {
    case SIF_FILE_FORMAT_ASCII:
      status = sif__file_check_ascii(c);
      break;
    case SIF_FILE_FORMAT_FITS:
      status = sif__file_check_fits(c);
      break;
    case SIF_FILE_FORMAT_HDF5:
      status = sif__file_check_hdf5(c);
      break;
    case SIF_FILE_FORMAT_GADGET:
      status = sif__file_check_gadget(c);
      break;
    case SIF_FILE_FORMAT_XFIELD:
      status = sif__file_check_xfield(c);
      break;
    case SIF_FILE_FORMAT_XGRID:
      status = sif__file_check_xgrid(c);
      break;
    case SIF_FILE_FORMAT_CHECKPOINT:
      status = sif__file_check_checkpoint(c);
      break;
    case SIF_FILE_FORMAT_UNKNOWN:
      break;
    }
  }
  if (c->info && status == SIF_OK)
    check_catalogue(c);

  sif__log_capture_end(&saved);

  if (!c->content)
    sif__file_check_content(c, "");
  if (status == SIF_ERR_ALLOC || !c->content) {
    SIF_LOG_ERROR(TAG, "%s: out of memory checking the file", path);
    sif_file_check_free(c);
    return NULL;
  }
  return c;
}

/* ------------------------------------------------------------------------ */
/* printing                                                                  */
/* ------------------------------------------------------------------------ */

static const char* plural(uint64_t n) { return n == 1 ? "" : "s"; }

int sif_file_check_print(const sif_file_check_t* c, FILE* s) {
  if (!c || !s) {
    SIF_LOG_ERROR(TAG, "sif_file_check_print: NULL argument");
    return SIF_ERR_INVALID;
  }

  if (c->info) {
    const int status =
      sif__file_info_print_stats(c->info, c->column_stats, c->array_stats, s);
    if (status != SIF_OK)
      return status;
    if (c->content[0])
      fprintf(s, "\nchecked as %s\n", c->content);
  }

  if (c->n_problems) {
    fprintf(s, "%sproblems\n", c->info ? "\n" : "");
    for (uint32_t i = 0; i < c->n_problems; i++)
      fprintf(s, "  %-8s %s\n",
        c->problems[i].severity == SIF_FILE_PROBLEM_ERROR ? "error" : "warning",
        c->problems[i].message);
    const uint64_t total = c->n_errors + c->n_warnings;
    if (total > c->n_problems)
      fprintf(s, "  ... and %" PRIu64 " more\n", total - c->n_problems);
  }

  fprintf(s, "\nresult    ");
  if (c->n_errors == 0 && c->n_warnings == 0)
    fputs("sound\n", s);
  else
    fprintf(s, "%" PRIu64 " error%s, %" PRIu64 " warning%s\n", c->n_errors,
      plural(c->n_errors), c->n_warnings, plural(c->n_warnings));

  fflush(s);
  return ferror(s) ? SIF_ERR_IO : SIF_OK;
}
