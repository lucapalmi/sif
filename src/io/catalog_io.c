/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/catalog_io.h"

#include "sif/utils/logger.h"
#include "structures/catalog_internal.h"

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/*
 * Longest line the reader accepts. A row of every column at the widest
 * SIF_PRI_REAL is about 150 characters, so this is room to spare, not a limit
 * a written file can reach; a longer line is refused as malformed rather than
 * split.
 */
#define CATALOG_LINE_MAX 4096

/* Values a row may hold: the six sif writes, and room for columns another
 * tool added that the reader skips. Past the last column a layout places,
 * a row may run on with anything. */
#define CATALOG_MAX_COLS 64

int sif_catalog_write_ascii(
  const char* filepath, const sif_catalog_t* catalog) {
  if (!catalog || !filepath) {
    SIF_LOG_ERROR("io", "invalid arguments for write_catalog_ascii");
    return SIF_ERR_INVALID;
  }

  FILE* file = fopen(filepath, "w");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for writing", filepath);
    return SIF_ERR_IO;
  }

  /* The count goes first so the reader can allocate the catalogue once,
   * instead of growing it a void at a time or scanning the file twice; the
   * names say what each column is, sky or Cartesian, footprint or not. Both
   * behind '#', so numpy.loadtxt and friends read the rows as they are. */
  fprintf(file, "#n=%" PRIu64 "\n", catalog->n_voids);

  /* The catalogue's metadata, one key a line, before the names: strings in
   * quotes, so that "10" comes back a string, numbers as they are. */
  for (uint32_t m = 0; m < sif_catalog_meta_count(catalog); m++) {
    const char* key = sif_catalog_meta_name(catalog, m);
    switch (sif_catalog_meta_kind(catalog, key)) {
    case SIF_CATALOG_META_INT:
      fprintf(file, "#%s=%lld\n", key,
        (long long)sif_catalog_meta_int_get(catalog, key));
      break;
    case SIF_CATALOG_META_REAL:
      fprintf(
        file, "#%s=%.17g\n", key, sif_catalog_meta_real_get(catalog, key));
      break;
    case SIF_CATALOG_META_STRING:
      fprintf(
        file, "#%s=\"%s\"\n", key, sif_catalog_meta_string_get(catalog, key));
      break;
    case SIF_CATALOG_META_MISSING:
      break;
    }
  }
  fputs(catalog->units == SIF_COORDINATES_SKY ? "#ra dec z r" : "#cx cy cz r",
    file);
  if (catalog->footprint)
    fputs(" footprint footprint_shell", file);
  fputc('\n', file);

  /* SIF_PRI_REAL round-trips: a catalogue written and read back gives the same
   * radii bit for bit, which is what lets a size function computed from the
   * file match one computed in memory. */
  for (uint64_t i = 0; i < catalog->n_voids; i++) {
    fprintf(file,
      SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL,
      catalog->cx[i], catalog->cy[i], catalog->cz[i], catalog->radii[i]);
    if (catalog->footprint)
      fprintf(file, " " SIF_PRI_REAL " " SIF_PRI_REAL, catalog->footprint[i],
        catalog->footprint_shell[i]);
    fputc('\n', file);
  }

  /* fprintf() reports nothing useful per call, so the stream's error flag and
   * the flush inside fclose() are what say whether the file actually reached
   * the disk. Without this a full quota reads back as a short catalogue. */
  const bool ok = (ferror(file) == 0);
  if (fclose(file) != 0 || !ok) {
    SIF_LOG_ERROR("io", "failed to flush %s to disk", filepath);
    return SIF_ERR_IO;
  }

  SIF_LOG_INFO(
    "io", "saved %" PRIu64 " voids to %s (ASCII)", catalog->n_voids, filepath);
  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* reading                                                                   */
/* ------------------------------------------------------------------------ */

/* What a line of the file is. */
typedef enum { LINE_END, LINE_TOO_LONG, LINE_COMMENT, LINE_DATA } line_kind_t;

/*
 * The next line that holds anything, into `line`, and what it is. Blank lines
 * are skipped, as the whitespace-driven reader this replaced skipped them. A
 * comment comes back without its '#' and the blanks around it.
 */
static line_kind_t next_line(FILE* file, char* line, char** text) {
  while (fgets(line, CATALOG_LINE_MAX, file)) {
    const size_t len = strlen(line);
    if (len == CATALOG_LINE_MAX - 1 && line[len - 1] != '\n' && !feof(file))
      return LINE_TOO_LONG;

    char* p = line + strspn(line, " \t\r\n");
    if (!*p)
      continue;
    if (*p == '#') {
      p++;
      p += strspn(p, " \t");
      char* end = p + strlen(p);
      while (end > p && isspace((unsigned char)end[-1]))
        *--end = '\0';
      *text = p;
      return LINE_COMMENT;
    }
    *text = p;
    return LINE_DATA;
  }
  return LINE_END;
}

/* One value, parsed at the precision it is kept in: a float read through a
 * double and then rounded can land one ulp off the float that was written. */
static bool parse_real(const char* s, char** end, sif_real* out) {
  if (sizeof(sif_real) == 4)
    *out = (sif_real)strtof(s, end);
  else
    *out = (sif_real)strtod(s, end);
  return *end != s;
}

/*
 * The values of a data row, separated by blanks or commas, and how many.
 *
 * Without a `limit` (0) every column has to be a number, and a row with more
 * than CATALOG_MAX_COLS, or with anything else, is -1. With one, only the
 * first `limit` columns are looked at, and one that is not a number sets its
 * bit in `bad` rather than failing: the caller knows which columns it reads,
 * and another tool's file can have a name or a flag in one it skips.
 */
static int parse_row(const char* text, sif_real* v, int limit, uint64_t* bad) {
  static const char SEPARATORS[] = " \t\r\n,";
  int n = 0;
  const char* p = text;
  *bad = 0;
  for (;;) {
    p += strspn(p, SEPARATORS);
    if (!*p || (limit > 0 && n == limit))
      return n;
    if (n == CATALOG_MAX_COLS)
      return -1;
    const size_t len = strcspn(p, SEPARATORS);
    char* end;
    if (!parse_real(p, &end, &v[n]) || (size_t)(end - p) != len) {
      if (limit == 0)
        return -1;
      *bad |= UINT64_C(1) << n;
    }
    n++;
    p += len;
  }
}

/* Where each quantity sits in a row, or -1. */
typedef struct {
  int x, y, z, r, fp, fp_shell;
  bool sky;
} layout_t;

/* Whether a column name is one of `names`: sif's own first, then what other
 * tools, or sif's other formats, call the same quantity. */
static bool name_is(const char* name, const char* const* names) {
  for (; *names; names++)
    if (strcasecmp(name, *names) == 0)
      return true;
  return false;
}

/*
 * A comment that names the columns. It has to name the three centre
 * coordinates and a radius, one way or the other, to count: any other
 * comment is just a comment, and columns with names nobody reads are
 * skipped. Returns 1 for a layout, 0 for a comment that is not one, -1 for
 * one that mixes sky and Cartesian names.
 */
static int parse_names(char* text, layout_t* out) {
  static const char* const X[] = {"cx", "x", NULL};
  static const char* const Y[] = {"cy", "y", NULL};
  static const char* const Z[] = {"cz", "z", NULL};
  static const char* const RA[] = {"ra", NULL};
  static const char* const DEC[] = {"dec", NULL};
  static const char* const R[] = {"r", "radius", NULL};
  static const char* const FP[] = {"footprint", NULL};
  static const char* const FPS[] = {"footprint_shell", NULL};

  layout_t l = {-1, -1, -1, -1, -1, -1, false};
  bool cartesian = false;
  int col = 0;
  char* save = NULL;
  for (char* tok = strtok_r(text, " \t,", &save); tok;
    tok = strtok_r(NULL, " \t,", &save), col++) {
    if (name_is(tok, X)) {
      l.x = col;
      cartesian = true;
    } else if (name_is(tok, Y)) {
      l.y = col;
      cartesian = true;
    } else if (name_is(tok, RA)) {
      l.x = col;
      l.sky = true;
    } else if (name_is(tok, DEC)) {
      l.y = col;
      l.sky = true;
    } else if (name_is(tok, Z)) {
      l.z = col;
    } else if (name_is(tok, R)) {
      l.r = col;
    } else if (name_is(tok, FP)) {
      l.fp = col;
    } else if (name_is(tok, FPS)) {
      l.fp_shell = col;
    }
  }

  if (l.x < 0 || l.y < 0 || l.z < 0 || l.r < 0)
    return 0;
  if (cartesian && l.sky)
    return -1;
  if ((l.fp < 0) != (l.fp_shell < 0))
    l.fp = l.fp_shell = -1; /* half a footprint is none */
  *out = l;
  return 1;
}

/*
 * A column format: the columns in order, as for the field readers -- `x y z`
 * (or `cx cy cz`) or `ra dec z` for the centre, `r` for the radius, `*` for
 * a column not read. Blanks and commas between names are optional. Every
 * mistake is refused, with where it is.
 */
static int parse_format(const char* fmt, layout_t* out) {
  layout_t l = {-1, -1, -1, -1, -1, -1, false};
  bool cartesian = false;
  int col = 0;
  for (const char* p = fmt; *p;) {
    const char c = (char)tolower((unsigned char)*p);
    if (c == ' ' || c == '\t' || c == ',') {
      p++;
      continue;
    }
    const long at = (long)(p - fmt) + 1;
    int* slot = NULL;
    const char* name = NULL;
    int len = 1;
    if (c == 'c' && p[1] && strchr("xyz", tolower((unsigned char)p[1])))
      p++; /* cx cy cz: the same as x y z */
    const char d = (char)tolower((unsigned char)*p);
    if (d == '*') {
      p++;
      col++;
      continue;
    } else if (d == 'x' || d == 'y') {
      slot = d == 'x' ? &l.x : &l.y;
      name = d == 'x' ? "x" : "y";
      cartesian = true;
    } else if (d == 'z') {
      slot = &l.z;
      name = "z";
    } else if (d == 'r' && tolower((unsigned char)p[1]) == 'a') {
      slot = &l.x;
      name = "ra";
      len = 2;
      l.sky = true;
    } else if (d == 'r') {
      slot = &l.r;
      name = "r";
    } else if (d == 'd' && tolower((unsigned char)p[1]) == 'e' &&
               tolower((unsigned char)p[2]) == 'c') {
      slot = &l.y;
      name = "dec";
      len = 3;
      l.sky = true;
    } else {
      SIF_LOG_ERROR("io",
        "catalogue format '%s', position %ld: '%c' names no column -- use x "
        "y z (or ra dec z), r, and * for a column not read",
        fmt, at, *p);
      return SIF_ERR_INVALID;
    }
    if (*slot >= 0) {
      SIF_LOG_ERROR("io",
        "catalogue format '%s', position %ld: %s is named twice", fmt, at,
        name);
      return SIF_ERR_INVALID;
    }
    *slot = col++;
    p += len;
  }

  if (cartesian && l.sky) {
    SIF_LOG_ERROR("io",
      "catalogue format '%s' mixes sky (ra, dec) and Cartesian (x, y) "
      "centres",
      fmt);
    return SIF_ERR_INVALID;
  }
  if (l.x < 0 || l.y < 0 || l.z < 0 || l.r < 0) {
    SIF_LOG_ERROR("io",
      "catalogue format '%s' has to name the centre, x y z or ra dec z, and "
      "the radius, r",
      fmt);
    return SIF_ERR_INVALID;
  }
  *out = l;
  return SIF_OK;
}

/* "n=123", with blanks allowed around the '='. */
static bool parse_count(const char* text, uint64_t* n) {
  if (tolower((unsigned char)text[0]) != 'n')
    return false;
  const char* p = text + 1;
  p += strspn(p, " \t");
  if (*p != '=')
    return false;
  p++;
  p += strspn(p, " \t");
  if (!isdigit((unsigned char)*p))
    return false;
  char* end;
  const unsigned long long v = strtoull(p, &end, 10);
  end += strspn(end, " \t");
  if (*end)
    return false;
  *n = (uint64_t)v;
  return true;
}

/*
 * A "key=value" comment, into the metadata of `into`: a quoted value is a
 * string, one that parses whole as an integer or a number is that, anything
 * else a string as written. Returns whether the comment was one; a key the
 * catalogue does not allow is logged and skipped, not fatal.
 */
static bool parse_meta(char* text, sif_catalog_t* into) {
  char* eq = strchr(text, '=');
  if (!eq)
    return false;
  *eq = '\0';
  char* key = text;
  char* end = eq;
  while (end > key && isspace((unsigned char)end[-1]))
    *--end = '\0';
  char* value = eq + 1;
  value += strspn(value, " \t");

  const size_t len = strlen(value);
  if (len >= 2 && value[0] == '"' && value[len - 1] == '"') {
    value[len - 1] = '\0';
    (void)sif_catalog_meta_string_set(into, key, value + 1);
    return true;
  }
  char* stop;
  const long long i = strtoll(value, &stop, 10);
  if (stop != value && !*stop) {
    (void)sif_catalog_meta_int_set(into, key, (int64_t)i);
    return true;
  }
  const double d = strtod(value, &stop);
  if (stop != value && !*stop && isfinite(d)) {
    (void)sif_catalog_meta_real_set(into, key, d);
    return true;
  }
  (void)sif_catalog_meta_string_set(into, key, value);
  return true;
}

/* Every exit after the file is open goes through here. */
static sif_catalog_t* fail(FILE* file, sif_catalog_t* catalog) {
  sif_catalog_free(catalog);
  fclose(file);
  return NULL;
}

sif_catalog_t* sif_catalog_read_ascii(const char* filepath, const char* fmt) {
  if (!filepath) {
    SIF_LOG_ERROR("io", "invalid filepath for read_catalog_ascii");
    return NULL;
  }

  /* A format places the columns, and any names in the file are then just a
   * comment. It is checked before the file is opened. */
  layout_t layout = {0, 1, 2, 3, -1, -1, false};
  if (fmt && parse_format(fmt, &layout) != SIF_OK)
    return NULL;
  const bool have_format = fmt != NULL;

  FILE* file = fopen(filepath, "r");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for reading", filepath);
    return NULL;
  }

  char line[CATALOG_LINE_MAX];
  char* text = NULL;
  line_kind_t kind;

  /* --- the header: the comments before the first row --- */

  uint64_t n_voids = 0;
  bool have_count = false, have_names = have_format;
  long data_at = 0; /* where the first row starts */

  /* The metadata, held until the catalogue exists: its size is only known
   * once the header is read. */
  sif_catalog_t* meta = sif_catalog_alloc(1);
  if (!meta)
    return fail(file, NULL);

  for (;;) {
    data_at = ftell(file);
    kind = next_line(file, line, &text);
    if (kind != LINE_COMMENT)
      break;
    if (!have_count && parse_count(text, &n_voids)) {
      have_count = true;
    } else if (strchr(text, '=')) {
      (void)parse_meta(text, meta);
    } else if (!have_names) { /* never with a format */
      const int named = parse_names(text, &layout);
      if (named < 0) {
        SIF_LOG_ERROR("io",
          "%s: the column names mix sky (ra, dec) and Cartesian (cx, cy) "
          "centres",
          filepath);
        sif_catalog_free(meta);
        return fail(file, NULL);
      }
      have_names = named == 1;
    }
  }

  /* A file from before the header: its first line is the count alone. */
  if (!have_count && !have_names && kind == LINE_DATA) {
    char* end;
    const unsigned long long legacy = strtoull(text, &end, 10);
    const char* rest = end + strspn(end, " \t\r\n");
    if (end != text && !*rest) {
      n_voids = (uint64_t)legacy;
      have_count = true;
      data_at = ftell(file);
      kind = next_line(file, line, &text);
    }
  }

  /* Without a count the rows are counted first, and then read. */
  if (!have_count) {
    for (line_kind_t k = kind; k != LINE_END;
      k = next_line(file, line, &text)) {
      if (k == LINE_TOO_LONG)
        break;
      n_voids += k == LINE_DATA;
    }
    if (fseek(file, data_at, SEEK_SET) != 0) {
      SIF_LOG_ERROR("io", "%s: cannot read the rows a second time", filepath);
      sif_catalog_free(meta);
      return fail(file, NULL);
    }
    kind = next_line(file, line, &text);
  }

  sif_catalog_t* catalog = sif_catalog_alloc(n_voids);
  const int copied = catalog ? sif__catalog_meta_copy(catalog, meta) : SIF_OK;
  sif_catalog_free(meta);
  if (!catalog || copied != SIF_OK) {
    SIF_LOG_ERROR("io", "failed to allocate catalog for loading");
    return fail(file, catalog);
  }
  if (layout.sky)
    catalog->units = SIF_COORDINATES_SKY;

  /*
   * Rows are required to be there: the count said how many, and a file that
   * stops short is truncated rather than merely small. Reading fewer would
   * hand back a catalogue whose tail is uninitialized memory.
   *
   * Without names, the first row decides the layout -- four columns, or six
   * with the footprint -- and every row after it has to agree. With them,
   * every row needs at least the columns they place.
   */
  int n_columns = 0;
  int needed = 0;
  const int placed[] = {
    layout.x, layout.y, layout.z, layout.r, layout.fp, layout.fp_shell};
  for (int c = 0; c < 6; c++)
    if (placed[c] + 1 > needed)
      needed = placed[c] + 1;

  sif_real v[CATALOG_MAX_COLS];
  uint64_t bad = 0;
  for (uint64_t i = 0; i < n_voids; i++) {
    while (kind == LINE_COMMENT)
      kind = next_line(file, line, &text);
    if (kind == LINE_TOO_LONG) {
      SIF_LOG_ERROR(
        "io", "%s: a line is too long to be a catalogue row", filepath);
      return fail(file, catalog);
    }
    const int got = kind == LINE_DATA
                      ? parse_row(text, v, have_names ? needed : 0, &bad)
                      : -1;

    if (i == 0 && !have_names) {
      n_columns = got == 6 ? 6 : 4;
      if (got == 6) {
        layout.fp = 4;
        layout.fp_shell = 5;
      }
    }
    if (i == 0 && layout.fp >= 0 &&
        sif_catalog_reserve_footprint(catalog) != SIF_OK)
      return fail(file, catalog);

    if (have_names ? got < needed : got != n_columns) {
      SIF_LOG_ERROR("io",
        "void %" PRIu64 " of %s has %d readable columns, expected %d", i,
        filepath, got < 0 ? 0 : got, have_names ? needed : n_columns);
      return fail(file, catalog);
    }
    for (int c = 0; c < 6; c++)
      if (placed[c] >= 0 && (bad >> placed[c] & 1)) {
        SIF_LOG_ERROR("io", "void %" PRIu64 " of %s: column %d is not a number",
          i, filepath, placed[c] + 1);
        return fail(file, catalog);
      }

    catalog->cx[i] = v[layout.x];
    catalog->cy[i] = v[layout.y];
    catalog->cz[i] = v[layout.z];
    catalog->radii[i] = v[layout.r];
    if (layout.fp >= 0) {
      catalog->footprint[i] = v[layout.fp];
      catalog->footprint_shell[i] = v[layout.fp_shell];
    }
    kind = next_line(file, line, &text);
  }

  /* More rows than the count said is as wrong as fewer. */
  while (kind == LINE_COMMENT)
    kind = next_line(file, line, &text);
  if (kind != LINE_END) {
    SIF_LOG_ERROR(
      "io", "%s holds more rows than its count of %" PRIu64, filepath, n_voids);
    return fail(file, catalog);
  }

  /* Filled in directly rather than through sif_catalog_append(), so the size
   * has to be set by hand. */
  catalog->n_voids = n_voids;

  fclose(file);
  SIF_LOG_INFO("io", "loaded %" PRIu64 " voids from %s (ASCII%s%s)", n_voids,
    filepath, catalog->footprint ? ", with footprint" : "",
    layout.sky ? ", on the sky" : "");

  return catalog;
}

#undef CATALOG_LINE_MAX
#undef CATALOG_MAX_COLS
