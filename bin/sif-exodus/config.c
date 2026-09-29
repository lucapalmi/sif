/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "config.h"

#include "sif/core/macros.h"
#include "sif/finder/exodus_finder.h"
#include "sif/io/field_io.h"
#include "sif/io/fits_io.h"
#include "sif/io/gadget_io.h"
#include "sif/utils/logger.h"

#include "lauxlib.h"
#include "lua.h"
#include "lualib.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* More radii than any ladder needs; a bound on what a helper will build. */
#define MAX_RADII 100000

/* --- what a configuration may say ------------------------------------ */

static const char* const SECTIONS[] = {
  "input", "grid", "mesh", "finder", "output", "run", NULL};

static const char* const INPUT_KEYS[] = {"path", "format", "box_length",
  "ascii", "binary", "gadget", "fits", "hdf5", NULL};
static const char* const ASCII_KEYS[] = {
  "columns", "delimiter", "skip_header", NULL};
static const char* const BINARY_KEYS[] = {
  "columns", "precision", "layout", "endian", "header_bytes", NULL};
static const char* const GADGET_KEYS[] = {
  "snapformat", "ptype", "length", "masses", "fraction", "seed", NULL};
static const char* const FITS_KEYS[] = {
  "columns", "where", "hdu", "fraction", "seed", NULL};
static const char* const HDF5_KEYS[] = {
  "columns", "length_scale", "fraction", "seed", NULL};
static const char* const FITS_COLUMN_KEYS[] = {"x", "y", "z", "w", NULL};
/* Keys refused with a message of their own, rather than as unknown. */
static const char* const FITS_COLUMN_REFUSED[] = {
  "x", "y", "z", "w", "vx", "vy", "vz", "ra", "dec", NULL};
static const char* const GRID_KEYS[] = {"n_cells", NULL};
static const char* const MESH_KEYS[] = {"n_cells", NULL};
static const char* const FINDER_KEYS[] = {"radii", "radii_units", "threshold",
  "overlap_fraction", "search_factor", NULL};
static const char* const OUTPUT_KEYS[] = {"path", "format", NULL};
static const char* const RUN_KEYS[] = {
  "threads", "log_level", "tune_fft", NULL};

/* Each section's keys, to tell a setting written at the top level where it
 * belongs. */
static const struct {
  const char* section;
  const char* const* keys;
} SECTION_KEYS[] = {{"input", INPUT_KEYS}, {"grid", GRID_KEYS},
  {"mesh", MESH_KEYS}, {"finder", FINDER_KEYS}, {"output", OUTPUT_KEYS},
  {"run", RUN_KEYS}};

/* Enumerations, in the order of the C enums they stand for. */
static const char* const INPUT_FORMATS[] = {
  "xfield", "ascii", "binary", "gadget", "fits", "hdf5", NULL};
static const char* const OUTPUT_FORMATS[] = {"hdf5", "ascii", "fits", NULL};
static const char* const RADII_UNITS[] = {"physical", "mps", NULL};
static const char* const LAYOUTS[] = {"rows", "blocks", NULL};
static const char* const PRECISIONS[] = {"float32", "float64", NULL};
static const char* const ENDIANS[] = {"native", "little", "big", NULL};
static const char* const LENGTHS[] = {"kpc", "mpc", "auto", NULL};
static const char* const LOG_LEVELS[] = {
  "trace", "debug", "info", "warning", "error", "none", NULL};

/* What the script may call, besides its own functions. */
static const char* const BASE_FUNCTIONS[] = {"assert", "error", "getmetatable",
  "ipairs", "next", "pairs", "pcall", "print", "rawequal", "rawget", "rawlen",
  "rawset", "select", "setmetatable", "tonumber", "tostring", "type", "xpcall",
  "_VERSION", NULL};
static const char* const OS_FUNCTIONS[] = {
  "clock", "date", "difftime", "getenv", "time", NULL};

static const char* const LUA_KEYWORDS[] = {"and", "break", "do", "else",
  "elseif", "end", "false", "for", "function", "global", "goto", "if", "in",
  "local", "nil", "not", "or", "repeat", "return", "then", "true", "until",
  "while", NULL};

/* --- small utilities ------------------------------------------------- */

static int list_index(const char* s, const char* const* list) {
  for (int i = 0; list[i]; i++)
    if (strcmp(s, list[i]) == 0)
      return i;
  return -1;
}

static bool ends_with(const char* s, const char* suffix) {
  const size_t n = strlen(s), m = strlen(suffix);
  return n >= m && strcmp(s + n - m, suffix) == 0;
}

/* Edit distance, for "did you mean": insertions, deletions, substitutions
 * and swaps of neighbours each cost one, so gird is one edit from grid.
 * Names are short; longer ones are simply never close. */
static size_t edit_distance(const char* a, const char* b) {
  const size_t la = strlen(a), lb = strlen(b);
  size_t d[32][32];
  if (la >= 32 || lb >= 32)
    return 32;

  for (size_t i = 0; i <= la; i++)
    d[i][0] = i;
  for (size_t j = 0; j <= lb; j++)
    d[0][j] = j;
  for (size_t i = 1; i <= la; i++) {
    for (size_t j = 1; j <= lb; j++) {
      size_t best = d[i - 1][j - 1] + (a[i - 1] != b[j - 1]);
      if (d[i - 1][j] + 1 < best)
        best = d[i - 1][j] + 1;
      if (d[i][j - 1] + 1 < best)
        best = d[i][j - 1] + 1;
      if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] &&
          d[i - 2][j - 2] + 1 < best)
        best = d[i - 2][j - 2] + 1;
      d[i][j] = best;
    }
  }
  return d[la][lb];
}

/* The candidate a name is most likely a misspelling of, or NULL. */
static const char* closest(const char* name, const char* const* candidates) {
  const char* best = NULL;
  size_t best_d = 3;
  for (int i = 0; candidates[i]; i++) {
    const size_t d = edit_distance(name, candidates[i]);
    const size_t allowed = strlen(candidates[i]) > 4 ? 2 : 1;
    if (d <= allowed && d < best_d) {
      best = candidates[i];
      best_d = d;
    }
  }
  return best;
}

static bool is_identifier(const char* s) {
  if (!(*s == '_' || (*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z')))
    return false;
  for (s++; *s; s++)
    if (!(*s == '_' || (*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') ||
          (*s >= '0' && *s <= '9')))
      return false;
  return true;
}

/* The shortest %g that reads back as the same double; whole numbers plainly,
 * so that 10 is not 1e+01. */
static void format_double(char* buf, size_t n, double v) {
  if (v == floor(v) && fabs(v) < 1e15) {
    snprintf(buf, n, "%.0f", v);
    return;
  }
  for (int p = 1; p <= 17; p++) {
    snprintf(buf, n, "%.*g", p, v);
    if (strtod(buf, NULL) == v)
      return;
  }
}

/* The shortest %g that reads back as the same sif_real. */
static void format_real(char* buf, size_t n, sif_real v) {
  if (v == floor(v) && fabs(v) < 1e7) {
    snprintf(buf, n, "%.0f", (double)v);
    return;
  }
  for (int p = 1; p <= 17; p++) {
    snprintf(buf, n, "%.*g", p, (double)v);
    if ((sif_real)strtod(buf, NULL) == v)
      return;
  }
}

static void write_lua_string(FILE* out, const char* s) {
  fputc('"', out);
  for (const unsigned char* c = (const unsigned char*)s; *c; c++) {
    if (*c == '"' || *c == '\\')
      fprintf(out, "\\%c", *c);
    else if (*c == '\n')
      fputs("\\n", out);
    else if (*c == '\t')
      fputs("\\t", out);
    else if (*c < 0x20 || *c == 0x7f)
      fprintf(out, "\\%03u", *c);
    else
      fputc(*c, out);
  }
  fputc('"', out);
}

/* --- reading, and saying what is wrong ------------------------------- */

typedef struct {
  lua_State* L;
  const char* file;
  int n_errors;
} reader_t;

static void report(reader_t* r, const char* where, const char* fmt, ...) {
  va_list ap;
  fprintf(stderr, "%s: ", r->file);
  if (where && *where)
    fprintf(stderr, "%s: ", where);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  r->n_errors++;
}

/* "path.key", for messages; "key" alone at the top level, whose path is
 * empty. */
static const char* at(char* buf, size_t n, const char* path, const char* key) {
  snprintf(buf, n, "%s%s%s", path, *path ? "." : "", key);
  return buf;
}

static void list_names(char* buf, size_t n, const char* const* names) {
  buf[0] = '\0';
  for (int i = 0; names[i]; i++) {
    if (i > 0)
      strncat(buf, ", ", n - strlen(buf) - 1);
    strncat(buf, names[i], n - strlen(buf) - 1);
  }
}

/* Push t[key], raw, and return its type. */
static int field(lua_State* L, int t, const char* key) {
  lua_pushstring(L, key);
  return lua_rawget(L, t);
}

/* Every key of the table at t has to be one of `keys`. */
static void check_keys(
  reader_t* r, int t, const char* path, const char* const* keys) {

  lua_State* L = r->L;
  char w[160], names[256];

  lua_pushnil(L);
  while (lua_next(L, t)) {
    if (lua_type(L, -2) != LUA_TSTRING) {
      report(r, path,
        "holds name = value settings, not a list; found an entry with a %s "
        "key",
        luaL_typename(L, -2));
    } else {
      const char* key = lua_tostring(L, -2);
      if (list_index(key, keys) < 0) {
        const char* guess = closest(key, keys);
        if (guess) {
          report(r, at(w, sizeof w, path, key),
            "unknown key (did you mean %s?)", guess);
        } else {
          list_names(names, sizeof names, keys);
          report(r, at(w, sizeof w, path, key), "unknown key; %s takes %s",
            *path ? path : "a configuration", names);
        }
      }
    }
    lua_pop(L, 1);
  }
}

static void type_error(
  reader_t* r, const char* path, const char* key, const char* want, int type) {
  char w[160];
  report(r, at(w, sizeof w, path, key), "must be %s, got a %s", want,
    lua_typename(r->L, type));
}

/* Each reader below leaves the stack as it found it and returns whether the
 * setting was given (and valid); *out is only written then. */

static bool get_number(
  reader_t* r, int t, const char* path, const char* key, double* out) {

  lua_State* L = r->L;
  char w[160];
  bool given = false;

  const int type = field(L, t, key);
  if (type == LUA_TNUMBER) {
    const double v = lua_tonumber(L, -1);
    if (isfinite(v)) {
      *out = v;
      given = true;
    } else {
      report(r, at(w, sizeof w, path, key), "must be finite, got %g", v);
    }
  } else if (type != LUA_TNIL) {
    type_error(r, path, key, "a number", type);
  }
  lua_pop(L, 1);
  return given;
}

static bool get_integer(reader_t* r, int t, const char* path, const char* key,
  int64_t lo, int64_t hi, int64_t* out) {

  lua_State* L = r->L;
  char w[160];
  bool given = false;

  const int type = field(L, t, key);
  if (type == LUA_TNUMBER) {
    int is_int = 0;
    const lua_Integer v = lua_tointegerx(L, -1, &is_int);
    if (!is_int)
      report(r, at(w, sizeof w, path, key), "must be a whole number, got %g",
        lua_tonumber(L, -1));
    else if (v < lo || v > hi)
      report(r, at(w, sizeof w, path, key),
        "must be between %lld and %lld, got %lld", (long long)lo, (long long)hi,
        (long long)v);
    else {
      *out = v;
      given = true;
    }
  } else if (type != LUA_TNIL) {
    type_error(r, path, key, "a whole number", type);
  }
  lua_pop(L, 1);
  return given;
}

/* A count that may also be "auto", which reads as 0. */
static bool get_count(reader_t* r, int t, const char* path, const char* key,
  int64_t lo, int64_t hi, uint32_t* out) {

  lua_State* L = r->L;
  char w[160];

  const int type = field(L, t, key);
  if (type == LUA_TSTRING) {
    const bool is_auto = strcmp(lua_tostring(L, -1), "auto") == 0;
    if (is_auto)
      *out = 0;
    else
      report(r, at(w, sizeof w, path, key),
        "must be a whole number or \"auto\", got \"%s\"", lua_tostring(L, -1));
    lua_pop(L, 1);
    return is_auto;
  }
  lua_pop(L, 1);

  int64_t v;
  if (!get_integer(r, t, path, key, lo, hi, &v))
    return false;
  *out = (uint32_t)v;
  return true;
}

static bool get_bool(
  reader_t* r, int t, const char* path, const char* key, bool* out) {

  lua_State* L = r->L;
  bool given = false;

  const int type = field(L, t, key);
  if (type == LUA_TBOOLEAN) {
    *out = lua_toboolean(L, -1);
    given = true;
  } else if (type != LUA_TNIL) {
    type_error(r, path, key, "true or false", type);
  }
  lua_pop(L, 1);
  return given;
}

/* The string stays valid while the table holding it does, which is until
 * the state is closed; callers copy it before that. */
static bool get_string(
  reader_t* r, int t, const char* path, const char* key, const char** out) {

  lua_State* L = r->L;
  bool given = false;

  const int type = field(L, t, key);
  if (type == LUA_TSTRING) {
    *out = lua_tostring(L, -1);
    given = true;
  } else if (type != LUA_TNIL) {
    type_error(r, path, key, "a string", type);
  }
  lua_pop(L, 1);
  return given;
}

static bool get_enum(reader_t* r, int t, const char* path, const char* key,
  const char* const* names, int* out) {

  char w[160], all[256];
  const char* s;

  if (!get_string(r, t, path, key, &s))
    return false;

  const int i = list_index(s, names);
  if (i >= 0) {
    *out = i;
    return true;
  }

  list_names(all, sizeof all, names);
  const char* guess = closest(s, names);
  if (guess)
    report(r, at(w, sizeof w, path, key),
      "must be one of %s; got \"%s\" (did you mean %s?)", all, s, guess);
  else
    report(
      r, at(w, sizeof w, path, key), "must be one of %s; got \"%s\"", all, s);
  return false;
}

/* --- the sections ------------------------------------------------------ */

/* A copy of s that the configuration owns, or NULL when out of memory --
 * which the caller reports. */
static char* keep(exodus_config_t* c, const char* s) {
  char** grown =
    realloc(c->_strings, (c->_n_strings + 1) * sizeof(*c->_strings));
  if (!grown)
    return NULL;
  c->_strings = grown;
  char* copy = strdup(s);
  if (copy)
    c->_strings[c->_n_strings++] = copy;
  return copy;
}

static void set_defaults(exodus_config_t* c) {
  exodus_params_t* p = &c->params;
  memset(p, 0, sizeof *p);

  p->input.kind = EXODUS_INPUT_XFIELD;
  p->input.columns = "x y z";
  p->input.delimiter = ' ';
  p->input.binary.layout = SIF_BINARY_ROWS;
  p->input.binary.precision = SIF_BINARY_FLOAT32;
  p->input.binary.endian = SIF_BINARY_NATIVE;
  p->input.gadget.format = SIF_GADGET_FORMAT_AUTO;
  p->input.gadget.ptype = SIF_GADGET_PTYPE_1;
  p->input.gadget.length = SIF_GADGET_LENGTH_AUTO;
  p->input.gadget.fraction = 1.0;
  p->input.fits.fraction = 1.0;
  p->input.hdf5.fraction = 1.0;
  p->input.hdf5.length_scale = 1.0;

  p->finder.radii_units = EXODUS_RADII_PHYSICAL;
  p->finder.overlap_fraction = 0.0;
  p->finder.search_factor = 1.5;
  p->output.kind = EXODUS_OUTPUT_HDF5;

  c->run.n_threads = 0;
  c->run.log_level = SIF_LOG_LEVEL_INFO;
  c->run.tune_fft = true;
}

/* input.ascii or input.binary: the column format, and velocities refused
 * here as well as in the pipeline, since here the message can say where. */
static void read_columns(
  reader_t* r, int t, const char* path, exodus_config_t* c) {

  char w[160];
  const char* columns;
  if (!get_string(r, t, path, "columns", &columns))
    return;
  if (strchr(columns, 'v') || strchr(columns, 'V')) {
    report(r, at(w, sizeof w, path, "columns"),
      "\"%s\" reads velocities, which the finder never uses; skip them with *",
      columns);
    return;
  }
  /* Sky coordinates need the survey finder, which this program does not run
   * yet: a box is filled with positions. In the column language only ra and
   * dec have an r or a d in them. */
  if (strpbrk(columns, "rRdD")) {
    report(r, at(w, sizeof w, path, "columns"),
      "\"%s\" reads sky coordinates (ra dec z); sif-exodus finds voids in a "
      "box, from positions x y z",
      columns);
    return;
  }
  c->params.input.columns = keep(c, columns);
}

static void read_ascii(reader_t* r, int t, exodus_config_t* c) {
  const char* path = "input.ascii";
  char w[160];
  const char* delim;
  int64_t v;

  check_keys(r, t, path, ASCII_KEYS);
  read_columns(r, t, path, c);

  if (get_string(r, t, path, "delimiter", &delim)) {
    if (strlen(delim) != 1)
      report(r, at(w, sizeof w, path, "delimiter"),
        "must be one character, such as \" \", \",\" or \"\\t\"; got \"%s\"",
        delim);
    else
      c->params.input.delimiter = delim[0];
  }
  if (get_integer(r, t, path, "skip_header", 0, UINT32_MAX, &v))
    c->params.input.skip_header = (uint32_t)v;
}

static void read_binary(reader_t* r, int t, exodus_config_t* c) {
  const char* path = "input.binary";
  int i;
  int64_t v;

  check_keys(r, t, path, BINARY_KEYS);
  read_columns(r, t, path, c);

  if (get_enum(r, t, path, "precision", PRECISIONS, &i))
    c->params.input.binary.precision = SIF_BINARY_FLOAT32 + i;
  else if (field(r->L, t, "precision") == LUA_TNIL)
    report(r, "input.binary.precision",
      "required: \"float32\" or \"float64\", the file does not say");
  lua_settop(r->L, t);

  if (get_enum(r, t, path, "layout", LAYOUTS, &i))
    c->params.input.binary.layout = SIF_BINARY_ROWS + i;
  if (get_enum(r, t, path, "endian", ENDIANS, &i))
    c->params.input.binary.endian = SIF_BINARY_NATIVE + i;
  if (get_integer(r, t, path, "header_bytes", 0, INT64_MAX, &v))
    c->params.input.binary.header_bytes = (uint64_t)v;
}

static void read_gadget(reader_t* r, int t, exodus_config_t* c) {
  const char* path = "input.gadget";
  lua_State* L = r->L;
  char w[160];
  int i;
  int64_t v;
  double x;
  bool b;

  check_keys(r, t, path, GADGET_KEYS);

  /* SnapFormat, as GADGET's own parameter file spells it: 1, 2 or 3 (HDF5),
   * or "auto" to read it from the file. */
  const int type = field(L, t, "snapformat");
  if (type == LUA_TSTRING && strcmp(lua_tostring(L, -1), "auto") == 0) {
    c->params.input.gadget.format = SIF_GADGET_FORMAT_AUTO;
  } else if (type == LUA_TSTRING && strcmp(lua_tostring(L, -1), "hdf5") == 0) {
    c->params.input.gadget.format = SIF_GADGET_FORMAT_HDF5;
  } else if (type == LUA_TNUMBER && lua_isinteger(L, -1) &&
             lua_tointeger(L, -1) >= 1 && lua_tointeger(L, -1) <= 3) {
    c->params.input.gadget.format =
      SIF_GADGET_FORMAT_1 + (int)(lua_tointeger(L, -1) - 1);
  } else if (type != LUA_TNIL) {
    report(r, at(w, sizeof w, path, "snapformat"),
      "must be 1, 2, 3 (HDF5) or \"auto\"");
  }
  lua_pop(L, 1);

  if (get_integer(r, t, path, "ptype", 0, 5, &v))
    c->params.input.gadget.ptype = SIF_GADGET_PTYPE_0 + (int)v;
  if (get_enum(r, t, path, "length", LENGTHS, &i))
    c->params.input.gadget.length = SIF_GADGET_LENGTH_KPC + i;
  if (get_bool(r, t, path, "masses", &b))
    c->params.input.gadget.masses = b;
  if (get_number(r, t, path, "fraction", &x)) {
    if (x > 0.0 && x <= 1.0)
      c->params.input.gadget.fraction = x;
    else
      report(
        r, at(w, sizeof w, path, "fraction"), "must be in (0, 1], got %g", x);
  }
  if (get_integer(r, t, path, "seed", 0, INT64_MAX, &v))
    c->params.input.gadget.seed = (uint64_t)v;
}

/* input.fits.columns or input.hdf5.columns: what fills x, y, z and,
 * optionally, the weight -- a FITS column or expression, an HDF5 dataset. */
static void read_named_columns(
  reader_t* r, int t, const char* section, exodus_config_t* c) {
  char path[64];
  snprintf(path, sizeof path, "%s.columns", section);
  lua_State* L = r->L;
  sif_field_columns_t* cols = &c->params.input.named;
  const char* const* keys = FITS_COLUMN_KEYS;
  const char** slots[] = {&cols->x, &cols->y, &cols->z, &cols->w};

  const int type = field(L, t, "columns");
  const int ct = lua_gettop(L);
  if (type == LUA_TNIL) {
    report(r, path,
      "missing: the columns to read, as columns = { x = \"X\", y = \"Y\", "
      "z = \"Z\" }");
  } else if (type != LUA_TTABLE) {
    type_error(r, section, "columns", "a table", type);
  } else {
    /* Velocities have no key: the finder never reads them. */
    lua_pushnil(L);
    while (lua_next(L, ct)) {
      const char* key =
        lua_type(L, -2) == LUA_TSTRING ? lua_tostring(L, -2) : NULL;
      if (key && (strcmp(key, "vx") == 0 || strcmp(key, "vy") == 0 ||
                   strcmp(key, "vz") == 0)) {
        char w[160];
        report(r, at(w, sizeof w, path, key),
          "velocities are not read: the finder never uses them");
      } else if (key && (strcmp(key, "ra") == 0 || strcmp(key, "dec") == 0)) {
        char w[160];
        report(r, at(w, sizeof w, path, key),
          "sky coordinates are not read: sif-exodus finds voids in a box, "
          "from positions x y z");
      }
      lua_pop(L, 1);
    }
    lua_settop(L, ct);
    check_keys(r, ct, path, FITS_COLUMN_REFUSED);

    for (int i = 0; keys[i]; i++) {
      const char* v;
      char w[160];
      if (get_string(r, ct, path, keys[i], &v)) {
        if (!*v)
          report(r, at(w, sizeof w, path, keys[i]), "is empty");
        else if (!(*slots[i] = keep(c, v)))
          report(r, at(w, sizeof w, path, keys[i]), "out of memory");
      } else if (i < 3 && field(L, ct, keys[i]) == LUA_TNIL) {
        report(r, at(w, sizeof w, path, keys[i]),
          "missing: all three positions are required");
      }
      lua_settop(L, ct);
    }
  }
  lua_settop(L, ct - 1);
}

static void read_fits(reader_t* r, int t, exodus_config_t* c) {
  const char* path = "input.fits";
  lua_State* L = r->L;
  char w[160];
  const char* v;
  int64_t n;
  double x;

  check_keys(r, t, path, FITS_KEYS);
  read_named_columns(r, t, path, c);

  if (get_string(r, t, path, "where", &v) && *v)
    c->params.input.fits.where = keep(c, v);

  /* The table: its EXTNAME, or its extension number, 1 for the first. */
  const int type = field(L, t, "hdu");
  if (type == LUA_TSTRING && *lua_tostring(L, -1)) {
    c->params.input.fits.hdu = keep(c, lua_tostring(L, -1));
  } else if (type == LUA_TNUMBER && lua_isinteger(L, -1) &&
             lua_tointeger(L, -1) >= 0) {
    char num[32];
    snprintf(num, sizeof num, "%lld", (long long)lua_tointeger(L, -1));
    c->params.input.fits.hdu = keep(c, num);
  } else if (type != LUA_TNIL) {
    report(r, at(w, sizeof w, path, "hdu"),
      "must be the table's EXTNAME or its extension number (1 for the "
      "first)");
  }
  lua_settop(L, t);

  if (get_number(r, t, path, "fraction", &x)) {
    if (x > 0.0 && x <= 1.0)
      c->params.input.fits.fraction = x;
    else
      report(
        r, at(w, sizeof w, path, "fraction"), "must be in (0, 1], got %g", x);
  }
  if (get_integer(r, t, path, "seed", 0, INT64_MAX, &n))
    c->params.input.fits.seed = (uint64_t)n;
}

/* input.hdf5: datasets of any HDF5 file. */
static void read_hdf5_input(reader_t* r, int t, exodus_config_t* c) {
  const char* path = "input.hdf5";
  char w[160];
  int64_t n;
  double x;

  check_keys(r, t, path, HDF5_KEYS);
  read_named_columns(r, t, path, c);

  if (get_number(r, t, path, "length_scale", &x)) {
    if (x > 0.0)
      c->params.input.hdf5.length_scale = x;
    else
      report(r, at(w, sizeof w, path, "length_scale"),
        "must be positive, got %g", x);
  }
  if (get_number(r, t, path, "fraction", &x)) {
    if (x > 0.0 && x <= 1.0)
      c->params.input.hdf5.fraction = x;
    else
      report(
        r, at(w, sizeof w, path, "fraction"), "must be in (0, 1], got %g", x);
  }
  if (get_integer(r, t, path, "seed", 0, INT64_MAX, &n))
    c->params.input.hdf5.seed = (uint64_t)n;
}

/* input.path as a list of files, which only FITS and HDF5 inputs read.
 * Returns the first, or NULL after reporting what is wrong. */
static const char* read_path_list(reader_t* r, int t, exodus_config_t* c) {
  lua_State* L = r->L;
  field(L, t, "path");
  const int lt = lua_gettop(L);
  const char* first = NULL;
  char w[160];

  const lua_Integer n = (lua_Integer)lua_rawlen(L, lt);
  if (n == 0 || n > UINT32_MAX) {
    report(r, "input.path", "an empty list: name the files to read");
  } else if (!(c->_paths = calloc((size_t)n, sizeof(char*)))) {
    report(r, "input.path", "out of memory");
  } else {
    bool ok = true;
    for (lua_Integer i = 1; i <= n; i++) {
      const int et = lua_rawgeti(L, lt, i);
      snprintf(w, sizeof w, "input.path[%lld]", (long long)i);
      if (et != LUA_TSTRING || !*lua_tostring(L, -1)) {
        report(r, w, "must be a file name");
        ok = false;
      } else if (!(c->_paths[i - 1] = keep(c, lua_tostring(L, -1)))) {
        report(r, w, "out of memory");
        ok = false;
      }
      lua_pop(L, 1);
    }
    if (ok) {
      c->params.input.paths = c->_paths;
      c->params.input.n_paths = (uint32_t)n;
      first = c->_paths[0];
    }
  }
  lua_settop(L, t);
  return first;
}

static bool is_fits_name(const char* path) {
  return ends_with(path, ".fits") || ends_with(path, ".fit") ||
         ends_with(path, ".fits.gz") || ends_with(path, ".fit.gz");
}

static void read_input(reader_t* r, int root, exodus_config_t* c) {
  lua_State* L = r->L;
  exodus_params_t* p = &c->params;
  const int base = lua_gettop(L);
  const char* path = NULL;
  int format = -1;
  int t = 0;
  char all[128], w[160];

  bool path_list = false;
  const int type = field(L, root, "input");
  if (type == LUA_TSTRING) {
    path = lua_tostring(L, -1);
  } else if (type == LUA_TTABLE) {
    t = lua_gettop(L);
    check_keys(r, t, "input", INPUT_KEYS);
    const int path_type = field(L, t, "path");
    lua_settop(L, t);
    if (path_type == LUA_TTABLE) {
      path_list = true;
      path = read_path_list(r, t, c);
    } else if (!get_string(r, t, "input", "path", &path) &&
               path_type == LUA_TNIL) {
      report(r, "input.path", "missing: the file to read");
    }
    lua_settop(L, t);
    get_enum(r, t, "input", "format", INPUT_FORMATS, &format);
  } else if (type == LUA_TNIL) {
    report(r, "input",
      "missing: the tracers, as input = \"file.xfield\" or input = { path = "
      "..., format = ... }");
  } else {
    report(r, "input", "must be a file name or a table, got a %s",
      lua_typename(L, type));
  }

  if (path && !path_list)
    p->input.path = keep(c, path);
  else if (path)
    p->input.path = path; /* kept already, as the list's first */

  /* Only sif's own files, and FITS files, say what they are by their
   * name: a .dat or a .bin could be anything. */
  const bool format_given = t && field(L, t, "format") != LUA_TNIL;
  lua_settop(L, t ? t : base + 1);
  if (format < 0 && path && !format_given) {
    if (ends_with(path, ".xfield")) {
      format = EXODUS_INPUT_XFIELD;
    } else if (is_fits_name(path)) {
      format = EXODUS_INPUT_FITS;
    } else {
      list_names(all, sizeof all, INPUT_FORMATS);
      report(r, "input.format",
        "missing: the format of \"%s\" cannot be told from its name (%s)", path,
        all);
    }
  }
  if (format >= 0)
    p->input.kind = (exodus_input_kind_t)format;

  const bool named = format >= 0 && (p->input.kind == EXODUS_INPUT_FITS ||
                                      p->input.kind == EXODUS_INPUT_HDF5);
  if (path_list && format >= 0 && !named)
    report(r, "input.path",
      "a list of files, which only the fits and hdf5 formats read; %s reads "
      "one",
      INPUT_FORMATS[p->input.kind]);

  /* A single FITS or HDF5 file is a list of one. */
  if (named && !path_list && p->input.path) {
    c->_paths = malloc(sizeof(char*));
    if (c->_paths) {
      c->_paths[0] = p->input.path;
      p->input.paths = c->_paths;
      p->input.n_paths = 1;
    } else {
      report(r, "input.path", "out of memory");
    }
  }

  if (t) {
    double box;
    if (get_number(r, t, "input", "box_length", &box)) {
      if (box > 0.0)
        p->input.box_length = box;
      else
        report(r, "input.box_length", "must be positive, got %g", box);
    }

    /* One sub-table per format, and only the one the format reads. */
    static const char* const subs[] = {
      "ascii", "binary", "gadget", "fits", "hdf5"};
    static const exodus_input_kind_t kinds[] = {EXODUS_INPUT_ASCII,
      EXODUS_INPUT_BINARY, EXODUS_INPUT_GADGET, EXODUS_INPUT_FITS,
      EXODUS_INPUT_HDF5};
    for (int i = 0; i < 5; i++) {
      const int st = field(L, t, subs[i]);
      const int s = lua_gettop(L);
      if (st == LUA_TTABLE) {
        if (format >= 0 && p->input.kind != kinds[i])
          report(r, at(w, sizeof w, "input", subs[i]),
            "given, but input.format is %s", INPUT_FORMATS[p->input.kind]);
        else if (kinds[i] == EXODUS_INPUT_ASCII)
          read_ascii(r, s, c);
        else if (kinds[i] == EXODUS_INPUT_BINARY)
          read_binary(r, s, c);
        else if (kinds[i] == EXODUS_INPUT_GADGET)
          read_gadget(r, s, c);
        else if (kinds[i] == EXODUS_INPUT_FITS)
          read_fits(r, s, c);
        else
          read_hdf5_input(r, s, c);
      } else if (st != LUA_TNIL) {
        type_error(r, "input", subs[i], "a table", st);
      } else if (format >= 0 && p->input.kind == EXODUS_INPUT_BINARY &&
                 kinds[i] == EXODUS_INPUT_BINARY) {
        report(r, "input.binary",
          "missing: a binary file needs at least its precision, as binary = "
          "{ precision = \"float32\" }");
      } else if (format >= 0 && p->input.kind == EXODUS_INPUT_FITS &&
                 kinds[i] == EXODUS_INPUT_FITS) {
        report(r, "input.fits",
          "missing: a FITS table needs its columns named, as fits = { "
          "columns = { x = \"X\", y = \"Y\", z = \"Z\" } }");
      } else if (format >= 0 && p->input.kind == EXODUS_INPUT_HDF5 &&
                 kinds[i] == EXODUS_INPUT_HDF5) {
        report(r, "input.hdf5",
          "missing: an HDF5 file needs its datasets named, as hdf5 = { "
          "columns = { x = \"Group/Pos[0]\", ... } }");
      }
      lua_settop(L, t);
    }
  } else if (format >= 0 && p->input.kind == EXODUS_INPUT_BINARY) {
    report(r, "input.binary", "missing: a binary file needs its precision");
  } else if (format >= 0 && p->input.kind == EXODUS_INPUT_FITS) {
    report(r, "input.fits", "missing: a FITS table needs its columns named");
  } else if (format >= 0 && p->input.kind == EXODUS_INPUT_HDF5) {
    report(r, "input.hdf5", "missing: an HDF5 file needs its datasets named");
  }

  if (format >= 0 &&
      (p->input.kind == EXODUS_INPUT_ASCII ||
        p->input.kind == EXODUS_INPUT_BINARY ||
        p->input.kind == EXODUS_INPUT_FITS ||
        p->input.kind == EXODUS_INPUT_HDF5) &&
      !(p->input.box_length > 0.0))
    report(r, "input.box_length",
      "missing: %s files do not record their box, so the configuration "
      "has to",
      INPUT_FORMATS[p->input.kind]);

  lua_settop(L, base);
}

/* grid and mesh have one setting each, and the same one. */
static void read_cells(reader_t* r, int root, const char* name,
  const char* const* keys, int64_t min, uint32_t* n_cells) {

  lua_State* L = r->L;
  const int type = field(L, root, name);
  const int t = lua_gettop(L);
  if (type == LUA_TTABLE) {
    check_keys(r, t, name, keys);
    get_count(r, t, name, "n_cells", min, 1 << 16, n_cells);
  } else if (type != LUA_TNIL) {
    report(r, name, "must be a table, got a %s", lua_typename(L, type));
  }
  lua_settop(L, t - 1);
}

/* finder.radii: a number, or a list of them. */
static void read_radii(reader_t* r, int t, exodus_config_t* c) {
  lua_State* L = r->L;
  const int type = field(L, t, "radii");
  const int v = lua_gettop(L);
  lua_Integer n = 0;

  if (type == LUA_TNIL) {
    report(r, "finder.radii",
      "missing: the radius ladder, e.g. radii = ladder(10, 40, 0.05)");
  } else if (type == LUA_TNUMBER) {
    n = 1;
  } else if (type == LUA_TTABLE) {
    n = (lua_Integer)lua_rawlen(L, v);
    if (n == 0)
      report(r, "finder.radii", "is empty");
    else if (n > MAX_RADII)
      report(r, "finder.radii", "has %lld radii, more than %d", (long long)n,
        MAX_RADII);
  } else {
    report(r, "finder.radii", "must be a number or a list of numbers, got a %s",
      lua_typename(L, type));
  }

  if (n > 0 && n <= MAX_RADII) {
    c->_radii = malloc((size_t)n * sizeof(sif_real));
    if (!c->_radii) {
      report(r, "finder.radii", "out of memory");
      n = 0;
    }
  }

  int bad = 0;
  for (lua_Integer i = 0; c->_radii && i < n; i++) {
    if (type == LUA_TTABLE)
      lua_rawgeti(L, v, i + 1);
    else
      lua_pushvalue(L, v);
    const double x = lua_tonumber(L, -1);
    if (lua_type(L, -1) != LUA_TNUMBER || !isfinite(x) || !(x > 0.0)) {
      if (bad++ == 0)
        report(r, "finder.radii",
          "entry %lld is %s; every radius must be a positive number",
          (long long)(i + 1), luaL_tolstring(L, -1, NULL));
    } else {
      c->_radii[i] = (sif_real)x;
    }
    lua_settop(L, v);
  }

  if (c->_radii && bad == 0) {
    c->params.finder.radii = c->_radii;
    c->params.finder.n_radii = (uint32_t)n;
  }
  lua_settop(L, v - 1);
}

static void read_finder(reader_t* r, int root, exodus_config_t* c) {
  lua_State* L = r->L;
  exodus_params_t* p = &c->params;
  int i;
  double x;

  const int type = field(L, root, "finder");
  const int t = lua_gettop(L);
  if (type != LUA_TTABLE) {
    if (type == LUA_TNIL)
      report(r, "finder",
        "missing: at least the radii and the threshold, as finder = { radii "
        "= ladder(10, 40, 0.05), threshold = -0.7 }");
    else
      report(r, "finder", "must be a table, got a %s", lua_typename(L, type));
    lua_settop(L, t - 1);
    return;
  }

  check_keys(r, t, "finder", FINDER_KEYS);
  read_radii(r, t, c);

  if (get_enum(r, t, "finder", "radii_units", RADII_UNITS, &i))
    p->finder.radii_units = (exodus_radii_units_t)i;

  if (get_number(r, t, "finder", "threshold", &x)) {
    if (x > -1.0 && x < 0.0)
      p->finder.threshold = x;
    else
      report(r, "finder.threshold",
        "must be a density contrast between -1 and 0, got %g", x);
  } else if (field(L, t, "threshold") == LUA_TNIL) {
    report(r, "finder.threshold",
      "missing: the density contrast a void is grown to, e.g. -0.7");
  }
  lua_settop(L, t);

  if (get_number(r, t, "finder", "overlap_fraction", &x)) {
    if (x >= 0.0 && x <= 1.0)
      p->finder.overlap_fraction = x;
    else
      report(r, "finder.overlap_fraction", "must be in [0, 1], got %g", x);
  }
  if (get_number(r, t, "finder", "search_factor", &x)) {
    if (x == 1.25 || x == 1.5 || x == 1.75 || x == 2.0)
      p->finder.search_factor = x;
    else
      report(
        r, "finder.search_factor", "must be 1.25, 1.5, 1.75 or 2, got %g", x);
  }
  lua_settop(L, t - 1);
}

static void read_output(reader_t* r, int root, exodus_config_t* c) {
  lua_State* L = r->L;
  exodus_params_t* p = &c->params;
  const char* path = NULL;
  int format = -1;

  const int type = field(L, root, "output");
  const int t = lua_gettop(L);
  if (type == LUA_TSTRING) {
    path = lua_tostring(L, -1);
  } else if (type == LUA_TTABLE) {
    check_keys(r, t, "output", OUTPUT_KEYS);
    if (!get_string(r, t, "output", "path", &path) &&
        field(L, t, "path") == LUA_TNIL)
      report(r, "output.path", "missing: the file to write");
    lua_settop(L, t);
    get_enum(r, t, "output", "format", OUTPUT_FORMATS, &format);
  } else if (type == LUA_TNIL) {
    report(r, "output",
      "missing: where the catalogue goes, as output = \"voids.h5\"");
  } else {
    report(r, "output", "must be a file name or a table, got a %s",
      lua_typename(L, type));
  }

  if (path) {
    p->output.path = keep(c, path);
    if (format < 0 && (ends_with(path, ".h5") || ends_with(path, ".hdf5") ||
                        ends_with(path, ".he5")))
      format = EXODUS_OUTPUT_HDF5;
    else if (format < 0 &&
             (ends_with(path, ".fits") || ends_with(path, ".fit")))
      format = EXODUS_OUTPUT_FITS;
    else if (format < 0)
      format = EXODUS_OUTPUT_ASCII;
  }
  if (format >= 0)
    p->output.kind = (exodus_output_kind_t)format;

#if !defined(SIF_HAVE_HDF5)
  /* The library would write text next to the path instead, which is not the
   * file asked for; better to say so now. */
  if (path && p->output.kind == EXODUS_OUTPUT_HDF5)
    report(r, type == LUA_TSTRING ? "output" : "output.format",
      "this sif-exodus was built without HDF5: write an ASCII catalogue "
      "(e.g. \"voids.txt\"), or rebuild with SIF_HDF5_SUPPORT=ON");
#endif
#if !defined(SIF_HAVE_FITS)
  if (path && p->output.kind == EXODUS_OUTPUT_FITS)
    report(r, type == LUA_TSTRING ? "output" : "output.format",
      "this sif-exodus was built without FITS support: write an HDF5 or an "
      "ASCII catalogue, or rebuild with SIF_FITS_SUPPORT=ON");
#endif

  lua_settop(L, t - 1);
}

static void read_run(reader_t* r, int root, exodus_config_t* c) {
  lua_State* L = r->L;
  int i;
  bool b;

  const int type = field(L, root, "run");
  const int t = lua_gettop(L);
  if (type == LUA_TTABLE) {
    check_keys(r, t, "run", RUN_KEYS);
    get_count(r, t, "run", "threads", 1, 1 << 20, &c->run.n_threads);
    if (get_enum(r, t, "run", "log_level", LOG_LEVELS, &i))
      c->run.log_level = (uint8_t)(SIF_LOG_LEVEL_TRACE + i);
    if (get_bool(r, t, "run", "tune_fft", &b))
      c->run.tune_fft = b;
  } else if (type != LUA_TNIL) {
    report(r, "run", "must be a table, got a %s", lua_typename(L, type));
  }
  lua_settop(L, t - 1);
}

/* A name the script left at the top level that is not a section. Its own
 * variables are its business -- `h = 0.6774` is fine -- but a setting
 * written outside its section, or a misspelt section, would otherwise be
 * silently ignored. */
static void check_top_level(
  reader_t* r, int root, int preset, const config_define_t* d, size_t n_d) {

  lua_State* L = r->L;

  lua_pushnil(L);
  while (lua_next(L, root)) {
    if (lua_type(L, -2) == LUA_TSTRING) {
      const char* k = lua_tostring(L, -2);
      bool skip = list_index(k, SECTIONS) >= 0;

      lua_pushvalue(L, -2);
      skip = skip || lua_rawget(L, preset) != LUA_TNIL;
      lua_pop(L, 1);
      for (size_t i = 0; !skip && i < n_d; i++)
        skip = strcmp(k, d[i].name) == 0;

      for (size_t s = 0;
        !skip && s < sizeof(SECTION_KEYS) / sizeof(SECTION_KEYS[0]); s++) {
        if (list_index(k, SECTION_KEYS[s].keys) >= 0) {
          report(r, k, "is a setting of %s: write %s = { %s = ... }",
            SECTION_KEYS[s].section, SECTION_KEYS[s].section, k);
          skip = true;
        }
      }

      const char* guess = skip ? NULL : closest(k, SECTIONS);
      if (guess)
        report(r, k, "is not a section (did you mean %s?)", guess);
    }
    lua_pop(L, 1);
  }
}

/* --- the sandbox ------------------------------------------------------- */

/* ladder(r_min, r_max, step): r_min, r_min (1 + step), ... up to r_max. */
static int lua_ladder(lua_State* L) {
  const double lo = luaL_checknumber(L, 1);
  const double hi = luaL_checknumber(L, 2);
  const double step = luaL_checknumber(L, 3);
  luaL_argcheck(L, isfinite(lo) && lo > 0.0, 1, "r_min must be positive");
  luaL_argcheck(L, isfinite(hi) && hi >= lo, 2, "r_max must be at least r_min");
  luaL_argcheck(L, isfinite(step) && step > 0.0, 3,
    "the step must be positive: 0.05 for 5% between rungs");

  const double ratio = 1.0 + step;
  const double count = floor(log(hi / lo) / log(ratio) + 1e-9) + 1.0;
  luaL_argcheck(L, count <= MAX_RADII, 3, "too many rungs");

  const int n = (int)count;
  lua_createtable(L, n, 0);
  for (int i = 0; i < n; i++) {
    lua_pushnumber(L, lo * pow(ratio, i));
    lua_rawseti(L, -2, i + 1);
  }
  return 1;
}

/* linspace(a, b, n): n values evenly spaced from a to b, both included. */
static int lua_linspace(lua_State* L) {
  const double a = luaL_checknumber(L, 1);
  const double b = luaL_checknumber(L, 2);
  const lua_Integer n = luaL_checkinteger(L, 3);
  luaL_argcheck(L, n >= 1 && n <= MAX_RADII, 3, "out of range");

  lua_createtable(L, (int)n, 0);
  for (lua_Integer i = 0; i < n; i++) {
    lua_pushnumber(L, n == 1 ? a : a + (b - a) * (double)i / (double)(n - 1));
    lua_rawseti(L, -2, i + 1);
  }
  return 1;
}

/* geomspace(a, b, n): n values evenly spaced in log from a to b, both
 * included -- numpy's geomspace, not its logspace. */
static int lua_geomspace(lua_State* L) {
  const double a = luaL_checknumber(L, 1);
  const double b = luaL_checknumber(L, 2);
  const lua_Integer n = luaL_checkinteger(L, 3);
  luaL_argcheck(L, a > 0.0, 1, "must be positive");
  luaL_argcheck(L, b > 0.0, 2, "must be positive");
  luaL_argcheck(L, n >= 1 && n <= MAX_RADII, 3, "out of range");

  lua_createtable(L, (int)n, 0);
  for (lua_Integer i = 0; i < n; i++) {
    const double f = n == 1 ? 0.0 : (double)i / (double)(n - 1);
    lua_pushnumber(L, a * pow(b / a, f));
    lua_rawseti(L, -2, i + 1);
  }
  return 1;
}

/* Push the environment the script runs in: the parts of the standard
 * library a computation needs, and the helpers above. */
static void push_sandbox(lua_State* L) {
  lua_newtable(L);
  const int env = lua_gettop(L);

  luaL_requiref(L, LUA_GNAME, luaopen_base, 0);
  for (int i = 0; BASE_FUNCTIONS[i]; i++) {
    lua_getfield(L, -1, BASE_FUNCTIONS[i]);
    lua_setfield(L, env, BASE_FUNCTIONS[i]);
  }
  lua_pop(L, 1);

  luaL_requiref(L, LUA_MATHLIBNAME, luaopen_math, 0);
  lua_setfield(L, env, LUA_MATHLIBNAME);
  luaL_requiref(L, LUA_STRLIBNAME, luaopen_string, 0);
  lua_setfield(L, env, LUA_STRLIBNAME);
  luaL_requiref(L, LUA_TABLIBNAME, luaopen_table, 0);
  lua_setfield(L, env, LUA_TABLIBNAME);
  luaL_requiref(L, LUA_UTF8LIBNAME, luaopen_utf8, 0);
  lua_setfield(L, env, LUA_UTF8LIBNAME);

  /* os, read-only: the environment and the clock. */
  luaL_requiref(L, LUA_OSLIBNAME, luaopen_os, 0);
  lua_newtable(L);
  for (int i = 0; OS_FUNCTIONS[i]; i++) {
    lua_getfield(L, -2, OS_FUNCTIONS[i]);
    lua_setfield(L, -2, OS_FUNCTIONS[i]);
  }
  lua_setfield(L, env, LUA_OSLIBNAME);
  lua_pop(L, 1);

  lua_pushcfunction(L, lua_ladder);
  lua_setfield(L, env, "ladder");
  lua_pushcfunction(L, lua_linspace);
  lua_setfield(L, env, "linspace");
  lua_pushcfunction(L, lua_geomspace);
  lua_setfield(L, env, "geomspace");

  lua_pushvalue(L, env);
  lua_setfield(L, env, LUA_GNAME);
}

/* --- the resolved configuration ---------------------------------------- */

/* The columns = { ... } line of a fits or hdf5 table. */
static void write_named_columns(FILE* out, const sif_field_columns_t* cols) {
  fputs("    columns = { x = ", out);
  write_lua_string(out, cols->x);
  fputs(", y = ", out);
  write_lua_string(out, cols->y);
  fputs(", z = ", out);
  write_lua_string(out, cols->z);
  if (cols->w) {
    fputs(", w = ", out);
    write_lua_string(out, cols->w);
  }
  fputs(" },\n", out);
}

static void write_resolved(FILE* out, const exodus_config_t* c,
  const char* file, const config_define_t* d, size_t n_d) {

  const exodus_params_t* p = &c->params;
  char num[40];

  fprintf(out, "-- sif-exodus %s: %s as it resolved", SIF_VERSION_STRING, file);
  for (size_t i = 0; i < n_d; i++)
    fprintf(out, "%s -D %s=%s", i ? "" : ", with", d[i].name, d[i].value);
  fputs(".\n-- Every setting, defaults included; runs as a configuration of "
        "its own.\n\n",
    out);

  fputs("input = {\n  path = ", out);
  if ((p->input.kind == EXODUS_INPUT_FITS ||
        p->input.kind == EXODUS_INPUT_HDF5) &&
      p->input.n_paths > 1) {
    fputs("{\n", out);
    for (uint32_t i = 0; i < p->input.n_paths; i++) {
      fputs("    ", out);
      write_lua_string(out, p->input.paths[i]);
      fputs(",\n", out);
    }
    fputs("  }", out);
  } else {
    write_lua_string(out, p->input.path);
  }
  fprintf(out, ",\n  format = \"%s\",\n", INPUT_FORMATS[p->input.kind]);
  if (p->input.box_length > 0.0) {
    format_double(num, sizeof num, p->input.box_length);
    fprintf(out, "  box_length = %s,\n", num);
  } else {
    fputs("  -- box_length: from the file\n", out);
  }

  switch (p->input.kind) {
  case EXODUS_INPUT_ASCII: {
    const char delim[2] = {p->input.delimiter, '\0'};
    fputs("  ascii = { columns = ", out);
    write_lua_string(out, p->input.columns);
    fputs(", delimiter = ", out);
    write_lua_string(out, delim);
    fprintf(out, ", skip_header = %u },\n", p->input.skip_header);
    break;
  }
  case EXODUS_INPUT_BINARY:
    fputs("  binary = {\n    columns = ", out);
    write_lua_string(out, p->input.columns);
    fprintf(out,
      ",\n    precision = \"%s\",\n    layout = \"%s\",\n    endian = "
      "\"%s\",\n    header_bytes = %llu,\n  },\n",
      PRECISIONS[p->input.binary.precision - SIF_BINARY_FLOAT32],
      LAYOUTS[p->input.binary.layout - SIF_BINARY_ROWS],
      ENDIANS[p->input.binary.endian - SIF_BINARY_NATIVE],
      (unsigned long long)p->input.binary.header_bytes);
    break;
  case EXODUS_INPUT_GADGET: {
    const sif_gadget_format_t f = p->input.gadget.format;
    fputs("  gadget = {\n    snapformat = ", out);
    if (f == SIF_GADGET_FORMAT_AUTO)
      fputs("\"auto\"", out);
    else
      fprintf(out, "%d", (int)(f - SIF_GADGET_FORMAT_1) + 1);
    format_double(num, sizeof num, p->input.gadget.fraction);
    fprintf(out,
      ",\n    ptype = %d,\n    length = \"%s\",\n    masses = %s,\n    "
      "fraction = %s,\n    seed = %llu,\n  },\n",
      (int)(p->input.gadget.ptype - SIF_GADGET_PTYPE_0),
      LENGTHS[p->input.gadget.length - SIF_GADGET_LENGTH_KPC],
      p->input.gadget.masses ? "true" : "false", num,
      (unsigned long long)p->input.gadget.seed);
    break;
  }
  case EXODUS_INPUT_FITS:
    fputs("  fits = {\n", out);
    write_named_columns(out, &p->input.named);
    if (p->input.fits.where) {
      fputs("    where = ", out);
      write_lua_string(out, p->input.fits.where);
      fputs(",\n", out);
    }
    if (p->input.fits.hdu) {
      fputs("    hdu = ", out);
      write_lua_string(out, p->input.fits.hdu);
      fputs(",\n", out);
    }
    format_double(num, sizeof num, p->input.fits.fraction);
    fprintf(out, "    fraction = %s,\n    seed = %llu,\n  },\n", num,
      (unsigned long long)p->input.fits.seed);
    break;
  case EXODUS_INPUT_HDF5: {
    char scale[40];
    fputs("  hdf5 = {\n", out);
    write_named_columns(out, &p->input.named);
    format_double(scale, sizeof scale, p->input.hdf5.length_scale);
    format_double(num, sizeof num, p->input.hdf5.fraction);
    fprintf(out,
      "    length_scale = %s,\n    fraction = %s,\n    seed = %llu,\n  },\n",
      scale, num, (unsigned long long)p->input.hdf5.seed);
    break;
  }
  case EXODUS_INPUT_XFIELD:
    break;
  }
  fputs("}\n\n", out);

  if (p->grid.n_cells)
    fprintf(out, "grid = { n_cells = %u }\n", p->grid.n_cells);
  else
    fputs("grid = { n_cells = \"auto\" }\n", out);
  if (p->mesh.n_cells)
    fprintf(out, "mesh = { n_cells = %u }\n\n", p->mesh.n_cells);
  else
    fputs("mesh = { n_cells = \"auto\" }\n\n", out);

  fputs("finder = {\n  radii = {", out);
  for (uint32_t i = 0; i < p->finder.n_radii; i++) {
    format_real(num, sizeof num, p->finder.radii[i]);
    fprintf(out, "%s%s%s", i % 6 == 0 ? "\n    " : " ", num,
      i + 1 < p->finder.n_radii ? "," : ",\n  ");
  }
  fprintf(
    out, "},\n  radii_units = \"%s\",\n", RADII_UNITS[p->finder.radii_units]);
  format_double(num, sizeof num, p->finder.threshold);
  fprintf(out, "  threshold = %s,\n", num);
  format_double(num, sizeof num, p->finder.overlap_fraction);
  fprintf(out, "  overlap_fraction = %s,\n", num);
  format_double(num, sizeof num, p->finder.search_factor);
  fprintf(out, "  search_factor = %s,\n}\n\n", num);

  fputs("output = { path = ", out);
  write_lua_string(out, p->output.path);
  fprintf(out, ", format = \"%s\" }\n\n", OUTPUT_FORMATS[p->output.kind]);

  fputs("run = {\n  threads = ", out);
  if (c->run.n_threads)
    fprintf(out, "%u", c->run.n_threads);
  else
    fputs("\"auto\"", out);
  fprintf(out, ",\n  log_level = \"%s\",\n  tune_fft = %s,\n}\n",
    LOG_LEVELS[c->run.log_level - SIF_LOG_LEVEL_TRACE],
    c->run.tune_fft ? "true" : "false");
}

/* --- the entry points -------------------------------------------------- */

int config_load(const char* path, const config_define_t* defines,
  size_t n_defines, exodus_config_t* c) {

  memset(c, 0, sizeof *c);
  set_defaults(c);

  reader_t r = {NULL, path, 0};

  /* -D names are globals in the script, so they have to be usable as such,
   * and must not shadow what the configuration is made of. */
  for (size_t i = 0; i < n_defines; i++) {
    const char* n = defines[i].name;
    if (!is_identifier(n) || list_index(n, LUA_KEYWORDS) >= 0)
      report(&r, "-D", "%s is not a name a script can use", n);
    else if (list_index(n, SECTIONS) >= 0)
      report(&r, "-D", "%s is a section; set it in the file", n);
  }
  if (r.n_errors)
    return -1;

  lua_State* L = luaL_newstate();
  if (!L) {
    fprintf(stderr, "%s: cannot create a Lua state\n", path);
    return -1;
  }
  r.L = L;

  push_sandbox(L);
  const int env = lua_gettop(L);

  /* What the sandbox holds before the script runs, so that what it adds can
   * be told apart. */
  lua_newtable(L);
  const int preset = lua_gettop(L);
  lua_pushnil(L);
  while (lua_next(L, env)) {
    lua_pop(L, 1);
    lua_pushvalue(L, -1);
    lua_pushboolean(L, 1);
    lua_rawset(L, preset);
  }

  for (size_t i = 0; i < n_defines; i++) {
    lua_pushstring(L, defines[i].value);
    lua_setfield(L, env, defines[i].name);
  }

  /* Text only: a precompiled chunk could do what the sandbox forbids. */
  if (luaL_loadfilex(L, path, "t") != LUA_OK) {
    fprintf(stderr, "%s\n", lua_tostring(L, -1));
    lua_close(L);
    return -1;
  }
  lua_pushvalue(L, env);
  lua_setupvalue(L, -2, 1);

  if (lua_pcall(L, 0, 1, 0) != LUA_OK) {
    fprintf(stderr, "%s\n", luaL_tolstring(L, -1, NULL));
    lua_close(L);
    return -1;
  }

  /* A script may also return its configuration as one table, in which case
   * nothing else is read and every key has to be a section. */
  int root = env;
  if (lua_type(L, -1) == LUA_TTABLE) {
    root = lua_gettop(L);
    check_keys(&r, root, "", SECTIONS);
  } else if (lua_type(L, -1) != LUA_TNIL) {
    report(&r, NULL, "the script returned a %s; return nothing, or a table",
      luaL_typename(L, -1));
  } else {
    check_top_level(&r, env, preset, defines, n_defines);
  }

  read_input(&r, root, c);
  read_cells(&r, root, "grid", GRID_KEYS, 8, &c->params.grid.n_cells);
  read_cells(&r, root, "mesh", MESH_KEYS, 1, &c->params.mesh.n_cells);
  read_finder(&r, root, c);
  read_output(&r, root, c);
  read_run(&r, root, c);

  lua_close(L);

  if (r.n_errors == 0 && (!c->params.input.path || !c->params.output.path)) {
    fprintf(stderr, "%s: out of memory\n", path);
    r.n_errors++;
  }
  if (r.n_errors == 0) {
    char* buf = NULL;
    size_t size = 0;
    FILE* out = open_memstream(&buf, &size);
    if (out) {
      write_resolved(out, c, path, defines, n_defines);
      fclose(out);
    }
    c->resolved = buf;
    if (!c->resolved) {
      fprintf(stderr, "%s: out of memory\n", path);
      r.n_errors++;
    }
  }

  if (r.n_errors) {
    if (r.n_errors > 1)
      fprintf(stderr, "%s: %d problems\n", path, r.n_errors);
    config_free(c);
    return -1;
  }
  return 0;
}

/* --- the check before a run -------------------------------------------- */

/* Where the check's findings go: stderr, as the configuration's own errors,
 * for --check; the library's log when a run is about to start. */
typedef struct {
  const char* file;
  bool use_log;
  int n_errors;
} checker_t;

static void check_say(
  checker_t* ch, int level, const char* where, const char* fmt, ...) {

  char msg[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);

  if (level == SIF_LOG_LEVEL_ERROR)
    ch->n_errors++;

  if (ch->use_log) {
    if (level == SIF_LOG_LEVEL_ERROR)
      SIF_LOG_ERROR(
        "sif-exodus", "%s%s%s", where ? where : "", where ? ": " : "", msg);
    else if (level == SIF_LOG_LEVEL_WARNING)
      SIF_LOG_WARNING(
        "sif-exodus", "%s%s%s", where ? where : "", where ? ": " : "", msg);
    else
      SIF_LOG_INFO("sif-exodus", "%s", msg);
    return;
  }

  fprintf(stderr, "%s: %s%s%s%s\n", ch->file,
    level == SIF_LOG_LEVEL_WARNING ? "warning: " : "", where ? where : "",
    where ? ": " : "", msg);
}

/* What the input says about itself before it is read. Zeroes for what it
 * does not say. */
typedef struct {
  uint64_t n_tracers;
  double box;
  bool weighted;
} input_facts_t;

/* A column format's bytes per particle in a binary file, or 0 for one this
 * cannot size -- the reader will say what is wrong with it. */
static uint64_t binary_record_bytes(const char* columns, uint64_t value_bytes) {
  uint64_t bytes = 0;
  for (const char* c = columns; *c;) {
    const char ch = (char)(*c | 0x20); /* lower case, for letters */
    if (*c == ' ' || *c == '\t' || *c == ',') {
      c++;
    } else if (ch == 'x' || ch == 'y' || ch == 'z' || ch == 'w') {
      bytes += value_bytes;
      c++;
    } else if (ch == 'v' && c[1]) {
      bytes += value_bytes;
      c += 2;
    } else if (*c == '*') {
      char* end;
      const unsigned long width = strtoul(c + 1, &end, 10);
      bytes += end == c + 1 ? value_bytes : width;
      c = end;
    } else {
      return 0;
    }
  }
  return bytes;
}

static void inspect_xfield(checker_t* ch, const char* path, input_facts_t* in) {
  sif_xfield_header_t h;
  FILE* f = fopen(path, "rb");
  if (!f) {
    check_say(ch, SIF_LOG_LEVEL_ERROR, "input.path", "cannot open %s: %s", path,
      strerror(errno));
    return;
  }
  const bool whole = fread(&h, sizeof h, 1, f) == 1;
  fclose(f);

  if (!whole || memcmp(h.magic, SIF_XFIELD_MAGIC, 4) != 0) {
    check_say(
      ch, SIF_LOG_LEVEL_ERROR, "input.path", "%s is not an .xfield file", path);
    return;
  }
  if ((h.is_double != 0) != (sizeof(sif_real) == 8)) {
    check_say(ch, SIF_LOG_LEVEL_ERROR, "input.path",
      "%s was written in %s precision, and this sif-exodus works in %s", path,
      h.is_double ? "double" : "single",
      sizeof(sif_real) == 8 ? "double" : "single");
    return;
  }
  in->n_tracers = h.n_particles;
  in->box = h.box_length;
  in->weighted = h.has_weights != 0;
}

static void inspect_gadget(
  checker_t* ch, const exodus_params_t* p, input_facts_t* in) {

  sif_gadget_header_t h;
  if (sif_gadget_read_header(p->input.path, p->input.gadget.format, &h) !=
      SIF_OK) {
    check_say(ch, SIF_LOG_LEVEL_ERROR, "input.path",
      "%s cannot be read as a GADGET snapshot (see above)", p->input.path);
    return;
  }

  const int type = (int)(p->input.gadget.ptype - SIF_GADGET_PTYPE_0);
  if (h.n_part_total[type] == 0) {
    check_say(ch, SIF_LOG_LEVEL_ERROR, "input.gadget.ptype",
      "%s has no particles of type %d", p->input.path, type);
    return;
  }

  /* The box in Mpc/h, as the reader converts it. */
  double to_mpc = 0.0;
  if (p->input.gadget.length == SIF_GADGET_LENGTH_KPC)
    to_mpc = 1e-3;
  else if (p->input.gadget.length == SIF_GADGET_LENGTH_MPC)
    to_mpc = 1.0;
  else if (h.format != SIF_GADGET_FORMAT_HDF5)
    check_say(ch, SIF_LOG_LEVEL_ERROR, "input.gadget.length",
      "a binary snapshot does not record its length unit: say \"kpc\" or "
      "\"mpc\"");
  else if (h.unit_length_in_cm > 0.0)
    to_mpc = h.unit_length_in_cm / 3.085677581e24;
  else
    check_say(ch, SIF_LOG_LEVEL_ERROR, "input.gadget.length",
      "%s does not record its length unit: say \"kpc\" or \"mpc\"",
      p->input.path);

  in->n_tracers =
    (uint64_t)llround(p->input.gadget.fraction * (double)h.n_part_total[type]);
  in->box = h.box_size * to_mpc;
  in->weighted = p->input.gadget.masses && h.mass_table[type] == 0.0;
}

static void inspect_table(
  checker_t* ch, const exodus_params_t* p, input_facts_t* in) {

  struct stat st;
  FILE* f = fopen(p->input.path, "rb");
  if (!f || fstat(fileno(f), &st) != 0) {
    check_say(ch, SIF_LOG_LEVEL_ERROR, "input.path", "cannot open %s: %s",
      p->input.path, strerror(errno));
    if (f)
      fclose(f);
    return;
  }
  fclose(f);

  in->weighted = strchr(p->input.columns, 'w') || strchr(p->input.columns, 'W');
  if (p->input.kind == EXODUS_INPUT_ASCII)
    return; /* counted only by reading it */

  const uint64_t size = (uint64_t)st.st_size;
  const uint64_t header = p->input.binary.header_bytes;
  const uint64_t record = binary_record_bytes(
    p->input.columns, p->input.binary.precision == SIF_BINARY_FLOAT64 ? 8 : 4);
  if (record == 0)
    return;
  if (size < header || (size - header) % record != 0) {
    check_say(ch, SIF_LOG_LEVEL_ERROR, "input.binary",
      "%s holds %llu bytes after its %llu-byte header, which is not a whole "
      "number of %llu-byte particles: check columns, precision and "
      "header_bytes",
      p->input.path, (unsigned long long)(size < header ? 0 : size - header),
      (unsigned long long)header, (unsigned long long)record);
    return;
  }
  in->n_tracers = (size - header) / record;
}

/* Every file of a FITS catalogue, opened and its structure read: whether the
 * columns are there is found when it is read. */
static void inspect_fits(
  checker_t* ch, const exodus_params_t* p, input_facts_t* in) {
  for (uint32_t i = 0; i < p->input.n_paths; i++) {
    const char* path = p->input.paths[i];
    FILE* sink = tmpfile();
    const int status = sink ? sif_fits_print_summary(path, sink) : SIF_OK;
    if (sink)
      fclose(sink);
    if (status == SIF_ERR_UNSUPPORTED) {
      check_say(ch, SIF_LOG_LEVEL_ERROR, "input.format",
        "this sif-exodus was built without FITS support");
      return;
    }
    if (status != SIF_OK)
      check_say(ch, SIF_LOG_LEVEL_ERROR, "input.path",
        "%s cannot be read as a FITS file (see above)", path);
  }
  in->weighted = p->input.named.w != NULL;
}

/* Every file of an HDF5 input, there and an HDF5 file: whether the datasets
 * are in them is found when they are read. */
static void inspect_hdf5(
  checker_t* ch, const exodus_params_t* p, input_facts_t* in) {
  static const unsigned char sig[8] = {
    0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
  for (uint32_t i = 0; i < p->input.n_paths; i++) {
    const char* path = p->input.paths[i];
    unsigned char head[8];
    FILE* f = fopen(path, "rb");
    if (!f) {
      check_say(ch, SIF_LOG_LEVEL_ERROR, "input.path", "cannot open %s: %s",
        path, strerror(errno));
      continue;
    }
    const size_t n = fread(head, 1, 8, f);
    fclose(f);
    if (n != 8 || memcmp(head, sig, 8) != 0)
      check_say(
        ch, SIF_LOG_LEVEL_ERROR, "input.path", "%s is not an HDF5 file", path);
  }
#if !defined(SIF_HAVE_HDF5)
  check_say(ch, SIF_LOG_LEVEL_ERROR, "input.format",
    "this sif-exodus was built without HDF5 support");
#endif
  in->weighted = p->input.named.w != NULL;
}

/* "1.3 GiB", "250 MiB": a size to read, not to compute with. */
static const char* human_bytes(char* buf, size_t n, uint64_t bytes) {
  const double mib = (double)bytes / (1024.0 * 1024.0);
  if (mib < 1024.0)
    snprintf(buf, n, "%.0f MiB", ceil(mib));
  else
    snprintf(buf, n, "%.1f GiB", mib / 1024.0);
  return buf;
}

static uint64_t physical_memory(void) {
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page = sysconf(_SC_PAGESIZE);
  return pages > 0 && page > 0 ? (uint64_t)pages * (uint64_t)page : 0;
}

int config_check(const exodus_config_t* c, const char* file, bool summary) {
  const exodus_params_t* p = &c->params;
  checker_t ch = {file, !summary, 0};
  input_facts_t in = {0, 0.0, false};

  /* 1. The input, as far as its header goes. */
  if (p->input.kind == EXODUS_INPUT_XFIELD)
    inspect_xfield(&ch, p->input.path, &in);
  else if (p->input.kind == EXODUS_INPUT_GADGET)
    inspect_gadget(&ch, p, &in);
  else if (p->input.kind == EXODUS_INPUT_FITS)
    inspect_fits(&ch, p, &in);
  else if (p->input.kind == EXODUS_INPUT_HDF5)
    inspect_hdf5(&ch, p, &in);
  else
    inspect_table(&ch, p, &in);

  const double box = p->input.box_length > 0.0 ? p->input.box_length : in.box;
  const uint64_t n = in.n_tracers;
  const double mps = n && box > 0.0 ? box / cbrt((double)n) : 0.0;

  if (summary && n)
    check_say(&ch, SIF_LOG_LEVEL_INFO, NULL,
      "input: %llu %s tracers, box %g, mean separation %g",
      (unsigned long long)n, in.weighted ? "weighted" : "unweighted", box, mps);
  else if (summary && box > 0.0)
    check_say(&ch, SIF_LOG_LEVEL_INFO, NULL,
      "input: box %g; the tracers are counted when read", box);

  /* 2. The radii in the box's units, which mean separations need N for. */
  double r_min = 0.0, r_max = 0.0;
  if (p->finder.radii_units == EXODUS_RADII_PHYSICAL || mps > 0.0) {
    const double scale = p->finder.radii_units == EXODUS_RADII_MPS ? mps : 1.0;
    r_min = r_max = p->finder.radii[0] * scale;
    for (uint32_t i = 1; i < p->finder.n_radii; i++) {
      r_min = fmin(r_min, p->finder.radii[i] * scale);
      r_max = fmax(r_max, p->finder.radii[i] * scale);
    }
  }
  if (summary && r_max > 0.0 && p->finder.radii_units == EXODUS_RADII_MPS)
    check_say(&ch, SIF_LOG_LEVEL_INFO, NULL,
      "radii: %u, from %g to %g (%g to %g mean separations)", p->finder.n_radii,
      r_min, r_max, r_min / mps, r_max / mps);
  else if (summary && r_max > 0.0)
    check_say(&ch, SIF_LOG_LEVEL_INFO, NULL, "radii: %u, from %g to %g",
      p->finder.n_radii, r_min, r_max);
  else if (summary)
    check_say(&ch, SIF_LOG_LEVEL_INFO, NULL,
      "radii: %u, in mean separations, sized once the tracers are counted",
      p->finder.n_radii);

  /* 3. Grid and mesh, as the run will size them. */
  const uint32_t grid = p->grid.n_cells
                          ? p->grid.n_cells
                          : (n ? pipeline_default_grid_cells(n) : 0);
  uint32_t mesh = p->mesh.n_cells;
  if (box > 0.0 && r_max > 0.0) {
    /* Whatever the count, a search sphere wider than the box is fatal. */
    const uint32_t suggested =
      sif_finder_suggest_mesh_cells(n ? n : 1, (sif_real)box, (sif_real)r_max);
    if (suggested == 0)
      check_say(&ch, SIF_LOG_LEVEL_ERROR, "finder.radii",
        "the largest radius, %g, needs a search sphere wider than the box "
        "(%g)",
        r_max, box);
    else if (!mesh && n)
      mesh = suggested;
  }
  if (summary && grid)
    check_say(&ch, SIF_LOG_LEVEL_INFO, NULL, "grid: %u^3 cells of %g%s", grid,
      box / grid, p->grid.n_cells ? "" : " (auto)");
  if (summary && mesh)
    check_say(&ch, SIF_LOG_LEVEL_INFO, NULL, "mesh: %u^3 cells%s", mesh,
      p->mesh.n_cells ? "" : " (auto)");

  /* The finder deconvolves the CIC window, which amplifies the smallest
   * scales; a top-hat under two cells no longer suppresses them. */
  if (grid && box > 0.0 && r_min > 0.0 && r_min < 2.0 * box / grid)
    check_say(&ch, SIF_LOG_LEVEL_WARNING, "finder.radii",
      "the smallest radius, %g, spans under two cells of the grid (%g): "
      "start the ladder at %g or above, or give grid.n_cells = %u or more",
      r_min, box / grid, 2.0 * box / grid, (unsigned)ceil(2.0 * box / r_min));

  /* 4. Memory. */
  if (n && grid && mesh) {
    const uint64_t peak = pipeline_peak_bytes(n, in.weighted, grid, mesh);
    char a[32], b[32];
    check_say(&ch, SIF_LOG_LEVEL_INFO, NULL, "memory: about %s at the peak",
      human_bytes(a, sizeof a, peak));

    const uint64_t ram = physical_memory();
    if (ram && peak > ram)
      check_say(&ch, SIF_LOG_LEVEL_WARNING, NULL,
        "the run needs about %s and this machine has %s",
        human_bytes(a, sizeof a, peak), human_bytes(b, sizeof b, ram));
  }

  /* 5. The output: a directory that takes files, and nothing lost silently
   * when it already exists. */
  char* dir = strdup(p->output.path);
  if (dir) {
    char* slash = strrchr(dir, '/');
    if (!slash)
      strcpy(dir, ".");
    else if (slash == dir)
      slash[1] = '\0';
    else
      *slash = '\0';
    if (access(dir, W_OK) != 0)
      check_say(&ch, SIF_LOG_LEVEL_ERROR, "output.path",
        "cannot write into %s: %s", dir, strerror(errno));
    free(dir);
  }
  if (access(p->output.path, F_OK) == 0)
    check_say(&ch, SIF_LOG_LEVEL_WARNING, "output.path",
      "%s exists, and will be replaced when the run completes", p->output.path);

  return ch.n_errors ? -1 : 0;
}

void config_free(exodus_config_t* c) {
  if (!c)
    return;
  free(c->_radii);
  for (size_t i = 0; i < c->_n_strings; i++)
    free(c->_strings[i]);
  free(c->_strings);
  free((void*)c->_paths);
  free(c->resolved);
  memset(c, 0, sizeof *c);
}
