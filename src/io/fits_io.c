/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * FITS tables, through cfitsio. The only file that includes fitsio.h.
 *
 * Every part of the field has a source: a plain column, read with
 * fits_read_col(), or an expression, evaluated with fits_calc_rows(). A plain
 * name takes the first path because it is several times faster and because
 * it is what nearly every source is; an expression pays for cfitsio's
 * evaluator, which is what it is there for.
 *
 * Two passes. The first evaluates the filter over the whole table into a
 * bitmask, which gives the count the field is allocated at; the second reads
 * the table chunk by chunk and keeps the rows the mask and the subsample
 * pass. Without a filter the first pass is skipped, and the count is the
 * table's.
 */

#include "sif/io/fits_io.h"

#include "io/fits_internal.h"
#include "measure/profiles_internal.h"
#include "sif/structures/bitmask.h"
#include "sif/utils/logger.h"
#include "sif/utils/random.h"
#include "structures/catalog_internal.h"
#include "structures/results_internal.h"

#include <fitsio.h>

#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define TAG "fits"

/* Rows per read. As in the GADGET reader: large enough that the per-call
 * overhead vanishes, small enough that the buffers (at most 7 doubles a row,
 * about 15 MB) never matter next to the field. */
#define CHUNK ((long)1 << 18)

int sif__fits_describe(char* buf, size_t len) {
  float version = 0;
  fits_get_version(&version);
  /* The version comes back as a float, major + minor / 100: 4.07 is 4.7. */
  const int major = (int)version;
  const int minor = (int)lround(((double)version - major) * 100.0);
  snprintf(buf, len, "cfitsio %d.%d", major, minor);
  return 1;
}

/* ------------------------------------------------------------------------ */
/* errors                                                                    */
/* ------------------------------------------------------------------------ */

/* Logs what went wrong, followed by cfitsio's own account of it: the status
 * text and the messages it stacked up on the way, which for a parse error
 * point at the offending token. */
static void log_fits(int fstatus, const char* fmt, ...) {
  char what[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(what, sizeof(what), fmt, ap);
  va_end(ap);

  char text[FLEN_STATUS];
  fits_get_errstatus(fstatus, text);

  char detail[512] = "";
  char msg[FLEN_ERRMSG];
  size_t used = 0;
  while (fits_read_errmsg(msg) && used + 3 < sizeof(detail)) {
    const int n = snprintf(
      detail + used, sizeof(detail) - used, "%s%s", used ? "; " : "", msg);
    if (n < 0)
      break;
    used += (size_t)n;
  }

  if (detail[0])
    SIF_LOG_ERROR(TAG, "%s: %s (%s)", what, text, detail);
  else
    SIF_LOG_ERROR(TAG, "%s: %s", what, text);
}

/* ------------------------------------------------------------------------ */
/* the table                                                                 */
/* ------------------------------------------------------------------------ */

static bool is_table(int hdutype) {
  return hdutype == BINARY_TBL || hdutype == ASCII_TBL;
}

/* Moves to the HDU @p hdu names -- an extension number or an EXTNAME -- of
 * whatever kind. */
static int move_to_hdu(
  fitsfile* f, const char* path, const char* hdu, int* hdutype) {
  int fstatus = 0;

  bool is_number = true;
  for (const char* c = hdu; *c; c++)
    is_number = is_number && isdigit((unsigned char)*c);

  if (is_number) {
    const long ext = strtol(hdu, NULL, 10);
    if (ext > INT_MAX - 1) {
      SIF_LOG_ERROR(TAG, "%s has no extension %s", path, hdu);
      return SIF_ERR_INVALID;
    }
    /* cfitsio counts from 1 at the primary HDU; the extension number, as
     * astropy and fitsio count, from 0. */
    fits_movabs_hdu(f, (int)ext + 1, hdutype, &fstatus);
  } else {
    char name[FLEN_VALUE];
    snprintf(name, sizeof(name), "%s", hdu);
    fits_movnam_hdu(f, ANY_HDU, name, 0, &fstatus);
    if (fstatus == 0)
      fits_get_hdu_type(f, hdutype, &fstatus);
  }

  if (fstatus == END_OF_FILE || fstatus == BAD_HDU_NUM) {
    fits_clear_errmsg();
    SIF_LOG_ERROR(TAG, "%s has no HDU %s", path, hdu);
    return SIF_ERR_INVALID;
  }
  if (fstatus) {
    log_fits(fstatus, "failed to move to HDU %s of %s", hdu, path);
    return SIF_ERR_IO;
  }
  return SIF_OK;
}

/* Moves to the table asked for: the first one in the file, an extension
 * number, or an EXTNAME. */
static int move_to_table(fitsfile* f, const char* path, const char* hdu) {
  int fstatus = 0;
  int hdutype = 0;

  if (!hdu || !*hdu) {
    int n_hdus = 0;
    fits_get_num_hdus(f, &n_hdus, &fstatus);
    for (int i = 2; fstatus == 0 && i <= n_hdus; i++) {
      fits_movabs_hdu(f, i, &hdutype, &fstatus);
      if (fstatus == 0 && is_table(hdutype))
        return SIF_OK;
    }
    if (fstatus) {
      log_fits(fstatus, "failed to read the HDUs of %s", path);
      return SIF_ERR_IO;
    }
    SIF_LOG_ERROR(TAG, "%s holds no table", path);
    return SIF_ERR_INVALID;
  }

  const int status = move_to_hdu(f, path, hdu, &hdutype);
  if (status != SIF_OK)
    return status;
  if (!is_table(hdutype)) {
    SIF_LOG_ERROR(TAG, "HDU %s of %s is an image, not a table", hdu, path);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* sources                                                                   */
/* ------------------------------------------------------------------------ */

enum { ROLE_X, ROLE_Y, ROLE_Z, ROLE_VX, ROLE_VY, ROLE_VZ, ROLE_W, N_ROLES };

/* What each part is called, for messages: as positions, and as sky
 * coordinates, which fill the same parts of the field. */
static const char* const ROLE_NAMES[N_ROLES] = {
  "x", "y", "z", "vx", "vy", "vz", "w"};
static const char* const ROLE_NAMES_SKY[N_ROLES] = {
  "ra", "dec", "z", "vx", "vy", "vz", "w"};

/* Where one part of the field comes from. */
typedef struct {
  /* As the caller wrote it, for messages. NULL for a part not read. */
  const char* spec;
  /* A copy cfitsio may hold as non-const, as its prototypes ask. */
  char* expr;
  /* The column, for a plain name; 0 for an expression. */
  int colnum;
  double* buf;
} source_t;

/* A spec that could be nothing but a column name. Anything else -- an
 * operator, a parenthesis, a leading digit -- is an expression, which
 * keeps "1" or "2*W" from being looked up as names, and keeps cfitsio's
 * wildcards (`*`, `?`, `#`) out of fits_get_colnum(), where "A*B" would
 * otherwise match a column called AXB. */
static bool is_plain_name(const char* s) {
  if (!(isalpha((unsigned char)*s) || *s == '_'))
    return false;
  for (; *s; s++)
    if (!(isalnum((unsigned char)*s) || *s == '_'))
      return false;
  return true;
}

static bool type_is_numeric(int type) {
  return type != TSTRING && type != TLOGICAL && type != TBIT &&
         type != TCOMPLEX && type != TDBLCOMPLEX;
}

static int source_open(fitsfile* f, const char* path, const char* what,
  const char* spec, source_t* s) {
  s->spec = spec;
  s->expr = malloc(strlen(spec) + 1);
  s->buf = malloc((size_t)CHUNK * sizeof(double));
  if (!s->expr || !s->buf)
    return SIF_ERR_ALLOC;
  strcpy(s->expr, spec);

  int fstatus = 0;

  if (is_plain_name(spec)) {
    fits_get_colnum(f, CASEINSEN, s->expr, &s->colnum, &fstatus);
    if (fstatus) {
      fits_clear_errmsg();
      SIF_LOG_ERROR(TAG, "%s: no column %s, for %s", path, spec, what);
      return SIF_ERR_INVALID;
    }

    int type = 0;
    LONGLONG repeat = 0, width = 0;
    fits_get_eqcoltypell(f, s->colnum, &type, &repeat, &width, &fstatus);
    if (fstatus) {
      log_fits(fstatus, "%s: failed to read the type of column %s", path, spec);
      return SIF_ERR_IO;
    }
    if (type < 0 || !type_is_numeric(type)) {
      SIF_LOG_ERROR(
        TAG, "%s: column %s, for %s, is not numeric", path, spec, what);
      return SIF_ERR_INVALID;
    }
    if (repeat != 1) {
      SIF_LOG_ERROR(TAG,
        "%s: column %s, for %s, holds %lld values a row; only scalar columns "
        "are read",
        path, spec, what, (long long)repeat);
      return SIF_ERR_INVALID;
    }
    return SIF_OK;
  }

  /* An expression: parsed here, once, so a mistake is reported before
   * anything is read. A negative count is cfitsio's way of saying the
   * expression is a constant, which fits_calc_rows() repeats down the
   * rows. */
  int type = 0, naxis = 0;
  long n_elem = 0, naxes[1];
  fits_test_expr(f, s->expr, 1, &type, &n_elem, &naxis, naxes, &fstatus);
  if (fstatus) {
    log_fits(fstatus, "%s: cannot evaluate \"%s\", for %s", path, spec, what);
    return SIF_ERR_INVALID;
  }
  if (!type_is_numeric(type) || (n_elem != 1 && n_elem != -1)) {
    SIF_LOG_ERROR(TAG, "%s: \"%s\", for %s, is not a number for each row", path,
      spec, what);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

static void source_close(source_t* s) {
  free(s->expr);
  free(s->buf);
}

/* Rows [first, first + n) of a source, 0-based, as doubles. An undefined
 * value comes back as NaN. */
static int source_read(fitsfile* f, source_t* s, long first, long n) {
  double nulval = NAN;
  int anynul = 0, fstatus = 0;

  if (s->colnum > 0)
    fits_read_col(f, TDOUBLE, s->colnum, (LONGLONG)first + 1, 1, n, &nulval,
      s->buf, &anynul, &fstatus);
  else
    fits_calc_rows(
      f, TDOUBLE, s->expr, first + 1, n, &nulval, s->buf, &anynul, &fstatus);

  if (fstatus) {
    log_fits(fstatus, "failed to read rows %ld-%ld of %s", first + 1, first + n,
      s->spec);
    return SIF_ERR_IO;
  }
  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* one file                                                                  */
/* ------------------------------------------------------------------------ */

/* One file of the catalogue, open at its table, with a source for every part
 * of the field. Columns are looked up file by file: a column's number, or
 * whether it is there at all, may differ between files of one catalogue. */
typedef struct {
  const char* path;
  fitsfile* f;
  long n_rows;
  source_t src[N_ROLES];
} table_t;

static void table_close(table_t* t) {
  for (int r = 0; r < N_ROLES; r++)
    source_close(&t->src[r]);
  if (t->f) {
    int fstatus = 0;
    fits_close_file(t->f, &fstatus);
  }
  memset(t, 0, sizeof(*t));
}

/* Checks @p where parses on this file's table and gives a boolean a row. */
static int filter_check(const table_t* t, char* where) {
  int fstatus = 0, type = 0, naxis = 0;
  long n_elem = 0, naxes[1];
  fits_test_expr(t->f, where, 1, &type, &n_elem, &naxis, naxes, &fstatus);
  if (fstatus) {
    log_fits(fstatus, "%s: cannot evaluate the filter \"%s\"", t->path, where);
    return SIF_ERR_INVALID;
  }
  if (type != TLOGICAL || (n_elem != 1 && n_elem != -1)) {
    SIF_LOG_ERROR(TAG,
      "%s: the filter \"%s\" is not true or false for each row", t->path,
      where);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/* Opens a file at its table and sets up every source, and the filter if
 * there is one: everything that can be wrong with a file short of its data.
 * On failure the table is closed. */
static int table_open(table_t* t, const char* path, const char* hdu,
  const char* const specs[N_ROLES], const char* const names[N_ROLES],
  char* where) {
  memset(t, 0, sizeof(*t));
  t->path = path;

  int fstatus = 0;
  fits_open_diskfile(&t->f, path, READONLY, &fstatus);
  if (fstatus) {
    log_fits(fstatus, "failed to open %s", path);
    t->f = NULL;
    return SIF_ERR_IO;
  }

  int status = move_to_table(t->f, path, hdu);
  if (status == SIF_OK) {
    fits_get_num_rows(t->f, &t->n_rows, &fstatus);
    if (fstatus) {
      log_fits(fstatus, "failed to read the row count of %s", path);
      status = SIF_ERR_IO;
    }
  }
  for (int r = 0; status == SIF_OK && r < N_ROLES; r++)
    if (specs[r])
      status = source_open(t->f, path, names[r], specs[r], &t->src[r]);
  if (status == SIF_OK && where)
    status = filter_check(t, where);

  if (status != SIF_OK)
    table_close(t);
  return status;
}

/* Evaluates @p where over every row of the table, setting the bits of the
 * rows it keeps from @p offset on. */
static int filter_rows(const table_t* t, char* where, sif_bitmask_t* mask,
  uint64_t offset, char* row_status, uint64_t* n_pass) {
  for (long first = 0; first < t->n_rows; first += CHUNK) {
    const long n = (t->n_rows - first < CHUNK) ? t->n_rows - first : CHUNK;
    long n_good = 0;
    int fstatus = 0;
    fits_find_rows(t->f, where, first + 1, n, &n_good, row_status, &fstatus);
    if (fstatus) {
      log_fits(fstatus, "%s: failed to evaluate the filter on rows %ld-%ld",
        t->path, first + 1, first + n);
      return SIF_ERR_IO;
    }
    for (long j = 0; j < n; j++)
      if (row_status[j])
        sif_bitmask_set(mask, offset + (uint64_t)(first + j));
    *n_pass += (uint64_t)n_good;
  }
  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* the reader                                                                */
/* ------------------------------------------------------------------------ */

/* What is carried from file to file while the rows are streamed. */
typedef struct {
  sif_field_t* field;
  const sif_bitmask_t* mask;
  sif_prng_state_t prng;
  uint64_t n_pass, n_keep, seen, kept;

  /* Rows with an undefined value, and where the first one was. */
  uint64_t n_bad;
  const char* bad_path;
  const char* bad_spec;
  long bad_row;
  int bad_role;
} stream_t;

static int stream_table(stream_t* st, table_t* t, uint64_t offset) {
  const sif_bitmask_t* mask = st->mask;
  const bool subsample = st->n_keep < st->n_pass;
  source_t* src = t->src;
  sif_field_t* fl = st->field;

  for (long first = 0; first < t->n_rows; first += CHUNK) {
    const long n = (t->n_rows - first < CHUNK) ? t->n_rows - first : CHUNK;
    const uint64_t base = offset + (uint64_t)first;

    /* A chunk the filter empties is not read at all: a catalogue sorted by
     * redshift, cut to a slice of it, skips most of the file. */
    if (mask) {
      bool any = false;
      for (long j = 0; j < n && !any; j++)
        any = sif_bitmask_get(mask, base + (uint64_t)j);
      if (!any)
        continue;
    }

    for (int r = 0; r < N_ROLES; r++) {
      if (!src[r].spec)
        continue;
      const int status = source_read(t->f, &src[r], first, n);
      if (status != SIF_OK) {
        SIF_LOG_ERROR(TAG, "in %s", t->path);
        return status;
      }
    }

    for (long j = 0; j < n; j++) {
      if (mask && !sif_bitmask_get(mask, base + (uint64_t)j))
        continue;

      /* Checked on every row the filter keeps, before the subsample draws,
       * so whether a catalogue reads does not depend on the seed. */
      bool row_ok = true;
      for (int r = 0; r < N_ROLES; r++) {
        if (src[r].spec && !isfinite(src[r].buf[j])) {
          if (st->n_bad == 0) {
            st->bad_path = t->path;
            st->bad_spec = src[r].colnum > 0 ? src[r].spec : NULL;
            st->bad_row = first + j + 1;
            st->bad_role = r;
          }
          row_ok = false;
        }
      }
      if (!row_ok) {
        st->n_bad++;
        continue;
      }
      if (st->n_bad)
        continue;

      /* Algorithm S, as in the GADGET reader: keep this row with probability
       * (still wanted) / (still to see). Run over the rows of every file in
       * turn, so the subsample is one draw from the whole catalogue. */
      if (subsample) {
        const double u = sif_prng_next_double(&st->prng);
        const bool keep =
          (double)(st->n_pass - st->seen) * u < (double)(st->n_keep - st->kept);
        st->seen++;
        if (!keep)
          continue;
      }

      const uint64_t k = st->kept++;
      fl->x[k] = (sif_real)src[ROLE_X].buf[j];
      fl->y[k] = (sif_real)src[ROLE_Y].buf[j];
      fl->z[k] = (sif_real)src[ROLE_Z].buf[j];
      if (src[ROLE_VX].spec) {
        fl->vx[k] = (sif_real)src[ROLE_VX].buf[j];
        fl->vy[k] = (sif_real)src[ROLE_VY].buf[j];
        fl->vz[k] = (sif_real)src[ROLE_VZ].buf[j];
      }
      if (src[ROLE_W].spec)
        fl->weights[k] = (sif_real)src[ROLE_W].buf[j];
    }
  }
  return SIF_OK;
}

static int check_args(const char* const* paths, uint32_t n_paths,
  const sif_field_columns_t* columns, double fraction) {
  if (!paths || n_paths == 0 || !columns) {
    SIF_LOG_ERROR(TAG, "at least one path, and the columns, are required");
    return SIF_ERR_INVALID;
  }
  for (uint32_t i = 0; i < n_paths; i++) {
    if (!paths[i] || !*paths[i]) {
      SIF_LOG_ERROR(TAG, "path %u of %u is empty", i + 1, n_paths);
      return SIF_ERR_INVALID;
    }
  }
  /* Positions or sky coordinates, never some of each; z is either. */
  const bool xy = columns->x || columns->y;
  const bool radec = columns->ra || columns->dec;
  if (xy && radec) {
    SIF_LOG_ERROR(TAG,
      "columns name both x, y and ra, dec: positions are x y z, sky "
      "coordinates ra dec z");
    return SIF_ERR_INVALID;
  }
  if (radec ? !(columns->ra && columns->dec && columns->z)
            : !(columns->x && columns->y && columns->z)) {
    SIF_LOG_ERROR(TAG, "columns for all three of %s are required",
      radec ? "ra, dec and z" : "x, y and z");
    return SIF_ERR_INVALID;
  }
  const int n_vel = !!columns->vx + !!columns->vy + !!columns->vz;
  if (n_vel != 0 && n_vel != 3) {
    SIF_LOG_ERROR(TAG,
      "velocities are read all three or not at all; %d of vx, vy and vz were "
      "named",
      n_vel);
    return SIF_ERR_INVALID;
  }
  if (!(fraction > 0 && fraction <= 1)) {
    SIF_LOG_ERROR(TAG, "fraction must be in (0, 1], not %g", fraction);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/* The reader, with the kind of failure returned alongside: the public entry
 * point only says whether it worked. */
static int read_fits(const char* const* paths, uint32_t n_paths,
  const char* hdu, const sif_field_columns_t* columns, const char* where,
  double fraction, uint64_t seed, sif_field_t** out_field) {

  *out_field = NULL;
  int status = check_args(paths, n_paths, columns, fraction);
  if (status != SIF_OK)
    return status;

  /* Sky coordinates fill the same parts of the field as positions do. */
  const bool sky = columns->ra != NULL;
  const char* const* names = sky ? ROLE_NAMES_SKY : ROLE_NAMES;
  const char* specs[N_ROLES] = {sky ? columns->ra : columns->x,
    sky ? columns->dec : columns->y, columns->z, columns->vx, columns->vy,
    columns->vz, columns->w};
  for (int r = 0; r < N_ROLES; r++) {
    if (specs[r] && !*specs[r]) {
      SIF_LOG_ERROR(TAG,
        "the column for %s is an empty string; leave it NULL to read none",
        names[r]);
      return SIF_ERR_INVALID;
    }
  }

  /* A copy cfitsio may hold as non-const, as its prototypes ask. */
  char* expr = NULL;
  if (where && *where) {
    expr = malloc(strlen(where) + 1);
    if (!expr)
      return SIF_ERR_ALLOC;
    strcpy(expr, where);
  }

  uint64_t* offsets = calloc((size_t)n_paths + 1, sizeof(uint64_t));
  if (!offsets) {
    free(expr);
    return SIF_ERR_ALLOC;
  }

  /* --- pass 0: every file, checked before anything is read --- */

  table_t t;
  for (uint32_t i = 0; status == SIF_OK && i < n_paths; i++) {
    status = table_open(&t, paths[i], hdu, specs, names, expr);
    if (status == SIF_OK) {
      offsets[i + 1] = offsets[i] + (uint64_t)t.n_rows;
      table_close(&t);
    }
  }
  const uint64_t n_rows = offsets[n_paths];
  if (status == SIF_OK && n_rows == 0) {
    SIF_LOG_ERROR(TAG, "the table%s in %s%s %s no rows",
      n_paths == 1 ? "" : "s", paths[0], n_paths == 1 ? "" : " and the rest",
      n_paths == 1 ? "has" : "have");
    status = SIF_ERR_INVALID;
  }

  /* --- pass 1: the filter, to size the field --- */

  sif_bitmask_t* mask = NULL;
  uint64_t n_pass = n_rows;
  if (status == SIF_OK && expr) {
    mask = sif_bitmask_alloc(n_rows);
    char* row_status = malloc((size_t)CHUNK);
    status = (mask && row_status) ? SIF_OK : SIF_ERR_ALLOC;
    n_pass = 0;
    for (uint32_t i = 0; status == SIF_OK && i < n_paths; i++) {
      status = table_open(&t, paths[i], hdu, specs, names, expr);
      if (status == SIF_OK) {
        status = filter_rows(&t, expr, mask, offsets[i], row_status, &n_pass);
        table_close(&t);
      }
    }
    free(row_status);
    if (status == SIF_OK && n_pass == 0) {
      SIF_LOG_ERROR(TAG, "the filter \"%s\" keeps none of the %" PRIu64 " rows",
        where, n_rows);
      status = SIF_ERR_INVALID;
    }
  }

  uint64_t n_keep = 0;
  if (status == SIF_OK) {
    n_keep =
      fraction < 1 ? (uint64_t)llround(fraction * (double)n_pass) : n_pass;
    if (n_keep == 0) {
      SIF_LOG_ERROR(TAG, "a fraction of %g keeps none of %" PRIu64 " rows",
        fraction, n_pass);
      status = SIF_ERR_INVALID;
    }
  }

  /* --- pass 2: the rows --- */

  stream_t st;
  memset(&st, 0, sizeof(st));
  if (status == SIF_OK) {
    st.field = sif_field_alloc(n_keep);
    status = st.field ? sif_field_reserve_positions(st.field) : SIF_ERR_ALLOC;
    if (status == SIF_OK && specs[ROLE_VX])
      status = sif_field_reserve_velocities(st.field);
    if (status == SIF_OK && specs[ROLE_W])
      status = sif_field_reserve_weights(st.field);
  }

  if (status == SIF_OK) {
    st.mask = mask;
    st.n_pass = n_pass;
    st.n_keep = n_keep;
    sif_prng_init(&st.prng, seed);
    for (uint32_t i = 0; status == SIF_OK && i < n_paths; i++) {
      status = table_open(&t, paths[i], hdu, specs, names, NULL);
      if (status == SIF_OK) {
        status = stream_table(&st, &t, offsets[i]);
        table_close(&t);
      }
    }
  }

  if (status == SIF_OK && st.n_bad) {
    char hint[160] = "";
    if (st.bad_spec)
      snprintf(hint, sizeof(hint),
        "; drop them with a filter such as "
        "\"!ISNULL(%s)\"",
        st.bad_spec);
    SIF_LOG_ERROR(TAG,
      "%" PRIu64 " row%s the filter keeps hold an undefined value, the first "
      "at row %ld of %s, in %s (\"%s\")%s",
      st.n_bad, st.n_bad == 1 ? "" : "s", st.bad_row, st.bad_path,
      names[st.bad_role], specs[st.bad_role], hint);
    status = SIF_ERR_INVALID;
  }
  if (status == SIF_OK && st.kept != n_keep) {
    SIF_LOG_ERROR(
      TAG, "read %" PRIu64 " rows, expected %" PRIu64, st.kept, n_keep);
    status = SIF_ERR_IO;
  }

  sif_bitmask_free(mask);
  free(offsets);
  free(expr);

  if (status != SIF_OK) {
    sif_field_free(st.field);
    return status;
  }

  if (sky)
    st.field->units = SIF_COORDINATES_SKY;

  char files[64];
  if (n_paths == 1)
    snprintf(files, sizeof(files), "%s", "");
  else
    snprintf(files, sizeof(files), " and %u more file%s", n_paths - 1,
      n_paths == 2 ? "" : "s");
  if (where && *where)
    SIF_LOG_INFO(TAG,
      "loaded %" PRIu64 " of %" PRIu64 " rows passing \"%s\" (of %" PRIu64
      ") from %s%s",
      n_keep, n_pass, where, n_rows, paths[0], files);
  else
    SIF_LOG_INFO(TAG, "loaded %" PRIu64 " of %" PRIu64 " rows from %s%s",
      n_keep, n_rows, paths[0], files);

  *out_field = st.field;
  return SIF_OK;
}

sif_field_t* sif_field_read_fits(const char* const* paths, uint32_t n_paths,
  const char* hdu, const sif_field_columns_t* columns, const char* where,
  double fraction, uint64_t seed) {
  sif_field_t* field = NULL;
  (void)read_fits(paths, n_paths, hdu, columns, where, fraction, seed, &field);
  return field;
}

/* ------------------------------------------------------------------------ */
/* inspection                                                                */
/* ------------------------------------------------------------------------ */

/* A keyword's value as text, or "" if the header has none. */
static void key_string(fitsfile* f, const char* key, char* out) {
  int fstatus = 0;
  out[0] = '\0';
  fits_read_key(f, TSTRING, key, out, NULL, &fstatus);
  if (fstatus) {
    fits_clear_errmsg();
    out[0] = '\0';
  }
}

static int print_table(fitsfile* f, FILE* stream) {
  int fstatus = 0, n_cols = 0;
  LONGLONG n_rows = 0;
  fits_get_num_rowsll(f, &n_rows, &fstatus);
  fits_get_num_cols(f, &n_cols, &fstatus);
  if (fstatus)
    return fstatus;
  fprintf(stream, "%lld row%s, %d column%s\n", (long long)n_rows,
    n_rows == 1 ? "" : "s", n_cols, n_cols == 1 ? "" : "s");

  for (int c = 1; c <= n_cols; c++) {
    char key[FLEN_KEYWORD], name[FLEN_VALUE], form[FLEN_VALUE],
      unit[FLEN_VALUE];
    fits_make_keyn("TTYPE", c, key, &fstatus);
    key_string(f, key, name);
    fits_make_keyn("TFORM", c, key, &fstatus);
    key_string(f, key, form);
    fits_make_keyn("TUNIT", c, key, &fstatus);
    key_string(f, key, unit);

    int type = 0;
    LONGLONG repeat = 0, width = 0;
    fits_get_eqcoltypell(f, c, &type, &repeat, &width, &fstatus);
    if (fstatus)
      return fstatus;

    /* What the reader makes of it: a plain column, a vector whose elements
     * are named NAME[1] ... NAME[n], or something it cannot read. */
    char line[256];
    int len =
      snprintf(line, sizeof(line), "      %-20s %-8s %-12s", name, form, unit);
    if (type < 0 || !type_is_numeric(type))
      snprintf(line + len, sizeof(line) - (size_t)len, "not readable");
    else if (repeat > 1)
      snprintf(line + len, sizeof(line) - (size_t)len,
        "vector: %s[1] ... %s[%lld]", name, name, (long long)repeat);
    len = (int)strlen(line);
    while (len > 0 && line[len - 1] == ' ')
      line[--len] = '\0';
    fputs(line, stream);
    fputc('\n', stream);
  }
  return 0;
}

int sif_fits_print_summary(const char* path, FILE* stream) {
  if (!path || !stream) {
    SIF_LOG_ERROR(TAG, "path and stream are required");
    return SIF_ERR_INVALID;
  }

  fitsfile* f = NULL;
  int fstatus = 0;
  fits_open_diskfile(&f, path, READONLY, &fstatus);
  if (fstatus) {
    log_fits(fstatus, "failed to open %s", path);
    return SIF_ERR_IO;
  }

  int n_hdus = 0;
  fits_get_num_hdus(f, &n_hdus, &fstatus);
  if (!fstatus)
    fprintf(stream, "%s: %d HDU%s\n", path, n_hdus, n_hdus == 1 ? "" : "s");

  for (int i = 1; fstatus == 0 && i <= n_hdus; i++) {
    int hdutype = 0;
    fits_movabs_hdu(f, i, &hdutype, &fstatus);
    if (fstatus)
      break;

    char name[FLEN_VALUE];
    key_string(f, "EXTNAME", name);
    fprintf(stream, "  [%d] %-16s ", i - 1, name[0] ? name : "");

    if (is_table(hdutype)) {
      fprintf(stream, "%s table, ", hdutype == BINARY_TBL ? "binary" : "ASCII");
      fstatus = print_table(f, stream);
    } else {
      int naxis = 0;
      long naxes[9] = {0};
      fits_get_img_dim(f, &naxis, &fstatus);
      if (naxis > 9)
        naxis = 9;
      fits_get_img_size(f, naxis, naxes, &fstatus);
      if (naxis == 0) {
        fputs("image, empty\n", stream);
      } else {
        fputs("image, ", stream);
        for (int a = 0; a < naxis; a++)
          fprintf(stream, "%s%ld", a ? " x " : "", naxes[a]);
        fputc('\n', stream);
      }
    }
  }

  int close_status = 0;
  fits_close_file(f, &close_status);
  fflush(stream);

  if (fstatus) {
    log_fits(fstatus, "failed to read the structure of %s", path);
    return SIF_ERR_IO;
  }
  return SIF_OK;
}

int sif_fits_inspect(const char* path) {
  return sif_fits_print_summary(path, stdout);
}

/* ------------------------------------------------------------------------ */
/* products: the file                                                        */
/* ------------------------------------------------------------------------ */

/* The HDUs of a sif FITS file, one per product, as the HDF5 file has one
 * group per product. */
#define VOIDS_EXTNAME "VOIDS"
#define DENS_EXTNAME  "DENSITY_PROFILES"
#define VEL_EXTNAME   "VELOCITY_PROFILES"
#define VSF_EXTNAME   "SIZE_FUNCTION"

/* What the primary header of every file sif writes says about it. */
#define FORMAT_KEY  "SIFFMT"
#define FORMAT_NAME "sif"

/* sif_real as cfitsio names it, in memory and in a column. */
#define REAL_TYPE (sizeof(sif_real) == 8 ? TDOUBLE : TFLOAT)
#define REAL_CODE (sizeof(sif_real) == 8 ? 'D' : 'E')
#define REAL_FORM (sizeof(sif_real) == 8 ? "1D" : "1E")

/* The file a product goes into, open for writing: created, with an empty
 * primary HDU that says it is sif's, if it is missing; refused if it exists
 * and is not sif's -- a file somebody else wrote is never written into. */
static int product_open(const char* path, fitsfile** out) {
  *out = NULL;
  fitsfile* f = NULL;
  int st = 0;
  struct stat sb;

  if (stat(path, &sb) != 0) {
    fits_create_diskfile(&f, path, &st);
    fits_create_img(f, BYTE_IMG, 0, NULL, &st);
    fits_update_key(f, TSTRING, FORMAT_KEY, FORMAT_NAME, "written by sif", &st);
    fits_update_key(f, TSTRING, "SIFVER", SIF_VERSION_STRING,
      "the sif version that wrote the file", &st);
    if (st) {
      log_fits(st, "failed to create %s", path);
      if (f) {
        int close_st = 0;
        fits_close_file(f, &close_st);
      }
      return SIF_ERR_IO;
    }
    *out = f;
    return SIF_OK;
  }

  fits_open_diskfile(&f, path, READWRITE, &st);
  if (st) {
    log_fits(st, "failed to open %s for writing", path);
    return SIF_ERR_IO;
  }
  char format[FLEN_VALUE];
  key_string(f, FORMAT_KEY, format);
  if (strcmp(format, FORMAT_NAME) != 0) {
    SIF_LOG_ERROR(TAG,
      "%s is a FITS file sif did not write, and is not written into; give "
      "another path",
      path);
    fits_close_file(f, &st);
    return SIF_ERR_IO;
  }
  *out = f;
  return SIF_OK;
}

/* Deletes the HDU named @p extname if the file has one, so a product written
 * again replaces its old self and nothing else. */
static int hdu_remove(fitsfile* f, const char* extname) {
  char name[FLEN_VALUE];
  snprintf(name, sizeof name, "%s", extname);
  int st = 0;
  fits_movnam_hdu(f, ANY_HDU, name, 0, &st);
  if (st == BAD_HDU_NUM) {
    fits_clear_errmsg();
    return 0;
  }
  if (!st)
    fits_delete_hdu(f, NULL, &st);
  return st;
}

/* Closes a file written into, and says whether everything reached it. */
static int product_close(fitsfile* f, int st, const char* path) {
  int close_st = 0;
  fits_close_file(f, &close_st);
  if (st || close_st) {
    log_fits(st ? st : close_st, "failed to write %s", path);
    return SIF_ERR_IO;
  }
  return SIF_OK;
}

/* Opens a file for reading at a product's HDU. SIF_ERR_INVALID for a file
 * without it, SIF_ERR_IO for one that cannot be read. */
static int product_read_open(
  const char* path, const char* extname, fitsfile** out) {
  *out = NULL;
  fitsfile* f = NULL;
  int st = 0;
  fits_open_diskfile(&f, path, READONLY, &st);
  if (st) {
    log_fits(st, "failed to open %s", path);
    return SIF_ERR_IO;
  }
  char name[FLEN_VALUE];
  snprintf(name, sizeof name, "%s", extname);
  fits_movnam_hdu(f, ANY_HDU, name, 0, &st);
  if (st) {
    fits_clear_errmsg();
    SIF_LOG_ERROR(TAG, "%s holds no %s table", path, extname);
    st = 0;
    fits_close_file(f, &st);
    return SIF_ERR_INVALID;
  }
  *out = f;
  return SIF_OK;
}

/* Warns when a product's rows and the catalogue's disagree -- a product
 * written for another catalogue -- as the HDF5 file does; never refuses. */
static void warn_rows(
  fitsfile* f, const char* path, uint64_t n_rows, const char* extname) {
  char name[] = VOIDS_EXTNAME;
  int st = 0;
  fits_movnam_hdu(f, BINARY_TBL, name, 0, &st);
  LONGLONG n = 0;
  if (!st)
    fits_get_num_rowsll(f, &n, &st);
  if (st) {
    fits_clear_errmsg();
    return;
  }
  if ((uint64_t)n != n_rows)
    SIF_LOG_WARNING(TAG,
      "%s: %s has %" PRIu64 " rows but the catalogue %lld voids; they do not "
      "belong together",
      path, extname, n_rows, (long long)n);
}

/* ------------------------------------------------------------------------ */
/* catalogues                                                                */
/* ------------------------------------------------------------------------ */

/* The catalogue's metadata, as keywords of the current header. */
static void meta_write(fitsfile* f, const sif_catalog_t* catalog, int* st) {
  for (uint32_t m = 0; m < sif_catalog_meta_count(catalog) && !*st; m++) {
    char key[FLEN_KEYWORD + 64];
    snprintf(key, sizeof key, "%s", sif_catalog_meta_name(catalog, m));
    switch (sif_catalog_meta_kind(catalog, key)) {
    case SIF_CATALOG_META_INT: {
      LONGLONG v = (LONGLONG)sif_catalog_meta_int_get(catalog, key);
      fits_update_key(f, TLONGLONG, key, &v, NULL, st);
      break;
    }
    case SIF_CATALOG_META_REAL: {
      double v = sif_catalog_meta_real_get(catalog, key);
      fits_update_key(f, TDOUBLE, key, &v, NULL, st);
      break;
    }
    case SIF_CATALOG_META_STRING:
      fits_update_key_longstr(
        f, key, (char*)sif_catalog_meta_string_get(catalog, key), NULL, st);
      break;
    case SIF_CATALOG_META_MISSING:
      break;
    }
  }
}

/* Every keyword of the current header that is not structure, as the
 * catalogue's metadata: what sif wrote, and whatever anyone added. */
static void meta_read(fitsfile* f, sif_catalog_t* catalog) {
  int n_keys = 0, st = 0;
  fits_get_hdrspace(f, &n_keys, NULL, &st);
  for (int k = 1; k <= n_keys && !st; k++) {
    char name[FLEN_KEYWORD], value[FLEN_VALUE], lower[FLEN_KEYWORD];
    fits_read_keyn(f, k, name, value, NULL, &st);
    if (st || !name[0])
      break;
    size_t c = 0;
    for (; name[c] && c < sizeof lower - 1; c++)
      lower[c] = (char)tolower((unsigned char)name[c]);
    lower[c] = '\0';
    if (sif__catalog_meta_reserved(lower))
      continue;

    char kind = 0;
    int kst = 0;
    fits_get_keytype(value, &kind, &kst);
    if (kst) {
      fits_clear_errmsg();
      continue;
    }
    if (kind == 'C') {
      char* text = NULL;
      fits_read_key_longstr(f, name, &text, NULL, &kst);
      if (!kst && text)
        (void)sif_catalog_meta_string_set(catalog, lower, text);
      if (text)
        fits_free_memory(text, &kst);
      fits_clear_errmsg();
    } else if (kind == 'L') {
      (void)sif_catalog_meta_int_set(catalog, lower, value[0] == 'T');
    } else if (kind == 'I') {
      (void)sif_catalog_meta_int_set(catalog, lower, strtoll(value, NULL, 10));
    } else if (kind == 'F') {
      /* FITS may write the exponent with a D. */
      for (char* d = value; *d; d++)
        if (*d == 'D' || *d == 'd')
          *d = 'E';
      (void)sif_catalog_meta_real_set(catalog, lower, strtod(value, NULL));
    }
  }
  fits_clear_errmsg();
}

int sif_catalog_write_fits(const char* filepath, const sif_catalog_t* catalog) {
  if (!filepath || !catalog) {
    SIF_LOG_ERROR(TAG, "invalid arguments for sif_catalog_write_fits");
    return SIF_ERR_INVALID;
  }

  const bool sky = catalog->units == SIF_COORDINATES_SKY;
  const bool fp = catalog->footprint != NULL;
  /* The names the ASCII header gives the columns, as FITS spells them. */
  char* names_cart[] = {"CX", "CY", "CZ", "R", "FOOTPRINT", "FOOTPRINT_SHELL"};
  char* names_sky[] = {"RA", "DEC", "Z", "R", "FOOTPRINT", "FOOTPRINT_SHELL"};
  char* units_cart[] = {"", "", "", "", "", ""};
  char* units_sky[] = {"deg", "deg", "", "Mpc/h", "", ""};
  char* form[6];
  for (int c = 0; c < 6; c++)
    form[c] = (char*)REAL_FORM;
  const int n_cols = fp ? 6 : 4;
  sif_real* const cols[6] = {catalog->cx, catalog->cy, catalog->cz,
    catalog->radii, catalog->footprint, catalog->footprint_shell};

  fitsfile* f;
  if (product_open(filepath, &f) != SIF_OK)
    return SIF_ERR_IO;

  int st = hdu_remove(f, VOIDS_EXTNAME);
  fits_create_tbl(f, BINARY_TBL, (LONGLONG)catalog->n_voids, n_cols,
    sky ? names_sky : names_cart, form, sky ? units_sky : units_cart,
    VOIDS_EXTNAME, &st);
  fits_update_key(f, TSTRING, "COORDS", sky ? "sky" : "cartesian",
    "what the centres are", &st);
  meta_write(f, catalog, &st);
  for (int c = 0; c < n_cols && catalog->n_voids > 0; c++)
    fits_write_col(
      f, REAL_TYPE, c + 1, 1, 1, (LONGLONG)catalog->n_voids, cols[c], &st);

  if (product_close(f, st, filepath) != SIF_OK)
    return SIF_ERR_IO;
  SIF_LOG_INFO(TAG, "saved %" PRIu64 " voids to %s (FITS%s)", catalog->n_voids,
    filepath, sky ? ", on the sky" : "");
  return SIF_OK;
}

/* A column's number, or 0 if the table has none of that name. */
static int column_of(fitsfile* f, const char* name) {
  char buf[FLEN_VALUE];
  snprintf(buf, sizeof buf, "%s", name);
  int col = 0, fstatus = 0;
  fits_get_colnum(f, CASEINSEN, buf, &col, &fstatus);
  if (fstatus) {
    fits_clear_errmsg();
    return 0;
  }
  return col;
}

sif_catalog_t* sif_catalog_read_fits(const char* filepath) {
  if (!filepath) {
    SIF_LOG_ERROR(TAG, "invalid filepath for sif_catalog_read_fits");
    return NULL;
  }

  fitsfile* f = NULL;
  int fstatus = 0;
  fits_open_diskfile(&f, filepath, READONLY, &fstatus);
  if (fstatus) {
    log_fits(fstatus, "failed to open %s", filepath);
    return NULL;
  }

  sif_catalog_t* cat = NULL;
  int status = move_to_table(f, filepath, VOIDS_EXTNAME);
  LONGLONG n = 0;
  if (status == SIF_OK) {
    fits_get_num_rowsll(f, &n, &fstatus);
    if (fstatus) {
      log_fits(fstatus, "failed to read the row count of %s", filepath);
      status = SIF_ERR_IO;
    }
  }

  /* The centres' names say what they are: RA, DEC and Z on the sky, CX, CY
   * and CZ in a box -- or X, Y and Z, as other tools write them. R or
   * RADIUS for the radius. */
  int cols[6] = {0};
  bool sky = false;
  if (status == SIF_OK) {
    sky = column_of(f, "RA") > 0;
    if (sky) {
      cols[0] = column_of(f, "RA");
      cols[1] = column_of(f, "DEC");
      cols[2] = column_of(f, "Z");
    } else {
      cols[0] = column_of(f, "CX") ? column_of(f, "CX") : column_of(f, "X");
      cols[1] = column_of(f, "CY") ? column_of(f, "CY") : column_of(f, "Y");
      cols[2] = column_of(f, "CZ") ? column_of(f, "CZ") : column_of(f, "Z");
    }
    cols[3] = column_of(f, "R") ? column_of(f, "R") : column_of(f, "RADIUS");
    cols[4] = column_of(f, "FOOTPRINT");
    cols[5] = column_of(f, "FOOTPRINT_SHELL");
    if (!cols[0] || !cols[1] || !cols[2] || !cols[3]) {
      SIF_LOG_ERROR(TAG,
        "%s: the " VOIDS_EXTNAME " table has no %s and R columns", filepath,
        sky ? "RA, DEC, Z" : "CX, CY, CZ");
      status = SIF_ERR_INVALID;
    } else if (!cols[4] != !cols[5]) {
      SIF_LOG_ERROR(
        TAG, "%s: one footprint column without the other", filepath);
      status = SIF_ERR_INVALID;
    }
  }

  if (status == SIF_OK) {
    cat = sif_catalog_alloc((uint64_t)n);
    if (!cat || (cols[4] && sif_catalog_reserve_footprint(cat) != SIF_OK))
      status = SIF_ERR_ALLOC;
  }

  if (status == SIF_OK && n > 0) {
    sif_real* const out[6] = {cat->cx, cat->cy, cat->cz, cat->radii,
      cat->footprint, cat->footprint_shell};
    for (int c = 0; c < 6 && !fstatus; c++) {
      if (!cols[c])
        continue;
      int anynul = 0;
      fits_read_col(
        f, REAL_TYPE, cols[c], 1, 1, n, NULL, out[c], &anynul, &fstatus);
    }
    if (fstatus) {
      log_fits(fstatus, "failed to read the voids of %s", filepath);
      status = SIF_ERR_IO;
    }
  }

  if (status == SIF_OK)
    meta_read(f, cat);

  int close_status = 0;
  fits_close_file(f, &close_status);

  if (status != SIF_OK) {
    sif_catalog_free(cat);
    return NULL;
  }

  /* Filled in directly rather than through sif_catalog_append(). */
  cat->n_voids = (uint64_t)n;
  if (sky)
    cat->units = SIF_COORDINATES_SKY;

  SIF_LOG_INFO(TAG, "loaded %" PRIu64 " voids from %s (FITS%s)", cat->n_voids,
    filepath, sky ? ", on the sky" : "");
  return cat;
}

/* ------------------------------------------------------------------------ */
/* profiles                                                                  */
/* ------------------------------------------------------------------------ */

/* The widest ladder of bin edges the EDGE0 ... keywords can name. */
#define MAX_PROFILE_BINS 9999u

/* One profile set as a table: a row per void, one vector column of n_bins,
 * the shape and the bin edges as keywords. */
static void profiles_table_write(fitsfile* f, const char* extname,
  const char* column, uint64_t n_voids, uint32_t n_bins, sif_real ext,
  const sif_real* r_edges, const sif_real* rows, int differential, int* st) {

  char form[16];
  snprintf(form, sizeof form, "%" PRIu32 "%c", n_bins, REAL_CODE);
  char* ttype[] = {(char*)column};
  char* tform[] = {form};
  *st = *st ? *st : hdu_remove(f, extname);
  fits_create_tbl(
    f, BINARY_TBL, (LONGLONG)n_voids, 1, ttype, tform, NULL, extname, st);

  LONGLONG nb = n_bins;
  double e = (double)ext;
  fits_update_key(f, TLONGLONG, "N_BINS", &nb, "bins per profile", st);
  fits_update_key(f, TDOUBLE, "EXT", &e, "outer edge, in void radii", st);
  if (differential >= 0) {
    int d = differential;
    fits_update_key(f, TLOGICAL, "DIFFERENTIAL", &d,
      "a bin holds its own shell (T) or all it encloses (F)", st);
  }
  for (uint32_t j = 0; j <= n_bins; j++) {
    char key[FLEN_KEYWORD];
    snprintf(key, sizeof key, "EDGE%" PRIu32, j);
    double v = (double)r_edges[j];
    fits_update_key(f, TDOUBLE, key, &v, NULL, st);
  }
  if (n_voids > 0)
    fits_write_col(
      f, REAL_TYPE, 1, 1, 1, (LONGLONG)(n_voids * n_bins), (void*)rows, st);
}

int sif_profiles_write_fits(const char* filepath,
  const sif_density_profiles_t* dens, const sif_velocity_profiles_t* vel) {

  if (!filepath || (!dens && !vel)) {
    SIF_LOG_ERROR(
      TAG, "sif_profiles_write_fits needs a path and at least one profile set");
    return SIF_ERR_INVALID;
  }
  if ((dens && dens->n_bins > MAX_PROFILE_BINS) ||
      (vel && vel->n_bins > MAX_PROFILE_BINS)) {
    SIF_LOG_ERROR(TAG, "profiles of more than %u bins do not fit a FITS header",
      MAX_PROFILE_BINS);
    return SIF_ERR_INVALID;
  }

  fitsfile* f;
  if (product_open(filepath, &f) != SIF_OK)
    return SIF_ERR_IO;

  int st = 0;
  if (dens)
    profiles_table_write(f, DENS_EXTNAME, "DENSITY", dens->n_voids,
      dens->n_bins, dens->ext, dens->r_edges, dens->profiles,
      dens->differential ? 1 : 0, &st);
  if (vel)
    profiles_table_write(f, VEL_EXTNAME, "V_RAD", vel->n_voids, vel->n_bins,
      vel->ext, vel->r_edges, vel->v_rad, -1, &st);
  if (!st && dens)
    warn_rows(f, filepath, dens->n_voids, DENS_EXTNAME);
  if (!st && vel)
    warn_rows(f, filepath, vel->n_voids, VEL_EXTNAME);

  if (product_close(f, st, filepath) != SIF_OK)
    return SIF_ERR_IO;
  SIF_LOG_INFO(TAG, "saved profiles to %s (FITS)", filepath);
  return SIF_OK;
}

/* Whether the file has an HDU of that name. */
static bool has_hdu(fitsfile* f, const char* extname) {
  char name[FLEN_VALUE];
  snprintf(name, sizeof name, "%s", extname);
  int st = 0;
  fits_movnam_hdu(f, ANY_HDU, name, 0, &st);
  fits_clear_errmsg();
  return st == 0;
}

int sif_profiles_read_header_fits(
  const char* filepath, int* out_has_density, int* out_has_velocity) {
  if (!filepath) {
    SIF_LOG_ERROR(TAG, "invalid filepath for sif_profiles_read_header_fits");
    return SIF_ERR_INVALID;
  }
  fitsfile* f = NULL;
  int st = 0;
  fits_open_diskfile(&f, filepath, READONLY, &st);
  if (st) {
    log_fits(st, "failed to open %s", filepath);
    return SIF_ERR_IO;
  }
  if (out_has_density)
    *out_has_density = has_hdu(f, DENS_EXTNAME);
  if (out_has_velocity)
    *out_has_velocity = has_hdu(f, VEL_EXTNAME);
  fits_close_file(f, &st);
  return SIF_OK;
}

/* A profile table's shape and edges, from its header; the rows are read by
 * the caller once the set is allocated. */
static int profiles_shape(fitsfile* f, uint64_t* n_voids, uint32_t* n_bins,
  sif_real* ext, int* differential) {
  int st = 0;
  LONGLONG n = 0, nb = 0;
  double e = 0.0;
  fits_get_num_rowsll(f, &n, &st);
  fits_read_key(f, TLONGLONG, "N_BINS", &nb, NULL, &st);
  fits_read_key(f, TDOUBLE, "EXT", &e, NULL, &st);
  if (differential) {
    int d = 0;
    fits_read_key(f, TLOGICAL, "DIFFERENTIAL", &d, NULL, &st);
    *differential = d;
  }
  if (st || nb <= 0 || nb > (LONGLONG)MAX_PROFILE_BINS) {
    fits_clear_errmsg();
    return SIF_ERR_IO;
  }
  *n_voids = (uint64_t)n;
  *n_bins = (uint32_t)nb;
  *ext = (sif_real)e;
  return SIF_OK;
}

static int profiles_rows_read(fitsfile* f, uint64_t n_voids, uint32_t n_bins,
  sif_real* edges, sif_real* rows) {
  int st = 0;
  for (uint32_t j = 0; j <= n_bins && !st; j++) {
    char key[FLEN_KEYWORD];
    snprintf(key, sizeof key, "EDGE%" PRIu32, j);
    double v = 0.0;
    fits_read_key(f, TDOUBLE, key, &v, NULL, &st);
    edges[j] = (sif_real)v;
  }
  int anynul = 0;
  if (!st && n_voids > 0)
    fits_read_col(f, REAL_TYPE, 1, 1, 1, (LONGLONG)(n_voids * n_bins), NULL,
      rows, &anynul, &st);
  if (st) {
    fits_clear_errmsg();
    return SIF_ERR_IO;
  }
  return SIF_OK;
}

int sif_profiles_read_fits(const char* filepath,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {

  if (out_dens)
    *out_dens = NULL;
  if (out_vel)
    *out_vel = NULL;
  if (!filepath || (!out_dens && !out_vel)) {
    SIF_LOG_ERROR(
      TAG, "sif_profiles_read_fits needs a path and at least one output");
    return SIF_ERR_INVALID;
  }

  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;
  fitsfile* f = NULL;
  int status = SIF_OK;

  if (out_dens) {
    uint64_t n;
    uint32_t b;
    sif_real ext;
    int differential;
    status = product_read_open(filepath, DENS_EXTNAME, &f);
    if (status == SIF_OK &&
        profiles_shape(f, &n, &b, &ext, &differential) != SIF_OK)
      status = SIF_ERR_IO;
    if (status == SIF_OK &&
        !(dens = sif__density_profiles_alloc(n, b, ext, differential != 0)))
      status = SIF_ERR_ALLOC;
    if (status == SIF_OK)
      status = profiles_rows_read(f, n, b, dens->r_edges, dens->profiles);
    if (f) {
      int st = 0;
      fits_close_file(f, &st);
      f = NULL;
    }
  }

  if (status == SIF_OK && out_vel) {
    uint64_t n;
    uint32_t b;
    sif_real ext;
    status = product_read_open(filepath, VEL_EXTNAME, &f);
    if (status == SIF_OK && profiles_shape(f, &n, &b, &ext, NULL) != SIF_OK)
      status = SIF_ERR_IO;
    if (status == SIF_OK && !(vel = sif__velocity_profiles_alloc(n, b, ext)))
      status = SIF_ERR_ALLOC;
    if (status == SIF_OK)
      status = profiles_rows_read(f, n, b, vel->r_edges, vel->v_rad);
    if (f) {
      int st = 0;
      fits_close_file(f, &st);
    }
  }

  if (status != SIF_OK) {
    if (status == SIF_ERR_IO)
      SIF_LOG_ERROR(TAG, "failed to read profiles from %s", filepath);
    sif_density_profiles_free(dens);
    sif_velocity_profiles_free(vel);
    return status;
  }
  if (out_dens)
    *out_dens = dens;
  if (out_vel)
    *out_vel = vel;
  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* size function                                                             */
/* ------------------------------------------------------------------------ */

int sif_size_function_write_fits(
  const char* filepath, const sif_size_function_t* vsf) {
  if (!filepath || !vsf) {
    SIF_LOG_ERROR(TAG, "invalid arguments for sif_size_function_write_fits");
    return SIF_ERR_INVALID;
  }

  /* A row per bin: its edges and centre, the count and the size function
   * with its error -- the table a plot reads. */
  char* ttype[] = {"R_LOW", "R_HIGH", "R_CENTER", "COUNT", "VSF", "ERR"};
  char* tform[] = {(char*)REAL_FORM, (char*)REAL_FORM, (char*)REAL_FORM, "1K",
    (char*)REAL_FORM, (char*)REAL_FORM};
  const uint32_t b = vsf->n_bins;

  fitsfile* f;
  if (product_open(filepath, &f) != SIF_OK)
    return SIF_ERR_IO;

  int st = hdu_remove(f, VSF_EXTNAME);
  fits_create_tbl(f, BINARY_TBL, b, 6, ttype, tform, NULL, VSF_EXTNAME, &st);

  /* The binning is also inside OPTIONS, but spelled out: per unit ln R or
   * per unit R is the one thing a reader needs to read the values. */
  const char* binning =
    ((vsf->options & SIF__VSF_BIN_MASK) == SIF_VSF_BIN_LINEAR) ? "linear"
                                                               : "ln";
  LONGLONG nb = b, opt = (LONGLONG)vsf->options;
  double r_min = (double)vsf->r_min, r_max = (double)vsf->r_max;
  fits_update_key(f, TLONGLONG, "N_BINS", &nb, NULL, &st);
  fits_update_key(f, TDOUBLE, "R_MIN", &r_min, NULL, &st);
  fits_update_key(f, TDOUBLE, "R_MAX", &r_max, NULL, &st);
  fits_update_key(f, TLONGLONG, "OPTIONS", &opt, "sif's options", &st);
  fits_update_key(f, TSTRING, "BINNING", (char*)binning,
    "VSF per unit ln R, or per unit R", &st);
  if (b > 0) {
    fits_write_col(f, REAL_TYPE, 1, 1, 1, b, vsf->r_edges, &st);
    fits_write_col(f, REAL_TYPE, 2, 1, 1, b, vsf->r_edges + 1, &st);
    fits_write_col(f, REAL_TYPE, 3, 1, 1, b, vsf->r_centers, &st);
    fits_write_col(f, TULONGLONG, 4, 1, 1, b, vsf->counts, &st);
    fits_write_col(f, REAL_TYPE, 5, 1, 1, b, vsf->vsf, &st);
    fits_write_col(f, REAL_TYPE, 6, 1, 1, b, vsf->err, &st);
  }

  if (product_close(f, st, filepath) != SIF_OK)
    return SIF_ERR_IO;
  SIF_LOG_INFO(TAG, "saved the size function to %s (FITS)", filepath);
  return SIF_OK;
}

sif_size_function_t* sif_size_function_read_fits(const char* filepath) {
  if (!filepath) {
    SIF_LOG_ERROR(TAG, "invalid filepath for sif_size_function_read_fits");
    return NULL;
  }
  fitsfile* f;
  if (product_read_open(filepath, VSF_EXTNAME, &f) != SIF_OK)
    return NULL;

  int st = 0;
  LONGLONG nb = 0, opt = 0;
  double r_min = 0.0, r_max = 0.0;
  fits_read_key(f, TLONGLONG, "N_BINS", &nb, NULL, &st);
  fits_read_key(f, TDOUBLE, "R_MIN", &r_min, NULL, &st);
  fits_read_key(f, TDOUBLE, "R_MAX", &r_max, NULL, &st);
  fits_read_key(f, TLONGLONG, "OPTIONS", &opt, NULL, &st);

  sif_size_function_t* vsf = !st && nb > 0 && nb <= UINT32_MAX
                               ? sif__size_function_alloc((uint32_t)nb)
                               : NULL;
  if (vsf) {
    const LONGLONG b = nb;
    int anynul = 0;
    vsf->options = (sif_option)opt;
    vsf->r_min = (sif_real)r_min;
    vsf->r_max = (sif_real)r_max;
    fits_read_col(f, REAL_TYPE, 1, 1, 1, b, NULL, vsf->r_edges, &anynul, &st);
    fits_read_col(f, REAL_TYPE, 2, b, 1, 1, NULL, vsf->r_edges + b, &anynul,
      &st); /* the last upper edge */
    fits_read_col(f, REAL_TYPE, 3, 1, 1, b, NULL, vsf->r_centers, &anynul, &st);
    fits_read_col(f, TULONGLONG, 4, 1, 1, b, NULL, vsf->counts, &anynul, &st);
    fits_read_col(f, REAL_TYPE, 5, 1, 1, b, NULL, vsf->vsf, &anynul, &st);
    fits_read_col(f, REAL_TYPE, 6, 1, 1, b, NULL, vsf->err, &anynul, &st);
  }

  int close_st = 0;
  fits_close_file(f, &close_st);
  if (st || !vsf) {
    if (st)
      log_fits(st, "failed to read the size function of %s", filepath);
    else
      SIF_LOG_ERROR(TAG, "%s: the size function table is malformed", filepath);
    sif_size_function_free(vsf);
    return NULL;
  }
  return vsf;
}

/* ------------------------------------------------------------------------ */
/* keywords                                                                  */
/* ------------------------------------------------------------------------ */

/* Opens a file at the HDU a keyword lives in: the primary header for NULL or
 * "", otherwise an extension number or an EXTNAME. */
static int open_at(
  const char* path, const char* hdu, int mode, fitsfile** out) {
  *out = NULL;
  if (!path) {
    SIF_LOG_ERROR(TAG, "a path is required");
    return SIF_ERR_INVALID;
  }

  fitsfile* f = NULL;
  int fstatus = 0;
  fits_open_diskfile(&f, path, mode, &fstatus);
  if (fstatus) {
    log_fits(fstatus, "failed to open %s", path);
    return SIF_ERR_IO;
  }
  if (hdu && *hdu) {
    int hdutype = 0;
    const int status = move_to_hdu(f, path, hdu, &hdutype);
    if (status != SIF_OK) {
      fstatus = 0;
      fits_close_file(f, &fstatus);
      return status;
    }
  }
  *out = f;
  return SIF_OK;
}

/* A keyword's value as written, and its kind, or MISSING. Names longer than
 * eight characters are the HIERARCH keywords cfitsio writes for them. */
static sif_fits_key_kind_t key_raw(
  fitsfile* f, const char* key, char value[FLEN_VALUE]) {
  char name[FLEN_KEYWORD + 16];
  snprintf(name, sizeof name, "%s", key);
  int fstatus = 0;
  fits_read_keyword(f, name, value, NULL, &fstatus);
  if (fstatus) {
    fits_clear_errmsg();
    return SIF_FITS_KEY_MISSING;
  }

  char dtype = 0;
  fits_get_keytype(value, &dtype, &fstatus);
  if (fstatus) {
    fits_clear_errmsg();
    return SIF_FITS_KEY_MISSING; /* a keyword with no value */
  }
  switch (dtype) {
  case 'C':
    return SIF_FITS_KEY_STRING;
  case 'L':
    return SIF_FITS_KEY_LOGICAL;
  case 'I':
    return SIF_FITS_KEY_INT;
  case 'F':
    return SIF_FITS_KEY_REAL;
  default:
    return SIF_FITS_KEY_MISSING; /* complex: nothing here reads it */
  }
}

sif_fits_key_kind_t sif_fits_key_kind(
  const char* path, const char* hdu, const char* key) {
  fitsfile* f;
  if (!key || open_at(path, hdu, READONLY, &f) != SIF_OK)
    return SIF_FITS_KEY_MISSING;
  char value[FLEN_VALUE];
  const sif_fits_key_kind_t kind = key_raw(f, key, value);
  int fstatus = 0;
  fits_close_file(f, &fstatus);
  return kind;
}

double sif_fits_get_key_real(
  const char* path, const char* hdu, const char* key) {
  fitsfile* f;
  if (!key || open_at(path, hdu, READONLY, &f) != SIF_OK)
    return 0.0;

  char value[FLEN_VALUE];
  double out = 0.0;
  const sif_fits_key_kind_t kind = key_raw(f, key, value);
  if (kind == SIF_FITS_KEY_LOGICAL) {
    out = value[0] == 'T' ? 1.0 : 0.0;
  } else if (kind == SIF_FITS_KEY_INT || kind == SIF_FITS_KEY_REAL) {
    char name[FLEN_KEYWORD + 16];
    snprintf(name, sizeof name, "%s", key);
    int fstatus = 0;
    fits_read_key(f, TDOUBLE, name, &out, NULL, &fstatus);
    if (fstatus) {
      fits_clear_errmsg();
      out = 0.0;
    }
  } else if (kind == SIF_FITS_KEY_STRING) {
    SIF_LOG_WARNING(TAG, "%s: keyword %s holds text, not a number", path, key);
  }

  int fstatus = 0;
  fits_close_file(f, &fstatus);
  return out;
}

void sif_fits_get_key_string(
  const char* path, const char* hdu, const char* key, char* buf, size_t len) {
  if (!buf || len == 0)
    return;
  buf[0] = '\0';

  fitsfile* f;
  if (!key || open_at(path, hdu, READONLY, &f) != SIF_OK)
    return;

  char value[FLEN_VALUE];
  const sif_fits_key_kind_t kind = key_raw(f, key, value);
  if (kind == SIF_FITS_KEY_STRING) {
    /* The value without its quotes, and a long string whole: cfitsio follows
     * CONTINUE cards and allocates what it finds. */
    char name[FLEN_KEYWORD + 16];
    snprintf(name, sizeof name, "%s", key);
    char* text = NULL;
    int fstatus = 0;
    fits_read_key_longstr(f, name, &text, NULL, &fstatus);
    if (!fstatus && text)
      snprintf(buf, len, "%s", text);
    else
      fits_clear_errmsg();
    if (text)
      fits_free_memory(text, &fstatus);
  } else if (kind != SIF_FITS_KEY_MISSING) {
    /* A number or a logical, as written: "1.5", "T". */
    snprintf(buf, len, "%s", value);
  }

  int fstatus = 0;
  fits_close_file(f, &fstatus);
}

/* Writes one keyword, replacing any of the same name. */
static int set_key(
  const char* path, const char* hdu, const char* key, int type, void* value) {
  if (!key || !*key) {
    SIF_LOG_ERROR(TAG, "a keyword name is required");
    return SIF_ERR_INVALID;
  }
  fitsfile* f;
  int status = open_at(path, hdu, READWRITE, &f);
  if (status != SIF_OK)
    return status;

  char name[FLEN_KEYWORD + 16];
  snprintf(name, sizeof name, "%s", key);
  int fstatus = 0;
  /* A string longer than a card holds goes over several, with the CONTINUE
   * convention -- a path, say -- rather than cut at 68 characters. */
  if (type == TSTRING)
    fits_update_key_longstr(f, name, (char*)value, NULL, &fstatus);
  else
    fits_update_key(f, type, name, value, NULL, &fstatus);
  if (fstatus) {
    log_fits(fstatus, "%s: failed to write keyword %s", path, key);
    status = fstatus == BAD_KEYCHAR || fstatus == BAD_ORDER ? SIF_ERR_INVALID
                                                            : SIF_ERR_IO;
  }
  int close_status = 0;
  fits_close_file(f, &close_status);
  if (status == SIF_OK && close_status) {
    log_fits(close_status, "failed to write %s", path);
    status = SIF_ERR_IO;
  }
  return status;
}

int sif_fits_set_key_int(
  const char* path, const char* hdu, const char* key, int64_t value) {
  LONGLONG v = (LONGLONG)value;
  return set_key(path, hdu, key, TLONGLONG, &v);
}

int sif_fits_set_key_real(
  const char* path, const char* hdu, const char* key, double value) {
  if (!isfinite(value)) {
    SIF_LOG_ERROR(TAG, "keyword %s: FITS has no value for %g", key, value);
    return SIF_ERR_INVALID;
  }
  return set_key(path, hdu, key, TDOUBLE, &value);
}

int sif_fits_set_key_string(
  const char* path, const char* hdu, const char* key, const char* value) {
  if (!value) {
    SIF_LOG_ERROR(TAG, "keyword %s: no value", key ? key : "(null)");
    return SIF_ERR_INVALID;
  }
  return set_key(path, hdu, key, TSTRING, (void*)value);
}

#undef VOIDS_EXTNAME
#undef DENS_EXTNAME
#undef VEL_EXTNAME
#undef VSF_EXTNAME
#undef FORMAT_KEY
#undef FORMAT_NAME
#undef MAX_PROFILE_BINS
#undef REAL_TYPE
#undef REAL_CODE
#undef REAL_FORM
#undef TAG
#undef CHUNK
