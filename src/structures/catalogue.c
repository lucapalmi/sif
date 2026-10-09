/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/catalogue.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model/cosmology_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "structures/catalogue_internal.h"
#include "utils/logger_internal.h"

/*
 * Number of sif_real views packed into the arena: cx, cy, cz, radii.
 *
 * One allocation rather than four, so growth is a single copy and the whole
 * catalogue is one free. Structure-of-arrays rather than an array of structs
 * because every consumer sweeps one field at a time -- the size function reads
 * only radii, the profiles read only the centres -- and a struct would drag
 * three unused values through cache on each.
 */
#define CATALOGUE_N_VIEWS 4

/* Each view starts on a cache-line boundary, so the per-view stride is the
 * capacity rounded up to a whole number of sif_real per cache line. Without
 * the rounding, two views would share the line at their boundary and the
 * threads writing them would contend over it. */
static inline uint64_t catalogue_stride(uint64_t capacity) {
  const uint64_t align_elements = SIF_CACHE_LINE / sizeof(sif_real);
  return (capacity + align_elements - 1) & ~(align_elements - 1);
}

/* Bytes of the main arena for `capacity` voids, for the messages that report
 * a failure to allocate one. */
static inline uint64_t catalogue_block_bytes(uint64_t capacity) {
  return CATALOGUE_N_VIEWS * catalogue_stride(capacity) * sizeof(sif_real);
}

/*
 * Allocates one arena for `capacity` voids and hands out the four views.
 * Returns NULL on failure without touching the outputs.
 */
static sif_real* catalogue_block_alloc(uint64_t capacity, sif_real** out_cx,
  sif_real** out_cy, sif_real** out_cz, sif_real** out_radii) {

  const uint64_t stride = catalogue_stride(capacity);

  sif_real* block =
    sif_malloc_aligned(CATALOGUE_N_VIEWS * stride * sizeof(sif_real));
  if (!block)
    return NULL;

  *out_cx = block;
  *out_cy = block + stride;
  *out_cz = block + (2 * stride);
  *out_radii = block + (3 * stride);

  return block;
}

/* The two optional footprint views, laid out the same way. */
#define CATALOGUE_N_FOOTPRINT_VIEWS 2

static sif_real* catalogue_footprint_alloc(
  uint64_t capacity, sif_real** out_footprint, sif_real** out_shell) {

  const uint64_t stride = catalogue_stride(capacity);

  sif_real* block =
    sif_malloc_aligned(CATALOGUE_N_FOOTPRINT_VIEWS * stride * sizeof(sif_real));
  if (!block)
    return NULL;

  *out_footprint = block;
  *out_shell = block + stride;

  return block;
}

/*
 * Moves the stored voids into a freshly allocated arena of `new_capacity` and
 * swaps it in. On failure the catalogue is left exactly as it was.
 *
 * Always a fresh arena and a copy, never a realloc: realloc may extend in
 * place, but the address it extends is only guaranteed to keep malloc's
 * alignment, and the whole point of the arena is that each view starts on a
 * cache line. Growing in place would also have to move three of the four views
 * anyway, since the stride changes with the capacity.
 *
 * The new arena is filled before the old one is released, so a failure leaves
 * the catalogue untouched and the caller still holding something valid -- which
 * is what lets sif_catalogue_trim() treat a failed trim as merely disappointing.
 */
static int catalogue_resize(sif_catalogue_t* catalogue, uint64_t new_capacity) {
  sif_real *new_cx, *new_cy, *new_cz, *new_radii;
  sif_real *new_fp = NULL, *new_fp_shell = NULL;

  sif_real* new_block =
    catalogue_block_alloc(new_capacity, &new_cx, &new_cy, &new_cz, &new_radii);

  /* The footprint arena moves with the main one or neither moves, so a
   * failure on either leaves the catalogue exactly as it was. */
  sif_real* new_fp_block = NULL;
  if (new_block && catalogue->_footprint_block) {
    new_fp_block =
      catalogue_footprint_alloc(new_capacity, &new_fp, &new_fp_shell);
    if (!new_fp_block) {
      sif_free_aligned(new_block);
      new_block = NULL;
    }
  }

  if (!new_block) {
    SIF_LOG_ERROR("void_catalogue",
      "catalogue: cannot resize from %" PRIu64 " to %" PRIu64
      " voids (%.2f GiB); the catalogue is unchanged",
      catalogue->capacity, new_capacity,
      SIF__GIB(catalogue_block_bytes(new_capacity)));
    return SIF_ERR_ALLOC;
  }

  /* Copying only what is in use, and only what fits: the trim path shrinks,
   * so new_capacity can be below n_voids and the excess is dropped rather
   * than run off the end of the new arena. */
  const uint64_t n =
    (catalogue->n_voids < new_capacity) ? catalogue->n_voids : new_capacity;

  if (n > 0) {
    memcpy(new_cx, catalogue->cx, n * sizeof(sif_real));
    memcpy(new_cy, catalogue->cy, n * sizeof(sif_real));
    memcpy(new_cz, catalogue->cz, n * sizeof(sif_real));
    memcpy(new_radii, catalogue->radii, n * sizeof(sif_real));
    if (new_fp_block) {
      memcpy(new_fp, catalogue->footprint, n * sizeof(sif_real));
      memcpy(new_fp_shell, catalogue->footprint_shell, n * sizeof(sif_real));
    }
  }

  sif_free_aligned(catalogue->_block);
  if (new_fp_block) {
    sif_free_aligned(catalogue->_footprint_block);
    catalogue->_footprint_block = new_fp_block;
    catalogue->footprint = new_fp;
    catalogue->footprint_shell = new_fp_shell;
  }

  catalogue->_block = new_block;
  catalogue->cx = new_cx;
  catalogue->cy = new_cy;
  catalogue->cz = new_cz;
  catalogue->radii = new_radii;
  catalogue->capacity = new_capacity;
  catalogue->n_voids = n;

  return SIF_OK;
}

sif_catalogue_t* sif_catalogue_alloc(uint64_t initial_capacity) {
  /* Clamped rather than rejected, so that a catalogue which comes back
   * non-NULL is always appendable. A caller sizing from an estimate that came
   * out zero gets something usable instead of a NULL it has to special-case,
   * and the doubling in append covers the growth from there. */
  if (initial_capacity == 0)
    initial_capacity = 1;

  sif_catalogue_t* cat = malloc(sizeof(sif_catalogue_t));
  if (!cat) {
    SIF_LOG_ERROR("void_catalogue", "catalogue: cannot allocate (%zu bytes)",
      sizeof(sif_catalogue_t));
    return NULL;
  }

  cat->n_voids = 0;
  cat->capacity = initial_capacity;
  cat->_meta = NULL;
  cat->_n_meta = 0;
  cat->units = SIF_COORDINATES_CARTESIAN;
  cat->_footprint_block = NULL;
  cat->footprint = NULL;
  cat->footprint_shell = NULL;

  cat->_block = catalogue_block_alloc(
    initial_capacity, &cat->cx, &cat->cy, &cat->cz, &cat->radii);

  if (!cat->_block) {
    SIF_LOG_ERROR("void_catalogue",
      "catalogue: cannot allocate room for %" PRIu64 " voids (%.2f GiB)",
      initial_capacity, SIF__GIB(catalogue_block_bytes(initial_capacity)));
    free(cat);
    return NULL;
  }

  SIF_LOG_TRACE("void_catalogue", "void catalogue initialised");
  return cat;
}

void sif_catalogue_free(sif_catalogue_t* catalogue) {
  if (!catalogue)
    return;

  /* One arena, so one free -- the four views point into it and must not be
   * released individually. */
  sif_free_aligned(catalogue->_block);
  sif_free_aligned(catalogue->_footprint_block);
  for (uint32_t i = 0; i < catalogue->_n_meta; i++) {
    free(catalogue->_meta[i].key);
    free(catalogue->_meta[i].text);
  }
  free(catalogue->_meta);
  free(catalogue);
}

int sif_catalogue_append(
  sif_catalogue_t* catalogue, sif_real x, sif_real y, sif_real z, sif_real r) {

  if (!catalogue) {
    SIF_LOG_ERROR("void_catalogue", "catalogue: NULL");
    return SIF_ERR_INVALID;
  }

  /* Doubling, so a finder appending one void at a time pays an amortised
   * constant per append rather than a copy of the whole catalogue. A finder
   * cannot know its void count in advance, which is what rules out sizing the
   * arena once up front. */
  if (catalogue->n_voids >= catalogue->capacity) {
    int status = catalogue_resize(catalogue, catalogue->capacity << 1);
    if (status != SIF_OK)
      return status;

    SIF_LOG_TRACE(
      "void_catalogue", "resized catalogue to %" PRIu64, catalogue->capacity);
  }

  const uint64_t n = catalogue->n_voids;
  catalogue->cx[n] = x;
  catalogue->cy[n] = y;
  catalogue->cz[n] = z;
  catalogue->radii[n] = r;
  if (catalogue->footprint) {
    catalogue->footprint[n] = SIF_CATALOGUE_FOOTPRINT_UNKNOWN;
    catalogue->footprint_shell[n] = SIF_CATALOGUE_FOOTPRINT_UNKNOWN;
  }
  catalogue->n_voids++;

  return SIF_OK;
}

int sif_catalogue_trim(sif_catalogue_t* catalogue) {
  if (!catalogue) {
    SIF_LOG_ERROR("void_catalogue", "catalogue: NULL");
    return SIF_ERR_INVALID;
  }

  /* An empty catalogue is a legitimate result, not an error. Trim it to the
   * minimum usable allocation rather than warning about it. */
  const uint64_t target = (catalogue->n_voids > 0) ? catalogue->n_voids : 1;

  if (target == catalogue->capacity) {
    SIF_LOG_TRACE(
      "void_catalogue", "catalogue already exact, no trimming required");
    return SIF_OK;
  }

  /* Not fatal: the catalogue is still correct, just larger than necessary,
   * and the resize has said so. */
  int status = catalogue_resize(catalogue, target);
  if (status != SIF_OK)
    return status;

  SIF_LOG_TRACE(
    "void_catalogue", "trimmed catalogue to %" PRIu64 " voids", catalogue->n_voids);
  return SIF_OK;
}

int sif_catalogue_reserve_footprint(sif_catalogue_t* catalogue) {
  if (!catalogue) {
    SIF_LOG_ERROR("void_catalogue", "catalogue: NULL");
    return SIF_ERR_INVALID;
  }

  if (catalogue->_footprint_block)
    return SIF_OK;

  sif_real *fp, *fp_shell;
  sif_real* block = catalogue_footprint_alloc(catalogue->capacity, &fp, &fp_shell);
  if (!block) {
    SIF_LOG_ERROR("void_catalogue",
      "catalogue: cannot allocate the footprint columns for %" PRIu64
      " voids (%.2f GiB)",
      catalogue->capacity,
      SIF__GIB(CATALOGUE_N_FOOTPRINT_VIEWS *
               catalogue_stride(catalogue->capacity) * sizeof(sif_real)));
    return SIF_ERR_ALLOC;
  }

  for (uint64_t i = 0; i < catalogue->n_voids; i++) {
    fp[i] = SIF_CATALOGUE_FOOTPRINT_UNKNOWN;
    fp_shell[i] = SIF_CATALOGUE_FOOTPRINT_UNKNOWN;
  }

  catalogue->_footprint_block = block;
  catalogue->footprint = fp;
  catalogue->footprint_shell = fp_shell;

  return SIF_OK;
}

int sif_catalogue_translate(sif_catalogue_t* catalogue, const sif_real offset[3]) {
  if (!catalogue || !offset) {
    SIF_LOG_ERROR(
      "void_catalogue", "%s: NULL", !catalogue ? "catalogue" : "offset");
    return SIF_ERR_INVALID;
  }
  if (catalogue->units != SIF_COORDINATES_CARTESIAN) {
    SIF_LOG_ERROR("void_catalogue",
      "catalogue: sky coordinates (ra, dec, z), not positions; a translation "
      "would move angles");
    return SIF_ERR_INVALID;
  }

  for (uint64_t i = 0; i < catalogue->n_voids; i++) {
    catalogue->cx[i] += offset[0];
    catalogue->cy[i] += offset[1];
    catalogue->cz[i] += offset[2];
  }

  return SIF_OK;
}

int sif_catalogue_to_sky(sif_catalogue_t* catalogue, const sif_cosmology_t* cosmo) {
  if (!catalogue || !cosmo) {
    SIF_LOG_ERROR(
      "void_catalogue", "%s: NULL", !catalogue ? "catalogue" : "cosmology");
    return SIF_ERR_INVALID;
  }
  if (catalogue->units != SIF_COORDINATES_CARTESIAN) {
    SIF_LOG_ERROR("void_catalogue", "catalogue: already in sky coordinates");
    return SIF_ERR_INVALID;
  }

  /* Everything checked, and the table built, before anything is changed. */
  const uint64_t n = catalogue->n_voids;
  double d_max = 0.0;
  uint64_t bad = 0;
  uint64_t first = UINT64_MAX;
  for (uint64_t i = 0; i < n; i++) {
    const double x = catalogue->cx[i], y = catalogue->cy[i], z = catalogue->cz[i];
    const double d = sqrt(x * x + y * y + z * z);
    if (!isfinite(d)) {
      if (!bad)
        first = i;
      bad++;
    } else if (d > d_max) {
      d_max = d;
    }
  }
  if (bad) {
    SIF_LOG_ERROR("void_catalogue",
      "catalogue: %" PRIu64 " of %" PRIu64 " centres not finite (first: void "
      "%" PRIu64 " at (%g, %g, %g)); nothing converted",
      bad, n, first, (double)catalogue->cx[first],
      (double)catalogue->cy[first], (double)catalogue->cz[first]);
    return SIF_ERR_INVALID;
  }

  sif__distance_table_t table;
  const int status = sif__distance_table_build_to(cosmo, d_max, &table);
  if (status != SIF_OK)
    return status;

  const double deg = 180.0 / 3.14159265358979323846;

#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n; i++) {
    const double x = catalogue->cx[i], y = catalogue->cy[i], z = catalogue->cz[i];
    const double d = sqrt(x * x + y * y + z * z);
    double ra = atan2(y, x) * deg;
    if (ra < 0.0)
      ra += 360.0;
    /* A centre at the observer has no direction; it is put at (0, 0). */
    const double dec = d > 0.0 ? asin(z / d) * deg : 0.0;
    catalogue->cx[i] = (sif_real)ra;
    catalogue->cy[i] = (sif_real)dec;
    catalogue->cz[i] = (sif_real)sif__distance_table_invert(&table, d);
  }

  sif__distance_table_free(&table);
  catalogue->units = SIF_COORDINATES_SKY;

  SIF_LOG_INFO("void_catalogue",
    "catalogue: %" PRIu64 " void centres converted to the sky, out to D_C = %g "
    "Mpc/h",
    n, d_max);
  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* metadata                                                                  */
/* ------------------------------------------------------------------------ */

#define META_KEY_MAX 64

/* The FITS keywords a table header is made of, with or without a column
 * number after them: a key named like one would be read back as structure,
 * or break the header it was written into. */
static const char* const FITS_STRUCTURAL[] = {"simple", "bitpix", "naxis",
  "extend", "xtension", "pcount", "gcount", "tfields", "ttype", "tform",
  "tunit", "tnull", "tscal", "tzero", "tdisp", "tdim", "tbcol", "theap",
  "extname", "extver", "extlevel", "end", "comment", "history", "continue",
  "checksum", "datasum", "blank", "bscale", "bzero", "bunit", "hierarch", NULL};

bool sif__catalogue_meta_reserved(const char* key) {
  if (strcmp(key, "n") == 0 || strcmp(key, "n_voids") == 0 ||
      strcmp(key, "coordinates") == 0 || strcmp(key, "coords") == 0)
    return true;
  for (const char* const* s = FITS_STRUCTURAL; *s; s++) {
    const size_t len = strlen(*s);
    if (strncmp(key, *s, len) != 0)
      continue;
    const char* rest = key + len;
    while (isdigit((unsigned char)*rest))
      rest++;
    if (!*rest)
      return true;
  }
  return false;
}

/*
 * Why `key` cannot name a value, or NULL if it can: one reason, the first
 * that applies, written into `why`. The lower-cased key goes into `out`.
 */
static const char* meta_key_problem(
  const char* key, char out[META_KEY_MAX + 1], char* why, size_t why_cap) {

  if (!key)
    return "key is NULL";
  if (!key[0])
    return "key is empty";

  size_t i = 0;
  for (; key[i]; i++) {
    const unsigned char c = (unsigned char)key[i];
    if (i == META_KEY_MAX) {
      snprintf(why, why_cap, "key is %zu characters long (at most %d)",
        strlen(key), META_KEY_MAX);
      return why;
    }
    if (i == 0 && isdigit(c)) {
      snprintf(why, why_cap,
        "key starts with a digit ('%c'); it must start with a letter or '_'",
        c);
      return why;
    }
    if (!(isalnum(c) || c == '_')) {
      if (isprint(c))
        snprintf(why, why_cap,
          "key has '%c' at character %zu (expected letters, digits and '_')",
          c, i + 1);
      else
        snprintf(why, why_cap,
          "key has byte 0x%02x at character %zu (expected letters, digits and "
          "'_')",
          c, i + 1);
      return why;
    }
    out[i] = (char)tolower(c);
  }
  out[i] = '\0';

  if (sif__catalogue_meta_reserved(out))
    return "key is reserved by the catalogue file formats (n, n_voids, "
           "coordinates, and the FITS structural keywords)";
  return NULL;
}

/* The key in lower case, into `out`, or false for one that is not allowed. */
static bool meta_key(const char* key, char out[META_KEY_MAX + 1]) {
  char why[128];
  return meta_key_problem(key, out, why, sizeof why) == NULL;
}

bool sif__catalogue_meta_key_ok(const char* key) {
  char k[META_KEY_MAX + 1];
  return meta_key(key, k);
}

static int meta_find(const sif_catalogue_t* cat, const char* key) {
  char k[META_KEY_MAX + 1];
  size_t i = 0;
  for (; key[i] && i < META_KEY_MAX; i++)
    k[i] = (char)tolower((unsigned char)key[i]);
  k[i] = '\0';
  for (uint32_t m = 0; m < cat->_n_meta; m++)
    if (strcmp(cat->_meta[m].key, k) == 0)
      return (int)m;
  return -1;
}

/* The entry for a key, new at the end or the one it already has, emptied of
 * any string it held, into *slot. SIF_ERR_INVALID for a key that is refused,
 * SIF_ERR_ALLOC if the table could not grow; logged either way. */
static int meta_slot(sif_catalogue_t* cat, const char* key,
  sif_catalogue_meta_entry_t** slot) {

  if (!cat) {
    SIF_LOG_ERROR("void_catalogue", "catalogue: NULL");
    return SIF_ERR_INVALID;
  }

  char k[META_KEY_MAX + 1];
  char why[128];
  const char* problem = meta_key_problem(key, k, why, sizeof why);
  if (problem) {
    if (!key)
      SIF_LOG_ERROR("void_catalogue", "catalogue value: %s", problem);
    else
      SIF_LOG_ERROR("void_catalogue", "catalogue value \"%.*s%s\": %s",
        META_KEY_MAX, key, strlen(key) > META_KEY_MAX ? "..." : "", problem);
    return SIF_ERR_INVALID;
  }

  const int at = meta_find(cat, k);
  if (at >= 0) {
    sif_catalogue_meta_entry_t* e = &cat->_meta[at];
    free(e->text);
    e->text = NULL;
    *slot = e;
    return SIF_OK;
  }

  sif_catalogue_meta_entry_t* grown =
    realloc(cat->_meta, (cat->_n_meta + 1) * sizeof(*grown));
  if (!grown) {
    SIF_LOG_ERROR("void_catalogue",
      "catalogue value \"%s\": cannot grow the table to %u entries", k,
      cat->_n_meta + 1);
    return SIF_ERR_ALLOC;
  }
  cat->_meta = grown;
  sif_catalogue_meta_entry_t* e = &cat->_meta[cat->_n_meta];
  memset(e, 0, sizeof *e);
  e->key = strdup(k);
  if (!e->key) {
    SIF_LOG_ERROR(
      "void_catalogue", "catalogue value \"%s\": cannot copy the key", k);
    return SIF_ERR_ALLOC;
  }
  cat->_n_meta++;
  *slot = e;
  return SIF_OK;
}

int sif_catalogue_meta_int_set(
  sif_catalogue_t* catalogue, const char* key, int64_t value) {
  sif_catalogue_meta_entry_t* e = NULL;
  const int status = meta_slot(catalogue, key, &e);
  if (status != SIF_OK)
    return status;
  e->kind = SIF_CATALOGUE_META_INT;
  e->i = value;
  e->d = (double)value;
  return SIF_OK;
}

int sif_catalogue_meta_real_set(
  sif_catalogue_t* catalogue, const char* key, double value) {
  if (!isfinite(value)) {
    SIF_LOG_ERROR("void_catalogue",
      "catalogue value \"%s\": %g is not finite (the file formats cannot hold "
      "it)",
      key ? key : "(null)", value);
    return SIF_ERR_INVALID;
  }
  sif_catalogue_meta_entry_t* e = NULL;
  const int status = meta_slot(catalogue, key, &e);
  if (status != SIF_OK)
    return status;
  e->kind = SIF_CATALOGUE_META_REAL;
  e->d = value;
  e->i = 0;
  return SIF_OK;
}

int sif_catalogue_meta_string_set(
  sif_catalogue_t* catalogue, const char* key, const char* value) {
  if (!value) {
    SIF_LOG_ERROR(
      "void_catalogue", "catalogue value \"%s\": NULL", key ? key : "(null)");
    return SIF_ERR_INVALID;
  }
  const char* bad = strpbrk(value, "\n\r\"");
  if (bad) {
    SIF_LOG_ERROR("void_catalogue",
      "catalogue value \"%s\": contains %s at character %zu (the file formats "
      "cannot hold it)",
      key ? key : "(null)", *bad == '"' ? "a double quote" : "a line break",
      (size_t)(bad - value) + 1);
    return SIF_ERR_INVALID;
  }
  char* copy = strdup(value);
  if (!copy) {
    SIF_LOG_ERROR("void_catalogue",
      "catalogue value \"%s\": cannot copy %zu characters",
      key ? key : "(null)", strlen(value));
    return SIF_ERR_ALLOC;
  }
  sif_catalogue_meta_entry_t* e = NULL;
  const int status = meta_slot(catalogue, key, &e);
  if (status != SIF_OK) {
    free(copy);
    return status;
  }
  e->kind = SIF_CATALOGUE_META_STRING;
  e->text = copy;
  return SIF_OK;
}

int sif_catalogue_meta_remove(sif_catalogue_t* catalogue, const char* key) {
  if (!catalogue || !key) {
    SIF_LOG_ERROR(
      "void_catalogue", "%s: NULL", !catalogue ? "catalogue" : "key");
    return SIF_ERR_INVALID;
  }
  const int at = meta_find(catalogue, key);
  if (at < 0)
    return SIF_OK;
  free(catalogue->_meta[at].key);
  free(catalogue->_meta[at].text);
  memmove(&catalogue->_meta[at], &catalogue->_meta[at + 1],
    (catalogue->_n_meta - (uint32_t)at - 1) * sizeof(*catalogue->_meta));
  catalogue->_n_meta--;
  return SIF_OK;
}

sif_catalogue_meta_kind_t sif_catalogue_meta_kind(
  const sif_catalogue_t* catalogue, const char* key) {
  if (!catalogue || !key)
    return SIF_CATALOGUE_META_MISSING;
  const int at = meta_find(catalogue, key);
  return at < 0 ? SIF_CATALOGUE_META_MISSING : catalogue->_meta[at].kind;
}

int64_t sif_catalogue_meta_int_get(
  const sif_catalogue_t* catalogue, const char* key) {
  if (sif_catalogue_meta_kind(catalogue, key) != SIF_CATALOGUE_META_INT)
    return 0;
  return catalogue->_meta[meta_find(catalogue, key)].i;
}

double sif_catalogue_meta_real_get(
  const sif_catalogue_t* catalogue, const char* key) {
  const sif_catalogue_meta_kind_t kind = sif_catalogue_meta_kind(catalogue, key);
  if (kind != SIF_CATALOGUE_META_INT && kind != SIF_CATALOGUE_META_REAL)
    return 0.0;
  return catalogue->_meta[meta_find(catalogue, key)].d;
}

const char* sif_catalogue_meta_string_get(
  const sif_catalogue_t* catalogue, const char* key) {
  if (sif_catalogue_meta_kind(catalogue, key) != SIF_CATALOGUE_META_STRING)
    return NULL;
  return catalogue->_meta[meta_find(catalogue, key)].text;
}

uint32_t sif_catalogue_meta_count(const sif_catalogue_t* catalogue) {
  return catalogue ? catalogue->_n_meta : 0;
}

const char* sif_catalogue_meta_name(
  const sif_catalogue_t* catalogue, uint32_t index) {
  if (!catalogue || index >= catalogue->_n_meta)
    return NULL;
  return catalogue->_meta[index].key;
}

int sif__catalogue_meta_copy(sif_catalogue_t* to, const sif_catalogue_t* from) {
  for (uint32_t m = 0; m < from->_n_meta; m++) {
    const sif_catalogue_meta_entry_t* e = &from->_meta[m];
    int status = SIF_OK;
    if (e->kind == SIF_CATALOGUE_META_INT)
      status = sif_catalogue_meta_int_set(to, e->key, e->i);
    else if (e->kind == SIF_CATALOGUE_META_REAL)
      status = sif_catalogue_meta_real_set(to, e->key, e->d);
    else if (e->kind == SIF_CATALOGUE_META_STRING)
      status = sif_catalogue_meta_string_set(to, e->key, e->text);
    if (status != SIF_OK)
      return status;
  }
  return SIF_OK;
}

#undef META_KEY_MAX
