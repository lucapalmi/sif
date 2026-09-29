/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/structures/catalog.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "model/cosmology_internal.h"
#include "sif/utils/align.h"
#include "sif/utils/logger.h"
#include "structures/catalog_internal.h"

/*
 * Number of sif_real views packed into the arena: cx, cy, cz, radii.
 *
 * One allocation rather than four, so growth is a single copy and the whole
 * catalogue is one free. Structure-of-arrays rather than an array of structs
 * because every consumer sweeps one field at a time -- the size function reads
 * only radii, the profiles read only the centres -- and a struct would drag
 * three unused values through cache on each.
 */
#define CATALOG_N_VIEWS 4

/* Each view starts on a cache-line boundary, so the per-view stride is the
 * capacity rounded up to a whole number of sif_real per cache line. Without
 * the rounding, two views would share the line at their boundary and the
 * threads writing them would contend over it. */
static inline uint64_t catalog_stride(uint64_t capacity) {
  const uint64_t align_elements = SIF_CACHE_LINE / sizeof(sif_real);
  return (capacity + align_elements - 1) & ~(align_elements - 1);
}

/*
 * Allocates one arena for `capacity` voids and hands out the four views.
 * Returns NULL on failure without touching the outputs.
 */
static sif_real* catalog_block_alloc(uint64_t capacity, sif_real** out_cx,
  sif_real** out_cy, sif_real** out_cz, sif_real** out_radii) {

  const uint64_t stride = catalog_stride(capacity);

  sif_real* block =
    sif_malloc_aligned(CATALOG_N_VIEWS * stride * sizeof(sif_real));
  if (!block)
    return NULL;

  *out_cx = block;
  *out_cy = block + stride;
  *out_cz = block + (2 * stride);
  *out_radii = block + (3 * stride);

  return block;
}

/* The two optional footprint views, laid out the same way. */
#define CATALOG_N_FOOTPRINT_VIEWS 2

static sif_real* catalog_footprint_alloc(
  uint64_t capacity, sif_real** out_footprint, sif_real** out_shell) {

  const uint64_t stride = catalog_stride(capacity);

  sif_real* block =
    sif_malloc_aligned(CATALOG_N_FOOTPRINT_VIEWS * stride * sizeof(sif_real));
  if (!block)
    return NULL;

  *out_footprint = block;
  *out_shell = block + stride;

  return block;
}

/*
 * Moves the stored voids into a freshly allocated arena of `new_capacity` and
 * swaps it in. On failure the catalog is left exactly as it was.
 *
 * Always a fresh arena and a copy, never a realloc: realloc may extend in
 * place, but the address it extends is only guaranteed to keep malloc's
 * alignment, and the whole point of the arena is that each view starts on a
 * cache line. Growing in place would also have to move three of the four views
 * anyway, since the stride changes with the capacity.
 *
 * The new arena is filled before the old one is released, so a failure leaves
 * the catalog untouched and the caller still holding something valid -- which
 * is what lets sif_catalog_trim() treat a failed trim as merely disappointing.
 */
static int catalog_resize(sif_catalog_t* catalog, uint64_t new_capacity) {
  sif_real *new_cx, *new_cy, *new_cz, *new_radii;
  sif_real *new_fp = NULL, *new_fp_shell = NULL;

  sif_real* new_block =
    catalog_block_alloc(new_capacity, &new_cx, &new_cy, &new_cz, &new_radii);

  /* The footprint arena moves with the main one or neither moves, so a
   * failure on either leaves the catalogue exactly as it was. */
  sif_real* new_fp_block = NULL;
  if (new_block && catalog->_footprint_block) {
    new_fp_block =
      catalog_footprint_alloc(new_capacity, &new_fp, &new_fp_shell);
    if (!new_fp_block) {
      sif_free_aligned(new_block);
      new_block = NULL;
    }
  }

  if (!new_block) {
    SIF_LOG_ERROR("void_catalog",
      "failed to resize from %" PRIu64 " to %" PRIu64 " voids",
      catalog->capacity, new_capacity);
    return SIF_ERR_ALLOC;
  }

  /* Copying only what is in use, and only what fits: the trim path shrinks,
   * so new_capacity can be below n_voids and the excess is dropped rather
   * than run off the end of the new arena. */
  const uint64_t n =
    (catalog->n_voids < new_capacity) ? catalog->n_voids : new_capacity;

  if (n > 0) {
    memcpy(new_cx, catalog->cx, n * sizeof(sif_real));
    memcpy(new_cy, catalog->cy, n * sizeof(sif_real));
    memcpy(new_cz, catalog->cz, n * sizeof(sif_real));
    memcpy(new_radii, catalog->radii, n * sizeof(sif_real));
    if (new_fp_block) {
      memcpy(new_fp, catalog->footprint, n * sizeof(sif_real));
      memcpy(new_fp_shell, catalog->footprint_shell, n * sizeof(sif_real));
    }
  }

  sif_free_aligned(catalog->_block);
  if (new_fp_block) {
    sif_free_aligned(catalog->_footprint_block);
    catalog->_footprint_block = new_fp_block;
    catalog->footprint = new_fp;
    catalog->footprint_shell = new_fp_shell;
  }

  catalog->_block = new_block;
  catalog->cx = new_cx;
  catalog->cy = new_cy;
  catalog->cz = new_cz;
  catalog->radii = new_radii;
  catalog->capacity = new_capacity;
  catalog->n_voids = n;

  return SIF_OK;
}

sif_catalog_t* sif_catalog_alloc(uint64_t initial_capacity) {
  /* Clamped rather than rejected, so that a catalogue which comes back
   * non-NULL is always appendable. A caller sizing from an estimate that came
   * out zero gets something usable instead of a NULL it has to special-case,
   * and the doubling in append covers the growth from there. */
  if (initial_capacity == 0)
    initial_capacity = 1;

  sif_catalog_t* cat = malloc(sizeof(sif_catalog_t));
  if (!cat) {
    SIF_LOG_ERROR("void_catalog", "failed to allocate void catalog (%zu bytes)",
      sizeof(sif_catalog_t));
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

  cat->_block = catalog_block_alloc(
    initial_capacity, &cat->cx, &cat->cy, &cat->cz, &cat->radii);

  if (!cat->_block) {
    SIF_LOG_ERROR("void_catalog",
      "failed to allocate void catalog arena for %" PRIu64 " voids",
      initial_capacity);
    free(cat);
    return NULL;
  }

  SIF_LOG_TRACE("void_catalog", "void catalog initialized");
  return cat;
}

void sif_catalog_free(sif_catalog_t* catalog) {
  if (!catalog)
    return;

  /* One arena, so one free -- the four views point into it and must not be
   * released individually. */
  sif_free_aligned(catalog->_block);
  sif_free_aligned(catalog->_footprint_block);
  for (uint32_t i = 0; i < catalog->_n_meta; i++) {
    free(catalog->_meta[i].key);
    free(catalog->_meta[i].text);
  }
  free(catalog->_meta);
  free(catalog);
}

int sif_catalog_append(
  sif_catalog_t* catalog, sif_real x, sif_real y, sif_real z, sif_real r) {

  if (!catalog)
    return SIF_ERR_INVALID;

  /* Doubling, so a finder appending one void at a time pays an amortized
   * constant per append rather than a copy of the whole catalogue. A finder
   * cannot know its void count in advance, which is what rules out sizing the
   * arena once up front. */
  if (catalog->n_voids >= catalog->capacity) {
    int status = catalog_resize(catalog, catalog->capacity << 1);
    if (status != SIF_OK)
      return status;

    SIF_LOG_TRACE(
      "void_catalog", "resized catalog to %" PRIu64, catalog->capacity);
  }

  const uint64_t n = catalog->n_voids;
  catalog->cx[n] = x;
  catalog->cy[n] = y;
  catalog->cz[n] = z;
  catalog->radii[n] = r;
  if (catalog->footprint) {
    catalog->footprint[n] = SIF_CATALOG_FOOTPRINT_UNKNOWN;
    catalog->footprint_shell[n] = SIF_CATALOG_FOOTPRINT_UNKNOWN;
  }
  catalog->n_voids++;

  return SIF_OK;
}

int sif_catalog_trim(sif_catalog_t* catalog) {
  if (!catalog)
    return SIF_ERR_INVALID;

  /* An empty catalog is a legitimate result, not an error. Trim it to the
   * minimum usable allocation rather than warning about it. */
  const uint64_t target = (catalog->n_voids > 0) ? catalog->n_voids : 1;

  if (target == catalog->capacity) {
    SIF_LOG_TRACE(
      "void_catalog", "catalog already exact, no trimming required");
    return SIF_OK;
  }

  int status = catalog_resize(catalog, target);
  if (status != SIF_OK) {
    /* Not fatal: the catalog is still correct, just larger than necessary. */
    SIF_LOG_WARNING(
      "void_catalog", "could not trim catalog, keeping current allocation");
    return status;
  }

  SIF_LOG_TRACE(
    "void_catalog", "trimmed catalog to %" PRIu64 " voids", catalog->n_voids);
  return SIF_OK;
}

int sif_catalog_reserve_footprint(sif_catalog_t* catalog) {
  if (!catalog)
    return SIF_ERR_INVALID;

  if (catalog->_footprint_block)
    return SIF_OK;

  sif_real *fp, *fp_shell;
  sif_real* block = catalog_footprint_alloc(catalog->capacity, &fp, &fp_shell);
  if (!block) {
    SIF_LOG_ERROR("void_catalog",
      "failed to allocate the footprint columns for %" PRIu64 " voids",
      catalog->capacity);
    return SIF_ERR_ALLOC;
  }

  for (uint64_t i = 0; i < catalog->n_voids; i++) {
    fp[i] = SIF_CATALOG_FOOTPRINT_UNKNOWN;
    fp_shell[i] = SIF_CATALOG_FOOTPRINT_UNKNOWN;
  }

  catalog->_footprint_block = block;
  catalog->footprint = fp;
  catalog->footprint_shell = fp_shell;

  return SIF_OK;
}

int sif_catalog_translate(sif_catalog_t* catalog, const sif_real offset[3]) {
  if (!catalog || !offset)
    return SIF_ERR_INVALID;
  if (catalog->units != SIF_COORDINATES_CARTESIAN) {
    SIF_LOG_ERROR("void_catalog",
      "the catalogue holds sky coordinates; a translation would move angles");
    return SIF_ERR_INVALID;
  }

  for (uint64_t i = 0; i < catalog->n_voids; i++) {
    catalog->cx[i] += offset[0];
    catalog->cy[i] += offset[1];
    catalog->cz[i] += offset[2];
  }

  return SIF_OK;
}

int sif_catalog_to_sky(sif_catalog_t* catalog, const sif_cosmology_t* cosmo) {
  if (!catalog || !cosmo) {
    SIF_LOG_ERROR("void_catalog", "invalid catalogue or cosmology");
    return SIF_ERR_INVALID;
  }
  if (catalog->units != SIF_COORDINATES_CARTESIAN) {
    SIF_LOG_ERROR("void_catalog", "the catalogue is already on the sky");
    return SIF_ERR_INVALID;
  }

  /* Everything checked, and the table built, before anything is changed. */
  const uint64_t n = catalog->n_voids;
  double d_max = 0.0;
  uint64_t bad = 0;
  for (uint64_t i = 0; i < n; i++) {
    const double x = catalog->cx[i], y = catalog->cy[i], z = catalog->cz[i];
    const double d = sqrt(x * x + y * y + z * z);
    if (!isfinite(d))
      bad++;
    else if (d > d_max)
      d_max = d;
  }
  if (bad) {
    SIF_LOG_ERROR("void_catalog",
      "cannot convert: %" PRIu64 " of %" PRIu64 " centres are not finite", bad,
      n);
    return SIF_ERR_INVALID;
  }

  sif__distance_table_t table;
  const int status = sif__distance_table_build_to(cosmo, d_max, &table);
  if (status != SIF_OK)
    return status;

  const double deg = 180.0 / 3.14159265358979323846;

#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n; i++) {
    const double x = catalog->cx[i], y = catalog->cy[i], z = catalog->cz[i];
    const double d = sqrt(x * x + y * y + z * z);
    double ra = atan2(y, x) * deg;
    if (ra < 0.0)
      ra += 360.0;
    /* A centre at the observer has no direction; it is put at (0, 0). */
    const double dec = d > 0.0 ? asin(z / d) * deg : 0.0;
    catalog->cx[i] = (sif_real)ra;
    catalog->cy[i] = (sif_real)dec;
    catalog->cz[i] = (sif_real)sif__distance_table_invert(&table, d);
  }

  sif__distance_table_free(&table);
  catalog->units = SIF_COORDINATES_SKY;

  SIF_LOG_INFO("void_catalog",
    "converted %" PRIu64 " void centres to the sky, out to D_C = %g Mpc/h", n,
    d_max);
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

bool sif__catalog_meta_reserved(const char* key) {
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

/* The key in lower case, into `out`, or false for one that is not allowed. */
static bool meta_key(const char* key, char out[META_KEY_MAX + 1]) {
  if (!key || !(isalpha((unsigned char)key[0]) || key[0] == '_'))
    return false;
  size_t i = 0;
  for (; key[i]; i++) {
    if (i == META_KEY_MAX || !(isalnum((unsigned char)key[i]) || key[i] == '_'))
      return false;
    out[i] = (char)tolower((unsigned char)key[i]);
  }
  out[i] = '\0';
  return !sif__catalog_meta_reserved(out);
}

static int meta_find(const sif_catalog_t* cat, const char* key) {
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
 * any string it held. NULL, with the reason logged, if the key is refused. */
static sif_catalog_meta_entry_t* meta_slot(
  sif_catalog_t* cat, const char* key) {
  char k[META_KEY_MAX + 1];
  if (!cat || !meta_key(key, k)) {
    SIF_LOG_ERROR("void_catalog",
      "\"%s\" cannot name a catalogue value: it has to be an identifier of at "
      "most %d characters, and not one the file formats use themselves",
      key ? key : "(null)", META_KEY_MAX);
    return NULL;
  }
  const int at = meta_find(cat, k);
  if (at >= 0) {
    sif_catalog_meta_entry_t* e = &cat->_meta[at];
    free(e->text);
    e->text = NULL;
    return e;
  }
  sif_catalog_meta_entry_t* grown =
    realloc(cat->_meta, (cat->_n_meta + 1) * sizeof(*grown));
  if (!grown)
    return NULL;
  cat->_meta = grown;
  sif_catalog_meta_entry_t* e = &cat->_meta[cat->_n_meta];
  memset(e, 0, sizeof *e);
  e->key = strdup(k);
  if (!e->key)
    return NULL;
  cat->_n_meta++;
  return e;
}

int sif_catalog_meta_int_set(
  sif_catalog_t* catalog, const char* key, int64_t value) {
  sif_catalog_meta_entry_t* e = meta_slot(catalog, key);
  if (!e)
    return SIF_ERR_INVALID;
  e->kind = SIF_CATALOG_META_INT;
  e->i = value;
  e->d = (double)value;
  return SIF_OK;
}

int sif_catalog_meta_real_set(
  sif_catalog_t* catalog, const char* key, double value) {
  if (!isfinite(value)) {
    SIF_LOG_ERROR("void_catalog", "catalogue value %s is not finite (%g)",
      key ? key : "(null)", value);
    return SIF_ERR_INVALID;
  }
  sif_catalog_meta_entry_t* e = meta_slot(catalog, key);
  if (!e)
    return SIF_ERR_INVALID;
  e->kind = SIF_CATALOG_META_REAL;
  e->d = value;
  e->i = 0;
  return SIF_OK;
}

int sif_catalog_meta_string_set(
  sif_catalog_t* catalog, const char* key, const char* value) {
  if (!value || strpbrk(value, "\n\r\"")) {
    SIF_LOG_ERROR("void_catalog",
      "catalogue value %s: a string may not be NULL or hold a newline or a "
      "double quote",
      key ? key : "(null)");
    return SIF_ERR_INVALID;
  }
  char* copy = strdup(value);
  if (!copy)
    return SIF_ERR_ALLOC;
  sif_catalog_meta_entry_t* e = meta_slot(catalog, key);
  if (!e) {
    free(copy);
    return SIF_ERR_INVALID;
  }
  e->kind = SIF_CATALOG_META_STRING;
  e->text = copy;
  return SIF_OK;
}

int sif_catalog_meta_remove(sif_catalog_t* catalog, const char* key) {
  if (!catalog || !key)
    return SIF_ERR_INVALID;
  const int at = meta_find(catalog, key);
  if (at < 0)
    return SIF_OK;
  free(catalog->_meta[at].key);
  free(catalog->_meta[at].text);
  memmove(&catalog->_meta[at], &catalog->_meta[at + 1],
    (catalog->_n_meta - (uint32_t)at - 1) * sizeof(*catalog->_meta));
  catalog->_n_meta--;
  return SIF_OK;
}

sif_catalog_meta_kind_t sif_catalog_meta_kind(
  const sif_catalog_t* catalog, const char* key) {
  if (!catalog || !key)
    return SIF_CATALOG_META_MISSING;
  const int at = meta_find(catalog, key);
  return at < 0 ? SIF_CATALOG_META_MISSING : catalog->_meta[at].kind;
}

int64_t sif_catalog_meta_int_get(
  const sif_catalog_t* catalog, const char* key) {
  if (sif_catalog_meta_kind(catalog, key) != SIF_CATALOG_META_INT)
    return 0;
  return catalog->_meta[meta_find(catalog, key)].i;
}

double sif_catalog_meta_real_get(
  const sif_catalog_t* catalog, const char* key) {
  const sif_catalog_meta_kind_t kind = sif_catalog_meta_kind(catalog, key);
  if (kind != SIF_CATALOG_META_INT && kind != SIF_CATALOG_META_REAL)
    return 0.0;
  return catalog->_meta[meta_find(catalog, key)].d;
}

const char* sif_catalog_meta_string_get(
  const sif_catalog_t* catalog, const char* key) {
  if (sif_catalog_meta_kind(catalog, key) != SIF_CATALOG_META_STRING)
    return NULL;
  return catalog->_meta[meta_find(catalog, key)].text;
}

uint32_t sif_catalog_meta_count(const sif_catalog_t* catalog) {
  return catalog ? catalog->_n_meta : 0;
}

const char* sif_catalog_meta_name(
  const sif_catalog_t* catalog, uint32_t index) {
  if (!catalog || index >= catalog->_n_meta)
    return NULL;
  return catalog->_meta[index].key;
}

int sif__catalog_meta_copy(sif_catalog_t* to, const sif_catalog_t* from) {
  for (uint32_t m = 0; m < from->_n_meta; m++) {
    const sif_catalog_meta_entry_t* e = &from->_meta[m];
    int status = SIF_OK;
    if (e->kind == SIF_CATALOG_META_INT)
      status = sif_catalog_meta_int_set(to, e->key, e->i);
    else if (e->kind == SIF_CATALOG_META_REAL)
      status = sif_catalog_meta_real_set(to, e->key, e->d);
    else if (e->kind == SIF_CATALOG_META_STRING)
      status = sif_catalog_meta_string_set(to, e->key, e->text);
    if (status != SIF_OK)
      return status;
  }
  return SIF_OK;
}

#undef META_KEY_MAX
