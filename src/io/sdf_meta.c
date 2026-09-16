/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The metadata table: a block's typed key/value entries, in memory and on the
 * wire.
 *
 * This is the format's answer to a FITS header, and it is deliberately small.
 * Three value types cover everything a header card can express -- a count, a
 * measurement, a note -- with none of the card's limits: no eighty-column
 * budget, no eight-character keys, and no parsing text back into a number and
 * hoping it was one.
 *
 * The same table serves both directions, so reading a file's notes, adding to
 * them and writing them back is the obvious four lines rather than a loop that
 * copies keys between two types. Merging falls out of that: decoding several
 * encoded tables into one, in file order, gives later values the win, which is
 * the rule the format states for several metadata blocks.
 *
 * One table also does the library's own work. A product block keeps its shape
 * here under keys beginning `sif.`, which is why a new parameter on a profile
 * set costs a key rather than a payload version -- and why the public puts
 * refuse that prefix while sif__sdf_meta_put() does not.
 */

#include "sdf_internal.h"

#include "sif/utils/crc32.h"
#include "sif/utils/logger.h"

#include <stdlib.h>
#include <string.h>

/* Entry header: key length, value type, a reserved byte, value length. */
#define ENTRY_HEAD_BYTES 8

/* Entries are padded to this, so a table's values stay naturally aligned and
 * the data section after it starts on an 8-byte boundary. */
#define ENTRY_ALIGN 8

/* Keys the library keeps for a block's own shape. */
#define RESERVED_PREFIX "sif."

/* First allocation. A run's worth of notes fits in this without growing. */
#define FIRST_CAPACITY 8

/* Rounds an entry up to the padding the format puts between entries.
 *
 * In 64 bits, which is the point: an entry's length is a 32-bit head plus a
 * 16-bit key plus a value that may be as long as a uint32 can count, so the
 * sum is exactly the thing that overflows if it is added up in the width of
 * its parts. */
static uint64_t align_up(uint64_t value) {
  return (value + (ENTRY_ALIGN - 1)) & ~(uint64_t)(ENTRY_ALIGN - 1);
}

/* --- the table --- */

sif_sdf_meta_t* sif_sdf_meta_alloc(void) {
  sif_sdf_meta_t* meta = calloc(1, sizeof(sif_sdf_meta_t));
  if (!meta)
    SIF_LOG_ERROR("sdf", "failed to allocate a metadata table");
  return meta;
}

void sif_sdf_meta_free(sif_sdf_meta_t* meta) {
  if (!meta)
    return;

  for (uint32_t i = 0; i < meta->count; i++) {
    free(meta->items[i].key);
    free(meta->items[i].value);
  }
  free(meta->items);
  free(meta);
}

/* The entry a key names, or NULL. A scan: see the note on the struct. */
static sif__sdf_kv_t* lookup(const sif_sdf_meta_t* meta, const char* key) {
  if (!meta || !key)
    return NULL;

  for (uint32_t i = 0; i < meta->count; i++) {
    if (strcmp(meta->items[i].key, key) == 0)
      return &meta->items[i];
  }
  return NULL;
}

void sif__sdf_meta_put(sif_sdf_meta_t* meta, const char* key,
  sif_sdf_meta_type_t type, const void* value, uint32_t bytes) {

  if (!meta || meta->failed)
    return;

  if (!key || key[0] == '\0') {
    SIF_LOG_ERROR("sdf", "a metadata entry with no key");
    if (meta)
      meta->failed = 1;
    return;
  }
  if (!value && bytes > 0) {
    SIF_LOG_ERROR("sdf", "`%s` was given no value to store", key);
    meta->failed = 1;
    return;
  }

  /* A string is kept terminated so it can be handed straight out, and the
   * terminator sits past the length the wire records -- which is why a value
   * of exactly UINT32_MAX bytes cannot be stored: the length with the
   * terminator on it would wrap to nothing and the copy would run off the
   * end of a single byte. Checked here, at the one door both the public puts
   * and the decoder come through. */
  if (type == SIF_SDF_META_STR && bytes == UINT32_MAX) {
    SIF_LOG_ERROR("sdf", "the text under `%s` is too long to store", key);
    meta->failed = 1;
    return;
  }

  /* A key is length-prefixed with a uint16 on the wire. One that does not fit
   * would be written at its full length and described at a truncated one,
   * which encodes a table nothing can decode. */
  if (strlen(key) > UINT16_MAX) {
    SIF_LOG_ERROR("sdf", "a metadata key of %zu bytes is too long to store",
      strlen(key));
    meta->failed = 1;
    return;
  }

  const uint32_t stored = (type == SIF_SDF_META_STR) ? bytes + 1 : bytes;

  void* copy = malloc(stored ? stored : 1);
  if (!copy) {
    SIF_LOG_ERROR("sdf", "failed to store the value of `%s`", key);
    meta->failed = 1;
    return;
  }
  if (bytes > 0)
    memcpy(copy, value, bytes);
  if (type == SIF_SDF_META_STR)
    ((char*)copy)[bytes] = '\0';

  /* An existing key is replaced rather than repeated: a table is a mapping,
   * and the second `redshift` a caller writes is the one they mean. */
  sif__sdf_kv_t* existing = lookup(meta, key);
  if (existing) {
    free(existing->value);
    existing->type = (uint8_t)type;
    existing->bytes = bytes;
    existing->value = copy;
    return;
  }

  if (meta->count == meta->capacity) {
    const uint32_t grown = meta->capacity ? meta->capacity * 2 : FIRST_CAPACITY;
    sif__sdf_kv_t* items =
      realloc(meta->items, (size_t)grown * sizeof(sif__sdf_kv_t));
    if (!items) {
      SIF_LOG_ERROR("sdf", "failed to grow a metadata table to %u keys", grown);
      free(copy);
      meta->failed = 1;
      return;
    }
    meta->items = items;
    meta->capacity = grown;
  }

  const size_t key_bytes = strlen(key) + 1;
  char* key_copy = malloc(key_bytes);
  if (!key_copy) {
    SIF_LOG_ERROR("sdf", "failed to store the key `%s`", key);
    free(copy);
    meta->failed = 1;
    return;
  }
  memcpy(key_copy, key, key_bytes);

  meta->items[meta->count].key = key_copy;
  meta->items[meta->count].type = (uint8_t)type;
  meta->items[meta->count].bytes = bytes;
  meta->items[meta->count].value = copy;
  meta->count++;
}

void sif__sdf_meta_put_i64(
  sif_sdf_meta_t* meta, const char* key, int64_t value) {
  sif__sdf_meta_put(
    meta, key, SIF_SDF_META_I64, &value, (uint32_t)sizeof(int64_t));
}

void sif__sdf_meta_put_f64(
  sif_sdf_meta_t* meta, const char* key, double value) {
  /* Always a double, whatever sif_real is: a metadata table is a handful of
   * values and there is nothing to be gained by halving it. */
  sif__sdf_meta_put(
    meta, key, SIF_SDF_META_F64, &value, (uint32_t)sizeof(double));
}

/* --- the public puts, which are the above with one check in front --- */

/*
 * Keeps callers out of the namespace the library stores block shapes in.
 *
 * Not a matter of tidiness: a caller who put their own `sif.n_bins` into a
 * product block's table would change what the block claims to be, and the
 * reader would believe them.
 */
static int reserved(sif_sdf_meta_t* meta, const char* key) {
  if (key && strncmp(key, RESERVED_PREFIX, strlen(RESERVED_PREFIX)) == 0) {
    SIF_LOG_ERROR("sdf", "`%s` begins `%s`, which the library keeps for itself",
      key, RESERVED_PREFIX);
    if (meta)
      meta->failed = 1;
    return 1;
  }
  return 0;
}

void sif_sdf_meta_put_i64(
  sif_sdf_meta_t* meta, const char* key, int64_t value) {
  if (!reserved(meta, key))
    sif__sdf_meta_put_i64(meta, key, value);
}

void sif_sdf_meta_put_f64(sif_sdf_meta_t* meta, const char* key, double value) {
  if (!reserved(meta, key))
    sif__sdf_meta_put_f64(meta, key, value);
}

void sif_sdf_meta_put_str(
  sif_sdf_meta_t* meta, const char* key, const char* value) {
  if (reserved(meta, key))
    return;

  if (!value) {
    SIF_LOG_ERROR("sdf", "`%s` was given no text to store", key);
    if (meta)
      meta->failed = 1;
    return;
  }

  const size_t bytes = strlen(value);
  if (bytes > UINT32_MAX) {
    SIF_LOG_ERROR("sdf", "the text under `%s` is too long to store", key);
    if (meta)
      meta->failed = 1;
    return;
  }

  sif__sdf_meta_put(meta, key, SIF_SDF_META_STR, value, (uint32_t)bytes);
}

/* An array put, for either of the two eight-byte types. */
static void put_vector(sif_sdf_meta_t* meta, const char* key,
  sif_sdf_meta_type_t type, const void* values, uint32_t n_values) {

  if (reserved(meta, key))
    return;

  if (n_values > UINT32_MAX / 8u) {
    SIF_LOG_ERROR("sdf", "the array under `%s` is too long to store", key);
    if (meta)
      meta->failed = 1;
    return;
  }

  sif__sdf_meta_put(meta, key, type, values, n_values * 8u);
}

void sif_sdf_meta_put_i64v(sif_sdf_meta_t* meta, const char* key,
  const int64_t* values, uint32_t n_values) {
  put_vector(meta, key, SIF_SDF_META_I64, values, n_values);
}

void sif_sdf_meta_put_f64v(sif_sdf_meta_t* meta, const char* key,
  const double* values, uint32_t n_values) {
  put_vector(meta, key, SIF_SDF_META_F64, values, n_values);
}

/* --- reading a table --- */

/*
 * The one place a get can go wrong in two different ways.
 *
 * Absent and wrong-type are told apart because they are different mistakes: a
 * key nobody wrote is an ordinary answer, and a key written as text and read
 * as a number is a bug in one of the two.
 */
static sif_sdf_status_t fetch(const sif_sdf_meta_t* meta, const char* key,
  sif_sdf_meta_type_t want, uint32_t want_bytes, const sif__sdf_kv_t** out) {

  const sif__sdf_kv_t* item = lookup(meta, key);
  if (!item)
    return SIF_SDF_ERR_ABSENT;

  if (item->type != (uint8_t)want)
    return SIF_SDF_ERR_TYPE;

  /* A scalar get on an array is the same mistake as a wrong type: the caller
   * asked for one number and the table holds six. */
  if (want_bytes > 0 && item->bytes != want_bytes)
    return SIF_SDF_ERR_TYPE;

  *out = item;
  return SIF_SDF_OK;
}

sif_sdf_status_t sif_sdf_meta_get_i64(
  const sif_sdf_meta_t* meta, const char* key, int64_t* out) {
  const sif__sdf_kv_t* item = NULL;
  const sif_sdf_status_t status =
    fetch(meta, key, SIF_SDF_META_I64, sizeof(int64_t), &item);
  if (status == SIF_SDF_OK)
    memcpy(out, item->value, sizeof(int64_t));
  return status;
}

sif_sdf_status_t sif_sdf_meta_get_f64(
  const sif_sdf_meta_t* meta, const char* key, double* out) {
  const sif__sdf_kv_t* item = NULL;
  const sif_sdf_status_t status =
    fetch(meta, key, SIF_SDF_META_F64, sizeof(double), &item);
  if (status == SIF_SDF_OK)
    memcpy(out, item->value, sizeof(double));
  return status;
}

sif_sdf_status_t sif_sdf_meta_get_str(
  const sif_sdf_meta_t* meta, const char* key, const char** out) {
  const sif__sdf_kv_t* item = NULL;
  const sif_sdf_status_t status = fetch(meta, key, SIF_SDF_META_STR, 0, &item);
  if (status == SIF_SDF_OK)
    *out = (const char*)item->value;
  return status;
}

sif_sdf_status_t sif_sdf_meta_get_i64v(const sif_sdf_meta_t* meta,
  const char* key, const int64_t** out, uint32_t* out_n) {
  const sif__sdf_kv_t* item = NULL;
  const sif_sdf_status_t status = fetch(meta, key, SIF_SDF_META_I64, 0, &item);
  if (status == SIF_SDF_OK) {
    *out = (const int64_t*)item->value;
    *out_n = item->bytes / 8u;
  }
  return status;
}

sif_sdf_status_t sif_sdf_meta_get_f64v(const sif_sdf_meta_t* meta,
  const char* key, const double** out, uint32_t* out_n) {
  const sif__sdf_kv_t* item = NULL;
  const sif_sdf_status_t status = fetch(meta, key, SIF_SDF_META_F64, 0, &item);
  if (status == SIF_SDF_OK) {
    *out = (const double*)item->value;
    *out_n = item->bytes / 8u;
  }
  return status;
}

int sif_sdf_meta_has(const sif_sdf_meta_t* meta, const char* key) {
  return lookup(meta, key) != NULL;
}

sif_sdf_meta_type_t sif_sdf_meta_type(
  const sif_sdf_meta_t* meta, const char* key) {
  const sif__sdf_kv_t* item = lookup(meta, key);
  return item ? (sif_sdf_meta_type_t)item->type : SIF_SDF_META_NONE;
}

uint32_t sif_sdf_meta_length(const sif_sdf_meta_t* meta, const char* key) {
  const sif__sdf_kv_t* item = lookup(meta, key);
  if (!item)
    return 0;
  /* Bytes for text, elements for the two eight-byte types. */
  return (item->type == SIF_SDF_META_STR) ? item->bytes : item->bytes / 8u;
}

uint32_t sif_sdf_meta_count(const sif_sdf_meta_t* meta) {
  return meta ? meta->count : 0;
}

const char* sif_sdf_meta_key(const sif_sdf_meta_t* meta, uint32_t index) {
  if (!meta || index >= meta->count)
    return NULL;
  return meta->items[index].key;
}

/* --- the wire form --- */

sif_sdf_status_t sif__sdf_meta_encode(
  const sif_sdf_meta_t* meta, void** out, uint32_t* out_bytes) {

  *out = NULL;
  *out_bytes = 0;

  if (!meta || meta->count == 0)
    return SIF_SDF_OK;

  uint64_t total = 0;
  for (uint32_t i = 0; i < meta->count; i++) {
    /* An entry whose size wrapped would be a buffer allocated too short and a
     * value memcpy'd past the end of it. Measured wide, so the check below is
     * reachable rather than something a wrapped sum slips under. */
    total += align_up((uint64_t)ENTRY_HEAD_BYTES +
                      strlen(meta->items[i].key) + meta->items[i].bytes);
  }

  /* meta_bytes is a 32-bit field, and a table this large is a caller doing
   * something the format is not for. */
  if (total > UINT32_MAX) {
    SIF_LOG_ERROR("sdf", "a metadata table of %llu bytes is too large to write",
      (unsigned long long)total);
    return SIF_SDF_ERR_INVALID;
  }

  char* bytes = calloc(1, (size_t)total);
  if (!bytes) {
    SIF_LOG_ERROR("sdf", "failed to allocate %llu bytes of metadata",
      (unsigned long long)total);
    return SIF_SDF_ERR_ALLOC;
  }

  uint32_t offset = 0;
  for (uint32_t i = 0; i < meta->count; i++) {
    const sif__sdf_kv_t* item = &meta->items[i];
    const uint16_t key_bytes = (uint16_t)strlen(item->key);

    memcpy(bytes + offset, &key_bytes, sizeof(key_bytes));
    bytes[offset + 2] = (char)item->type;
    memcpy(bytes + offset + 4, &item->bytes, sizeof(item->bytes));
    memcpy(bytes + offset + ENTRY_HEAD_BYTES, item->key, key_bytes);
    memcpy(
      bytes + offset + ENTRY_HEAD_BYTES + key_bytes, item->value, item->bytes);

    /* Measured the way the sizing pass above measured it, so the two cannot
     * disagree about where the next entry starts. */
    offset += (uint32_t)align_up(
      (uint64_t)ENTRY_HEAD_BYTES + key_bytes + item->bytes);
  }

  *out = bytes;
  *out_bytes = (uint32_t)total;
  return SIF_SDF_OK;
}

sif_sdf_status_t sif__sdf_meta_decode(
  sif_sdf_meta_t* meta, const void* table, uint32_t bytes, const char* path) {

  const char* at = (const char*)table;
  uint32_t offset = 0;

  while (offset < bytes) {
    if (bytes - offset < ENTRY_HEAD_BYTES) {
      SIF_LOG_ERROR("sdf", "a metadata table in %s ends mid-entry", path);
      return SIF_SDF_ERR_CORRUPT;
    }

    uint16_t key_bytes;
    uint32_t value_bytes;
    memcpy(&key_bytes, at + offset, sizeof(key_bytes));
    const uint8_t type = (uint8_t)at[offset + 2];
    memcpy(&value_bytes, at + offset + 4, sizeof(value_bytes));

    /* The entry says how long it is whatever its type, so an entry of a type
     * this build has never heard of is stepped over rather than refused --
     * which is what lets a newer sif put something in a table an older one
     * still has to read past. */
    const uint64_t entry_bytes =
      (uint64_t)ENTRY_HEAD_BYTES + key_bytes + value_bytes;
    const uint64_t padded =
      (entry_bytes + ENTRY_ALIGN - 1) & ~(uint64_t)(ENTRY_ALIGN - 1);

    /* Checked against what is left of the table before it is used. The table
     * came out of a file and its checksum cannot be verified yet -- the
     * checksum covers the metadata and the data together, and the data has not
     * been read -- so an entry claiming to be longer than the table it sits in
     * would otherwise walk off the end of the buffer. */
    if (padded > bytes - offset) {
      SIF_LOG_ERROR("sdf", "a metadata entry in %s runs past its table", path);
      return SIF_SDF_ERR_CORRUPT;
    }

    if (key_bytes == 0) {
      SIF_LOG_ERROR("sdf", "a metadata entry in %s has no key", path);
      return SIF_SDF_ERR_CORRUPT;
    }

    if (type == SIF_SDF_META_I64 || type == SIF_SDF_META_F64 ||
        type == SIF_SDF_META_STR) {
      if (type != SIF_SDF_META_STR && value_bytes % 8u != 0) {
        SIF_LOG_ERROR("sdf",
          "a numeric metadata entry in %s is %u bytes, which is not a whole "
          "number of values",
          path, value_bytes);
        return SIF_SDF_ERR_CORRUPT;
      }

      /* The key on the wire is not terminated, so it is terminated here
       * before anything is done with it as a string. */
      char* key = malloc((size_t)key_bytes + 1);
      if (!key) {
        SIF_LOG_ERROR("sdf", "failed to allocate a metadata key");
        return SIF_SDF_ERR_ALLOC;
      }
      memcpy(key, at + offset + ENTRY_HEAD_BYTES, key_bytes);
      key[key_bytes] = '\0';

      /* One table, one entry per key. The table this decodes into starts
       * empty, so a key already in it is one this block wrote twice -- and a
       * block that says two things about the same key does not say which it
       * means. Across blocks the later value wins, but that is the merge,
       * which happens after each block has been read on its own. */
      if (sif_sdf_meta_has(meta, key)) {
        SIF_LOG_ERROR(
          "sdf", "a metadata table in %s declares `%s` twice", path, key);
        free(key);
        return SIF_SDF_ERR_CORRUPT;
      }

      /* Straight through the unchecked put: a `sif.` key in a block's table
       * is the library's own, and this is how it comes back. */
      sif__sdf_meta_put(meta, key, (sif_sdf_meta_type_t)type,
        at + offset + ENTRY_HEAD_BYTES + key_bytes, value_bytes);
      free(key);

      if (meta->failed)
        return SIF_SDF_ERR_ALLOC;
    }

    offset += (uint32_t)padded;
  }

  return SIF_SDF_OK;
}

void sif__sdf_meta_merge(sif_sdf_meta_t* dst, const sif_sdf_meta_t* src) {
  for (uint32_t i = 0; i < src->count; i++) {
    const sif__sdf_kv_t* item = &src->items[i];
    sif__sdf_meta_put(dst, item->key, (sif_sdf_meta_type_t)item->type,
      item->value, item->bytes);
  }
}

sif_sdf_status_t sif__sdf_meta_read_block(sif_sdf_t* file,
  const sif__sdf_entry_t* entry, sif_sdf_meta_t** out, uint32_t* crc) {

  *out = NULL;

  sif_sdf_meta_t* meta = sif_sdf_meta_alloc();
  if (!meta)
    return SIF_SDF_ERR_ALLOC;

  const uint32_t bytes = entry->header.meta_bytes;
  if (bytes == 0) {
    *out = meta;
    return SIF_SDF_OK;
  }

  /* A multiple of eight is a property of the encoding, so a table that is not
   * one was not written by this library and its entries cannot be walked. */
  if (bytes % ENTRY_ALIGN != 0) {
    SIF_LOG_ERROR("sdf",
      "%s has a metadata table of %u bytes, which is not a whole number of "
      "entries",
      file->path, bytes);
    sif_sdf_meta_free(meta);
    return SIF_SDF_ERR_CORRUPT;
  }

  void* table = malloc(bytes);
  if (!table) {
    SIF_LOG_ERROR("sdf", "failed to allocate %u bytes of metadata", bytes);
    sif_sdf_meta_free(meta);
    return SIF_SDF_ERR_ALLOC;
  }

  sif_sdf_status_t status =
    sif__sdf_read_raw(file, sif__sdf_meta_offset(entry), table, bytes, crc);
  if (status == SIF_SDF_OK)
    status = sif__sdf_meta_decode(meta, table, bytes, file->path);

  free(table);

  if (status != SIF_SDF_OK) {
    sif_sdf_meta_free(meta);
    return status;
  }

  *out = meta;
  return SIF_SDF_OK;
}

/* --- a file's metadata --- */

sif_sdf_meta_t* sif_sdf_meta_read(sif_sdf_t* file, sif_sdf_status_t* status) {
  if (!sif__sdf_status_accepts(status, "meta_read"))
    return NULL;

  if (!file) {
    SIF_LOG_ERROR("sdf", "no file given to read metadata from");
    *status = SIF_SDF_ERR_INVALID;
    return NULL;
  }

  sif_sdf_meta_t* meta = sif_sdf_meta_alloc();
  if (!meta) {
    *status = SIF_SDF_ERR_ALLOC;
    return NULL;
  }

  /* In file order, so that a key written twice comes back as the later of the
   * two. A file with no metadata blocks leaves the table empty, which is a
   * file without notes rather than a failure. */
  for (uint32_t i = 0; i < file->n_blocks; i++) {
    if (file->blocks[i].header.type != SIF_SDF_BLOCK_META)
      continue;

    uint32_t crc = SIF_CRC32_INIT;
    sif_sdf_meta_t* block = NULL;
    sif_sdf_status_t reason =
      sif__sdf_meta_read_block(file, &file->blocks[i], &block, &crc);

    /* A version-1 metadata block has no data section, and this build reads
     * none. The checksum covers whatever is there regardless, so one that
     * carries a section is folded in and stepped over rather than failing:
     * the alternative is that a single block from a newer sif takes down
     * every note in the file, including the ones already read, and metadata
     * is the one thing here that is meant to degrade gracefully. */
    if (reason == SIF_SDF_OK && file->blocks[i].header.data_bytes > 0)
      reason = sif__sdf_crc_range(file, sif__sdf_data_offset(&file->blocks[i]),
        file->blocks[i].header.data_bytes, &crc);

    if (reason == SIF_SDF_OK &&
        sif_crc32_final(crc) != file->blocks[i].header.crc32) {
      SIF_LOG_ERROR("sdf", "a metadata block in %s does not match its checksum",
        file->path);
      reason = SIF_SDF_ERR_CORRUPT;
    }

    if (reason == SIF_SDF_OK) {
      /* Each block is read on its own -- which is what lets a table that
       * declares a key twice be refused -- and only then folded in, where a
       * repeated key is the correction it is meant to be. */
      sif__sdf_meta_merge(meta, block);
      if (meta->failed)
        reason = SIF_SDF_ERR_ALLOC;
    }

    sif_sdf_meta_free(block);

    if (reason != SIF_SDF_OK) {
      sif_sdf_meta_free(meta);
      *status = reason;
      return NULL;
    }
  }

  return meta;
}

void sif_sdf_meta_write(
  sif_sdf_t* file, const sif_sdf_meta_t* meta, sif_sdf_status_t* status) {

  if (!sif__sdf_status_accepts(status, "meta_write"))
    return;

  if (!meta) {
    SIF_LOG_ERROR("sdf", "no metadata given to write");
    *status = SIF_SDF_ERR_INVALID;
    return;
  }

  /* Where a run of puts is answered for. A table that did not come out as the
   * caller asked is not written half-formed. */
  if (meta->failed) {
    SIF_LOG_ERROR("sdf",
      "the metadata offered to %s was not built successfully",
      file ? file->path : "a file");
    *status = SIF_SDF_ERR_INVALID;
    return;
  }

  sif_sdf_status_t reason = sif__sdf_write_gate(file, "metadata");
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return;
  }

  /* Nothing to say is not an error, and an empty block would be a block that
   * says nothing. */
  if (meta->count == 0)
    return;

  void* table = NULL;
  uint32_t bytes = 0;
  reason = sif__sdf_meta_encode(meta, &table, &bytes);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return;
  }

  sif_sdf_block_header_t header;
  memset(&header, 0, sizeof(sif_sdf_block_header_t));
  header.type = (uint16_t)SIF_SDF_BLOCK_META;
  header.real_dtype = (uint8_t)sif__sdf_native_dtype();
  /* No rows, no data, and no catalogue: a note belongs to the file rather
   * than to the voids in it. */
  header.n_items = 0;
  header.catalog_id = 0;

  *status = sif__sdf_block_write(file, &header, table, bytes, NULL, 0);
  free(table);
}
