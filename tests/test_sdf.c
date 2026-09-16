/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The .sdf container: the catalogue round trip, the identity that ties a file
 * together, and the files that have to be refused.
 *
 * The rejections are the point. A format whose whole purpose is to keep a
 * catalogue and its measurements together is worth nothing if it accepts a
 * file where they have come apart, so every rule the reader enforces is
 * exercised by actually producing a file that breaks it.
 */

#include "sif/core/macros.h"
#include "sif/core/system.h"
#include "sif/io/sdf.h"
#include "sif/measure/profiles.h"
#include "sif/measure/size_function.h"
#include "sif/structures/catalog.h"

/* Profile sets are only obtainable from an estimator, which wants a field and
 * a mesh. This is a test of the container rather than of the estimator, so it
 * builds a set through the allocator the estimator uses and stamps it the way
 * the estimator would. */
#include "measure/profiles_internal.h"

/* The on-disk structs and the checksum, so the tests below can assemble a
 * block by hand. Everything the format promises about files written by a
 * *later* sif is a promise about bytes this build cannot produce, so the only
 * way to exercise those rules is to write them directly. */
#include "io/sdf_internal.h"
#include "sif/utils/crc32.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_util.h"

static int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: ");                                                      \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#define N_VOIDS 500u
#define BOX     1000.0

static const char* SDF_PATH = "test_sdf.sdf";
static const char* ALT_PATH = "test_sdf_alt.sdf";

/* Offsets into a file holding one catalogue block, from the format's own
 * arithmetic: the file header, then the block header, then the payload. */
#define BLOCK_OFFSET 64
#define DATA_OFFSET  128

/* Where catalog_id sits inside a block header. */
#define CATALOG_ID_OFFSET 40

static sif_catalog_t* make_catalog(uint64_t n_voids) {
  sif_catalog_t* cat = sif_catalog_alloc(n_voids);
  if (!cat)
    return NULL;

  for (uint64_t i = 0; i < n_voids; i++) {
    const sif_real f = (sif_real)i;
    if (sif_catalog_append(cat, f * (sif_real)0.37, f * (sif_real)1.11,
          f * (sif_real)2.5, (sif_real)5.0 + f * (sif_real)0.013) != SIF_OK) {
      sif_catalog_free(cat);
      return NULL;
    }
  }
  return cat;
}

static long file_bytes(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f)
    return -1;
  fseek(f, 0, SEEK_END);
  const long bytes = ftell(f);
  fclose(f);
  return bytes;
}

/* Overwrites bytes in place, to damage a file the way a bad disk would. */
static void poke(const char* path, long offset, const void* src, size_t n) {
  FILE* f = fopen(path, "r+b");
  if (!f)
    return;
  fseek(f, offset, SEEK_SET);
  fwrite(src, 1, n, f);
  fclose(f);
}

/* Flips one bit, the way a bad disk or a tired stick of RAM does. */
static void flip_bit(const char* path, long offset, int bit) {
  FILE* f = fopen(path, "r+b");
  if (!f)
    return;
  fseek(f, offset, SEEK_SET);
  const int byte = fgetc(f);
  if (byte != EOF) {
    fseek(f, offset, SEEK_SET);
    fputc(byte ^ (1 << bit), f);
  }
  fclose(f);
}

/* Cuts a file short, the way a write that ran out of disk would. */
static void truncate_to(const char* path, long keep) {
  FILE* f = fopen(path, "rb");
  if (!f)
    return;
  char* buffer = malloc((size_t)keep);
  if (!buffer) {
    fclose(f);
    return;
  }
  const size_t got = fread(buffer, 1, (size_t)keep, f);
  fclose(f);

  f = fopen(path, "wb");
  if (f) {
    fwrite(buffer, 1, got, f);
    fclose(f);
  }
  free(buffer);
}

/* Writes a good file and closes it, for a test that then damages it. */
static int write_good_file(const char* path, const sif_catalog_t* cat) {
  sif_sdf_status_t status = SIF_SDF_OK;
  remove(path);
  sif_sdf_t* file = sif_sdf_create(path, BOX, cat, 0, &status);
  sif_sdf_close(file, &status);
  return status == SIF_SDF_OK;
}

/* --- blocks this build would never write --- */

/*
 * Appends a block assembled by hand.
 *
 * Fills in everything the format computes rather than chooses -- the magic,
 * the two lengths, the checksum, the padding -- so a test states only the
 * thing it is testing.
 */
static int append_forged(const char* path, sif_sdf_block_header_t header,
  const void* meta, uint32_t meta_bytes, const void* data,
  uint64_t data_bytes) {

  memcpy(header.magic, SIF__SDF_BLOCK_MAGIC, 4);
  header.meta_bytes = meta_bytes;
  header.data_bytes = data_bytes;

  uint32_t crc = SIF_CRC32_INIT;
  if (meta_bytes)
    crc = sif_crc32_update(crc, meta, meta_bytes);
  if (data_bytes)
    crc = sif_crc32_update(crc, data, (size_t)data_bytes);
  header.crc32 = sif_crc32_final(crc);

  /* Last, once the rest of the header is final: a forged block has to stand
   * behind its own header the way a real one does, or it is testing the
   * header checksum rather than whatever it meant to test. */
  header.header_crc32 = sif__sdf_block_checksum(&header);

  FILE* f = fopen(path, "ab");
  if (!f)
    return 0;

  fwrite(&header, 1, SIF__SDF_BLOCK_BYTES, f);
  if (meta_bytes)
    fwrite(meta, 1, meta_bytes, f);
  if (data_bytes)
    fwrite(data, 1, (size_t)data_bytes, f);

  const size_t payload = meta_bytes + (size_t)data_bytes;
  const size_t pad = (SIF__SDF_BLOCK_ALIGN - payload % SIF__SDF_BLOCK_ALIGN) %
                     SIF__SDF_BLOCK_ALIGN;
  if (pad) {
    static const char zeros[SIF__SDF_BLOCK_ALIGN] = {0};
    fwrite(zeros, 1, pad, f);
  }

  fclose(f);
  return 1;
}

/* One metadata entry as it goes on the wire, padded to eight. Returns what it
 * wrote, so a caller can lay several down in a row. */
static uint32_t forge_entry(char* out, const char* key, uint8_t type,
  const void* value, uint32_t value_bytes) {

  const uint16_t key_bytes = (uint16_t)strlen(key);
  const uint32_t total = (8u + key_bytes + value_bytes + 7u) & ~7u;

  memset(out, 0, total);
  memcpy(out, &key_bytes, sizeof(key_bytes));
  out[2] = (char)type;
  memcpy(out + 4, &value_bytes, sizeof(value_bytes));
  memcpy(out + 8, key, key_bytes);
  if (value_bytes)
    memcpy(out + 8 + key_bytes, value, value_bytes);

  return total;
}

/* A block header carrying what every forged block needs, at this build's own
 * precision unless the caller says otherwise. */
static sif_sdf_block_header_t forged_header(
  uint16_t type, uint64_t n_items, uint64_t catalog_id) {

  sif_sdf_block_header_t header;
  memset(&header, 0, sizeof(sif_sdf_block_header_t));
  header.type = type;
  header.type_version = (uint8_t)SIF__SDF_TYPE_VERSION;
  header.real_dtype =
    (uint8_t)((sizeof(sif_real) == 8) ? SIF_SDF_F64 : SIF_SDF_F32);
  header.n_items = n_items;
  header.catalog_id = catalog_id;
  return header;
}

/* The identity of the catalogue a file was built around, read off block 0 --
 * which is what a forged block has to repeat to belong to the file. */
static uint64_t catalog_id_of(const char* path) {
  uint64_t id = 0;
  FILE* f = fopen(path, "rb");
  if (!f)
    return 0;
  fseek(f, BLOCK_OFFSET + CATALOG_ID_OFFSET, SEEK_SET);
  if (fread(&id, 1, sizeof(id), f) != sizeof(id))
    id = 0;
  fclose(f);
  return id;
}

/* A bare file header, for a test that then forges block 0 itself. */
static int write_forged_file_header(const char* path) {
  sif_sdf_header_t header;
  memset(&header, 0, sizeof(sif_sdf_header_t));
  memcpy(header.magic, SIF_SDF_MAGIC, 4);
  header.version = SIF_SDF_VERSION;
  header.byte_order = SIF_SDF_BYTE_ORDER;
  header.box_length = BOX;
  header.header_crc32 = sif__sdf_header_checksum(&header);

  FILE* f = fopen(path, "wb");
  if (!f)
    return 0;
  const size_t wrote = fwrite(&header, 1, SIF__SDF_HEADER_BYTES, f);
  fclose(f);
  return wrote == SIF__SDF_HEADER_BYTES;
}

static void test_header_layout(void) {
  printf("header layout\n");

  /* Both headers are read back byte for byte, so either changing size would
   * shift every offset in every file already written. */
  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(4);
  CHECK(write_good_file(SDF_PATH, cat), "could not write a file to measure");

  /* Every offset the format document states, pinned. The sizes alone are
   * already checked at compile time; these are what catch a field being
   * reordered or widened, which would move every one after it. */
  CHECK(sizeof(sif_sdf_header_t) == 64, "the file header is %zu bytes",
    sizeof(sif_sdf_header_t));
  CHECK(sizeof(sif_sdf_block_header_t) == 64, "a block header is %zu bytes",
    sizeof(sif_sdf_block_header_t));

  CHECK(offsetof(sif_sdf_header_t, version) == 4, "file header: version");
  CHECK(offsetof(sif_sdf_header_t, byte_order) == 8, "file header: byte_order");
  CHECK(offsetof(sif_sdf_header_t, flags) == 12, "file header: flags");
  CHECK(offsetof(sif_sdf_header_t, box_length) == 16, "file header: box_length");
  CHECK(offsetof(sif_sdf_header_t, created) == 24, "file header: created");
  CHECK(offsetof(sif_sdf_header_t, writer) == 32, "file header: writer");
  CHECK(offsetof(sif_sdf_header_t, header_crc32) == 48,
    "file header: header_crc32");

  CHECK(offsetof(sif_sdf_block_header_t, type) == 4, "block header: type");
  CHECK(offsetof(sif_sdf_block_header_t, type_version) == 6,
    "block header: type_version");
  CHECK(offsetof(sif_sdf_block_header_t, real_dtype) == 7,
    "block header: real_dtype");
  CHECK(offsetof(sif_sdf_block_header_t, header_crc32) == 8,
    "block header: header_crc32");
  CHECK(offsetof(sif_sdf_block_header_t, flags) == 12, "block header: flags");
  CHECK(offsetof(sif_sdf_block_header_t, meta_bytes) == 16,
    "block header: meta_bytes");
  CHECK(offsetof(sif_sdf_block_header_t, crc32) == 20, "block header: crc32");
  CHECK(offsetof(sif_sdf_block_header_t, data_bytes) == 24,
    "block header: data_bytes");
  CHECK(offsetof(sif_sdf_block_header_t, n_items) == 32,
    "block header: n_items");
  CHECK(offsetof(sif_sdf_block_header_t, catalog_id) == CATALOG_ID_OFFSET,
    "block header: catalog_id");
  CHECK(offsetof(sif_sdf_block_header_t, name) == 48, "block header: name");

  const long payload = (long)(4 * 4 * sizeof(sif_real));
  const long expected = 64 + 64 + ((payload + 63) / 64) * 64;
  CHECK(file_bytes(SDF_PATH) == expected,
    "a four-void file is %ld bytes, expected %ld", file_bytes(SDF_PATH),
    expected);

  sif_catalog_free(cat);
  remove(SDF_PATH);
  (void)status;
}

static void test_catalog_identity(void) {
  printf("catalogue identity\n");

  sif_catalog_t* cat = make_catalog(N_VOIDS);
  sif_catalog_t* other = make_catalog(N_VOIDS);
  CHECK(cat && other, "catalogue allocation failed");
  if (!cat || !other)
    return;

  const uint64_t id = sif_catalog_id(cat);
  CHECK(id != 0, "a catalogue with voids in it has no identity");

  /* Two catalogues holding the very same voids are still two catalogues:
   * nothing measured from one belongs beside the other. */
  CHECK(sif_catalog_id(other) != id, "two catalogues share an identity");

  /* A trim moves every value in memory and changes nothing about which voids
   * they are. */
  CHECK(sif_catalog_trim(cat) == SIF_OK, "trim failed");
  CHECK(sif_catalog_id(cat) == id, "trimming changed the identity");

  /* An append does change which voids they are, and that is what makes a
   * measurement taken beforehand stale. */
  CHECK(sif_catalog_append(cat, 1, 2, 3, 4) == SIF_OK, "append failed");
  CHECK(sif_catalog_id(cat) != id, "an append left the identity alone");

  sif_catalog_free(cat);
  sif_catalog_free(other);
}

static void test_measurement_stamp(void) {
  printf("measurements carry their catalogue\n");

  sif_catalog_t* cat = make_catalog(N_VOIDS);
  sif_catalog_t* other = make_catalog(N_VOIDS);
  CHECK(cat && other, "catalogue allocation failed");
  if (!cat || !other)
    return;

  sif_size_function_t* vsf =
    sif_size_function_catalog(cat, (sif_real)BOX, 8, 0, 0, 0);
  CHECK(vsf != NULL, "size function failed");
  if (vsf)
    CHECK(vsf->source_id == sif_catalog_id(cat),
      "the size function does not name the catalogue it was binned from");
  sif_size_function_free(vsf);

  vsf = sif_size_function_catalog(other, (sif_real)BOX, 8, 0, 0, 0);
  if (vsf)
    CHECK(vsf->source_id != sif_catalog_id(cat),
      "two catalogues stamped their measurements the same");
  sif_size_function_free(vsf);

  sif_catalog_free(cat);
  sif_catalog_free(other);
}

static void test_catalog_roundtrip(void) {
  printf(".sdf catalogue round trip\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(N_VOIDS);
  CHECK(cat != NULL, "catalogue allocation failed");
  if (!cat)
    return;

  const uint64_t id = sif_catalog_id(cat);

  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);
  CHECK(file != NULL, "create failed: %s", sif_sdf_strerror(status));
  CHECK(sif_sdf_n_blocks(file) == 1, "a new file holds %u blocks, not one",
    sif_sdf_n_blocks(file));
  CHECK(sif_sdf_n_voids(file) == N_VOIDS, "void count not recorded");
  CHECK(sif_sdf_catalog_id(file) == id, "identity not recorded");
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "a clean create ended with %s",
    sif_sdf_strerror(status));

  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(file != NULL, "open failed: %s", sif_sdf_strerror(status));
  if (!file) {
    sif_catalog_free(cat);
    return;
  }
  CHECK(sif_sdf_box_length(file) == BOX, "box length did not round trip");

  sif_catalog_t* back = sif_sdf_catalog(file, &status);
  CHECK(back != NULL, "catalogue read failed: %s", sif_sdf_strerror(status));

  if (back) {
    CHECK(back->n_voids == N_VOIDS, "read %llu voids, expected %u",
      (unsigned long long)back->n_voids, N_VOIDS);
    CHECK(sif_catalog_id(back) == id,
      "the identity did not survive the round trip");

    /* Bit-exact, because the file was written at this build's precision and
     * read back into it: any difference here is a bug rather than a
     * conversion. */
    int identical = 1;
    for (uint64_t i = 0; i < N_VOIDS && identical; i++)
      identical = (back->cx[i] == cat->cx[i]) && (back->cy[i] == cat->cy[i]) &&
                  (back->cz[i] == cat->cz[i]) &&
                  (back->radii[i] == cat->radii[i]);
    CHECK(identical, "the voids did not survive the round trip");

    /* Every read allocates: two calls give two catalogues, and the file keeps
     * neither. */
    sif_catalog_t* again = sif_sdf_catalog(file, &status);
    CHECK(again != NULL && again != back,
      "a second read handed back the first catalogue");
    if (again)
      CHECK(sif_catalog_id(again) == id, "two reads of one file disagree");
    sif_catalog_free(again);
  }

  sif_catalog_free(back);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "a clean read ended with %s",
    sif_sdf_strerror(status));

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

static void test_empty_catalog(void) {
  printf("an empty catalogue\n");

  /* A run that found no voids has a result, and it is not the same thing as a
   * run that was never made. */
  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* empty = sif_catalog_alloc(1);
  CHECK(empty != NULL, "allocation failed");
  if (!empty)
    return;

  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, empty, 0, &status);
  CHECK(file != NULL, "an empty catalogue was refused: %s",
    sif_sdf_strerror(status));
  sif_sdf_close(file, &status);

  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  sif_catalog_t* back = sif_sdf_catalog(file, &status);
  CHECK(back != NULL && back->n_voids == 0,
    "an empty catalogue did not round trip");
  if (back)
    CHECK(sif_catalog_id(back) == sif_catalog_id(empty),
      "an empty catalogue lost its identity");

  sif_catalog_free(back);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "an empty file ended with %s",
    sif_sdf_strerror(status));

  sif_catalog_free(empty);
  remove(SDF_PATH);
}

static void test_creation_rejections(void) {
  printf("creation rejections\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(16);
  CHECK(cat != NULL, "catalogue allocation failed");
  if (!cat)
    return;

  /* There is no way to make a file without a catalogue, which is what makes
   * "a file always has one" a property rather than a rule. */
  remove(ALT_PATH);
  CHECK(sif_sdf_create(ALT_PATH, BOX, NULL, 0, &status) == NULL &&
          status == SIF_SDF_ERR_INVALID,
    "create accepted a NULL catalogue (%d)", (int)status);
  status = SIF_SDF_OK;
  CHECK(file_bytes(ALT_PATH) < 0, "a refused create still left a file behind");

  CHECK(sif_sdf_create(ALT_PATH, -1.0, cat, 0, &status) == NULL &&
          status == SIF_SDF_ERR_INVALID,
    "create accepted a negative box (%d)", (int)status);
  status = SIF_SDF_OK;

  /* An existing file is not overwritten unless the caller says so: it is
   * usually the output of a run that took hours. */
  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  CHECK(sif_sdf_create(SDF_PATH, BOX, cat, 0, &status) == NULL &&
          status == SIF_SDF_ERR_EXISTS,
    "create clobbered an existing file (%d)", (int)status);
  status = SIF_SDF_OK;

  sif_sdf_t* file =
    sif_sdf_create(SDF_PATH, BOX, cat, SIF_SDF_OVERWRITE, &status);
  CHECK(file != NULL, "overwrite failed: %s", sif_sdf_strerror(status));
  sif_sdf_close(file, &status);
  CHECK(
    status == SIF_SDF_OK, "overwrite ended with %s", sif_sdf_strerror(status));

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

static void test_missing_catalog(void) {
  printf("a file with no catalogue\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(16);
  CHECK(cat != NULL, "catalogue allocation failed");
  if (!cat)
    return;

  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");

  /* What a create that died between its two writes leaves behind. Reading it
   * is refused, since there is nothing in it to read; appending to it is not,
   * since that is how the run gets finished. */
  truncate_to(SDF_PATH, BLOCK_OFFSET);

  CHECK(sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status) == NULL &&
          status == SIF_SDF_ERR_NO_CATALOG,
    "a catalogue-less file opened for reading (%d)", (int)status);
  status = SIF_SDF_OK;

  sif_sdf_t* file = sif_sdf_open(SDF_PATH, SIF_SDF_APPEND, &status);
  CHECK(file != NULL, "a crashed create could not be reopened: %s",
    sif_sdf_strerror(status));
  if (file) {
    CHECK(sif_sdf_n_blocks(file) == 0, "the file should hold no blocks");
    CHECK(sif_sdf_catalog(file, &status) == NULL &&
            status == SIF_SDF_ERR_NO_CATALOG,
      "read a catalogue that is not there (%d)", (int)status);
    status = SIF_SDF_OK;
  }
  sif_sdf_close(file, &status);

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

static void test_damaged_files(void) {
  printf("damaged files\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(N_VOIDS);
  CHECK(cat != NULL, "catalogue allocation failed");
  if (!cat)
    return;

  /* A block cut short. The walk knows how long the block claimed to be, so
   * this is caught at open, before anything is allocated from the file. */
  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  truncate_to(SDF_PATH, file_bytes(SDF_PATH) - 64);
  CHECK(sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status) == NULL &&
          status == SIF_SDF_ERR_TRUNCATED,
    "a torn block opened (%d)", (int)status);
  status = SIF_SDF_OK;

  /* One flipped bit in the payload. The file is still structurally sound, so
   * it opens; the checksum is what stops the values reaching the caller. */
  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  FILE* f = fopen(SDF_PATH, "rb");
  unsigned char byte = 0;
  if (f) {
    fseek(f, DATA_OFFSET, SEEK_SET);
    if (fread(&byte, 1, 1, f) != 1)
      byte = 0;
    fclose(f);
  }
  byte ^= 0x01;
  poke(SDF_PATH, DATA_OFFSET, &byte, 1);

  sif_sdf_t* file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(file != NULL, "a flipped payload bit should not stop the file opening");
  CHECK(sif_sdf_catalog(file, &status) == NULL && status == SIF_SDF_ERR_CORRUPT,
    "a flipped bit read back as a clean catalogue (%d)", (int)status);
  status = SIF_SDF_OK;
  sif_sdf_close(file, &status);

  /* Something that is not a block where a block should be. */
  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  poke(SDF_PATH, BLOCK_OFFSET, "XXXX", 4);
  CHECK(sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status) == NULL &&
          status == SIF_SDF_ERR_CORRUPT,
    "a file with no block header at the first block opened (%d)", (int)status);
  status = SIF_SDF_OK;

  /* A catalogue that names itself nothing. Every block written afterwards
   * would claim to belong to it, and none of them could be checked. */
  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  const uint64_t nothing = 0;
  poke(SDF_PATH, BLOCK_OFFSET + CATALOG_ID_OFFSET, &nothing, sizeof(nothing));
  CHECK(sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status) == NULL &&
          status == SIF_SDF_ERR_CORRUPT,
    "a catalogue with no identity opened (%d)", (int)status);
  status = SIF_SDF_OK;

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

/* A density set of the shape the estimator would have produced, filled with
 * values that are easy to recognize on the way back. */
static sif_density_profiles_t* make_density(
  uint64_t n_voids, uint32_t n_bins, uint64_t source_id) {

  sif_density_profiles_t* profs =
    sif__density_profiles_alloc(n_voids, n_bins, (sif_real)3.0, true);
  if (!profs)
    return NULL;

  for (uint32_t b = 0; b <= n_bins; b++)
    profs->r_edges[b] = (sif_real)b * (sif_real)0.25;
  for (uint64_t i = 0; i < n_voids * n_bins; i++)
    profs->profiles[i] = (sif_real)((double)i * 1e-3 - 1.0);

  profs->source_id = source_id;
  return profs;
}

static sif_velocity_profiles_t* make_velocity(
  uint64_t n_voids, uint32_t n_bins, uint64_t source_id) {

  sif_velocity_profiles_t* profs =
    sif__velocity_profiles_alloc(n_voids, n_bins, (sif_real)3.0);
  if (!profs)
    return NULL;

  for (uint32_t b = 0; b <= n_bins; b++)
    profs->r_edges[b] = (sif_real)b * (sif_real)0.25;
  for (uint64_t i = 0; i < n_voids * n_bins; i++)
    profs->v_rad[i] = (sif_real)((double)i * 0.5);

  profs->source_id = source_id;
  return profs;
}

static void test_product_roundtrip(void) {
  printf("products round trip\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(N_VOIDS);
  CHECK(cat != NULL, "catalogue allocation failed");
  if (!cat)
    return;

  const uint64_t id = sif_catalog_id(cat);
  sif_size_function_t* vsf =
    sif_size_function_catalog(cat, (sif_real)BOX, 12, SIF_VSF_BIN_LINEAR, 0, 0);
  sif_density_profiles_t* dens = make_density(N_VOIDS, 10, id);
  sif_velocity_profiles_t* vel = make_velocity(N_VOIDS, 10, id);
  CHECK(vsf && dens && vel, "could not build the products");
  if (!vsf || !dens || !vel) {
    sif_catalog_free(cat);
    return;
  }

  /* One status across the whole run of calls, checked once at the end. */
  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);
  sif_sdf_append_size_function(file, vsf, "linear", &status);
  sif_sdf_append_density_profiles(file, dens, "fiducial", &status);
  sif_sdf_append_velocity_profiles(file, vel, "fiducial", &status);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "writing a full file ended with %s",
    sif_sdf_strerror(status));

  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(file != NULL, "open failed: %s", sif_sdf_strerror(status));
  if (!file) {
    sif_catalog_free(cat);
    return;
  }
  CHECK(sif_sdf_n_blocks(file) == 4, "expected four blocks, found %u",
    sif_sdf_n_blocks(file));

  /* A density set and a velocity set may share a name: the kind tells them
   * apart. */
  CHECK(sif_sdf_contains(file, SIF_SDF_BLOCK_DENSITY_PROFILES, "fiducial"),
    "the density set is not where it was put");
  CHECK(sif_sdf_contains(file, SIF_SDF_BLOCK_VELOCITY_PROFILES, "fiducial"),
    "the velocity set is not where it was put");
  CHECK(!sif_sdf_contains(file, SIF_SDF_BLOCK_SIZE_FUNCTION, "fiducial"),
    "a name matched across kinds");

  sif_size_function_t* vsf_back =
    sif_sdf_size_function(file, "linear", &status);
  CHECK(vsf_back != NULL, "size function read failed: %s",
    sif_sdf_strerror(status));
  if (vsf_back) {
    CHECK(vsf_back->n_bins == vsf->n_bins, "bin count did not round trip");
    CHECK(vsf_back->options == vsf->options,
      "the binning convention did not round trip");
    CHECK(vsf_back->r_min == vsf->r_min && vsf_back->r_max == vsf->r_max,
      "the radius bounds did not round trip");
    CHECK(vsf_back->source_id == id, "the size function lost its catalogue");

    int identical = 1;
    for (uint32_t b = 0; b < vsf->n_bins && identical; b++)
      identical = (vsf_back->vsf[b] == vsf->vsf[b]) &&
                  (vsf_back->err[b] == vsf->err[b]) &&
                  (vsf_back->counts[b] == vsf->counts[b]) &&
                  (vsf_back->r_centers[b] == vsf->r_centers[b]);
    CHECK(identical, "the size function values did not round trip");
    sif_size_function_free(vsf_back);
  }

  sif_density_profiles_t* dens_back =
    sif_sdf_density_profiles(file, "fiducial", &status);
  CHECK(dens_back != NULL, "density read failed: %s", sif_sdf_strerror(status));
  if (dens_back) {
    CHECK(dens_back->n_voids == N_VOIDS && dens_back->n_bins == 10,
      "the profile shape did not round trip");
    CHECK(dens_back->ext == dens->ext, "the extent did not round trip");
    /* Nothing in the values says which of the two binnings they are, which is
     * exactly why it is stored. */
    CHECK(dens_back->differential == dens->differential,
      "the binning did not round trip");
    CHECK(dens_back->source_id == id, "the profiles lost their catalogue");

    int identical = 1;
    for (uint64_t i = 0; i < (uint64_t)N_VOIDS * 10 && identical; i++)
      identical = dens_back->profiles[i] == dens->profiles[i];
    CHECK(identical, "the profile values did not round trip");
    sif_density_profiles_free(dens_back);
  }

  sif_velocity_profiles_t* vel_back =
    sif_sdf_velocity_profiles(file, NULL, &status);
  CHECK(vel_back != NULL, "velocity read failed: %s", sif_sdf_strerror(status));
  if (vel_back) {
    CHECK(vel_back->n_voids == N_VOIDS, "the row count did not round trip");
    int identical = 1;
    for (uint64_t i = 0; i < (uint64_t)N_VOIDS * 10 && identical; i++)
      identical = vel_back->v_rad[i] == vel->v_rad[i];
    CHECK(identical, "the velocity values did not round trip");
    sif_velocity_profiles_free(vel_back);
  }

  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "reading a full file ended with %s",
    sif_sdf_strerror(status));

  sif_size_function_free(vsf);
  sif_density_profiles_free(dens);
  sif_velocity_profiles_free(vel);
  sif_catalog_free(cat);
  remove(SDF_PATH);
}

static void test_names(void) {
  printf("names\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(N_VOIDS);
  if (!cat)
    return;

  sif_size_function_t* ln =
    sif_size_function_catalog(cat, (sif_real)BOX, 8, SIF_VSF_BIN_LN, 0, 0);
  sif_size_function_t* linear =
    sif_size_function_catalog(cat, (sif_real)BOX, 8, SIF_VSF_BIN_LINEAR, 0, 0);
  CHECK(ln && linear, "could not bin the size functions");

  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);
  sif_sdf_append_size_function(file, ln, "lnbins", &status);
  sif_sdf_append_size_function(file, linear, "linear", &status);
  CHECK(status == SIF_SDF_OK, "two names in one file ended with %s",
    sif_sdf_strerror(status));

  /* The same name twice for the same kind is what the check is for: the two
   * would be indistinguishable afterwards. */
  sif_sdf_append_size_function(file, linear, "lnbins", &status);
  CHECK(status == SIF_SDF_ERR_NAME_TAKEN, "a duplicate name was accepted (%d)",
    (int)status);
  status = SIF_SDF_OK;

  /* Cut down to fit, two names differing past the sixteenth byte would land
   * in the file as one. */
  sif_sdf_append_size_function(
    file, linear, "a name that is far too long to fit", &status);
  CHECK(status == SIF_SDF_ERR_INVALID, "an oversized name was accepted (%d)",
    (int)status);
  status = SIF_SDF_OK;

  CHECK(sif_sdf_n_blocks(file) == 3,
    "a refused append still added a block (%u)", sif_sdf_n_blocks(file));
  sif_sdf_close(file, &status);

  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  sif_size_function_t* by_name = sif_sdf_size_function(file, "linear", &status);
  CHECK(by_name != NULL && by_name->options == linear->options,
    "asking by name gave the wrong block");
  sif_size_function_free(by_name);

  /* No name asked for means the first of its kind. */
  sif_size_function_t* first = sif_sdf_size_function(file, NULL, &status);
  CHECK(first != NULL && first->options == ln->options,
    "asking for no name did not give the first block");
  sif_size_function_free(first);

  CHECK(sif_sdf_size_function(file, "absent", &status) == NULL &&
          status == SIF_SDF_ERR_ABSENT,
    "a block that is not there was read anyway (%d)", (int)status);
  status = SIF_SDF_OK;

  /* What a caller who did not write the file sees of it. */
  sif_sdf_block_info_t info;
  sif_sdf_info(file, 0, &info);
  CHECK(info.type == SIF_SDF_BLOCK_CATALOG && info.n_items == N_VOIDS,
    "block 0 is not the catalogue");
  sif_sdf_info(file, 2, &info);
  CHECK(info.type == SIF_SDF_BLOCK_SIZE_FUNCTION &&
          strcmp(info.name, "linear") == 0,
    "block 2 came back as `%s`", info.name);
  sif_sdf_info(file, 99, &info);
  CHECK(info.type == 0, "an index past the end described something");

  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "the names run ended with %s",
    sif_sdf_strerror(status));

  sif_size_function_free(ln);
  sif_size_function_free(linear);
  sif_catalog_free(cat);
  remove(SDF_PATH);
}

static void test_provenance(void) {
  printf("provenance\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(N_VOIDS);
  sif_catalog_t* other = make_catalog(N_VOIDS);
  CHECK(cat && other, "catalogue allocation failed");
  if (!cat || !other)
    return;

  /* Measured from a catalogue holding the very same voids -- and still not
   * this file's catalogue. */
  sif_size_function_t* foreign =
    sif_size_function_catalog(other, (sif_real)BOX, 8, 0, 0, 0);
  sif_size_function_t* mine =
    sif_size_function_catalog(cat, (sif_real)BOX, 8, 0, 0, 0);
  CHECK(foreign && mine, "could not bin the size functions");

  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);

  sif_sdf_append_size_function(file, foreign, "foreign", &status);
  CHECK(status == SIF_SDF_ERR_CATALOG_MISMATCH,
    "a size function from another catalogue was accepted (%d)", (int)status);
  status = SIF_SDF_OK;

  /* A model or a stitched size function names no catalogue at all, and cannot
   * prove it came from this one. */
  if (mine) {
    const uint64_t kept = mine->source_id;
    mine->source_id = 0;
    sif_sdf_append_size_function(file, mine, "anonymous", &status);
    CHECK(status == SIF_SDF_ERR_CATALOG_MISMATCH,
      "an unstamped size function was accepted (%d)", (int)status);
    status = SIF_SDF_OK;
    mine->source_id = kept;
  }

  /* Rows that do not line up with the catalogue, even carrying the right
   * identity. */
  sif_density_profiles_t* short_set =
    make_density(N_VOIDS - 1, 4, sif_catalog_id(cat));
  sif_sdf_append_density_profiles(file, short_set, "short", &status);
  CHECK(status == SIF_SDF_ERR_CATALOG_MISMATCH,
    "a set with the wrong number of rows was accepted (%d)", (int)status);
  status = SIF_SDF_OK;
  sif_density_profiles_free(short_set);

  CHECK(sif_sdf_n_blocks(file) == 1,
    "a refused append still added a block (%u)", sif_sdf_n_blocks(file));

  /* The one that does belong here. */
  sif_sdf_append_size_function(file, mine, "mine", &status);
  CHECK(status == SIF_SDF_OK,
    "the file refused its own catalogue's size "
    "function: %s",
    sif_sdf_strerror(status));
  sif_sdf_close(file, &status);

  /* A read handle is not a write handle. */
  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  sif_sdf_append_size_function(file, mine, "again", &status);
  CHECK(status == SIF_SDF_ERR_MODE, "a read handle accepted an append (%d)",
    (int)status);
  status = SIF_SDF_OK;

  /* A set read back out belongs to the file it came from, so it can go back
   * in -- under a name that is still free. */
  sif_size_function_t* back = sif_sdf_size_function(file, "mine", &status);
  CHECK(back != NULL && back->source_id == sif_catalog_id(cat),
    "a size function read back does not name the file's catalogue");
  sif_size_function_free(back);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "the provenance run ended with %s",
    sif_sdf_strerror(status));

  sif_size_function_free(foreign);
  sif_size_function_free(mine);
  sif_catalog_free(cat);
  sif_catalog_free(other);
  remove(SDF_PATH);
}

static void test_damaged_product(void) {
  printf("a damaged product block\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(64);
  if (!cat)
    return;

  sif_size_function_t* vsf =
    sif_size_function_catalog(cat, (sif_real)BOX, 8, 0, 0, 0);

  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);
  sif_sdf_append_size_function(file, vsf, "v", &status);
  const long block_at = (long)(64 + 64 + 4 * 64 * (long)sizeof(sif_real));
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "setup write failed");

  /* One bit inside the size function's metadata. The table is read before the
   * data and folded into the same checksum, so damage there is caught by the
   * same comparison that catches damage to the values. */
  unsigned char byte = 0;
  FILE* f = fopen(SDF_PATH, "rb");
  if (f) {
    fseek(f, block_at + 64, SEEK_SET);
    if (fread(&byte, 1, 1, f) != 1)
      byte = 0;
    fclose(f);
  }
  byte ^= 0x01;
  poke(SDF_PATH, block_at + 64, &byte, 1);

  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(
    file != NULL, "the file should still open: %s", sif_sdf_strerror(status));
  sif_size_function_t* damaged = sif_sdf_size_function(file, NULL, &status);
  CHECK(damaged == NULL && status != SIF_SDF_OK,
    "a damaged size function read back clean");
  status = SIF_SDF_OK;
  sif_sdf_close(file, &status);

  sif_size_function_free(vsf);
  sif_catalog_free(cat);
  remove(SDF_PATH);
}

static void test_metadata(void) {
  printf("metadata\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(32);
  if (!cat)
    return;

  const double cosmology[3] = {0.31, 0.049, 0.677};
  const int64_t seeds[2] = {12345, 67890};

  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);

  /* A file with no notes reads as a file with no notes, not as a failure. */
  sif_sdf_meta_t* empty = sif_sdf_meta_read(file, &status);
  CHECK(empty != NULL && sif_sdf_meta_count(empty) == 0,
    "a file without metadata did not read as an empty table");
  /* Writing nothing writes nothing, and is not an error either. */
  sif_sdf_meta_write(file, empty, &status);
  CHECK(status == SIF_SDF_OK && sif_sdf_n_blocks(file) == 1,
    "an empty table wrote a block");
  sif_sdf_meta_free(empty);

  sif_sdf_meta_t* meta = sif_sdf_meta_alloc();
  sif_sdf_meta_put_i64(meta, "n_tracers", 1048576);
  sif_sdf_meta_put_f64(meta, "redshift", 0.5);
  sif_sdf_meta_put_str(meta, "simulation", "quijote_fiducial_0");
  sif_sdf_meta_put_f64v(meta, "cosmology", cosmology, 3);
  sif_sdf_meta_put_i64v(meta, "seeds", seeds, 2);

  /* A table is a mapping: the second write of a key is the one meant. */
  sif_sdf_meta_put_f64(meta, "redshift", 0.55);
  CHECK(sif_sdf_meta_count(meta) == 5, "an overwrite added a key (%u)",
    sif_sdf_meta_count(meta));

  sif_sdf_meta_write(file, meta, &status);
  CHECK(status == SIF_SDF_OK, "writing metadata ended with %s",
    sif_sdf_strerror(status));
  sif_sdf_meta_free(meta);
  sif_sdf_close(file, &status);

  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  meta = sif_sdf_meta_read(file, &status);
  CHECK(meta != NULL, "reading metadata failed: %s", sif_sdf_strerror(status));
  if (!meta) {
    sif_catalog_free(cat);
    return;
  }

  CHECK(sif_sdf_meta_count(meta) == 5, "read back %u keys, expected 5",
    sif_sdf_meta_count(meta));

  int64_t n_tracers = 0;
  double redshift = 0.0;
  const char* simulation = NULL;
  const double* cosmo_back = NULL;
  const int64_t* seeds_back = NULL;
  uint32_t n = 0;

  CHECK(sif_sdf_meta_get_i64(meta, "n_tracers", &n_tracers) == SIF_SDF_OK &&
          n_tracers == 1048576,
    "an integer did not round trip");
  CHECK(sif_sdf_meta_get_f64(meta, "redshift", &redshift) == SIF_SDF_OK &&
          redshift == 0.55,
    "a double did not round trip, or the overwrite did not stick");
  CHECK(sif_sdf_meta_get_str(meta, "simulation", &simulation) == SIF_SDF_OK &&
          strcmp(simulation, "quijote_fiducial_0") == 0,
    "a string did not round trip");

  CHECK(sif_sdf_meta_get_f64v(meta, "cosmology", &cosmo_back, &n) == SIF_SDF_OK,
    "an array of doubles did not round trip");
  CHECK(n == 3 && cosmo_back && cosmo_back[0] == cosmology[0] &&
          cosmo_back[2] == cosmology[2],
    "the array of doubles came back wrong");
  CHECK(sif_sdf_meta_get_i64v(meta, "seeds", &seeds_back, &n) == SIF_SDF_OK &&
          n == 2 && seeds_back && seeds_back[1] == seeds[1],
    "an array of integers did not round trip");

  /* Types are not converted. An integer read as a double is a mistake in one
   * of the two places, and saying so is the only way either gets found. */
  double as_double = 0.0;
  int64_t as_int = 0;
  CHECK(sif_sdf_meta_get_f64(meta, "n_tracers", &as_double) == SIF_SDF_ERR_TYPE,
    "an integer was handed out as a double");
  CHECK(sif_sdf_meta_get_i64(meta, "simulation", &as_int) == SIF_SDF_ERR_TYPE,
    "a string was handed out as an integer");
  /* Asking for one number where the table holds three is the same mistake. */
  CHECK(sif_sdf_meta_get_f64(meta, "cosmology", &as_double) == SIF_SDF_ERR_TYPE,
    "an array was handed out as a scalar");
  CHECK(sif_sdf_meta_get_i64(meta, "absent", &as_int) == SIF_SDF_ERR_ABSENT,
    "a key that is not there was not reported as absent");

  CHECK(sif_sdf_meta_has(meta, "redshift") && !sif_sdf_meta_has(meta, "nope"),
    "has() disagrees with the table");
  CHECK(sif_sdf_meta_type(meta, "simulation") == SIF_SDF_META_STR,
    "a string does not know what it is");
  CHECK(sif_sdf_meta_type(meta, "nope") == SIF_SDF_META_NONE,
    "an absent key claims a type");
  CHECK(sif_sdf_meta_length(meta, "cosmology") == 3,
    "an array does not know how long it is");
  CHECK(sif_sdf_meta_length(meta, "simulation") == 18,
    "a string does not know how long it is (%u)",
    sif_sdf_meta_length(meta, "simulation"));

  /* Keys can be walked without knowing what is in the file. */
  int found = 0;
  for (uint32_t i = 0; i < sif_sdf_meta_count(meta); i++) {
    const char* key = sif_sdf_meta_key(meta, i);
    if (key && strcmp(key, "seeds") == 0)
      found = 1;
  }
  CHECK(found, "walking the keys did not find one that is there");
  CHECK(
    sif_sdf_meta_key(meta, 99) == NULL, "an index past the end named a key");

  sif_sdf_meta_free(meta);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "the metadata run ended with %s",
    sif_sdf_strerror(status));

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

static void test_metadata_merge(void) {
  printf("metadata merges and corrections\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(32);
  if (!cat)
    return;

  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);

  sif_sdf_meta_t* first = sif_sdf_meta_alloc();
  sif_sdf_meta_put_f64(first, "redshift", 0.5);
  sif_sdf_meta_put_str(first, "note", "first pass");
  sif_sdf_meta_write(file, first, &status);
  sif_sdf_meta_free(first);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "the first pass ended with %s",
    sif_sdf_strerror(status));

  /* Read it, correct it, write it back: the shape of the whole design. An
   * append-only file cannot rewrite the earlier block, so the correction is a
   * second one, and the later value is what a reader sees. */
  file = sif_sdf_open(SDF_PATH, SIF_SDF_APPEND, &status);
  sif_sdf_meta_t* meta = sif_sdf_meta_read(file, &status);
  CHECK(meta != NULL && sif_sdf_meta_count(meta) == 2,
    "the first pass did not read back");

  sif_sdf_meta_put_f64(meta, "redshift", 1.25);
  sif_sdf_meta_put_str(meta, "validated", "2026-08-22");
  sif_sdf_meta_write(file, meta, &status);
  sif_sdf_meta_free(meta);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "the correction ended with %s",
    sif_sdf_strerror(status));

  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(sif_sdf_n_blocks(file) == 3, "expected three blocks, found %u",
    sif_sdf_n_blocks(file));

  meta = sif_sdf_meta_read(file, &status);
  CHECK(meta != NULL, "the merge failed: %s", sif_sdf_strerror(status));

  double redshift = 0.0;
  const char* note = NULL;
  const char* validated = NULL;
  CHECK(sif_sdf_meta_count(meta) == 3,
    "the merged table has %u keys, expected 3", sif_sdf_meta_count(meta));
  CHECK(sif_sdf_meta_get_f64(meta, "redshift", &redshift) == SIF_SDF_OK &&
          redshift == 1.25,
    "the later value did not win (%g)", redshift);
  CHECK(sif_sdf_meta_get_str(meta, "note", &note) == SIF_SDF_OK &&
          strcmp(note, "first pass") == 0,
    "a key only the first block carries was lost");
  CHECK(sif_sdf_meta_get_str(meta, "validated", &validated) == SIF_SDF_OK,
    "a key only the second block carries was lost");

  sif_sdf_meta_free(meta);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "the merge run ended with %s",
    sif_sdf_strerror(status));

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

static void test_metadata_rejections(void) {
  printf("metadata rejections\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(32);
  if (!cat)
    return;

  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);

  /* The library keeps `sif.` for the parameters that say what a block is. A
   * caller who could write those could change what a block claims to be, and
   * the reader would believe them. */
  sif_sdf_meta_t* meta = sif_sdf_meta_alloc();
  sif_sdf_meta_put_i64(meta, "good", 1);
  sif_sdf_meta_put_i64(meta, "sif.n_bins", 99);
  CHECK(!sif_sdf_meta_has(meta, "sif.n_bins"),
    "a reserved key made it into the table");

  /* The failed put is answered for at the write, so a table that did not come
   * out as asked is never written half-formed. */
  sif_sdf_meta_write(file, meta, &status);
  CHECK(status == SIF_SDF_ERR_INVALID,
    "a table with a failed put was written anyway (%d)", (int)status);
  status = SIF_SDF_OK;
  CHECK(sif_sdf_n_blocks(file) == 1, "a refused write still added a block");
  sif_sdf_meta_free(meta);

  sif_sdf_close(file, &status);

  /* A read handle is not a write handle, for notes as much as for products. */
  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  meta = sif_sdf_meta_alloc();
  sif_sdf_meta_put_i64(meta, "n", 1);
  sif_sdf_meta_write(file, meta, &status);
  CHECK(status == SIF_SDF_ERR_MODE, "a read handle accepted metadata (%d)",
    (int)status);
  status = SIF_SDF_OK;
  sif_sdf_meta_free(meta);
  sif_sdf_close(file, &status);

  /* One flipped bit in a metadata block. It carries no data section, so its
   * checksum is over the table alone. */
  file = sif_sdf_open(SDF_PATH, SIF_SDF_APPEND, &status);
  meta = sif_sdf_meta_alloc();
  sif_sdf_meta_put_str(meta, "simulation", "quijote_fiducial_0");
  sif_sdf_meta_write(file, meta, &status);
  sif_sdf_meta_free(meta);
  const long meta_at = (long)(64 + 64 + 4 * 32 * (long)sizeof(sif_real));
  sif_sdf_close(file, &status);
  CHECK(
    status == SIF_SDF_OK, "setup write failed: %s", sif_sdf_strerror(status));

  unsigned char byte = 0;
  FILE* f = fopen(SDF_PATH, "rb");
  if (f) {
    fseek(f, meta_at + 64 + 8, SEEK_SET);
    if (fread(&byte, 1, 1, f) != 1)
      byte = 0;
    fclose(f);
  }
  byte ^= 0x01;
  poke(SDF_PATH, meta_at + 64 + 8, &byte, 1);

  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(
    file != NULL, "the file should still open: %s", sif_sdf_strerror(status));
  sif_sdf_meta_t* damaged = sif_sdf_meta_read(file, &status);
  CHECK(damaged == NULL && status != SIF_SDF_OK,
    "damaged metadata read back clean");
  status = SIF_SDF_OK;
  sif_sdf_close(file, &status);

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

static void test_repair(void) {
  printf("verify and repair\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(64);
  if (!cat)
    return;

  sif_size_function_t* vsf =
    sif_size_function_catalog(cat, (sif_real)BOX, 8, 0, 0, 0);
  CHECK(vsf != NULL, "could not bin a size function");
  if (!vsf) {
    sif_catalog_free(cat);
    return;
  }

  /* A sound file has nothing to recover, and saying so costs nothing. */
  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);
  sif_sdf_append_size_function(file, vsf, "v", &status);
  sif_sdf_close(file, &status);
  CHECK(
    status == SIF_SDF_OK, "setup write failed: %s", sif_sdf_strerror(status));

  const long whole = file_bytes(SDF_PATH);
  CHECK(sif_sdf_verify(SDF_PATH, &status) == 0 && status == SIF_SDF_OK,
    "a sound file was reported damaged (%d)", (int)status);
  CHECK(sif_sdf_repair(SDF_PATH, &status) == 0 && status == SIF_SDF_OK,
    "a sound file was cut back");
  CHECK(file_bytes(SDF_PATH) == whole, "a sound file changed size");

  /* What a run that died mid-append leaves. The file does not open at all,
   * which is the whole reason there has to be a way back. */
  truncate_to(SDF_PATH, whole - 32);
  CHECK(sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status) == NULL,
    "a file with a torn tail opened");
  status = SIF_SDF_OK;

  const uint64_t damaged = sif_sdf_verify(SDF_PATH, &status);
  CHECK(damaged > 0 && status == SIF_SDF_OK,
    "verify did not see the torn tail (%llu, %d)", (unsigned long long)damaged,
    (int)status);
  CHECK(file_bytes(SDF_PATH) == whole - 32, "verify changed the file");

  const uint64_t dropped = sif_sdf_repair(SDF_PATH, &status);
  CHECK(dropped == damaged && status == SIF_SDF_OK,
    "repair dropped %llu where verify said %llu", (unsigned long long)dropped,
    (unsigned long long)damaged);

  /* And what was underneath is readable again. */
  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(file != NULL, "the repaired file does not open: %s",
    sif_sdf_strerror(status));
  CHECK(sif_sdf_n_blocks(file) == 1, "the repaired file holds %u blocks",
    sif_sdf_n_blocks(file));
  sif_catalog_t* back = sif_sdf_catalog(file, &status);
  CHECK(back != NULL && back->n_voids == 64,
    "the catalogue did not survive the repair");
  sif_catalog_free(back);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "the repaired file ended with %s",
    sif_sdf_strerror(status));

  /* A damaged catalogue is not a damaged tail: there is nothing to keep, and
   * cutting the file back to a bare header would turn a broken file into an
   * empty one. */
  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  const long before = file_bytes(SDF_PATH);
  unsigned char byte = 0;
  FILE* f = fopen(SDF_PATH, "rb");
  if (f) {
    fseek(f, DATA_OFFSET, SEEK_SET);
    if (fread(&byte, 1, 1, f) != 1)
      byte = 0;
    fclose(f);
  }
  byte ^= 0x01;
  poke(SDF_PATH, DATA_OFFSET, &byte, 1);

  CHECK(
    sif_sdf_verify(SDF_PATH, &status) == 0 && status == SIF_SDF_ERR_NO_CATALOG,
    "a damaged catalogue was reported as a damaged tail (%d)", (int)status);
  status = SIF_SDF_OK;
  CHECK(
    sif_sdf_repair(SDF_PATH, &status) == 0 && status == SIF_SDF_ERR_NO_CATALOG,
    "repair tried to salvage a file with no sound catalogue (%d)", (int)status);
  status = SIF_SDF_OK;
  CHECK(file_bytes(SDF_PATH) == before,
    "a file with nothing to recover was truncated anyway");

  sif_size_function_free(vsf);
  sif_catalog_free(cat);
  remove(SDF_PATH);
}

/*
 * The rules that only matter years from now: what an old build does with a
 * file a newer one wrote. Nothing here can be produced by this build, which
 * is exactly why it is worth a test -- the first real one of these files will
 * turn up long after anyone remembers what was promised.
 */
static void test_forward_compatibility(void) {
  printf("blocks a later sif might add\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(32);
  if (!cat)
    return;

  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  const uint64_t id = catalog_id_of(SDF_PATH);
  CHECK(id != 0, "the catalogue block carries no identity");

  const char payload[48] = {1};

  /* A type sif has not assigned, carrying this file's catalogue. Stepped over
   * by length: `data_bytes` is what lets a build that has never heard of a
   * type know how far past it to seek. */
  sif_sdf_block_header_t unknown = forged_header(12, 7, id);
  memcpy(unknown.name, "future", 6);
  CHECK(append_forged(SDF_PATH, unknown, NULL, 0, payload, sizeof payload),
    "could not forge a block of an unknown type");

  /* The private range, deliberately naming a catalogue that is not this
   * file's. sif skips such a block whole and does not audit it, which is the
   * difference between "sif ignores what it did not write" and "sif refuses
   * the file that carries it". */
  sif_sdf_block_header_t private_block =
    forged_header((uint16_t)SIF_SDF_BLOCK_PRIVATE, 3, 0xDEADBEEFu);
  CHECK(append_forged(SDF_PATH, private_block, NULL, 0, payload, 16),
    "could not forge a private block");

  sif_sdf_t* file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(file != NULL && status == SIF_SDF_OK,
    "a file holding blocks from a later sif did not open: %s",
    sif_sdf_strerror(status));
  if (!file) {
    sif_catalog_free(cat);
    return;
  }

  CHECK(sif_sdf_n_blocks(file) == 3, "%u blocks, expected 3",
    (unsigned)sif_sdf_n_blocks(file));

  sif_sdf_block_info_t info;
  sif_sdf_info(file, 1, &info);
  CHECK(info.type == 12 && info.n_items == 7 &&
          strcmp(info.name, "future") == 0,
    "the unknown block reads as type %u, %llu items, named `%s`",
    (unsigned)info.type, (unsigned long long)info.n_items, info.name);

  /* The point of stepping over them rather than refusing the file: what this
   * build does understand is still there. */
  sif_catalog_t* back = sif_sdf_catalog(file, &status);
  CHECK(back != NULL && status == SIF_SDF_OK,
    "the catalogue could not be read past an unknown block: %s",
    sif_sdf_strerror(status));
  if (back) {
    CHECK(back->n_voids == 32, "read %llu voids, expected 32",
      (unsigned long long)back->n_voids);
    sif_catalog_free(back);
  }

  sif_sdf_close(file, NULL);
  sif_catalog_free(cat);
  remove(SDF_PATH);
}

/*
 * A block this build cannot interpret has to say so as a version, not as
 * damage.
 *
 * The distinction is the whole value of the check. "Corrupt" sends a caller
 * looking for a bad disk and reaching for the recovery tool; "written by a
 * newer sif" sends them to an upgrade, which is the thing that will actually
 * work.
 */
static void test_uninterpretable_blocks(void) {
  printf("blocks this build cannot interpret\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(16);
  if (!cat)
    return;

  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  const uint64_t id = catalog_id_of(SDF_PATH);
  const char payload[64] = {2};

  /* A known type laid out to a payload version from the future. Its metadata
   * is written to a scheme this build does not have, so a reader that parsed
   * the table first would miss the first shape key it looked for and call the
   * file corrupt -- which is what this build used to do. */
  sif_sdf_block_header_t future =
    forged_header((uint16_t)SIF_SDF_BLOCK_SIZE_FUNCTION, 4, id);
  future.type_version = (uint8_t)(SIF__SDF_TYPE_VERSION + 1);
  memcpy(future.name, "ahead", 5);
  CHECK(append_forged(SDF_PATH, future, NULL, 0, payload, sizeof payload),
    "could not forge a block from a later layout");

  /* A block asking for a feature bit in the half that may not be ignored. */
  sif_sdf_block_header_t flagged =
    forged_header((uint16_t)SIF_SDF_BLOCK_SIZE_FUNCTION, 4, id);
  flagged.flags = 1u;
  memcpy(flagged.name, "critical", 8);
  CHECK(append_forged(SDF_PATH, flagged, NULL, 0, payload, sizeof payload),
    "could not forge a block with a critical flag");

  /* And one in the half that may: this must not be refused for its flags,
   * whatever else is wrong with it. */
  sif_sdf_block_header_t ignorable =
    forged_header((uint16_t)SIF_SDF_BLOCK_SIZE_FUNCTION, 4, id);
  ignorable.flags = 0x10000u;
  memcpy(ignorable.name, "ignorable", 9);
  CHECK(append_forged(SDF_PATH, ignorable, NULL, 0, payload, sizeof payload),
    "could not forge a block with an ignorable flag");

  /* None of the three stops the file being opened: a block is only refused
   * when something asks for it. */
  sif_sdf_t* file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(file != NULL && status == SIF_SDF_OK,
    "a file holding a block from a later sif did not open: %s",
    sif_sdf_strerror(status));
  if (!file) {
    sif_catalog_free(cat);
    return;
  }

  status = SIF_SDF_OK;
  CHECK(sif_sdf_size_function(file, "ahead", &status) == NULL &&
          status == SIF_SDF_ERR_VERSION,
    "a later payload version reported %d (%s), expected the version error",
    (int)status, sif_sdf_strerror(status));

  status = SIF_SDF_OK;
  CHECK(sif_sdf_size_function(file, "critical", &status) == NULL &&
          status == SIF_SDF_ERR_VERSION,
    "an unknown critical flag reported %d (%s), expected the version error",
    (int)status, sif_sdf_strerror(status));

  status = SIF_SDF_OK;
  CHECK(sif_sdf_size_function(file, "ignorable", &status) == NULL &&
          status != SIF_SDF_ERR_VERSION,
    "a flag in the ignorable half was treated as a version this build "
    "does not know");

  sif_sdf_close(file, NULL);
  sif_catalog_free(cat);
  remove(SDF_PATH);
}

/*
 * The same question for a metadata table, where the answer matters more:
 * notes are the one thing in the format meant to degrade gracefully, and an
 * all-or-nothing failure costs a caller every note in the file rather than
 * the one it could not read.
 */
static void test_metadata_forward_compatibility(void) {
  printf("metadata a later sif might write\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(16);
  if (!cat)
    return;

  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");

  /* An entry of a type that does not exist, between two that do. The length
   * is in bytes rather than in elements precisely so this is steppable
   * without knowing the type -- and the test of that is the entry *after* it
   * still arriving. */
  char table[256];
  uint32_t n = 0;
  const double redshift = 0.5;
  const int64_t tracers = 42;
  const char blob[24] = {(char)0xAA};

  n += forge_entry(table + n, "redshift", SIF_SDF_META_F64, &redshift,
    (uint32_t)sizeof redshift);
  n += forge_entry(table + n, "exotic", 9, blob, (uint32_t)sizeof blob);
  n += forge_entry(
    table + n, "n_tracers", SIF_SDF_META_I64, &tracers, (uint32_t)sizeof tracers);

  sif_sdf_block_header_t notes =
    forged_header((uint16_t)SIF_SDF_BLOCK_META, 0, 0);
  CHECK(append_forged(SDF_PATH, notes, table, n, NULL, 0),
    "could not forge a metadata block");

  /* A metadata block carrying a data section, which version 1 never writes.
   * Its checksum covers that section, so a reader that skips the section
   * still has to fold it in to arrive at the same number. Getting this wrong
   * does not lose this block -- it loses every note in the file, the valid
   * one above included. */
  sif_sdf_block_header_t fat =
    forged_header((uint16_t)SIF_SDF_BLOCK_META, 0, 0);
  const int64_t later = 7;
  const uint32_t fat_bytes =
    forge_entry(table + n, "appended_later", SIF_SDF_META_I64, &later,
      (uint32_t)sizeof later);
  const char section[32] = {3};
  CHECK(append_forged(SDF_PATH, fat, table + n, fat_bytes, section,
          sizeof section),
    "could not forge a metadata block with a data section");

  sif_sdf_t* file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(file != NULL && status == SIF_SDF_OK, "the file did not open: %s",
    sif_sdf_strerror(status));
  if (!file) {
    sif_catalog_free(cat);
    return;
  }

  sif_sdf_meta_t* meta = sif_sdf_meta_read(file, &status);
  CHECK(meta != NULL && status == SIF_SDF_OK,
    "metadata from a later sif could not be read: %s",
    sif_sdf_strerror(status));

  if (meta) {
    double back_f = 0.0;
    int64_t back_i = 0;
    CHECK(sif_sdf_meta_get_f64(meta, "redshift", &back_f) == SIF_SDF_OK &&
            back_f == redshift,
      "the entry before an unknown one did not survive");
    CHECK(sif_sdf_meta_get_i64(meta, "n_tracers", &back_i) == SIF_SDF_OK &&
            back_i == tracers,
      "the entry after an unknown one did not survive");
    CHECK(sif_sdf_meta_get_i64(meta, "appended_later", &back_i) == SIF_SDF_OK &&
            back_i == later,
      "a metadata block carrying a data section lost its table");
    CHECK(!sif_sdf_meta_has(meta, "exotic"),
      "an entry of an unknown type came back as a value");
    sif_sdf_meta_free(meta);
  }

  sif_sdf_close(file, NULL);
  sif_catalog_free(cat);
  remove(SDF_PATH);
}

/*
 * A catalogue stored at the other precision.
 *
 * The format's one concession to interoperability, and the thing sif cannot
 * test against itself: the library is built single-precision, so a file at
 * the other width has to be assembled by hand. Written the other way round
 * on a double build, so this covers the conversion whichever way sif is
 * configured.
 */
static void test_other_precision(void) {
  printf("a catalogue at the other precision\n");

  const sif_sdf_dtype_t other =
    (sizeof(sif_real) == 8) ? SIF_SDF_F32 : SIF_SDF_F64;

  /* Exactly representable at either width, so a mismatch here is the
   * conversion going wrong and not rounding doing its job. */
  static const double centres[3] = {10.5, 20.25, 30.125};
  static const double radii[3] = {1.25, 2.5, 5.0};

  char payload[4 * 3 * sizeof(double)];
  size_t at = 0;
  for (int view = 0; view < 4; view++) {
    const double* src = (view == 3) ? radii : centres;
    for (int i = 0; i < 3; i++) {
      if (other == SIF_SDF_F64) {
        const double wide = src[i];
        memcpy(payload + at, &wide, sizeof wide);
        at += sizeof wide;
      } else {
        const float narrow = (float)src[i];
        memcpy(payload + at, &narrow, sizeof narrow);
        at += sizeof narrow;
      }
    }
  }

  remove(SDF_PATH);
  CHECK(write_forged_file_header(SDF_PATH), "could not forge a file header");

  sif_sdf_block_header_t header =
    forged_header((uint16_t)SIF_SDF_BLOCK_CATALOG, 3, 0x5DF00Du);
  header.real_dtype = (uint8_t)other;
  CHECK(append_forged(SDF_PATH, header, NULL, 0, payload, at),
    "could not forge a catalogue at the other precision");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_sdf_t* file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(file != NULL && status == SIF_SDF_OK,
    "a file at the other precision did not open: %s",
    sif_sdf_strerror(status));
  if (!file)
    return;

  /* The checksum is taken over the bytes as they sit in the file rather than
   * over the converted values, or a build of the other width would compute a
   * different one for an intact file and reject it. */
  sif_catalog_t* cat = sif_sdf_catalog(file, &status);
  CHECK(cat != NULL && status == SIF_SDF_OK,
    "a catalogue at the other precision could not be read: %s",
    sif_sdf_strerror(status));

  if (cat) {
    CHECK(cat->n_voids == 3, "read %llu voids, expected 3",
      (unsigned long long)cat->n_voids);
    for (int i = 0; i < 3; i++) {
      CHECK((double)cat->cx[i] == centres[i],
        "void %d came back at cx %.17g, expected %.17g", i, (double)cat->cx[i],
        centres[i]);
      CHECK((double)cat->radii[i] == radii[i],
        "void %d came back at r %.17g, expected %.17g", i,
        (double)cat->radii[i], radii[i]);
    }
    sif_catalog_free(cat);
  }

  sif_sdf_close(file, NULL);
  remove(SDF_PATH);
}

/*
 * The framing rules, which are checked at open rather than at whichever call
 * first wants the thing they describe.
 */
static void test_framing_rejections(void) {
  printf("framing rejections\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(16);
  if (!cat)
    return;

  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");

  /* Where the metadata block is about to land, so its header can be found
   * again afterwards. */
  const long notes_at = file_bytes(SDF_PATH);

  char table[64];
  const int64_t value = 1;
  const uint32_t bytes =
    forge_entry(table, "note", SIF_SDF_META_I64, &value, (uint32_t)sizeof value);

  sif_sdf_block_header_t notes =
    forged_header((uint16_t)SIF_SDF_BLOCK_META, 0, 0);
  CHECK(append_forged(SDF_PATH, notes, table, bytes, NULL, 0),
    "could not forge a metadata block");

  /* Entries are padded to eight, so a table whose length is not a multiple of
   * eight cannot be a sequence of them -- and the data section behind it does
   * not begin where the format says. A framing fault, so it is refused when
   * the chain is walked and not when someone asks for a table. */
  const uint32_t ragged = bytes + 1;
  poke(SDF_PATH, notes_at + 16, &ragged, sizeof ragged);

  CHECK(sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status) == NULL &&
          status == SIF_SDF_ERR_CORRUPT,
    "a metadata table of a ragged length opened (%d)", (int)status);
  status = SIF_SDF_OK;

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

/*
 * What recovery does with a file that has nothing to recover.
 *
 * Reporting a file sound is a promise that it opens. A prefix that does not
 * begin with a catalogue will not open however much is cut off the end of it,
 * so saying so is the difference between a recovery that helps and one that
 * reports success and hands back the same unopenable file.
 */
static void test_recovery_needs_a_catalogue(void) {
  printf("recovering a file with no catalogue\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(16);
  if (!cat)
    return;

  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  const long whole = file_bytes(SDF_PATH);

  /* Block 0 still checksums -- the checksum covers the payload, and this
   * changes the header -- so the scan finds a sound block that is not a
   * catalogue. */
  const uint16_t not_a_catalogue = (uint16_t)SIF_SDF_BLOCK_META;
  poke(SDF_PATH, BLOCK_OFFSET + 4, &not_a_catalogue, sizeof not_a_catalogue);

  CHECK(sif_sdf_verify(SDF_PATH, &status) == 0 &&
          status == SIF_SDF_ERR_NO_CATALOG,
    "verify called a file with no catalogue recoverable (%d)", (int)status);
  status = SIF_SDF_OK;

  CHECK(sif_sdf_repair(SDF_PATH, &status) == 0 &&
          status == SIF_SDF_ERR_NO_CATALOG,
    "repair claimed to recover a file with no catalogue (%d)", (int)status);
  status = SIF_SDF_OK;

  CHECK(file_bytes(SDF_PATH) == whole,
    "a file with nothing to recover was truncated anyway");

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

/*
 * Values too large to put on the wire.
 *
 * The guards are on the arithmetic rather than on the memory: a value long
 * enough to wrap the length of the entry carrying it would size a buffer too
 * short and then fill it past the end. Neither case here allocates anything
 * like what it names -- both are refused on the number alone.
 */
static void test_metadata_limits(void) {
  printf("metadata too large for the wire\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(8);
  if (!cat)
    return;

  remove(SDF_PATH);
  sif_sdf_t* file = sif_sdf_create(SDF_PATH, BOX, cat, 0, &status);
  CHECK(file != NULL, "setup create failed: %s", sif_sdf_strerror(status));
  if (!file) {
    sif_catalog_free(cat);
    return;
  }

  /* An array whose byte length would not fit the field that counts it. The
   * count is refused before the array is touched, so the pointer below is
   * never read. */
  sif_sdf_meta_t* huge = sif_sdf_meta_alloc();
  const int64_t one = 1;
  sif_sdf_meta_put_i64v(huge, "too_long", &one, UINT32_MAX);
  sif_sdf_meta_write(file, huge, &status);
  CHECK(status == SIF_SDF_ERR_INVALID,
    "an array too long for the wire was not refused (%d)", (int)status);
  status = SIF_SDF_OK;
  sif_sdf_meta_free(huge);

  /* A key is length-prefixed with a uint16, so one that does not fit would be
   * written at its full length and described at a truncated one -- a table
   * nothing could decode. */
  char* long_key = malloc(70000);
  if (long_key) {
    memset(long_key, 'k', 69999);
    long_key[69999] = '\0';

    sif_sdf_meta_t* keyed = sif_sdf_meta_alloc();
    sif_sdf_meta_put_i64(keyed, long_key, 1);
    sif_sdf_meta_write(file, keyed, &status);
    CHECK(status == SIF_SDF_ERR_INVALID,
      "a key too long for the wire was not refused (%d)", (int)status);
    status = SIF_SDF_OK;
    sif_sdf_meta_free(keyed);
    free(long_key);
  }

  sif_sdf_close(file, &status);
  sif_catalog_free(cat);
  remove(SDF_PATH);
}

/*
 * One bit, flipped in a header.
 *
 * These are the corruptions the format could not see. A block's payload
 * checksum covers what the block *holds*; until the headers carried checksums
 * of their own, nothing covered what a block *says it is*, and each of the
 * flips below changed an answer while every checksum in the file still
 * matched -- the box a run was measured in, whether a block exists at all,
 * which block a name refers to. A reader that accepts a damaged file is worse
 * than one that fails, because the numbers flow onward and whatever they
 * produce looks like a result.
 */
static void test_header_corruption(void) {
  printf("one bit flipped in a header\n");

  static const struct {
    const char* what;
    long offset;
    int bit;
  } flips[] = {
    /* The box every product in the file is expressed in. Silently became
     * 1000.00390625, and every measurement read afterwards was scaled to a
     * box the run was never in. */
    {"the box length", 20, 3},
    /* Into the private range, where sif skips a block whole: a profile set
     * would simply cease to exist, and the call for it would report that the
     * file never held one. */
    {"a block's type", BLOCK_OFFSET + 5, 7},
    /* `linear` and `linbar` are two different binnings. */
    {"a block's name", BLOCK_OFFSET + 48, 0},
    /* How many voids the file says it holds. */
    {"a block's item count", BLOCK_OFFSET + 32, 1},
    /* The identity that ties every block in the file to the catalogue. */
    {"the catalogue identity", BLOCK_OFFSET + CATALOG_ID_OFFSET, 5},
    /* Reserved space, which carries nothing today -- and is covered anyway,
     * so that it is still covered on the day it carries something. */
    {"the reserved space", 56, 2},
  };

  sif_catalog_t* cat = make_catalog(16);
  if (!cat)
    return;

  for (size_t i = 0; i < sizeof flips / sizeof flips[0]; i++) {
    sif_sdf_status_t status = SIF_SDF_OK;

    CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
    flip_bit(SDF_PATH, flips[i].offset, flips[i].bit);

    sif_sdf_t* file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
    CHECK(file == NULL && status == SIF_SDF_ERR_CORRUPT,
      "%s was flipped by one bit and the file still opened (%d)",
      flips[i].what, (int)status);

    sif_sdf_close(file, NULL);
    remove(SDF_PATH);
  }

  sif_catalog_free(cat);
}

/*
 * A header that fails its own checksum ends the sound part of the file.
 *
 * The lengths in a block header are what say where the next block starts, so
 * a header that does not stand behind itself is not something to read past --
 * following it would be the guessing this scan exists not to do.
 */
static void test_recovery_from_a_bad_header(void) {
  printf("recovering past a damaged block header\n");

  sif_sdf_status_t status = SIF_SDF_OK;
  sif_catalog_t* cat = make_catalog(64);
  if (!cat)
    return;

  sif_size_function_t* vsf =
    sif_size_function_catalog(cat, (sif_real)BOX, 8, 0, 0, 0);
  if (!vsf) {
    sif_catalog_free(cat);
    return;
  }

  /* Written in two steps, so the catalogue is on disk and measurable before
   * the second block goes on the end of it: a size taken while the stream is
   * still open is a size of whatever has reached the disk so far. */
  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");
  const long catalogue_ends = file_bytes(SDF_PATH);

  sif_sdf_t* file = sif_sdf_open(SDF_PATH, SIF_SDF_APPEND, &status);
  sif_sdf_append_size_function(file, vsf, "v", &status);
  sif_sdf_close(file, &status);
  CHECK(status == SIF_SDF_OK, "setup append failed: %s",
    sif_sdf_strerror(status));

  const long whole = file_bytes(SDF_PATH);

  /* One bit in the second block's header. Its payload is untouched and still
   * checksums, so nothing but the header check can see this. */
  flip_bit(SDF_PATH, catalogue_ends + 32, 1);

  CHECK(sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status) == NULL &&
          status == SIF_SDF_ERR_CORRUPT,
    "a file with a damaged block header opened (%d)", (int)status);
  status = SIF_SDF_OK;

  const uint64_t damaged = sif_sdf_verify(SDF_PATH, &status);
  CHECK(damaged == (uint64_t)(whole - catalogue_ends) && status == SIF_SDF_OK,
    "verify reported %llu damaged bytes, expected %ld",
    (unsigned long long)damaged, whole - catalogue_ends);

  const uint64_t dropped = sif_sdf_repair(SDF_PATH, &status);
  CHECK(dropped == damaged && status == SIF_SDF_OK,
    "repair dropped %llu where verify said %llu", (unsigned long long)dropped,
    (unsigned long long)damaged);

  /* And what is left is a file, not a prefix of one: the catalogue survives
   * the block that was lost. */
  file = sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status);
  CHECK(file != NULL && status == SIF_SDF_OK,
    "the recovered file did not open: %s", sif_sdf_strerror(status));
  if (file) {
    CHECK(sif_sdf_n_blocks(file) == 1, "%u blocks left, expected 1",
      (unsigned)sif_sdf_n_blocks(file));
    sif_catalog_t* back = sif_sdf_catalog(file, &status);
    CHECK(back != NULL && status == SIF_SDF_OK,
      "the catalogue did not survive: %s", sif_sdf_strerror(status));
    sif_catalog_free(back);
    sif_sdf_close(file, NULL);
  }

  sif_size_function_free(vsf);
  sif_catalog_free(cat);
  remove(SDF_PATH);
}

/*
 * A container from before the headers were checksummed.
 *
 * Refused rather than read on trust: version 1 laid its block headers out
 * differently, so reading one would mean believing 64 bytes that nothing ever
 * covered, at the offsets of a layout it does not use.
 */
static void test_older_container(void) {
  printf("a container older than this build\n");

  sif_catalog_t* cat = make_catalog(4);
  if (!cat)
    return;

  CHECK(write_good_file(SDF_PATH, cat), "setup write failed");

  /* Nothing else about the file changes: the point is that the version alone
   * is enough to turn it away, and that it is turned away as a version rather
   * than as damage. */
  const uint32_t older = SIF_SDF_VERSION - 1;
  poke(SDF_PATH, 4, &older, sizeof older);

  sif_sdf_status_t status = SIF_SDF_OK;
  CHECK(sif_sdf_open(SDF_PATH, SIF_SDF_READ, &status) == NULL &&
          status == SIF_SDF_ERR_VERSION,
    "a version %u container was not refused as a version (%d)",
    (unsigned)older, (int)status);

  sif_catalog_free(cat);
  remove(SDF_PATH);
}

int main(void) {
  sif_init(SIF_CONFIG_QUIET);

  test_header_layout();
  test_catalog_identity();
  test_measurement_stamp();
  test_catalog_roundtrip();
  test_empty_catalog();
  test_creation_rejections();
  test_missing_catalog();
  test_damaged_files();
  test_product_roundtrip();
  test_names();
  test_provenance();
  test_damaged_product();
  test_metadata();
  test_metadata_merge();
  test_metadata_rejections();
  test_metadata_limits();
  test_repair();
  test_recovery_needs_a_catalogue();
  test_framing_rejections();
  test_forward_compatibility();
  test_uninterpretable_blocks();
  test_metadata_forward_compatibility();
  test_other_precision();
  test_header_corruption();
  test_recovery_from_a_bad_header();
  test_older_container();

  remove(SDF_PATH);
  remove(ALT_PATH);

  sif_finalize();

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
    failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
