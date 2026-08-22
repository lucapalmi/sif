/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The measured products: stacked profiles and the size function.
 *
 * Every one of them follows the same shape, and the shape is the argument for
 * the format. Writing is a gate and a block: the file has a catalogue, the
 * product names that catalogue, the name is free. Reading is the mirror: the
 * block is there, its payload is the length its own shape implies, its
 * checksum matches, and only then does anything reach the caller.
 *
 * The parameters that give a block its shape -- how many bins, how far out,
 * which binning -- live in the block's metadata under keys beginning `sif.`,
 * not in a struct of their own. One mechanism serves both those and whatever
 * the caller wants to attach, and a product that grows a parameter later costs
 * a key rather than a payload version.
 */

#include "sdf_internal.h"

#include "measure/profiles_internal.h"
#include "sif/utils/crc32.h"
#include "sif/utils/logger.h"
#include "structures/results_internal.h"

#include <stdlib.h>
#include <string.h>

/* The structural keys. Spelled once here so the writer and the reader cannot
 * drift apart over a typo. */
#define KEY_N_BINS       "sif.n_bins"
#define KEY_EXT          "sif.ext"
#define KEY_DIFFERENTIAL "sif.differential"
#define KEY_OPTIONS      "sif.options"
#define KEY_R_MIN        "sif.r_min"
#define KEY_R_MAX        "sif.r_max"

/*
 * Everything a product writer does with its metadata: encode it, hand it to
 * the block writer, release it.
 *
 * The table is built by a run of puts that report nothing, so the encode is
 * where a failed one surfaces -- the same trade the public calls make with
 * their status argument.
 */
static sif_sdf_status_t block_write_with_meta(sif_sdf_t* file,
  sif_sdf_block_header_t* header, sif_sdf_meta_t* meta,
  const sif__sdf_span_t* spans, uint32_t n_spans) {

  void* table = NULL;
  uint32_t bytes = 0;

  sif_sdf_status_t status = sif__sdf_meta_encode(meta, &table, &bytes);
  if (status != SIF_SDF_OK)
    SIF_LOG_ERROR(
      "sdf", "could not build the metadata for a block of %s", file->path);
  else
    status = sif__sdf_block_write(file, header, table, bytes, spans, n_spans);

  free(table);
  sif_sdf_meta_free(meta);
  return status;
}

/*
 * A shape parameter the block has to carry.
 *
 * Absent, or there and of another type, is a corrupt block rather than the
 * ordinary empty answer a caller's own key would give: these are not notes,
 * they are what says how long the arrays are, and a block missing one cannot
 * be read at all.
 */
static sif_sdf_status_t need_i64(
  const sif_sdf_meta_t* meta, const char* key, int64_t* out, const char* path) {
  if (sif_sdf_meta_get_i64(meta, key, out) != SIF_SDF_OK) {
    SIF_LOG_ERROR("sdf", "a block in %s is missing its `%s`", path, key);
    return SIF_SDF_ERR_CORRUPT;
  }
  return SIF_SDF_OK;
}

static sif_sdf_status_t need_f64(
  const sif_sdf_meta_t* meta, const char* key, double* out, const char* path) {
  if (sif_sdf_meta_get_f64(meta, key, out) != SIF_SDF_OK) {
    SIF_LOG_ERROR("sdf", "a block in %s is missing its `%s`", path, key);
    return SIF_SDF_ERR_CORRUPT;
  }
  return SIF_SDF_OK;
}

/* Reads one block's metadata into a fresh table. */
static sif_sdf_status_t block_meta_read(sif_sdf_t* file,
  const sif__sdf_entry_t* entry, sif_sdf_meta_t** out, uint32_t* crc) {

  sif_sdf_meta_t* meta = sif_sdf_meta_alloc();
  if (!meta)
    return SIF_SDF_ERR_ALLOC;

  const sif_sdf_status_t status =
    sif__sdf_meta_read_block(file, entry, meta, crc);
  if (status != SIF_SDF_OK) {
    sif_sdf_meta_free(meta);
    return status;
  }

  *out = meta;
  return SIF_SDF_OK;
}

/* The header every product block starts from: its type, its name, and the
 * catalogue it belongs to. */
static sif_sdf_status_t product_header(sif_sdf_t* file,
  sif_sdf_block_type_t type, const char* name, uint64_t n_items,
  sif_sdf_block_header_t* out) {

  memset(out, 0, sizeof(sif_sdf_block_header_t));
  out->type = (uint16_t)type;
  out->real_dtype = (uint16_t)sif__sdf_native_dtype();
  out->n_items = n_items;
  out->catalog_id = file->catalog_id;

  return sif__sdf_name_pack(name, out->name, "an append");
}

/*
 * Finds the block a read was asked for, or says why it cannot.
 *
 * Absence is its own answer rather than an empty result: a caller asking for
 * the profiles measured with one extent and getting the ones measured with
 * another, or getting nothing at all, would go on to plot whichever it was
 * handed.
 */
static sif_sdf_status_t product_find(sif_sdf_t* file, sif_sdf_block_type_t type,
  const char* name, const sif__sdf_entry_t** out) {

  if (!file) {
    SIF_LOG_ERROR("sdf", "no file given to read from");
    return SIF_SDF_ERR_INVALID;
  }

  const int32_t index = sif__sdf_find(file, type, name);
  if (index < 0) {
    SIF_LOG_ERROR("sdf", "%s holds no block of that kind named `%s`",
      file->path, (name && name[0]) ? name : "");
    return SIF_SDF_ERR_ABSENT;
  }

  *out = &file->blocks[index];
  return SIF_SDF_OK;
}

/* --- density profiles --- */

void sif_sdf_append_density_profiles(sif_sdf_t* file,
  const sif_density_profiles_t* profs, const char* name,
  sif_sdf_status_t* status) {

  if (!sif__sdf_status_accepts(status, "append_density_profiles"))
    return;

  if (!profs) {
    SIF_LOG_ERROR("sdf", "no density profiles given to append");
    *status = SIF_SDF_ERR_INVALID;
    return;
  }

  sif_sdf_status_t reason = sif__sdf_append_gate(
    file, SIF_SDF_BLOCK_DENSITY_PROFILES, name, profs->source_id);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return;
  }

  if (profs->n_voids != file->n_voids) {
    SIF_LOG_ERROR("sdf", "%llu rows of profiles for the %llu voids in %s",
      (unsigned long long)profs->n_voids, (unsigned long long)file->n_voids,
      file->path);
    *status = SIF_SDF_ERR_CATALOG_MISMATCH;
    return;
  }

  sif_sdf_block_header_t header;
  reason = product_header(
    file, SIF_SDF_BLOCK_DENSITY_PROFILES, name, profs->n_voids, &header);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return;
  }

  sif_sdf_meta_t* meta = sif_sdf_meta_alloc();
  if (!meta) {
    *status = SIF_SDF_ERR_ALLOC;
    return;
  }
  sif__sdf_meta_put_i64(meta, KEY_N_BINS, profs->n_bins);
  sif__sdf_meta_put_f64(meta, KEY_EXT, (double)profs->ext);
  /* Not recoverable from the values, and it decides which radius a bin
   * belongs at: a cumulative bin is everything within its outer edge, a
   * differential one is a shell and belongs at its centre. */
  sif__sdf_meta_put_i64(meta, KEY_DIFFERENTIAL, profs->differential ? 1 : 0);

  const sif__sdf_span_t spans[2] = {
    {profs->r_edges, ((uint64_t)profs->n_bins + 1) * sizeof(sif_real)},
    {profs->profiles, profs->n_voids * profs->n_bins * sizeof(sif_real)}};

  *status = block_write_with_meta(file, &header, meta, spans, 2);
}

sif_density_profiles_t* sif_sdf_density_profiles(
  sif_sdf_t* file, const char* name, sif_sdf_status_t* status) {

  if (!sif__sdf_status_accepts(status, "density_profiles"))
    return NULL;

  const sif__sdf_entry_t* entry = NULL;
  sif_sdf_status_t reason =
    product_find(file, SIF_SDF_BLOCK_DENSITY_PROFILES, name, &entry);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  const sif_sdf_dtype_t dtype = (sif_sdf_dtype_t)entry->header.real_dtype;
  const uint64_t width = sif__sdf_dtype_bytes(dtype);
  const uint64_t n_voids = entry->header.n_items;

  /* The metadata comes first whether or not the shape needed it: the checksum
   * covers the table and then the data, in that order. */
  sif_sdf_meta_t* table = NULL;
  uint32_t crc = SIF_CRC32_INIT;
  reason = block_meta_read(file, entry, &table, &crc);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  int64_t n_bins = 0, differential = 0;
  double ext = 0.0;
  reason = need_i64(table, KEY_N_BINS, &n_bins, file->path);
  if (reason == SIF_SDF_OK)
    reason = need_f64(table, KEY_EXT, &ext, file->path);
  if (reason == SIF_SDF_OK)
    reason = need_i64(table, KEY_DIFFERENTIAL, &differential, file->path);
  sif_sdf_meta_free(table);

  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  if (n_bins <= 0 || n_bins > UINT32_MAX) {
    SIF_LOG_ERROR("sdf", "a profile block in %s claims %lld bins", file->path,
      (long long)n_bins);
    *status = SIF_SDF_ERR_CORRUPT;
    return NULL;
  }

  const uint64_t bins = (uint64_t)n_bins;

  /* Guarded before the product is formed, since the count and the bins both
   * come out of the file. */
  if (n_voids != 0 && bins > UINT64_MAX / n_voids / width) {
    SIF_LOG_ERROR(
      "sdf", "a profile block in %s is too large to be real", file->path);
    *status = SIF_SDF_ERR_CORRUPT;
    return NULL;
  }

  reason =
    sif__sdf_read_gate(file, entry, ((bins + 1) + n_voids * bins) * width);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  sif_density_profiles_t* profs = sif__density_profiles_alloc(
    n_voids, (uint32_t)bins, (sif_real)ext, differential != 0);
  if (!profs) {
    SIF_LOG_ERROR("sdf", "failed to allocate a profile set of %llu rows",
      (unsigned long long)n_voids);
    *status = SIF_SDF_ERR_ALLOC;
    return NULL;
  }

  uint64_t offset = sif__sdf_data_offset(entry);
  reason =
    sif__sdf_read_reals(file, offset, bins + 1, dtype, profs->r_edges, &crc);
  if (reason == SIF_SDF_OK) {
    offset += (bins + 1) * width;
    reason = sif__sdf_read_reals(
      file, offset, n_voids * bins, dtype, profs->profiles, &crc);
  }

  if (reason == SIF_SDF_OK && sif_crc32_final(crc) != entry->header.crc32) {
    SIF_LOG_ERROR(
      "sdf", "a profile block in %s does not match its checksum", file->path);
    reason = SIF_SDF_ERR_CORRUPT;
  }

  if (reason != SIF_SDF_OK) {
    sif_density_profiles_free(profs);
    *status = reason;
    return NULL;
  }

  /* Stamped with the catalogue the file holds, so a set read back out is as
   * good as one measured here: it can be written to this file again, and to
   * no other. */
  profs->source_id = entry->header.catalog_id;

  return profs;
}

/* --- velocity profiles --- */

void sif_sdf_append_velocity_profiles(sif_sdf_t* file,
  const sif_velocity_profiles_t* profs, const char* name,
  sif_sdf_status_t* status) {

  if (!sif__sdf_status_accepts(status, "append_velocity_profiles"))
    return;

  if (!profs) {
    SIF_LOG_ERROR("sdf", "no velocity profiles given to append");
    *status = SIF_SDF_ERR_INVALID;
    return;
  }

  sif_sdf_status_t reason = sif__sdf_append_gate(
    file, SIF_SDF_BLOCK_VELOCITY_PROFILES, name, profs->source_id);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return;
  }

  if (profs->n_voids != file->n_voids) {
    SIF_LOG_ERROR("sdf", "%llu rows of profiles for the %llu voids in %s",
      (unsigned long long)profs->n_voids, (unsigned long long)file->n_voids,
      file->path);
    *status = SIF_SDF_ERR_CATALOG_MISMATCH;
    return;
  }

  sif_sdf_block_header_t header;
  reason = product_header(
    file, SIF_SDF_BLOCK_VELOCITY_PROFILES, name, profs->n_voids, &header);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return;
  }

  sif_sdf_meta_t* meta = sif_sdf_meta_alloc();
  if (!meta) {
    *status = SIF_SDF_ERR_ALLOC;
    return;
  }
  sif__sdf_meta_put_i64(meta, KEY_N_BINS, profs->n_bins);
  sif__sdf_meta_put_f64(meta, KEY_EXT, (double)profs->ext);

  const sif__sdf_span_t spans[2] = {
    {profs->r_edges, ((uint64_t)profs->n_bins + 1) * sizeof(sif_real)},
    {profs->v_rad, profs->n_voids * profs->n_bins * sizeof(sif_real)}};

  *status = block_write_with_meta(file, &header, meta, spans, 2);
}

sif_velocity_profiles_t* sif_sdf_velocity_profiles(
  sif_sdf_t* file, const char* name, sif_sdf_status_t* status) {

  if (!sif__sdf_status_accepts(status, "velocity_profiles"))
    return NULL;

  const sif__sdf_entry_t* entry = NULL;
  sif_sdf_status_t reason =
    product_find(file, SIF_SDF_BLOCK_VELOCITY_PROFILES, name, &entry);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  const sif_sdf_dtype_t dtype = (sif_sdf_dtype_t)entry->header.real_dtype;
  const uint64_t width = sif__sdf_dtype_bytes(dtype);
  const uint64_t n_voids = entry->header.n_items;

  sif_sdf_meta_t* table = NULL;
  uint32_t crc = SIF_CRC32_INIT;
  reason = block_meta_read(file, entry, &table, &crc);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  int64_t n_bins = 0;
  double ext = 0.0;
  reason = need_i64(table, KEY_N_BINS, &n_bins, file->path);
  if (reason == SIF_SDF_OK)
    reason = need_f64(table, KEY_EXT, &ext, file->path);
  sif_sdf_meta_free(table);

  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  if (n_bins <= 0 || n_bins > UINT32_MAX) {
    SIF_LOG_ERROR("sdf", "a profile block in %s claims %lld bins", file->path,
      (long long)n_bins);
    *status = SIF_SDF_ERR_CORRUPT;
    return NULL;
  }

  const uint64_t bins = (uint64_t)n_bins;

  if (n_voids != 0 && bins > UINT64_MAX / n_voids / width) {
    SIF_LOG_ERROR(
      "sdf", "a profile block in %s is too large to be real", file->path);
    *status = SIF_SDF_ERR_CORRUPT;
    return NULL;
  }

  reason =
    sif__sdf_read_gate(file, entry, ((bins + 1) + n_voids * bins) * width);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  sif_velocity_profiles_t* profs =
    sif__velocity_profiles_alloc(n_voids, (uint32_t)bins, (sif_real)ext);
  if (!profs) {
    SIF_LOG_ERROR("sdf", "failed to allocate a profile set of %llu rows",
      (unsigned long long)n_voids);
    *status = SIF_SDF_ERR_ALLOC;
    return NULL;
  }

  uint64_t offset = sif__sdf_data_offset(entry);
  reason =
    sif__sdf_read_reals(file, offset, bins + 1, dtype, profs->r_edges, &crc);
  if (reason == SIF_SDF_OK) {
    offset += (bins + 1) * width;
    reason = sif__sdf_read_reals(
      file, offset, n_voids * bins, dtype, profs->v_rad, &crc);
  }

  if (reason == SIF_SDF_OK && sif_crc32_final(crc) != entry->header.crc32) {
    SIF_LOG_ERROR(
      "sdf", "a profile block in %s does not match its checksum", file->path);
    reason = SIF_SDF_ERR_CORRUPT;
  }

  if (reason != SIF_SDF_OK) {
    sif_velocity_profiles_free(profs);
    *status = reason;
    return NULL;
  }

  profs->source_id = entry->header.catalog_id;

  return profs;
}

/* --- the size function --- */

void sif_sdf_append_size_function(sif_sdf_t* file,
  const sif_size_function_t* vsf, const char* name, sif_sdf_status_t* status) {

  if (!sif__sdf_status_accepts(status, "append_size_function"))
    return;

  if (!vsf) {
    SIF_LOG_ERROR("sdf", "no size function given to append");
    *status = SIF_SDF_ERR_INVALID;
    return;
  }
  if (vsf->n_bins == 0) {
    SIF_LOG_ERROR("sdf", "a size function of no bins cannot be written");
    *status = SIF_SDF_ERR_INVALID;
    return;
  }

  /* A modelled or stitched size function names no catalogue, and the gate
   * turns it away for that reason: it was not measured from the voids this
   * file holds, and a curve sitting next to them would be read as though it
   * had been. */
  sif_sdf_status_t reason = sif__sdf_append_gate(
    file, SIF_SDF_BLOCK_SIZE_FUNCTION, name, vsf->source_id);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return;
  }

  sif_sdf_block_header_t header;
  reason = product_header(
    file, SIF_SDF_BLOCK_SIZE_FUNCTION, name, vsf->n_bins, &header);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return;
  }

  sif_sdf_meta_t* meta = sif_sdf_meta_alloc();
  if (!meta) {
    *status = SIF_SDF_ERR_ALLOC;
    return;
  }
  /* The options travel with the values because they record whether the
   * function is per unit ln R or per unit R, which nothing else in the block
   * reveals. */
  sif__sdf_meta_put_i64(meta, KEY_OPTIONS, (int64_t)vsf->options);
  sif__sdf_meta_put_f64(meta, KEY_R_MIN, (double)vsf->r_min);
  sif__sdf_meta_put_f64(meta, KEY_R_MAX, (double)vsf->r_max);

  const uint64_t bins = vsf->n_bins;
  const sif__sdf_span_t spans[5] = {
    {vsf->r_edges, (bins + 1) * sizeof(sif_real)},
    {vsf->r_centers, bins * sizeof(sif_real)},
    {vsf->counts, bins * sizeof(uint64_t)}, {vsf->vsf, bins * sizeof(sif_real)},
    {vsf->err, bins * sizeof(sif_real)}};

  *status = block_write_with_meta(file, &header, meta, spans, 5);
}

sif_size_function_t* sif_sdf_size_function(
  sif_sdf_t* file, const char* name, sif_sdf_status_t* status) {

  if (!sif__sdf_status_accepts(status, "size_function"))
    return NULL;

  const sif__sdf_entry_t* entry = NULL;
  sif_sdf_status_t reason =
    product_find(file, SIF_SDF_BLOCK_SIZE_FUNCTION, name, &entry);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  const sif_sdf_dtype_t dtype = (sif_sdf_dtype_t)entry->header.real_dtype;
  const uint64_t width = sif__sdf_dtype_bytes(dtype);
  const uint64_t bins = entry->header.n_items;

  if (bins == 0 || bins > UINT32_MAX) {
    SIF_LOG_ERROR("sdf", "a size function in %s claims %llu bins", file->path,
      (unsigned long long)bins);
    *status = SIF_SDF_ERR_CORRUPT;
    return NULL;
  }

  sif_sdf_meta_t* table = NULL;
  uint32_t crc = SIF_CRC32_INIT;
  reason = block_meta_read(file, entry, &table, &crc);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  int64_t options = 0;
  double r_min = 0.0, r_max = 0.0;
  reason = need_i64(table, KEY_OPTIONS, &options, file->path);
  if (reason == SIF_SDF_OK)
    reason = need_f64(table, KEY_R_MIN, &r_min, file->path);
  if (reason == SIF_SDF_OK)
    reason = need_f64(table, KEY_R_MAX, &r_max, file->path);
  sif_sdf_meta_free(table);

  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  /* Four arrays of reals, one of counts: the payload is not homogeneous, and
   * the counts are stored at their own width whatever the reals are. */
  const uint64_t expected =
    ((bins + 1) + 3 * bins) * width + bins * sizeof(uint64_t);

  reason = sif__sdf_read_gate(file, entry, expected);
  if (reason != SIF_SDF_OK) {
    *status = reason;
    return NULL;
  }

  sif_size_function_t* vsf = sif__size_function_alloc((uint32_t)bins);
  if (!vsf) {
    SIF_LOG_ERROR("sdf", "failed to allocate a size function of %llu bins",
      (unsigned long long)bins);
    *status = SIF_SDF_ERR_ALLOC;
    return NULL;
  }

  vsf->options = (sif_option)options;
  vsf->r_min = (sif_real)r_min;
  vsf->r_max = (sif_real)r_max;

  uint64_t offset = sif__sdf_data_offset(entry);

  reason =
    sif__sdf_read_reals(file, offset, bins + 1, dtype, vsf->r_edges, &crc);
  offset += (bins + 1) * width;

  if (reason == SIF_SDF_OK) {
    reason =
      sif__sdf_read_reals(file, offset, bins, dtype, vsf->r_centers, &crc);
    offset += bins * width;
  }
  if (reason == SIF_SDF_OK) {
    reason = sif__sdf_read_raw(
      file, offset, vsf->counts, (size_t)(bins * sizeof(uint64_t)), &crc);
    offset += bins * sizeof(uint64_t);
  }
  if (reason == SIF_SDF_OK) {
    reason = sif__sdf_read_reals(file, offset, bins, dtype, vsf->vsf, &crc);
    offset += bins * width;
  }
  if (reason == SIF_SDF_OK)
    reason = sif__sdf_read_reals(file, offset, bins, dtype, vsf->err, &crc);

  if (reason == SIF_SDF_OK && sif_crc32_final(crc) != entry->header.crc32) {
    SIF_LOG_ERROR(
      "sdf", "a size function in %s does not match its checksum", file->path);
    reason = SIF_SDF_ERR_CORRUPT;
  }

  if (reason != SIF_SDF_OK) {
    sif_size_function_free(vsf);
    *status = reason;
    return NULL;
  }

  vsf->source_id = entry->header.catalog_id;

  return vsf;
}
