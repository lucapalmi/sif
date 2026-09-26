/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Stands in for hdf5.c in a build without HDF5 (SIF_HDF5_SUPPORT=OFF, or AUTO
 * that found none). Every HDF5 entry point exists here too, so code calling
 * them compiles and runs the same either way.
 *
 * A writer is called at the end of a run, with the results in hand, so it
 * does not refuse: it dumps the product as plain text beside the requested
 * path, at <path>.<group>.txt, warns naming that file, and succeeds. The dumps
 * are the raw arrays and nothing more -- a comment header saying what the
 * columns are, then numbers numpy.loadtxt reads directly -- since their only
 * job is to keep the data. The catalogue goes through the ordinary ASCII
 * writer, so it can also be read back by sif.
 *
 * A reader has nothing to fall back to, and is called at the start of a run,
 * where failing costs nothing: it refuses with SIF_ERR_UNSUPPORTED.
 */

#include "sif/io/catalog_io.h"
#include "sif/io/hdf5_io.h"

#include "io/gadget_internal.h"
#include "io/hdf5_internal.h"
#include "sif/utils/logger.h"

#include <stdbool.h>
#include <stdio.h>

#define TAG "hdf5"

int sif__hdf5_describe(char* buf, size_t len) {
  snprintf(buf, len, "no HDF5");
  return 0;
}

/* --- the fallback --- */

/* <path>.<group>.txt, which never collides with the path itself: a text file
 * under the name the caller gave an HDF5 file would break whatever reads it
 * next, and confusingly. */
static int dump_path(
  char* buf, size_t len, const char* filepath, const char* group) {
  const int n = snprintf(buf, len, "%s.%s.txt", filepath, group);
  return (n > 0 && (size_t)n < len) ? SIF_OK : SIF_ERR_INVALID;
}

static void warn_dumped(const char* filepath, const char* dump) {
  SIF_LOG_WARNING(TAG,
    "this build of sif has no HDF5 support, so %s was not written; the data "
    "is saved as plain text in %s instead. Rebuild with "
    "-DSIF_HDF5_SUPPORT=ON to write HDF5",
    filepath, dump);
}

/* Closes the dump and says whether every byte reached it: fprintf() reports
 * nothing useful per call, so the stream's error flag and the flush in
 * fclose() are what tell a full disk from a written file. */
static int dump_close(FILE* f, const char* dump) {
  const bool ok = ferror(f) == 0;
  if (fclose(f) != 0 || !ok) {
    SIF_LOG_ERROR(TAG, "failed to write %s", dump);
    return SIF_ERR_IO;
  }
  return SIF_OK;
}

/* One profile set: its shape and edges as comments, then one row of n_bins
 * values per void. */
static int dump_profiles(const char* dump, const char* what, uint64_t n_voids,
  uint32_t n_bins, sif_real ext, int differential, const sif_real* r_edges,
  const sif_real* rows) {

  FILE* f = fopen(dump, "w");
  if (!f) {
    SIF_LOG_ERROR(TAG, "failed to open %s for writing", dump);
    return SIF_ERR_IO;
  }

  fprintf(f, "# sif %s: n_voids %" PRIu64 ", n_bins %u, ext " SIF_PRI_REAL,
    what, n_voids, n_bins, ext);
  if (differential >= 0)
    fprintf(f, ", differential %d", differential);
  fprintf(f, "\n# r_edges (units of each void's radius):");
  for (uint32_t b = 0; b <= n_bins; b++)
    fprintf(f, " " SIF_PRI_REAL, r_edges[b]);
  fprintf(f, "\n# one row per void, one column per bin\n");

  for (uint64_t v = 0; v < n_voids; v++) {
    const sif_real* row = rows + v * n_bins;
    for (uint32_t b = 0; b < n_bins; b++)
      fprintf(f, b ? " " SIF_PRI_REAL : SIF_PRI_REAL, row[b]);
    fputc('\n', f);
  }

  return dump_close(f, dump);
}

/* --- writers --- */

int sif_catalog_write_hdf5(const char* filepath, const sif_catalog_t* catalog) {
  if (!filepath || !catalog) {
    SIF_LOG_ERROR(TAG, "invalid arguments for sif_catalog_write_hdf5");
    return SIF_ERR_INVALID;
  }

  char dump[4096];
  if (dump_path(dump, sizeof(dump), filepath, "catalog") != SIF_OK)
    return SIF_ERR_INVALID;

  const int status = sif_catalog_write_ascii(catalog, dump);
  if (status == SIF_OK)
    warn_dumped(filepath, dump);
  return status;
}

int sif_profiles_write_hdf5(const char* filepath,
  const sif_density_profiles_t* dens, const sif_velocity_profiles_t* vel) {

  if (!filepath || (!dens && !vel)) {
    SIF_LOG_ERROR(
      TAG, "sif_profiles_write_hdf5 needs a path and at least one profile set");
    return SIF_ERR_INVALID;
  }

  char dump[4096];

  if (dens) {
    if (dump_path(dump, sizeof(dump), filepath, "density_profiles") != SIF_OK ||
        dump_profiles(dump, "density profiles", dens->n_voids, dens->n_bins,
          dens->ext, dens->differential ? 1 : 0, dens->r_edges,
          dens->profiles) != SIF_OK)
      return SIF_ERR_IO;
    warn_dumped(filepath, dump);
  }

  if (vel) {
    if (dump_path(dump, sizeof(dump), filepath, "velocity_profiles") !=
          SIF_OK ||
        dump_profiles(dump, "velocity profiles", vel->n_voids, vel->n_bins,
          vel->ext, -1, vel->r_edges, vel->v_rad) != SIF_OK)
      return SIF_ERR_IO;
    warn_dumped(filepath, dump);
  }

  return SIF_OK;
}

int sif_size_function_write_hdf5(
  const char* filepath, const sif_size_function_t* vsf) {

  if (!filepath || !vsf) {
    SIF_LOG_ERROR(TAG, "invalid arguments for sif_size_function_write_hdf5");
    return SIF_ERR_INVALID;
  }

  char dump[4096];
  if (dump_path(dump, sizeof(dump), filepath, "size_function") != SIF_OK)
    return SIF_ERR_INVALID;

  FILE* f = fopen(dump, "w");
  if (!f) {
    SIF_LOG_ERROR(TAG, "failed to open %s for writing", dump);
    return SIF_ERR_IO;
  }

  const bool linear = (vsf->options & SIF__VSF_BIN_MASK) == SIF_VSF_BIN_LINEAR;

  fprintf(f,
    "# sif size function: n_bins %u, binning %s (vsf per unit %s), options "
    "%u\n# r_lo r_hi r_center counts vsf err\n",
    vsf->n_bins, linear ? "linear" : "ln", linear ? "R" : "ln R",
    (unsigned)vsf->options);

  for (uint32_t b = 0; b < vsf->n_bins; b++) {
    fprintf(f,
      SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL " %" PRIu64
                   " " SIF_PRI_REAL " " SIF_PRI_REAL "\n",
      vsf->r_edges[b], vsf->r_edges[b + 1], vsf->r_centers[b], vsf->counts[b],
      vsf->vsf[b], vsf->err[b]);
  }

  const int status = dump_close(f, dump);
  if (status == SIF_OK)
    warn_dumped(filepath, dump);
  return status;
}

/* --- readers --- */

static void refuse(const char* filepath) {
  SIF_LOG_ERROR(TAG,
    "cannot read %s: this build of sif has no HDF5 support. Rebuild with "
    "-DSIF_HDF5_SUPPORT=ON",
    filepath ? filepath : "(null)");
}

sif_catalog_t* sif_catalog_read_hdf5(const char* filepath) {
  refuse(filepath);
  return NULL;
}

int sif_profiles_read_header_hdf5(
  const char* filepath, int* out_has_density, int* out_has_velocity) {

  if (out_has_density)
    *out_has_density = 0;
  if (out_has_velocity)
    *out_has_velocity = 0;

  refuse(filepath);
  return SIF_ERR_UNSUPPORTED;
}

int sif_profiles_read_hdf5(const char* filepath,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {

  if (out_dens)
    *out_dens = NULL;
  if (out_vel)
    *out_vel = NULL;

  refuse(filepath);
  return SIF_ERR_UNSUPPORTED;
}

sif_size_function_t* sif_size_function_read_hdf5(const char* filepath) {
  refuse(filepath);
  return NULL;
}

/* --- metadata --- */

/*
 * Setters append to <path>.attributes.txt -- `group key = value`, one entry a
 * line, in the order they were set -- so that a run's provenance is kept with
 * its dumps. The same keys are refused as in a build with HDF5.
 */
/* The kind of value an entry holds, as far as the dump is concerned. */
typedef enum { META_INT, META_REAL, META_STRING } meta_kind_t;

static int meta_dump(const char* filepath, const char* group, const char* key,
  meta_kind_t kind, int64_t i, double d, const char* str) {

  if (!filepath || !key || !*key || (kind == META_STRING && !str)) {
    SIF_LOG_ERROR(TAG, "invalid arguments for a metadata entry");
    return SIF_ERR_INVALID;
  }

  const char* g = sif__hdf5_group(group);
  if (sif__hdf5_key_reserved(g, key)) {
    SIF_LOG_ERROR(TAG,
      "'%s' on /%s is an attribute sif keeps for itself and cannot be set", key,
      g ? g : "");
    return SIF_ERR_INVALID;
  }

  char dump[4096];
  if (dump_path(dump, sizeof(dump), filepath, "attributes") != SIF_OK)
    return SIF_ERR_INVALID;

  FILE* f = fopen(dump, "a");
  if (!f) {
    SIF_LOG_ERROR(TAG, "failed to open %s for writing", dump);
    return SIF_ERR_IO;
  }

  fprintf(f, "/%s %s = ", g ? g : "", key);
  switch (kind) {
  case META_INT:
    fprintf(f, "%" PRId64 "\n", i);
    break;
  case META_REAL:
    fprintf(f, "%.17g\n", d);
    break;
  case META_STRING:
    fprintf(f, "\"%s\"\n", str);
    break;
  }

  const int status = dump_close(f, dump);
  if (status == SIF_OK)
    warn_dumped(filepath, dump);
  return status;
}

int sif_hdf5_set_attr_int(
  const char* filepath, const char* group, const char* key, int64_t value) {
  return meta_dump(filepath, group, key, META_INT, value, 0.0, NULL);
}

int sif_hdf5_set_attr_real(
  const char* filepath, const char* group, const char* key, double value) {
  return meta_dump(filepath, group, key, META_REAL, 0, value, NULL);
}

int sif_hdf5_set_attr_string(
  const char* filepath, const char* group, const char* key, const char* value) {
  return meta_dump(filepath, group, key, META_STRING, 0, 0.0, value);
}

int sif_hdf5_get_attr_int(
  const char* filepath, const char* group, const char* key, int64_t* out) {
  (void)group;
  (void)key;
  (void)out;
  refuse(filepath);
  return SIF_ERR_UNSUPPORTED;
}

int sif_hdf5_get_attr_real(
  const char* filepath, const char* group, const char* key, double* out) {
  (void)group;
  (void)key;
  (void)out;
  refuse(filepath);
  return SIF_ERR_UNSUPPORTED;
}

int sif_hdf5_get_attr_string(const char* filepath, const char* group,
  const char* key, char* buf, size_t len) {
  (void)group;
  (void)key;
  if (buf && len)
    buf[0] = '\0';
  refuse(filepath);
  return SIF_ERR_UNSUPPORTED;
}

int sif_hdf5_attr_kind(const char* filepath, const char* group, const char* key,
  sif_hdf5_attr_kind_t* out) {
  (void)group;
  (void)key;
  (void)out;
  refuse(filepath);
  return SIF_ERR_UNSUPPORTED;
}

int sif_hdf5_attr_count(
  const char* filepath, const char* group, uint32_t* out) {
  (void)group;
  if (out)
    *out = 0;
  refuse(filepath);
  return SIF_ERR_UNSUPPORTED;
}

int sif_hdf5_attr_name(const char* filepath, const char* group, uint32_t index,
  char* buf, size_t len) {
  (void)group;
  (void)index;
  if (buf && len)
    buf[0] = '\0';
  refuse(filepath);
  return SIF_ERR_UNSUPPORTED;
}

/* --- GADGET snapshots --- */

int sif__gadget_h5_open(const char* path, sif_gadget_h5_file_t** out_file,
  sif_gadget_header_t* out_header) {
  (void)out_header;
  *out_file = NULL;
  SIF_LOG_ERROR(TAG,
    "%s is an HDF5 snapshot, and this build of sif has no HDF5 support. "
    "Rebuild with -DSIF_HDF5_SUPPORT=ON to read it",
    path);
  return SIF_ERR_UNSUPPORTED;
}

int sif__gadget_h5_read(sif_gadget_h5_file_t* file, uint32_t ptype,
  sif_gadget_block_t block, uint64_t start, uint64_t count, double* out) {
  (void)file;
  (void)ptype;
  (void)block;
  (void)start;
  (void)count;
  (void)out;
  return SIF_ERR_UNSUPPORTED;
}

void sif__gadget_h5_close(sif_gadget_h5_file_t* file) { (void)file; }
