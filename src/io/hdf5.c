/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The HDF5 half of the I/O layer, compiled when SIF_HDF5_SUPPORT found a
 * usable HDF5. hdf5_off.c stands in for it otherwise, implementing the same
 * entry points as refusals (readers) and plain-text dumps (writers), so the
 * rest of the library never asks which one it got. The file layout is
 * documented once, in sif/io/hdf5_io.h.
 *
 * Every entry point opens the file, does its one thing and closes it: there
 * is no handle for a caller to hold, and nothing is left open between calls.
 * HDF5 is not thread-safe as commonly built, and none of this is meant to be
 * called from more than one thread at a time.
 */

#include "sif/io/hdf5_io.h"

#include "io/hdf5_internal.h"
#include "measure/profiles_internal.h"
#include "sif/utils/logger.h"
#include "structures/results_internal.h"

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define TAG "hdf5"

/* What the root of every sif file says about itself. The version is of the
 * layout, not of the library: it moves when a reader of this version could no
 * longer make sense of what a newer writer puts in the file. */
#define SIF_H5_FORMAT         "sif"
#define SIF_H5_FORMAT_VERSION 1u

#define G_CATALOG  "catalog"
#define G_DENSITY  "density_profiles"
#define G_VELOCITY "velocity_profiles"
#define G_VSF      "size_function"

int sif__hdf5_describe(char* buf, size_t len) {
  unsigned major = 0, minor = 0, release = 0;

  /* The library at run time, not H5_VERS_*: a shared HDF5 can be upgraded
   * under a build, and the one in use is the one worth reporting. */
  if (H5get_libversion(&major, &minor, &release) < 0)
    snprintf(buf, len, "HDF5 (version unknown)");
  else
    snprintf(buf, len, "HDF5 %u.%u.%u", major, minor, release);

  return 1;
}

/* ------------------------------------------------------------------------ */
/* plumbing                                                                  */
/* ------------------------------------------------------------------------ */

/*
 * HDF5 prints its whole error stack to stderr on every failed call, including
 * the ones that are merely questions ("is this attribute here?"). Each entry
 * point turns that off for its own duration and says what went wrong through
 * the sif logger instead -- and puts back whatever handler was there, since
 * the process may be using HDF5 for other things.
 */
typedef struct {
  H5E_auto2_t func;
  void* data;
} h5_quiet_t;

static void quiet_begin(h5_quiet_t* q) {
  H5Eget_auto2(H5E_DEFAULT, &q->func, &q->data);
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
}

static void quiet_end(const h5_quiet_t* q) {
  H5Eset_auto2(H5E_DEFAULT, q->func, q->data);
}

/* Closes any HDF5 identifier, or nothing for an invalid one, so cleanup paths
 * can close everything they might have opened without tracking what. */
static void close_id(hid_t id) {
  if (id < 0)
    return;

  switch (H5Iget_type(id)) {
  case H5I_FILE:
    H5Fclose(id);
    break;
  case H5I_GROUP:
    H5Gclose(id);
    break;
  case H5I_DATASET:
    H5Dclose(id);
    break;
  case H5I_DATASPACE:
    H5Sclose(id);
    break;
  case H5I_DATATYPE:
    H5Tclose(id);
    break;
  case H5I_ATTR:
    H5Aclose(id);
    break;
  default:
    break;
  }
}

/* sif_real in memory, and on disk. Little-endian on disk whatever the
 * machine, so a file means the same thing wherever it is read. */
static hid_t real_mem_type(void) {
  return sizeof(sif_real) == 8 ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
}

static hid_t real_file_type(void) {
  return sizeof(sif_real) == 8 ? H5T_IEEE_F64LE : H5T_IEEE_F32LE;
}

static int link_exists(hid_t loc, const char* name) {
  return H5Lexists(loc, name, H5P_DEFAULT) > 0;
}

/* --- attributes --- */

static int attr_write(hid_t obj, const char* name, hid_t file_type,
  hid_t mem_type, const void* value) {

  if (H5Aexists(obj, name) > 0 && H5Adelete(obj, name) < 0)
    return SIF_ERR_IO;

  hid_t space = H5Screate(H5S_SCALAR);
  hid_t attr =
    H5Acreate2(obj, name, file_type, space, H5P_DEFAULT, H5P_DEFAULT);
  const int ok = attr >= 0 && H5Awrite(attr, mem_type, value) >= 0;

  close_id(attr);
  close_id(space);
  return ok ? SIF_OK : SIF_ERR_IO;
}

static int attr_write_u64(hid_t obj, const char* name, uint64_t v) {
  return attr_write(obj, name, H5T_STD_U64LE, H5T_NATIVE_UINT64, &v);
}

static int attr_write_u32(hid_t obj, const char* name, uint32_t v) {
  return attr_write(obj, name, H5T_STD_U32LE, H5T_NATIVE_UINT32, &v);
}

/* Scalar metadata is kept in double whatever sif_real is: a float survives
 * the round trip through it exactly, and a reader never has to ask. */
static int attr_write_f64(hid_t obj, const char* name, double v) {
  return attr_write(obj, name, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &v);
}

/* A variable-length UTF-8 string, which h5py hands back as a str rather than
 * as bytes. */
static int attr_write_str(hid_t obj, const char* name, const char* v) {
  hid_t type = H5Tcopy(H5T_C_S1);
  if (type < 0)
    return SIF_ERR_IO;

  int status = SIF_ERR_IO;
  if (H5Tset_size(type, H5T_VARIABLE) >= 0 &&
      H5Tset_cset(type, H5T_CSET_UTF8) >= 0)
    status = attr_write(obj, name, type, type, &v);

  close_id(type);
  return status;
}

/* A numeric attribute, converted by HDF5 to `mem_type` on the way in. */
static int attr_read(hid_t obj, const char* name, hid_t mem_type, void* out) {
  if (H5Aexists(obj, name) <= 0)
    return SIF_ERR_IO;

  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  const int ok = attr >= 0 && H5Aread(attr, mem_type, out) >= 0;
  close_id(attr);
  return ok ? SIF_OK : SIF_ERR_IO;
}

/* A string attribute, fixed- or variable-length -- a file written by h5py may
 * hold either -- into `buf`. SIF_ERR_RANGE if it had to be truncated to fit,
 * with as much of it in `buf` as did. */
static int attr_read_str(hid_t obj, const char* name, char* buf, size_t len) {
  if (len == 0 || H5Aexists(obj, name) <= 0)
    return SIF_ERR_IO;

  int status = SIF_ERR_IO;
  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  hid_t ftype = attr >= 0 ? H5Aget_type(attr) : H5I_INVALID_HID;
  hid_t mtype = H5I_INVALID_HID;

  if (ftype < 0 || H5Tget_class(ftype) != H5T_STRING)
    goto done;

  mtype = H5Tcopy(H5T_C_S1);
  if (mtype < 0)
    goto done;

  if (H5Tis_variable_str(ftype) > 0) {
    char* s = NULL;
    if (H5Tset_size(mtype, H5T_VARIABLE) < 0 ||
        H5Tset_cset(mtype, H5Tget_cset(ftype)) < 0 ||
        H5Aread(attr, mtype, &s) < 0 || !s)
      goto done;
    const size_t need = strlen(s);
    snprintf(buf, len, "%s", s);
    H5free_memory(s);
    if (need >= len) {
      status = SIF_ERR_RANGE;
      goto done;
    }
  } else {
    const size_t n = H5Tget_size(ftype);
    char* tmp = calloc(n + 1, 1);
    if (!tmp || H5Tset_size(mtype, n + 1) < 0 ||
        H5Tset_strpad(mtype, H5T_STR_NULLTERM) < 0 ||
        H5Aread(attr, mtype, tmp) < 0) {
      free(tmp);
      goto done;
    }
    const size_t need = strlen(tmp);
    snprintf(buf, len, "%s", tmp);
    free(tmp);
    if (need >= len) {
      status = SIF_ERR_RANGE;
      goto done;
    }
  }
  status = SIF_OK;

done:
  close_id(mtype);
  close_id(ftype);
  close_id(attr);
  return status;
}

/* --- datasets --- */

/* A whole array, written in one go. A dataset with a zero dimension is still
 * created -- an empty catalogue is a catalogue -- but nothing is written to
 * it. */
static int dataset_write(hid_t loc, const char* name, int rank,
  const hsize_t* dims, hid_t file_type, hid_t mem_type, const void* data) {

  hsize_t total = 1;
  for (int i = 0; i < rank; i++)
    total *= dims[i];

  hid_t space = H5Screate_simple(rank, dims, NULL);
  hid_t dset = space >= 0 ? H5Dcreate2(loc, name, file_type, space, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT)
                          : H5I_INVALID_HID;

  int ok = dset >= 0;
  if (ok && total > 0)
    ok = H5Dwrite(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0;

  close_id(dset);
  close_id(space);
  return ok ? SIF_OK : SIF_ERR_IO;
}

/*
 * A whole array, read into `out` -- but only if its shape is exactly the one
 * expected. The metadata that says how large to make `out` came from the
 * attributes, and a dataset that disagrees with them is a malformed file, not
 * something to read part of.
 */
static int dataset_read(hid_t loc, const char* name, int rank,
  const hsize_t* expect, hid_t mem_type, void* out) {

  if (!link_exists(loc, name)) {
    SIF_LOG_ERROR(TAG, "dataset '%s' is missing", name);
    return SIF_ERR_IO;
  }

  int status = SIF_ERR_IO;
  hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
  hid_t space = dset >= 0 ? H5Dget_space(dset) : H5I_INVALID_HID;

  if (space < 0 || H5Sget_simple_extent_ndims(space) != rank)
    goto bad_shape;

  hsize_t dims[2] = {0, 0};
  H5Sget_simple_extent_dims(space, dims, NULL);

  hsize_t total = 1;
  for (int i = 0; i < rank; i++) {
    if (dims[i] != expect[i])
      goto bad_shape;
    total *= dims[i];
  }

  if (total == 0 ||
      H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) >= 0)
    status = SIF_OK;
  goto done;

bad_shape:
  SIF_LOG_ERROR(
    TAG, "dataset '%s' does not have the shape its group declares", name);

done:
  close_id(space);
  close_id(dset);
  return status;
}

/*
 * The (N, 3) centres, one column at a time from the catalogue's own x, y and
 * z arrays, so neither direction needs an interleaved copy of the catalogue.
 */
static int centers_io(
  hid_t dset, sif_real* const cols[3], uint64_t n, int writing) {

  if (n == 0)
    return SIF_OK;

  const hsize_t count[2] = {n, 1};
  const hsize_t mdim[1] = {n};
  int ok = 1;

  for (hsize_t k = 0; k < 3 && ok; k++) {
    const hsize_t start[2] = {0, k};
    hid_t fspace = H5Dget_space(dset);
    hid_t mspace = H5Screate_simple(1, mdim, NULL);

    ok = fspace >= 0 && mspace >= 0 &&
         H5Sselect_hyperslab(
           fspace, H5S_SELECT_SET, start, NULL, count, NULL) >= 0;
    if (ok) {
      ok = writing ? H5Dwrite(dset, real_mem_type(), mspace, fspace,
                       H5P_DEFAULT, cols[k]) >= 0
                   : H5Dread(dset, real_mem_type(), mspace, fspace, H5P_DEFAULT,
                       cols[k]) >= 0;
    }

    close_id(mspace);
    close_id(fspace);
  }

  return ok ? SIF_OK : SIF_ERR_IO;
}

/* --- files --- */

/*
 * Whether an open file is one this library wrote, and one it can read: the
 * root has to name the format, and not a layout newer than this build knows.
 * A file from anywhere else is left alone rather than guessed at.
 */
static int check_root(hid_t file, const char* path) {
  char format[16] = {0};
  uint32_t version = 0;

  if (attr_read_str(file, "sif_format", format, sizeof(format)) != SIF_OK ||
      strcmp(format, SIF_H5_FORMAT) != 0) {
    SIF_LOG_ERROR(TAG, "%s is an HDF5 file, but not one sif wrote", path);
    return SIF_ERR_IO;
  }

  if (attr_read(file, "sif_format_version", H5T_NATIVE_UINT32, &version) !=
        SIF_OK ||
      version > SIF_H5_FORMAT_VERSION) {
    SIF_LOG_ERROR(TAG,
      "%s is sif format version %u, newer than this build reads (%u)", path,
      version, SIF_H5_FORMAT_VERSION);
    return SIF_ERR_IO;
  }

  return SIF_OK;
}

static int stamp_root(hid_t file) {
  if (attr_write_str(file, "sif_format", SIF_H5_FORMAT) != SIF_OK ||
      attr_write_u32(file, "sif_format_version", SIF_H5_FORMAT_VERSION) !=
        SIF_OK ||
      attr_write_str(file, "sif_version", SIF_VERSION_STRING) != SIF_OK)
    return SIF_ERR_IO;
  return SIF_OK;
}

/*
 * The file a writer adds its group to: created and stamped if it does not
 * exist, opened for writing if it is a sif file, and refused otherwise. An
 * existing file that is not ours -- text, someone else's HDF5 -- is never
 * truncated to make room: that is somebody's data.
 */
static hid_t open_for_write(const char* path) {
  struct stat st;

  if (stat(path, &st) != 0) {
    hid_t file = H5Fcreate(path, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
      SIF_LOG_ERROR(TAG, "could not create %s", path);
      return H5I_INVALID_HID;
    }
    if (stamp_root(file) != SIF_OK) {
      SIF_LOG_ERROR(TAG, "could not write the header of %s", path);
      close_id(file);
      return H5I_INVALID_HID;
    }
    return file;
  }

  hid_t file = H5Fopen(path, H5F_ACC_RDWR, H5P_DEFAULT);
  if (file < 0) {
    SIF_LOG_ERROR(TAG,
      "%s exists but is not an HDF5 file that can be opened for writing; "
      "refusing to overwrite it",
      path);
    return H5I_INVALID_HID;
  }

  if (check_root(file, path) != SIF_OK) {
    close_id(file);
    return H5I_INVALID_HID;
  }

  return file;
}

static hid_t open_for_read(const char* path) {
  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    SIF_LOG_ERROR(TAG, "could not open %s as an HDF5 file", path);
    return H5I_INVALID_HID;
  }

  if (check_root(file, path) != SIF_OK) {
    close_id(file);
    return H5I_INVALID_HID;
  }

  return file;
}

/* A fresh, empty group in place of whatever was there under that name. */
static hid_t group_replace(hid_t file, const char* name) {
  if (link_exists(file, name) && H5Ldelete(file, name, H5P_DEFAULT) < 0)
    return H5I_INVALID_HID;
  return H5Gcreate2(file, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

static hid_t group_open(hid_t file, const char* name, const char* path) {
  if (!link_exists(file, name)) {
    SIF_LOG_ERROR(TAG, "%s holds no /%s", path, name);
    return H5I_INVALID_HID;
  }
  return H5Gopen2(file, name, H5P_DEFAULT);
}

/*
 * The void count a group in the file declares, or 0 if it has none. Only for
 * the row-count warnings: the products and the catalogue are the caller's to
 * keep in step, and a mismatch is worth a line in the log, not a refusal.
 */
static uint64_t group_n_voids(hid_t file, const char* name) {
  uint64_t n = 0;
  if (!link_exists(file, name))
    return 0;

  hid_t g = H5Gopen2(file, name, H5P_DEFAULT);
  if (g >= 0 && attr_read(g, "n_voids", H5T_NATIVE_UINT64, &n) != SIF_OK)
    n = 0;
  close_id(g);
  return n;
}

static void warn_rows(hid_t file, const char* path, const char* group,
  uint64_t n_voids, const char* written) {

  const uint64_t other = group_n_voids(file, group);
  if (other != 0 && other != n_voids) {
    SIF_LOG_WARNING(TAG,
      "%s: /%s was written with %" PRIu64 " voids, but /%s has %" PRIu64
      "; they cannot both describe the same voids",
      path, written, n_voids, group, other);
  }
}

/* ------------------------------------------------------------------------ */
/* catalogue                                                                 */
/* ------------------------------------------------------------------------ */

int sif_catalog_write_hdf5(const char* filepath, const sif_catalog_t* catalog) {
  if (!filepath || !catalog) {
    SIF_LOG_ERROR(TAG, "invalid arguments for sif_catalog_write_hdf5");
    return SIF_ERR_INVALID;
  }

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  int status = SIF_ERR_IO;
  hid_t file = open_for_write(filepath);
  hid_t group = H5I_INVALID_HID, dset = H5I_INVALID_HID,
        space = H5I_INVALID_HID;
  if (file < 0)
    goto done;

  group = group_replace(file, G_CATALOG);
  if (group < 0)
    goto fail;

  const uint64_t n = catalog->n_voids;
  const hsize_t dims_c[2] = {n, 3};
  const hsize_t dims_1[1] = {n};

  space = H5Screate_simple(2, dims_c, NULL);
  dset = space >= 0 ? H5Dcreate2(group, "centers", real_file_type(), space,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
                    : H5I_INVALID_HID;
  sif_real* const cols[3] = {catalog->cx, catalog->cy, catalog->cz};

  if (dset < 0 || centers_io(dset, cols, n, 1) != SIF_OK ||
      dataset_write(group, "radii", 1, dims_1, real_file_type(),
        real_mem_type(), catalog->radii) != SIF_OK ||
      attr_write_u64(group, "n_voids", n) != SIF_OK)
    goto fail;

  if (catalog->footprint &&
      (dataset_write(group, "footprint", 1, dims_1, real_file_type(),
         real_mem_type(), catalog->footprint) != SIF_OK ||
        dataset_write(group, "footprint_shell", 1, dims_1, real_file_type(),
          real_mem_type(), catalog->footprint_shell) != SIF_OK))
    goto fail;

  warn_rows(file, filepath, G_DENSITY, n, G_CATALOG);
  warn_rows(file, filepath, G_VELOCITY, n, G_CATALOG);

  status = SIF_OK;
  SIF_LOG_INFO(TAG, "saved %" PRIu64 " voids to %s", n, filepath);
  goto done;

fail:
  SIF_LOG_ERROR(TAG, "failed to write /%s to %s", G_CATALOG, filepath);

done:
  close_id(dset);
  close_id(space);
  close_id(group);
  close_id(file);
  quiet_end(&quiet);
  return status;
}

sif_catalog_t* sif_catalog_read_hdf5(const char* filepath) {
  if (!filepath) {
    SIF_LOG_ERROR(TAG, "invalid filepath for sif_catalog_read_hdf5");
    return NULL;
  }

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  sif_catalog_t* cat = NULL;
  hid_t file = open_for_read(filepath);
  hid_t group =
    file >= 0 ? group_open(file, G_CATALOG, filepath) : H5I_INVALID_HID;
  hid_t dset = H5I_INVALID_HID;
  uint64_t n = 0;

  if (group < 0 || attr_read(group, "n_voids", H5T_NATIVE_UINT64, &n) != SIF_OK)
    goto fail;

  cat = sif_catalog_alloc(n);
  if (!cat)
    goto fail;

  const hsize_t dims_c[2] = {n, 3};
  const hsize_t dims_1[1] = {n};
  sif_real* const cols[3] = {cat->cx, cat->cy, cat->cz};

  /* The centres are read by column, so their shape is checked here rather
   * than by dataset_read(). */
  if (!link_exists(group, "centers"))
    goto fail;
  dset = H5Dopen2(group, "centers", H5P_DEFAULT);
  {
    hid_t space = dset >= 0 ? H5Dget_space(dset) : H5I_INVALID_HID;
    hsize_t dims[2] = {0, 0};
    const int good = space >= 0 && H5Sget_simple_extent_ndims(space) == 2 &&
                     H5Sget_simple_extent_dims(space, dims, NULL) == 2 &&
                     dims[0] == dims_c[0] && dims[1] == dims_c[1];
    close_id(space);
    if (!good) {
      SIF_LOG_ERROR(TAG, "/%s/centers is not (n_voids, 3)", G_CATALOG);
      goto fail;
    }
  }

  if (centers_io(dset, cols, n, 0) != SIF_OK ||
      dataset_read(group, "radii", 1, dims_1, real_mem_type(), cat->radii) !=
        SIF_OK)
    goto fail;

  /* Filled in directly rather than through sif_catalog_append(). */
  cat->n_voids = n;

  if (link_exists(group, "footprint")) {
    if (sif_catalog_reserve_footprint(cat) != SIF_OK ||
        dataset_read(group, "footprint", 1, dims_1, real_mem_type(),
          cat->footprint) != SIF_OK ||
        dataset_read(group, "footprint_shell", 1, dims_1, real_mem_type(),
          cat->footprint_shell) != SIF_OK)
      goto fail;
  }

  close_id(dset);
  close_id(group);
  close_id(file);
  quiet_end(&quiet);
  SIF_LOG_INFO(TAG, "loaded %" PRIu64 " voids from %s", n, filepath);
  return cat;

fail:
  SIF_LOG_ERROR(TAG, "failed to read /%s from %s", G_CATALOG, filepath);
  sif_catalog_free(cat);
  close_id(dset);
  close_id(group);
  close_id(file);
  quiet_end(&quiet);
  return NULL;
}

/* ------------------------------------------------------------------------ */
/* profiles                                                                  */
/* ------------------------------------------------------------------------ */

/* One profile set's group: its shape as attributes, then the edges and the
 * (n_voids, n_bins) rows. */
static int profiles_group_write(hid_t file, const char* name, uint64_t n_voids,
  uint32_t n_bins, sif_real ext, const sif_real* r_edges, const char* rows_name,
  const sif_real* rows, int differential) {

  hid_t g = group_replace(file, name);
  if (g < 0)
    return SIF_ERR_IO;

  const hsize_t dims_e[1] = {(hsize_t)n_bins + 1};
  const hsize_t dims_r[2] = {n_voids, n_bins};

  int ok = attr_write_u64(g, "n_voids", n_voids) == SIF_OK &&
           attr_write_u32(g, "n_bins", n_bins) == SIF_OK &&
           attr_write_f64(g, "ext", (double)ext) == SIF_OK &&
           dataset_write(g, "r_edges", 1, dims_e, real_file_type(),
             real_mem_type(), r_edges) == SIF_OK &&
           dataset_write(g, rows_name, 2, dims_r, real_file_type(),
             real_mem_type(), rows) == SIF_OK;

  /* Stored as a small integer rather than an HDF5 enum: h5py reads it back
   * as a plain number, and 0 and 1 are all it needs to say. */
  if (ok && differential >= 0) {
    const uint8_t d = (uint8_t)differential;
    ok = attr_write(g, "differential", H5T_STD_U8LE, H5T_NATIVE_UINT8, &d) ==
         SIF_OK;
  }

  close_id(g);
  return ok ? SIF_OK : SIF_ERR_IO;
}

int sif_profiles_write_hdf5(const char* filepath,
  const sif_density_profiles_t* dens, const sif_velocity_profiles_t* vel) {

  if (!filepath || (!dens && !vel)) {
    SIF_LOG_ERROR(
      TAG, "sif_profiles_write_hdf5 needs a path and at least one profile set");
    return SIF_ERR_INVALID;
  }

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  int status = SIF_ERR_IO;
  hid_t file = open_for_write(filepath);
  if (file < 0)
    goto done;

  if (dens) {
    if (profiles_group_write(file, G_DENSITY, dens->n_voids, dens->n_bins,
          dens->ext, dens->r_edges, "profiles", dens->profiles,
          dens->differential ? 1 : 0) != SIF_OK) {
      SIF_LOG_ERROR(TAG, "failed to write /%s to %s", G_DENSITY, filepath);
      goto done;
    }
    warn_rows(file, filepath, G_CATALOG, dens->n_voids, G_DENSITY);
  }

  if (vel) {
    if (profiles_group_write(file, G_VELOCITY, vel->n_voids, vel->n_bins,
          vel->ext, vel->r_edges, "v_rad", vel->v_rad, -1) != SIF_OK) {
      SIF_LOG_ERROR(TAG, "failed to write /%s to %s", G_VELOCITY, filepath);
      goto done;
    }
    warn_rows(file, filepath, G_CATALOG, vel->n_voids, G_VELOCITY);
  }

  status = SIF_OK;
  SIF_LOG_INFO(TAG, "saved profiles to %s", filepath);

done:
  close_id(file);
  quiet_end(&quiet);
  return status;
}

int sif_profiles_read_header_hdf5(
  const char* filepath, int* out_has_density, int* out_has_velocity) {

  if (!filepath) {
    SIF_LOG_ERROR(TAG, "invalid filepath for sif_profiles_read_header_hdf5");
    return SIF_ERR_INVALID;
  }

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  hid_t file = open_for_read(filepath);
  if (file >= 0) {
    if (out_has_density)
      *out_has_density = link_exists(file, G_DENSITY);
    if (out_has_velocity)
      *out_has_velocity = link_exists(file, G_VELOCITY);
  }

  close_id(file);
  quiet_end(&quiet);
  return file >= 0 ? SIF_OK : SIF_ERR_IO;
}

/* One profile set's shape, from its group's attributes. */
static int profiles_shape(hid_t g, uint64_t* n_voids, uint32_t* n_bins,
  sif_real* ext, uint8_t* differential) {

  double e = 0.0;
  if (attr_read(g, "n_voids", H5T_NATIVE_UINT64, n_voids) != SIF_OK ||
      attr_read(g, "n_bins", H5T_NATIVE_UINT32, n_bins) != SIF_OK ||
      attr_read(g, "ext", H5T_NATIVE_DOUBLE, &e) != SIF_OK)
    return SIF_ERR_IO;
  *ext = (sif_real)e;

  if (differential &&
      attr_read(g, "differential", H5T_NATIVE_UINT8, differential) != SIF_OK)
    return SIF_ERR_IO;

  return SIF_OK;
}

int sif_profiles_read_hdf5(const char* filepath,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {

  if (out_dens)
    *out_dens = NULL;
  if (out_vel)
    *out_vel = NULL;

  if (!filepath || (!out_dens && !out_vel)) {
    SIF_LOG_ERROR(
      TAG, "sif_profiles_read_hdf5 needs a path and at least one output");
    return SIF_ERR_INVALID;
  }

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  int status = SIF_ERR_IO;
  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;
  hid_t g = H5I_INVALID_HID;

  hid_t file = open_for_read(filepath);
  if (file < 0)
    goto done;

  /* A set the caller asked for and the file does not hold is a request that
   * cannot be met -- different from a file that is broken. */
  if ((out_dens && !link_exists(file, G_DENSITY)) ||
      (out_vel && !link_exists(file, G_VELOCITY))) {
    SIF_LOG_ERROR(TAG, "%s holds no /%s", filepath,
      (out_dens && !link_exists(file, G_DENSITY)) ? G_DENSITY : G_VELOCITY);
    status = SIF_ERR_INVALID;
    goto done;
  }

  if (out_dens) {
    uint64_t n;
    uint32_t b;
    sif_real ext;
    uint8_t differential;

    g = H5Gopen2(file, G_DENSITY, H5P_DEFAULT);
    if (g < 0 || profiles_shape(g, &n, &b, &ext, &differential) != SIF_OK)
      goto done;

    dens = sif__density_profiles_alloc(n, b, ext, differential != 0);
    if (!dens) {
      status = SIF_ERR_ALLOC;
      goto done;
    }

    const hsize_t dims_e[1] = {(hsize_t)b + 1};
    const hsize_t dims_r[2] = {n, b};
    if (dataset_read(g, "r_edges", 1, dims_e, real_mem_type(), dens->r_edges) !=
          SIF_OK ||
        dataset_read(
          g, "profiles", 2, dims_r, real_mem_type(), dens->profiles) != SIF_OK)
      goto done;

    close_id(g);
    g = H5I_INVALID_HID;
  }

  if (out_vel) {
    uint64_t n;
    uint32_t b;
    sif_real ext;

    g = H5Gopen2(file, G_VELOCITY, H5P_DEFAULT);
    if (g < 0 || profiles_shape(g, &n, &b, &ext, NULL) != SIF_OK)
      goto done;

    vel = sif__velocity_profiles_alloc(n, b, ext);
    if (!vel) {
      status = SIF_ERR_ALLOC;
      goto done;
    }

    const hsize_t dims_e[1] = {(hsize_t)b + 1};
    const hsize_t dims_r[2] = {n, b};
    if (dataset_read(g, "r_edges", 1, dims_e, real_mem_type(), vel->r_edges) !=
          SIF_OK ||
        dataset_read(g, "v_rad", 2, dims_r, real_mem_type(), vel->v_rad) !=
          SIF_OK)
      goto done;
  }

  status = SIF_OK;

done:
  close_id(g);
  close_id(file);
  quiet_end(&quiet);

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

int sif_size_function_write_hdf5(
  const char* filepath, const sif_size_function_t* vsf) {

  if (!filepath || !vsf) {
    SIF_LOG_ERROR(TAG, "invalid arguments for sif_size_function_write_hdf5");
    return SIF_ERR_INVALID;
  }

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  int status = SIF_ERR_IO;
  hid_t file = open_for_write(filepath);
  hid_t g = file >= 0 ? group_replace(file, G_VSF) : H5I_INVALID_HID;

  if (g >= 0) {
    const uint32_t b = vsf->n_bins;
    const hsize_t dims_e[1] = {(hsize_t)b + 1};
    const hsize_t dims_b[1] = {b};

    /* The binning is also inside `options`, but spelled out: it is the one
     * thing a reader needs to know to interpret the values -- per unit ln R
     * or per unit R -- and nothing else in the file says it. */
    const char* binning =
      ((vsf->options & SIF__VSF_BIN_MASK) == SIF_VSF_BIN_LINEAR) ? "linear"
                                                                 : "ln";

    const int ok =
      attr_write_u32(g, "n_bins", b) == SIF_OK &&
      attr_write_f64(g, "r_min", (double)vsf->r_min) == SIF_OK &&
      attr_write_f64(g, "r_max", (double)vsf->r_max) == SIF_OK &&
      attr_write_u32(g, "options", (uint32_t)vsf->options) == SIF_OK &&
      attr_write_str(g, "binning", binning) == SIF_OK &&
      dataset_write(g, "r_edges", 1, dims_e, real_file_type(), real_mem_type(),
        vsf->r_edges) == SIF_OK &&
      dataset_write(g, "r_centers", 1, dims_b, real_file_type(),
        real_mem_type(), vsf->r_centers) == SIF_OK &&
      dataset_write(g, "counts", 1, dims_b, H5T_STD_U64LE, H5T_NATIVE_UINT64,
        vsf->counts) == SIF_OK &&
      dataset_write(g, "vsf", 1, dims_b, real_file_type(), real_mem_type(),
        vsf->vsf) == SIF_OK &&
      dataset_write(g, "err", 1, dims_b, real_file_type(), real_mem_type(),
        vsf->err) == SIF_OK;

    if (ok)
      status = SIF_OK;
  }

  if (status == SIF_OK)
    SIF_LOG_INFO(TAG, "saved the size function to %s", filepath);
  else if (file >= 0)
    SIF_LOG_ERROR(TAG, "failed to write /%s to %s", G_VSF, filepath);

  close_id(g);
  close_id(file);
  quiet_end(&quiet);
  return status;
}

sif_size_function_t* sif_size_function_read_hdf5(const char* filepath) {
  if (!filepath) {
    SIF_LOG_ERROR(TAG, "invalid filepath for sif_size_function_read_hdf5");
    return NULL;
  }

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  sif_size_function_t* vsf = NULL;
  hid_t file = open_for_read(filepath);
  hid_t g = file >= 0 ? group_open(file, G_VSF, filepath) : H5I_INVALID_HID;

  uint32_t b = 0, options = 0;
  double r_min = 0.0, r_max = 0.0;

  if (g < 0 || attr_read(g, "n_bins", H5T_NATIVE_UINT32, &b) != SIF_OK ||
      attr_read(g, "r_min", H5T_NATIVE_DOUBLE, &r_min) != SIF_OK ||
      attr_read(g, "r_max", H5T_NATIVE_DOUBLE, &r_max) != SIF_OK ||
      attr_read(g, "options", H5T_NATIVE_UINT32, &options) != SIF_OK)
    goto fail;

  vsf = sif__size_function_alloc(b);
  if (!vsf)
    goto fail;

  vsf->options = (sif_option)options;
  vsf->r_min = (sif_real)r_min;
  vsf->r_max = (sif_real)r_max;

  const hsize_t dims_e[1] = {(hsize_t)b + 1};
  const hsize_t dims_b[1] = {b};

  if (dataset_read(g, "r_edges", 1, dims_e, real_mem_type(), vsf->r_edges) !=
        SIF_OK ||
      dataset_read(
        g, "r_centers", 1, dims_b, real_mem_type(), vsf->r_centers) != SIF_OK ||
      dataset_read(g, "counts", 1, dims_b, H5T_NATIVE_UINT64, vsf->counts) !=
        SIF_OK ||
      dataset_read(g, "vsf", 1, dims_b, real_mem_type(), vsf->vsf) != SIF_OK ||
      dataset_read(g, "err", 1, dims_b, real_mem_type(), vsf->err) != SIF_OK)
    goto fail;

  close_id(g);
  close_id(file);
  quiet_end(&quiet);
  return vsf;

fail:
  SIF_LOG_ERROR(TAG, "failed to read /%s from %s", G_VSF, filepath);
  sif_size_function_free(vsf);
  close_id(g);
  close_id(file);
  quiet_end(&quiet);
  return NULL;
}

/* ------------------------------------------------------------------------ */
/* free-form metadata                                                        */
/* ------------------------------------------------------------------------ */

/*
 * The object a metadata call acts on -- the file itself for the root, a
 * product's group otherwise -- and the file it lives in. For writing, the
 * root of a missing file is created, as the product writers would; a product
 * group has to be there already, since there is nothing yet to describe.
 */
static int attr_target_open(const char* path, const char* group, int writing,
  hid_t* file_out, hid_t* obj_out) {

  *file_out = *obj_out = H5I_INVALID_HID;

  struct stat st;
  if (writing && group && stat(path, &st) != 0) {
    SIF_LOG_ERROR(
      TAG, "%s does not exist; write /%s before describing it", path, group);
    return SIF_ERR_INVALID;
  }

  hid_t file = writing ? open_for_write(path) : open_for_read(path);
  if (file < 0)
    return SIF_ERR_IO;

  if (!group) {
    *file_out = *obj_out = file;
    return SIF_OK;
  }

  if (!link_exists(file, group)) {
    SIF_LOG_ERROR(TAG, "%s holds no /%s", path, group);
    close_id(file);
    return SIF_ERR_INVALID;
  }

  hid_t obj = H5Gopen2(file, group, H5P_DEFAULT);
  if (obj < 0) {
    SIF_LOG_ERROR(TAG, "/%s in %s is not a group", group, path);
    close_id(file);
    return SIF_ERR_INVALID;
  }

  *file_out = file;
  *obj_out = obj;
  return SIF_OK;
}

static void attr_target_close(hid_t file, hid_t obj) {
  if (obj != file)
    close_id(obj);
  close_id(file);
}

/* The one kind of value these setters write. */
typedef enum { META_INT, META_REAL, META_STRING } meta_kind_t;

static int meta_set(const char* path, const char* group, const char* key,
  meta_kind_t kind, int64_t i, double d, const char* str) {

  if (!path || !key || !*key || (kind == META_STRING && !str)) {
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

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  int status = attr_target_open(path, g, 1, &file, &obj);
  if (status == SIF_OK) {
    switch (kind) {
    case META_INT:
      status = attr_write(obj, key, H5T_STD_I64LE, H5T_NATIVE_INT64, &i);
      break;
    case META_REAL:
      status = attr_write_f64(obj, key, d);
      break;
    case META_STRING:
      status = attr_write_str(obj, key, str);
      break;
    }
    if (status != SIF_OK)
      SIF_LOG_ERROR(
        TAG, "failed to set '%s' on /%s in %s", key, g ? g : "", path);
  }

  attr_target_close(file, obj);
  quiet_end(&quiet);
  return status;
}

int sif_hdf5_set_attr_int(
  const char* filepath, const char* group, const char* key, int64_t value) {
  return meta_set(filepath, group, key, META_INT, value, 0.0, NULL);
}

int sif_hdf5_set_attr_real(
  const char* filepath, const char* group, const char* key, double value) {
  return meta_set(filepath, group, key, META_REAL, 0, value, NULL);
}

int sif_hdf5_set_attr_string(
  const char* filepath, const char* group, const char* key, const char* value) {
  return meta_set(filepath, group, key, META_STRING, 0, 0.0, value);
}

/* The kind of an open attribute. One value only: a scalar, or an array of
 * one, which is what numpy hands h5py for a single number. */
static sif_hdf5_attr_kind_t attr_kind_of(hid_t attr) {
  hid_t type = H5Aget_type(attr);
  hid_t space = H5Aget_space(attr);
  sif_hdf5_attr_kind_t kind = SIF_HDF5_ATTR_OTHER;

  if (type >= 0 && space >= 0 && H5Sget_simple_extent_npoints(space) == 1) {
    switch (H5Tget_class(type)) {
    case H5T_INTEGER:
      kind = SIF_HDF5_ATTR_INT;
      break;
    case H5T_FLOAT:
      kind = SIF_HDF5_ATTR_REAL;
      break;
    case H5T_STRING:
      kind = SIF_HDF5_ATTR_STRING;
      break;
    default:
      break;
    }
  }

  close_id(space);
  close_id(type);
  return kind;
}

/*
 * Opens a metadata entry for reading and says what kind it is. Every getter
 * goes through here, so a missing entry and a missing group are reported the
 * same way whichever is asked for.
 */
static int meta_open(const char* path, const char* group, const char* key,
  hid_t* file, hid_t* obj, sif_hdf5_attr_kind_t* kind) {

  *file = *obj = H5I_INVALID_HID;
  if (!path || !key) {
    SIF_LOG_ERROR(TAG, "invalid arguments for a metadata entry");
    return SIF_ERR_INVALID;
  }

  const char* g = sif__hdf5_group(group);
  int status = attr_target_open(path, g, 0, file, obj);
  if (status != SIF_OK)
    return status;

  if (H5Aexists(*obj, key) <= 0) {
    SIF_LOG_ERROR(TAG, "/%s in %s has no entry '%s'", g ? g : "", path, key);
    return SIF_ERR_INVALID;
  }

  hid_t attr = H5Aopen(*obj, key, H5P_DEFAULT);
  if (attr < 0)
    return SIF_ERR_IO;
  *kind = attr_kind_of(attr);
  close_id(attr);
  return SIF_OK;
}

static int kind_mismatch(const char* key, const char* want) {
  SIF_LOG_ERROR(TAG, "'%s' is not %s", key, want);
  return SIF_ERR_INVALID;
}

int sif_hdf5_get_attr_int(
  const char* filepath, const char* group, const char* key, int64_t* out) {

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  sif_hdf5_attr_kind_t kind;
  int status =
    out ? meta_open(filepath, group, key, &file, &obj, &kind) : SIF_ERR_INVALID;
  if (status == SIF_OK) {
    status = kind == SIF_HDF5_ATTR_INT
               ? attr_read(obj, key, H5T_NATIVE_INT64, out)
               : kind_mismatch(key, "an integer");
  }

  attr_target_close(file, obj);
  quiet_end(&quiet);
  return status;
}

int sif_hdf5_get_attr_real(
  const char* filepath, const char* group, const char* key, double* out) {

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  sif_hdf5_attr_kind_t kind;
  int status =
    out ? meta_open(filepath, group, key, &file, &obj, &kind) : SIF_ERR_INVALID;
  if (status == SIF_OK) {
    /* An integer is a number too; HDF5 converts it on the way in. */
    status = (kind == SIF_HDF5_ATTR_REAL || kind == SIF_HDF5_ATTR_INT)
               ? attr_read(obj, key, H5T_NATIVE_DOUBLE, out)
               : kind_mismatch(key, "a number");
  }

  attr_target_close(file, obj);
  quiet_end(&quiet);
  return status;
}

int sif_hdf5_get_attr_string(const char* filepath, const char* group,
  const char* key, char* buf, size_t len) {

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  sif_hdf5_attr_kind_t kind;
  int status = (buf && len)
                 ? meta_open(filepath, group, key, &file, &obj, &kind)
                 : SIF_ERR_INVALID;
  if (status == SIF_OK) {
    status = kind == SIF_HDF5_ATTR_STRING ? attr_read_str(obj, key, buf, len)
                                          : kind_mismatch(key, "a string");
  }

  attr_target_close(file, obj);
  quiet_end(&quiet);
  return status;
}

int sif_hdf5_attr_kind(const char* filepath, const char* group, const char* key,
  sif_hdf5_attr_kind_t* out) {

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  sif_hdf5_attr_kind_t kind = SIF_HDF5_ATTR_OTHER;
  const int status =
    out ? meta_open(filepath, group, key, &file, &obj, &kind) : SIF_ERR_INVALID;
  if (status == SIF_OK)
    *out = kind;

  attr_target_close(file, obj);
  quiet_end(&quiet);
  return status;
}

static herr_t count_attr(
  hid_t loc, const char* name, const H5A_info_t* info, void* data) {
  (void)loc;
  (void)name;
  (void)info;
  (*(uint32_t*)data)++;
  return 0;
}

int sif_hdf5_attr_count(
  const char* filepath, const char* group, uint32_t* out) {

  if (!filepath || !out) {
    SIF_LOG_ERROR(TAG, "invalid arguments for sif_hdf5_attr_count");
    return SIF_ERR_INVALID;
  }

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  int status =
    attr_target_open(filepath, sif__hdf5_group(group), 0, &file, &obj);
  if (status == SIF_OK) {
    /* Counted by iterating rather than asked of H5Oget_info, whose signature
     * has changed twice across the HDF5 versions sif builds against. */
    uint32_t n = 0;
    hsize_t idx = 0;
    if (H5Aiterate2(obj, H5_INDEX_NAME, H5_ITER_INC, &idx, count_attr, &n) < 0)
      status = SIF_ERR_IO;
    else
      *out = n;
  }

  attr_target_close(file, obj);
  quiet_end(&quiet);
  return status;
}

int sif_hdf5_attr_name(const char* filepath, const char* group, uint32_t index,
  char* buf, size_t len) {

  if (!filepath || !buf || len == 0) {
    SIF_LOG_ERROR(TAG, "invalid arguments for sif_hdf5_attr_name");
    return SIF_ERR_INVALID;
  }

  h5_quiet_t quiet;
  quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  int status =
    attr_target_open(filepath, sif__hdf5_group(group), 0, &file, &obj);
  if (status == SIF_OK) {
    const ssize_t n = H5Aget_name_by_idx(obj, ".", H5_INDEX_NAME, H5_ITER_INC,
      (hsize_t)index, buf, len, H5P_DEFAULT);
    if (n < 0) {
      buf[0] = '\0';
      status = SIF_ERR_INVALID;
    } else if ((size_t)n >= len) {
      status = SIF_ERR_RANGE;
    }
  }

  attr_target_close(file, obj);
  quiet_end(&quiet);
  return status;
}
