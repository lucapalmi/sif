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
#include "io/hdf5_util.h"
#include "io/internal.h"
#include "measure/profiles_internal.h"
#include "sif/utils/logger.h"
#include "sif/utils/random.h"
#include "structures/catalogue_internal.h"
#include "structures/results_internal.h"
#include "utils/logger_internal.h"

#include <hdf5.h>

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
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

#define G_CATALOGUE  "catalogue"
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
    return sif__h5_keep();

  hid_t space = H5Screate(H5S_SCALAR);
  hid_t attr =
    H5Acreate2(obj, name, file_type, space, H5P_DEFAULT, H5P_DEFAULT);
  const int ok = attr >= 0 && H5Awrite(attr, mem_type, value) >= 0;
  const int status = ok ? SIF_OK : sif__h5_keep();

  close_id(attr);
  close_id(space);
  return status;
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
    return sif__h5_keep();

  int status;
  if (H5Tset_size(type, H5T_VARIABLE) >= 0 &&
      H5Tset_cset(type, H5T_CSET_UTF8) >= 0)
    status = attr_write(obj, name, type, type, &v);
  else
    status = sif__h5_keep();

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
  const int status = ok ? SIF_OK : sif__h5_keep();

  close_id(dset);
  close_id(space);
  return status;
}

/* "/group/name", or "/name" in the root, for messages. */
static const char* obj_path(const char* group, const char* name) {
  static SIF_THREAD_LOCAL char buf[256];
  snprintf(
    buf, sizeof(buf), "/%s%s%s", group ? group : "", group ? "/" : "", name);
  return buf;
}

/* A numeric attribute the file has to have, logged as "<path>: /group/name:
 * <reason>" if it is missing or cannot be read. */
static int attr_need(hid_t obj, const char* path, const char* group,
  const char* name, hid_t mem_type, void* out) {
  if (H5Aexists(obj, name) <= 0) {
    SIF_LOG_ERROR(TAG, "%s: %s: missing", path, obj_path(group, name));
    return SIF_ERR_IO;
  }
  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  const int ok = attr >= 0 && H5Aread(attr, mem_type, out) >= 0;
  if (!ok)
    SIF_LOG_ERROR(
      TAG, "%s: %s: %s", path, obj_path(group, name), sif__h5_cause());
  close_id(attr);
  return ok ? SIF_OK : SIF_ERR_IO;
}

/* "(a)" or "(a, b)". */
static const char* shape_text(
  int rank, const hsize_t* dims, char* buf, size_t len) {
  if (rank == 1)
    snprintf(buf, len, "(%llu)", (unsigned long long)dims[0]);
  else
    snprintf(buf, len, "(%llu, %llu)", (unsigned long long)dims[0],
      (unsigned long long)dims[1]);
  return buf;
}

/*
 * A whole array, read into `out` -- but only if its shape is exactly the one
 * expected. The metadata that says how large to make `out` came from the
 * attributes, and a dataset that disagrees with them is a malformed file, not
 * something to read part of.
 */
static int dataset_read(hid_t loc, const char* path, const char* group,
  const char* name, int rank, const hsize_t* expect, hid_t mem_type,
  void* out) {

  const char* where = obj_path(group, name);
  if (!link_exists(loc, name)) {
    SIF_LOG_ERROR(TAG, "%s: %s: missing", path, where);
    return SIF_ERR_IO;
  }

  int status = SIF_ERR_IO;
  hid_t dset = H5Dopen2(loc, name, H5P_DEFAULT);
  hid_t space = dset >= 0 ? H5Dget_space(dset) : H5I_INVALID_HID;
  if (space < 0) {
    SIF_LOG_ERROR(TAG, "%s: %s: %s", path, where, sif__h5_cause());
    goto done;
  }

  const int got_rank = H5Sget_simple_extent_ndims(space);
  if (got_rank != rank) {
    SIF_LOG_ERROR(
      TAG, "%s: %s: rank %d, expected %d", path, where, got_rank, rank);
    goto done;
  }

  hsize_t dims[2] = {0, 0};
  H5Sget_simple_extent_dims(space, dims, NULL);

  hsize_t total = 1;
  bool same = true;
  for (int i = 0; i < rank; i++) {
    same = same && dims[i] == expect[i];
    total *= dims[i];
  }
  if (!same) {
    char a[64], b[64];
    SIF_LOG_ERROR(TAG, "%s: %s: shape %s, expected %s", path, where,
      shape_text(rank, dims, a, sizeof a),
      shape_text(rank, expect, b, sizeof b));
    goto done;
  }

  if (total == 0 ||
      H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) >= 0)
    status = SIF_OK;
  else
    SIF_LOG_ERROR(TAG, "%s: %s: %s", path, where, sif__h5_cause());

done:
  close_id(space);
  close_id(dset);
  return status;
}

/*
 * The (N, 3) centres, one column at a time from the catalogue's own x, y and
 * z arrays, so neither direction needs an interleaved copy of the catalogue.
 */
static int centres_io(
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
    if (!ok)
      (void)sif__h5_keep();

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

  if (H5Aexists(file, "sif_format") <= 0) {
    SIF_LOG_ERROR(TAG, "%s: not a sif file (no /sif_format)%s", path,
      link_exists(file, "Header") ? "; a GADGET snapshot?" : "");
    return SIF_ERR_IO;
  }
  if (attr_read_str(file, "sif_format", format, sizeof(format)) != SIF_OK ||
      strcmp(format, SIF_H5_FORMAT) != 0) {
    SIF_LOG_ERROR(TAG, "%s: /sif_format is '%s', expected '%s'", path, format,
      SIF_H5_FORMAT);
    return SIF_ERR_IO;
  }

  if (attr_need(file, path, NULL, "sif_format_version", H5T_NATIVE_UINT32,
        &version) != SIF_OK)
    return SIF_ERR_IO;
  if (version > SIF_H5_FORMAT_VERSION) {
    SIF_LOG_ERROR(TAG, "%s: sif format version %u, this build reads up to %u",
      path, version, SIF_H5_FORMAT_VERSION);
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
    if (errno != ENOENT) {
      (void)sif__io_os_error(TAG, path, NULL, errno);
      return H5I_INVALID_HID;
    }
    hid_t file = H5Fcreate(path, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
      SIF_LOG_ERROR(TAG, "%s: create: %s", path, sif__h5_cause());
      return H5I_INVALID_HID;
    }
    if (stamp_root(file) != SIF_OK) {
      SIF_LOG_ERROR(TAG, "%s: root attributes: %s", path, sif__h5_kept());
      close_id(file);
      return H5I_INVALID_HID;
    }
    return file;
  }

  if (S_ISDIR(st.st_mode)) {
    (void)sif__io_os_error(TAG, path, NULL, EISDIR);
    return H5I_INVALID_HID;
  }

  hid_t file = H5Fopen(path, H5F_ACC_RDWR, H5P_DEFAULT);
  if (file < 0) {
    SIF_LOG_ERROR(
      TAG, "%s: %s; existing file left untouched", path, sif__h5_cause());
    return H5I_INVALID_HID;
  }

  if (check_root(file, path) != SIF_OK) {
    close_id(file);
    return H5I_INVALID_HID;
  }

  return file;
}

static hid_t open_for_read(const char* path) {
  hid_t file = sif__h5_open_read(TAG, path);
  if (file < 0)
    return H5I_INVALID_HID;

  if (check_root(file, path) != SIF_OK) {
    close_id(file);
    return H5I_INVALID_HID;
  }

  return file;
}

/* A fresh, empty group in place of whatever was there under that name. */
static hid_t group_replace(hid_t file, const char* name) {
  if (link_exists(file, name) && H5Ldelete(file, name, H5P_DEFAULT) < 0) {
    (void)sif__h5_keep();
    return H5I_INVALID_HID;
  }
  const hid_t g = H5Gcreate2(file, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (g < 0)
    (void)sif__h5_keep();
  return g;
}

static hid_t group_open(hid_t file, const char* name, const char* path) {
  if (!link_exists(file, name)) {
    SIF_LOG_ERROR(TAG, "%s: /%s: missing", path, name);
    return H5I_INVALID_HID;
  }
  const hid_t g = H5Gopen2(file, name, H5P_DEFAULT);
  if (g < 0)
    SIF_LOG_ERROR(TAG, "%s: /%s: %s", path, name, sif__h5_cause());
  return g;
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
      "%s: /%s has %" PRIu64 " voids, /%s %" PRIu64 " (not the same voids)",
      path, written, n_voids, group, other);
  }
}

/* ------------------------------------------------------------------------ */
/* catalogue                                                                 */
/* ------------------------------------------------------------------------ */

static sif_hdf5_attr_kind_t attr_kind_of(hid_t attr);

/* The catalogue's metadata, as attributes of its group. */
static int meta_write(hid_t group, const sif_catalogue_t* catalogue) {
  for (uint32_t m = 0; m < sif_catalogue_meta_count(catalogue); m++) {
    const char* key = sif_catalogue_meta_name(catalogue, m);
    int status = SIF_OK;
    switch (sif_catalogue_meta_kind(catalogue, key)) {
    case SIF_CATALOGUE_META_INT: {
      const int64_t v = sif_catalogue_meta_int_get(catalogue, key);
      status = attr_write(group, key, H5T_STD_I64LE, H5T_NATIVE_INT64, &v);
      break;
    }
    case SIF_CATALOGUE_META_REAL:
      status =
        attr_write_f64(group, key, sif_catalogue_meta_real_get(catalogue, key));
      break;
    case SIF_CATALOGUE_META_STRING:
      status =
        attr_write_str(group, key, sif_catalogue_meta_string_get(catalogue, key));
      break;
    case SIF_CATALOGUE_META_MISSING:
      break;
    }
    if (status != SIF_OK)
      return status;
  }
  return SIF_OK;
}

static herr_t count_attr(
  hid_t loc, const char* name, const H5A_info_t* info, void* data) {
  (void)loc;
  (void)name;
  (void)info;
  (*(uint32_t*)data)++;
  return 0;
}

/* Every attribute of the group the catalogue does not use itself, as its
 * metadata -- what sif wrote, and what anyone added with
 * sif_hdf5_set_attr_string() and its kin. One whose name is no metadata key,
 * or whose value is not a scalar, is left in the file and not read. */
static int meta_read(hid_t group, sif_catalogue_t* catalogue) {
  /* Counted by iterating, as in sif_hdf5_attr_count(): H5Oget_info3 and
   * H5O_info2_t only exist from HDF5 1.12. */
  uint32_t n_attrs = 0;
  hsize_t idx = 0;
  if (H5Aiterate2(group, H5_INDEX_NAME, H5_ITER_INC, &idx, count_attr,
        &n_attrs) < 0)
    return sif__h5_keep();

  for (hsize_t i = 0; i < n_attrs; i++) {
    hid_t attr = H5Aopen_by_idx(
      group, ".", H5_INDEX_NAME, H5_ITER_INC, i, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
      return sif__h5_keep();
    char name[128];
    const ssize_t len = H5Aget_name(attr, sizeof name, name);
    char lower[128] = "";
    for (ssize_t c = 0; c <= len && c < (ssize_t)sizeof lower; c++)
      lower[c] = (char)tolower((unsigned char)name[c]);
    lower[sizeof lower - 1] = '\0';

    if (len > 0 && len < (ssize_t)sizeof name &&
        sif__catalogue_meta_key_ok(name)) {
      int64_t iv;
      double dv;
      char text[4096];
      switch (attr_kind_of(attr)) {
      case SIF_HDF5_ATTR_INT:
        if (H5Aread(attr, H5T_NATIVE_INT64, &iv) >= 0)
          (void)sif_catalogue_meta_int_set(catalogue, name, iv);
        break;
      case SIF_HDF5_ATTR_REAL:
        if (H5Aread(attr, H5T_NATIVE_DOUBLE, &dv) >= 0 && isfinite(dv))
          (void)sif_catalogue_meta_real_set(catalogue, name, dv);
        break;
      case SIF_HDF5_ATTR_STRING:
        if (attr_read_str(group, name, text, sizeof text) == SIF_OK &&
            !strpbrk(text, "\n\r\""))
          (void)sif_catalogue_meta_string_set(catalogue, name, text);
        break;
      default:
        break;
      }
    }
    close_id(attr);
  }
  return SIF_OK;
}

int sif_catalogue_write_hdf5(const char* filepath, const sif_catalogue_t* catalogue) {
  if (!filepath || !catalogue) {
    SIF_LOG_ERROR(TAG, "sif_catalogue_write_hdf5: NULL argument");
    return SIF_ERR_INVALID;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  int status = SIF_ERR_IO;
  hid_t file = open_for_write(filepath);
  hid_t group = H5I_INVALID_HID, dset = H5I_INVALID_HID,
        space = H5I_INVALID_HID;
  if (file < 0)
    goto done;

  group = group_replace(file, G_CATALOGUE);
  if (group < 0)
    goto fail;

  const uint64_t n = catalogue->n_voids;
  const hsize_t dims_c[2] = {n, 3};
  const hsize_t dims_1[1] = {n};

  space = H5Screate_simple(2, dims_c, NULL);
  dset = space >= 0 ? H5Dcreate2(group, "centres", real_file_type(), space,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
                    : H5I_INVALID_HID;
  if (dset < 0)
    (void)sif__h5_keep();
  sif_real* const cols[3] = {catalogue->cx, catalogue->cy, catalogue->cz};

  /* The three columns of centres are otherwise unnamed: say which is
   * which, as the other formats do in their headers. */
  if (dset < 0 || centres_io(dset, cols, n, 1) != SIF_OK ||
      attr_write_str(dset, "columns",
        catalogue->units == SIF_COORDINATES_SKY ? "ra dec z" : "cx cy cz") !=
        SIF_OK ||
      dataset_write(group, "radii", 1, dims_1, real_file_type(),
        real_mem_type(), catalogue->radii) != SIF_OK ||
      attr_write_u64(group, "n_voids", n) != SIF_OK ||
      attr_write_str(group, "coordinates",
        catalogue->units == SIF_COORDINATES_SKY ? "sky" : "cartesian") !=
        SIF_OK ||
      meta_write(group, catalogue) != SIF_OK)
    goto fail;

  if (catalogue->footprint &&
      (dataset_write(group, "footprint", 1, dims_1, real_file_type(),
         real_mem_type(), catalogue->footprint) != SIF_OK ||
        dataset_write(group, "footprint_shell", 1, dims_1, real_file_type(),
          real_mem_type(), catalogue->footprint_shell) != SIF_OK))
    goto fail;

  warn_rows(file, filepath, G_DENSITY, n, G_CATALOGUE);
  warn_rows(file, filepath, G_VELOCITY, n, G_CATALOGUE);

  status = SIF_OK;
  SIF_LOG_INFO(TAG, "saved %" PRIu64 " voids to %s", n, filepath);
  goto done;

fail:
  SIF_LOG_ERROR(
    TAG, "%s: write /%s: %s", filepath, G_CATALOGUE, sif__h5_kept());

done:
  close_id(dset);
  close_id(space);
  close_id(group);
  close_id(file);
  sif__h5_quiet_end(&quiet);
  return status;
}

sif_catalogue_t* sif_catalogue_read_hdf5(const char* filepath) {
  if (!filepath) {
    SIF_LOG_ERROR(TAG, "sif_catalogue_read_hdf5: NULL path");
    return NULL;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  sif_catalogue_t* cat = NULL;
  hid_t file = open_for_read(filepath);
  hid_t group =
    file >= 0 ? group_open(file, G_CATALOGUE, filepath) : H5I_INVALID_HID;
  hid_t dset = H5I_INVALID_HID;
  uint64_t n = 0;

  if (group < 0 || attr_need(group, filepath, G_CATALOGUE, "n_voids",
                     H5T_NATIVE_UINT64, &n) != SIF_OK)
    goto fail;

  cat = sif_catalogue_alloc(n);
  if (!cat) {
    SIF_LOG_ERROR(TAG, "%s: %" PRIu64 " voids: out of memory", filepath, n);
    goto fail;
  }

  const hsize_t dims_c[2] = {n, 3};
  const hsize_t dims_1[1] = {n};
  sif_real* const cols[3] = {cat->cx, cat->cy, cat->cz};

  /* The centres are read by column, so their shape is checked here rather
   * than by dataset_read(). */
  const char* centres = obj_path(G_CATALOGUE, "centres");
  if (!link_exists(group, "centres")) {
    SIF_LOG_ERROR(TAG, "%s: %s: missing", filepath, centres);
    goto fail;
  }
  dset = H5Dopen2(group, "centres", H5P_DEFAULT);
  {
    hid_t space = dset >= 0 ? H5Dget_space(dset) : H5I_INVALID_HID;
    hsize_t dims[2] = {0, 0};
    const int rank = space >= 0 ? H5Sget_simple_extent_ndims(space) : -1;
    if (rank == 1 || rank == 2)
      H5Sget_simple_extent_dims(space, dims, NULL);
    close_id(space);
    if (rank != 2 || dims[0] != dims_c[0] || dims[1] != dims_c[1]) {
      char a[64], b[64];
      if (rank == 1 || rank == 2)
        SIF_LOG_ERROR(TAG, "%s: %s: shape %s, expected %s", filepath,
          obj_path(G_CATALOGUE, "centres"), shape_text(rank, dims, a, sizeof a),
          shape_text(2, dims_c, b, sizeof b));
      else
        SIF_LOG_ERROR(TAG, "%s: %s: rank %d, expected 2", filepath,
          obj_path(G_CATALOGUE, "centres"), rank);
      goto fail;
    }
  }

  if (centres_io(dset, cols, n, 0) != SIF_OK) {
    SIF_LOG_ERROR(TAG, "%s: %s: %s", filepath, obj_path(G_CATALOGUE, "centres"),
      sif__h5_kept());
    goto fail;
  }
  if (dataset_read(group, filepath, G_CATALOGUE, "radii", 1, dims_1,
        real_mem_type(), cat->radii) != SIF_OK)
    goto fail;

  /* Filled in directly rather than through sif_catalogue_append(). */
  cat->n_voids = n;

  /* A file written before the attribute existed holds Cartesian centres,
   * which is all a catalogue could then hold. */
  char coords[16];
  if (attr_read_str(group, "coordinates", coords, sizeof coords) == SIF_OK) {
    if (strcmp(coords, "sky") == 0) {
      cat->units = SIF_COORDINATES_SKY;
    } else if (strcmp(coords, "cartesian") != 0) {
      SIF_LOG_ERROR(TAG,
        "%s: /%s/coordinates is '%s', expected 'cartesian' or 'sky'", filepath,
        G_CATALOGUE, coords);
      goto fail;
    }
  }

  if (meta_read(group, cat) != SIF_OK) {
    SIF_LOG_ERROR(
      TAG, "%s: /%s attributes: %s", filepath, G_CATALOGUE, sif__h5_kept());
    goto fail;
  }

  if (link_exists(group, "footprint")) {
    if (sif_catalogue_reserve_footprint(cat) != SIF_OK) {
      SIF_LOG_ERROR(TAG, "%s: footprint: out of memory", filepath);
      goto fail;
    }
    if (dataset_read(group, filepath, G_CATALOGUE, "footprint", 1, dims_1,
          real_mem_type(), cat->footprint) != SIF_OK ||
        dataset_read(group, filepath, G_CATALOGUE, "footprint_shell", 1, dims_1,
          real_mem_type(), cat->footprint_shell) != SIF_OK)
      goto fail;
  }

  close_id(dset);
  close_id(group);
  close_id(file);
  sif__h5_quiet_end(&quiet);
  SIF_LOG_INFO(TAG, "loaded %" PRIu64 " voids from %s", n, filepath);
  return cat;

fail:
  sif_catalogue_free(cat);
  close_id(dset);
  close_id(group);
  close_id(file);
  sif__h5_quiet_end(&quiet);
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
      TAG, "sif_profiles_write_hdf5: needs a path and a profile set");
    return SIF_ERR_INVALID;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  int status = SIF_ERR_IO;
  hid_t file = open_for_write(filepath);
  if (file < 0)
    goto done;

  if (dens) {
    if (profiles_group_write(file, G_DENSITY, dens->n_voids, dens->n_bins,
          dens->ext, dens->r_edges, "profiles", dens->profiles,
          dens->differential ? 1 : 0) != SIF_OK) {
      SIF_LOG_ERROR(
        TAG, "%s: write /%s: %s", filepath, G_DENSITY, sif__h5_kept());
      goto done;
    }
    warn_rows(file, filepath, G_CATALOGUE, dens->n_voids, G_DENSITY);
  }

  if (vel) {
    if (profiles_group_write(file, G_VELOCITY, vel->n_voids, vel->n_bins,
          vel->ext, vel->r_edges, "v_rad", vel->v_rad, -1) != SIF_OK) {
      SIF_LOG_ERROR(
        TAG, "%s: write /%s: %s", filepath, G_VELOCITY, sif__h5_kept());
      goto done;
    }
    warn_rows(file, filepath, G_CATALOGUE, vel->n_voids, G_VELOCITY);
  }

  status = SIF_OK;
  SIF_LOG_INFO(TAG, "saved profiles to %s", filepath);

done:
  close_id(file);
  sif__h5_quiet_end(&quiet);
  return status;
}

int sif_profiles_read_header_hdf5(
  const char* filepath, int* out_has_density, int* out_has_velocity) {

  if (!filepath) {
    SIF_LOG_ERROR(TAG, "sif_profiles_read_header_hdf5: NULL path");
    return SIF_ERR_INVALID;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  hid_t file = open_for_read(filepath);
  if (file >= 0) {
    if (out_has_density)
      *out_has_density = link_exists(file, G_DENSITY);
    if (out_has_velocity)
      *out_has_velocity = link_exists(file, G_VELOCITY);
  }

  close_id(file);
  sif__h5_quiet_end(&quiet);
  return file >= 0 ? SIF_OK : SIF_ERR_IO;
}

/* One profile set's shape, from its group's attributes. */
static int profiles_shape(hid_t g, const char* path, const char* group,
  uint64_t* n_voids, uint32_t* n_bins, sif_real* ext, uint8_t* differential) {

  double e = 0.0;
  if (attr_need(g, path, group, "n_voids", H5T_NATIVE_UINT64, n_voids) !=
        SIF_OK ||
      attr_need(g, path, group, "n_bins", H5T_NATIVE_UINT32, n_bins) !=
        SIF_OK ||
      attr_need(g, path, group, "ext", H5T_NATIVE_DOUBLE, &e) != SIF_OK)
    return SIF_ERR_IO;
  *ext = (sif_real)e;

  if (differential && attr_need(g, path, group, "differential",
                        H5T_NATIVE_UINT8, differential) != SIF_OK)
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
    SIF_LOG_ERROR(TAG, "sif_profiles_read_hdf5: needs a path and an output");
    return SIF_ERR_INVALID;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

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
    SIF_LOG_ERROR(TAG, "%s: /%s: missing", filepath,
      (out_dens && !link_exists(file, G_DENSITY)) ? G_DENSITY : G_VELOCITY);
    status = SIF_ERR_INVALID;
    goto done;
  }

  if (out_dens) {
    uint64_t n;
    uint32_t b;
    sif_real ext;
    uint8_t differential;

    g = group_open(file, G_DENSITY, filepath);
    if (g < 0 || profiles_shape(g, filepath, G_DENSITY, &n, &b, &ext,
                   &differential) != SIF_OK)
      goto done;

    dens = sif__density_profiles_alloc(n, b, ext, differential != 0);
    if (!dens) {
      SIF_LOG_ERROR(TAG, "%s: /%s: out of memory", filepath, G_DENSITY);
      status = SIF_ERR_ALLOC;
      goto done;
    }

    const hsize_t dims_e[1] = {(hsize_t)b + 1};
    const hsize_t dims_r[2] = {n, b};
    if (dataset_read(g, filepath, G_DENSITY, "r_edges", 1, dims_e,
          real_mem_type(), dens->r_edges) != SIF_OK ||
        dataset_read(g, filepath, G_DENSITY, "profiles", 2, dims_r,
          real_mem_type(), dens->profiles) != SIF_OK)
      goto done;

    close_id(g);
    g = H5I_INVALID_HID;
  }

  if (out_vel) {
    uint64_t n;
    uint32_t b;
    sif_real ext;

    g = group_open(file, G_VELOCITY, filepath);
    if (g < 0 ||
        profiles_shape(g, filepath, G_VELOCITY, &n, &b, &ext, NULL) != SIF_OK)
      goto done;

    vel = sif__velocity_profiles_alloc(n, b, ext);
    if (!vel) {
      SIF_LOG_ERROR(TAG, "%s: /%s: out of memory", filepath, G_VELOCITY);
      status = SIF_ERR_ALLOC;
      goto done;
    }

    const hsize_t dims_e[1] = {(hsize_t)b + 1};
    const hsize_t dims_r[2] = {n, b};
    if (dataset_read(g, filepath, G_VELOCITY, "r_edges", 1, dims_e,
          real_mem_type(), vel->r_edges) != SIF_OK ||
        dataset_read(g, filepath, G_VELOCITY, "v_rad", 2, dims_r,
          real_mem_type(), vel->v_rad) != SIF_OK)
      goto done;
  }

  status = SIF_OK;

done:
  close_id(g);
  close_id(file);
  sif__h5_quiet_end(&quiet);

  if (status != SIF_OK) {
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
    SIF_LOG_ERROR(TAG, "sif_size_function_write_hdf5: NULL argument");
    return SIF_ERR_INVALID;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

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
      dataset_write(g, "r_centres", 1, dims_b, real_file_type(),
        real_mem_type(), vsf->r_centres) == SIF_OK &&
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
    SIF_LOG_ERROR(TAG, "%s: write /%s: %s", filepath, G_VSF, sif__h5_kept());

  close_id(g);
  close_id(file);
  sif__h5_quiet_end(&quiet);
  return status;
}

sif_size_function_t* sif_size_function_read_hdf5(const char* filepath) {
  if (!filepath) {
    SIF_LOG_ERROR(TAG, "sif_size_function_read_hdf5: NULL path");
    return NULL;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  sif_size_function_t* vsf = NULL;
  hid_t file = open_for_read(filepath);
  hid_t g = file >= 0 ? group_open(file, G_VSF, filepath) : H5I_INVALID_HID;

  uint32_t b = 0, options = 0;
  double r_min = 0.0, r_max = 0.0;

  if (g < 0 ||
      attr_need(g, filepath, G_VSF, "n_bins", H5T_NATIVE_UINT32, &b) !=
        SIF_OK ||
      attr_need(g, filepath, G_VSF, "r_min", H5T_NATIVE_DOUBLE, &r_min) !=
        SIF_OK ||
      attr_need(g, filepath, G_VSF, "r_max", H5T_NATIVE_DOUBLE, &r_max) !=
        SIF_OK ||
      attr_need(g, filepath, G_VSF, "options", H5T_NATIVE_UINT32, &options) !=
        SIF_OK)
    goto fail;

  vsf = sif__size_function_alloc(b);
  if (!vsf) {
    SIF_LOG_ERROR(TAG, "%s: /%s: out of memory", filepath, G_VSF);
    goto fail;
  }

  vsf->options = (sif_option)options;
  vsf->r_min = (sif_real)r_min;
  vsf->r_max = (sif_real)r_max;

  const hsize_t dims_e[1] = {(hsize_t)b + 1};
  const hsize_t dims_b[1] = {b};

  if (dataset_read(g, filepath, G_VSF, "r_edges", 1, dims_e, real_mem_type(),
        vsf->r_edges) != SIF_OK ||
      dataset_read(g, filepath, G_VSF, "r_centres", 1, dims_b, real_mem_type(),
        vsf->r_centres) != SIF_OK ||
      dataset_read(g, filepath, G_VSF, "counts", 1, dims_b, H5T_NATIVE_UINT64,
        vsf->counts) != SIF_OK ||
      dataset_read(g, filepath, G_VSF, "vsf", 1, dims_b, real_mem_type(),
        vsf->vsf) != SIF_OK ||
      dataset_read(g, filepath, G_VSF, "err", 1, dims_b, real_mem_type(),
        vsf->err) != SIF_OK)
    goto fail;

  close_id(g);
  close_id(file);
  sif__h5_quiet_end(&quiet);
  return vsf;

fail:
  sif_size_function_free(vsf);
  close_id(g);
  close_id(file);
  sif__h5_quiet_end(&quiet);
  return NULL;
}

/* ------------------------------------------------------------------------ */
/* particles from any HDF5 file                                              */
/* ------------------------------------------------------------------------ */

/* Rows per read, as in the other streaming readers. */
#define ROWS_CHUNK ((hsize_t)1 << 18)

enum { P_X, P_Y, P_Z, P_VX, P_VY, P_VZ, P_W, P_N };

/* One part of the field: a dataset, and the column of it for a 2D one. */
typedef struct {
  const char* spec; /* as the caller wrote it, for messages */
  char path[512];
  int column; /* -1 for a 1D dataset */
  hid_t dset;
  double* buf;
} h5_source_t;

/* "Group/Dataset" or "Group/Dataset[k]": the path, and the column. */
static int source_parse(const char* spec, h5_source_t* s) {
  s->spec = spec;
  s->column = -1;
  const char* open = strrchr(spec, '[');
  size_t len = strlen(spec);
  if (open && spec[len - 1] == ']') {
    char* end;
    const long k = strtol(open + 1, &end, 10);
    if (end == open + 1 || *end != ']' || k < 0) {
      SIF_LOG_ERROR(
        TAG, "column spec '%s': [] must hold a column number", spec);
      return SIF_ERR_INVALID;
    }
    s->column = (int)k;
    len = (size_t)(open - spec);
  }
  if (len == 0 || len >= sizeof s->path) {
    SIF_LOG_ERROR(TAG, "column spec '%s': no dataset path", spec);
    return SIF_ERR_INVALID;
  }
  memcpy(s->path, spec, len);
  s->path[len] = '\0';
  return SIF_OK;
}

/* Opens a source's dataset in one file and says how many rows it has. */
static int source_open(hid_t file, const char* filepath, const char* what,
  h5_source_t* s, hsize_t* rows) {
  s->dset = H5I_INVALID_HID;
  if (H5Lexists(file, s->path, H5P_DEFAULT) <= 0) {
    SIF_LOG_ERROR(TAG, "%s: %s (%s): missing", filepath, s->path, what);
    return SIF_ERR_INVALID;
  }
  s->dset = H5Dopen2(file, s->path, H5P_DEFAULT);
  if (s->dset < 0) {
    SIF_LOG_ERROR(TAG, "%s: %s (%s): not a dataset", filepath, s->path, what);
    return SIF_ERR_INVALID;
  }

  hid_t type = H5Dget_type(s->dset);
  const H5T_class_t cls = type >= 0 ? H5Tget_class(type) : H5T_NO_CLASS;
  close_id(type);
  hid_t space = H5Dget_space(s->dset);
  const int rank = space >= 0 ? H5Sget_simple_extent_ndims(space) : -1;
  hsize_t dims[2] = {0, 0};
  if (rank == 1 || rank == 2)
    H5Sget_simple_extent_dims(space, dims, NULL);
  close_id(space);

  if (cls != H5T_INTEGER && cls != H5T_FLOAT) {
    SIF_LOG_ERROR(TAG, "%s: %s (%s): not numeric", filepath, s->path, what);
    return SIF_ERR_INVALID;
  }
  if (rank == 1 && s->column >= 0) {
    SIF_LOG_ERROR(TAG, "%s: %s (%s): 1-D, but '%s' names a column", filepath,
      s->path, what, s->spec);
    return SIF_ERR_INVALID;
  }
  if (rank == 2 && s->column < 0) {
    SIF_LOG_ERROR(TAG, "%s: %s (%s): %llu columns; name one, as %s[0]",
      filepath, s->path, what, (unsigned long long)dims[1], s->path);
    return SIF_ERR_INVALID;
  }
  if (rank == 2 && (hsize_t)s->column >= dims[1]) {
    SIF_LOG_ERROR(TAG, "%s: %s (%s): column %d, dataset has %llu", filepath,
      s->path, what, s->column, (unsigned long long)dims[1]);
    return SIF_ERR_INVALID;
  }
  if (rank != 1 && rank != 2) {
    SIF_LOG_ERROR(TAG, "%s: %s (%s): rank %d, expected 1 or 2", filepath,
      s->path, what, rank);
    return SIF_ERR_INVALID;
  }
  *rows = dims[0];
  return SIF_OK;
}

/* Rows [first, first + n) of a source, as doubles, converted by HDF5. */
static int source_read(h5_source_t* s, hsize_t first, hsize_t n) {
  hid_t fspace = H5Dget_space(s->dset);
  hsize_t start[2] = {first, s->column < 0 ? 0 : (hsize_t)s->column};
  hsize_t count[2] = {n, 1};
  hid_t mspace = H5Screate_simple(1, &n, NULL);
  const int ok = fspace >= 0 && mspace >= 0 &&
                 H5Sselect_hyperslab(
                   fspace, H5S_SELECT_SET, start, NULL, count, NULL) >= 0 &&
                 H5Dread(s->dset, H5T_NATIVE_DOUBLE, mspace, fspace,
                   H5P_DEFAULT, s->buf) >= 0;
  const int status = ok ? SIF_OK : sif__h5_keep();
  close_id(mspace);
  close_id(fspace);
  return status;
}

static void sources_close(h5_source_t* src) {
  for (int r = 0; r < P_N; r++) {
    if (src[r].spec)
      close_id(src[r].dset);
    src[r].dset = H5I_INVALID_HID;
  }
}

/* Opens one file and every source in it; the rows they share. */
static int file_open(const char* path, h5_source_t* src,
  const char* const* names, hid_t* out_file, hsize_t* out_rows) {
  *out_file = sif__h5_open_read(TAG, path);
  if (*out_file < 0)
    return SIF_ERR_IO;
  hsize_t rows = 0;
  bool first = true;
  for (int r = 0; r < P_N; r++) {
    if (!src[r].spec)
      continue;
    hsize_t n = 0;
    const int status = source_open(*out_file, path, names[r], &src[r], &n);
    if (status != SIF_OK) {
      sources_close(src);
      close_id(*out_file);
      return status;
    }
    if (!first && n != rows) {
      SIF_LOG_ERROR(TAG, "%s: %s (%s): %llu rows, preceding columns %llu", path,
        src[r].path, names[r], (unsigned long long)n, (unsigned long long)rows);
      sources_close(src);
      close_id(*out_file);
      return SIF_ERR_INVALID;
    }
    rows = n;
    first = false;
  }
  *out_rows = rows;
  return SIF_OK;
}

sif_field_t* sif_field_read_hdf5(const char* const* paths, uint32_t n_paths,
  const sif_field_columns_t* columns, double length_scale, double fraction,
  uint64_t seed) {

  if (!paths || n_paths == 0 || !columns) {
    SIF_LOG_ERROR(TAG, "sif_field_read_hdf5: needs a path and the columns");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  for (uint32_t i = 0; i < n_paths; i++) {
    if (!paths[i] || !*paths[i]) {
      SIF_LOG_ERROR(TAG, "path %u of %u: empty", i + 1, n_paths);
      sif__error_status_set(SIF_ERR_INVALID);
      return NULL;
    }
  }
  const bool xy = columns->x || columns->y;
  const bool sky = columns->ra || columns->dec;
  if (xy && sky) {
    SIF_LOG_ERROR(TAG, "columns: both x y and ra dec given");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  if (sky ? !(columns->ra && columns->dec && columns->z)
          : !(columns->x && columns->y && columns->z)) {
    SIF_LOG_ERROR(TAG, "columns: %s required", sky ? "ra, dec, z" : "x, y, z");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  const int n_vel = !!columns->vx + !!columns->vy + !!columns->vz;
  if (n_vel != 0 && n_vel != 3) {
    SIF_LOG_ERROR(TAG, "columns: vx, vy, vz go together (%d given)", n_vel);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  if (!(fraction > 0 && fraction <= 1)) {
    SIF_LOG_ERROR(TAG, "fraction: %g, expected (0, 1]", fraction);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }
  if (!(length_scale > 0) || !isfinite(length_scale) ||
      (sky && length_scale != 1.0)) {
    SIF_LOG_ERROR(TAG, "length_scale: %g, expected > 0 (1 for sky coordinates)",
      length_scale);
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  static const char* const NAMES[P_N] = {"x", "y", "z", "vx", "vy", "vz", "w"};
  static const char* const NAMES_SKY[P_N] = {
    "ra", "dec", "z", "vx", "vy", "vz", "w"};
  const char* const* names = sky ? NAMES_SKY : NAMES;
  const char* specs[P_N] = {sky ? columns->ra : columns->x,
    sky ? columns->dec : columns->y, columns->z, columns->vx, columns->vy,
    columns->vz, columns->w};

  h5_source_t src[P_N];
  memset(src, 0, sizeof src);
  for (int r = 0; r < P_N; r++) {
    src[r].dset = H5I_INVALID_HID;
    if (specs[r] && source_parse(specs[r], &src[r]) != SIF_OK) {
      sif__error_status_set(SIF_ERR_INVALID);
      return NULL;
    }
    src[r].spec = specs[r];
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  sif_field_t* field = NULL;
  hsize_t* rows_of = calloc(n_paths, sizeof(hsize_t));
  int status = rows_of ? SIF_OK : SIF_ERR_ALLOC;
  if (!rows_of)
    SIF_LOG_ERROR(TAG, "out of memory");
  hid_t file = H5I_INVALID_HID;

  /* --- pass 0: every file, checked before anything is read --- */

  uint64_t n_rows = 0;
  for (uint32_t i = 0; status == SIF_OK && i < n_paths; i++) {
    status = file_open(paths[i], src, names, &file, &rows_of[i]);
    if (status == SIF_OK) {
      n_rows += rows_of[i];
      sources_close(src);
      close_id(file);
    }
  }
  if (status == SIF_OK && n_rows == 0) {
    SIF_LOG_ERROR(TAG, "%s%s: no rows", paths[0],
      n_paths > 1 ? " (and the other files)" : "");
    status = SIF_ERR_INVALID;
  }

  const uint64_t n_keep = status != SIF_OK ? 0
                          : fraction < 1
                            ? (uint64_t)llround(fraction * (double)n_rows)
                            : n_rows;
  if (status == SIF_OK && n_keep == 0) {
    SIF_LOG_ERROR(
      TAG, "fraction %g keeps none of %" PRIu64 " rows", fraction, n_rows);
    status = SIF_ERR_INVALID;
  }

  if (status == SIF_OK) {
    field = sif_field_alloc(n_keep);
    status = field ? sif_field_reserve_positions(field) : SIF_ERR_ALLOC;
    if (status == SIF_OK && specs[P_VX])
      status = sif_field_reserve_velocities(field);
    if (status == SIF_OK && specs[P_W])
      status = sif_field_reserve_weights(field);
  }
  for (int r = 0; status == SIF_OK && r < P_N; r++)
    if (specs[r] && !(src[r].buf = malloc(ROWS_CHUNK * sizeof(double))))
      status = SIF_ERR_ALLOC;
  if (status == SIF_ERR_ALLOC && rows_of)
    SIF_LOG_ERROR(TAG, "%s: %" PRIu64 " rows: out of memory", paths[0], n_keep);

  /* --- pass 1: the rows, subsampled over every file together --- */

  sif_prng_state_t prng;
  sif_prng_init(&prng, seed);
  const bool subsample = n_keep < n_rows;
  uint64_t seen = 0, kept = 0, n_bad = 0;
  const char* bad_path = NULL;
  hsize_t bad_row = 0;
  int bad_role = 0;

  for (uint32_t i = 0; status == SIF_OK && i < n_paths; i++) {
    hsize_t rows;
    status = file_open(paths[i], src, names, &file, &rows);
    for (hsize_t first = 0; status == SIF_OK && first < rows;
      first += ROWS_CHUNK) {
      const hsize_t n = rows - first < ROWS_CHUNK ? rows - first : ROWS_CHUNK;
      for (int r = 0; status == SIF_OK && r < P_N; r++)
        if (specs[r] && source_read(&src[r], first, n) != SIF_OK) {
          SIF_LOG_ERROR(TAG, "%s: %s (%s): rows %llu-%llu: %s", paths[i],
            src[r].path, names[r], (unsigned long long)first,
            (unsigned long long)(first + n - 1), sif__h5_kept());
          status = SIF_ERR_IO;
        }
      for (hsize_t j = 0; status == SIF_OK && j < n; j++) {
        /* Checked on every row, before the subsample draws, so whether a
         * file reads does not depend on the seed. */
        bool row_ok = true;
        for (int r = 0; r < P_N; r++) {
          if (specs[r] && !isfinite(src[r].buf[j])) {
            if (n_bad == 0) {
              bad_path = paths[i];
              bad_row = first + j;
              bad_role = r;
            }
            row_ok = false;
          }
        }
        if (!row_ok) {
          n_bad++;
          continue;
        }
        if (n_bad)
          continue;
        if (subsample) {
          const double u = sif_prng_next_double(&prng);
          const bool keep =
            (double)(n_rows - seen) * u < (double)(n_keep - kept);
          seen++;
          if (!keep)
            continue;
        }
        const uint64_t k = kept++;
        const double pos_scale = sky ? 1.0 : length_scale;
        field->x[k] = (sif_real)(src[P_X].buf[j] * pos_scale);
        field->y[k] = (sif_real)(src[P_Y].buf[j] * pos_scale);
        field->z[k] = (sif_real)(src[P_Z].buf[j] * pos_scale);
        if (specs[P_VX]) {
          field->vx[k] = (sif_real)src[P_VX].buf[j];
          field->vy[k] = (sif_real)src[P_VY].buf[j];
          field->vz[k] = (sif_real)src[P_VZ].buf[j];
        }
        if (specs[P_W])
          field->weights[k] = (sif_real)src[P_W].buf[j];
      }
    }
    if (file >= 0) {
      sources_close(src);
      close_id(file);
      file = H5I_INVALID_HID;
    }
  }

  if (status == SIF_OK && n_bad) {
    SIF_LOG_ERROR(TAG,
      "%s: %s (%s): row %llu: not finite (%" PRIu64 " such rows in all)",
      bad_path, specs[bad_role], names[bad_role], (unsigned long long)bad_row,
      n_bad);
    status = SIF_ERR_INVALID;
  }
  if (status == SIF_OK && kept != n_keep) {
    SIF_LOG_ERROR(TAG, "%s: read %" PRIu64 " rows, expected %" PRIu64, paths[0],
      kept, n_keep);
    status = SIF_ERR_IO;
  }

  for (int r = 0; r < P_N; r++)
    free(src[r].buf);
  free(rows_of);
  sif__h5_quiet_end(&quiet);

  if (status != SIF_OK) {
    sif__error_status_set(status);
    sif_field_free(field);
    return NULL;
  }
  if (sky)
    field->units = SIF_COORDINATES_SKY;
  SIF_LOG_INFO(TAG, "loaded %" PRIu64 " of %" PRIu64 " rows from %s%s", n_keep,
    n_rows, paths[0], n_paths > 1 ? " and the rest" : "");
  return field;
}

#undef ROWS_CHUNK

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
      TAG, "%s: does not exist (write /%s before its attributes)", path, group);
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
    SIF_LOG_ERROR(TAG, "%s: /%s: missing", path, group);
    close_id(file);
    return SIF_ERR_INVALID;
  }

  hid_t obj = H5Gopen2(file, group, H5P_DEFAULT);
  if (obj < 0) {
    SIF_LOG_ERROR(TAG, "%s: /%s: not a group", path, group);
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
    SIF_LOG_ERROR(TAG, "metadata: NULL path or key");
    return SIF_ERR_INVALID;
  }

  const char* g = sif__hdf5_group(group);
  if (sif__hdf5_key_reserved(g, key)) {
    SIF_LOG_ERROR(TAG, "%s: %s: reserved by sif", path, obj_path(g, key));
    return SIF_ERR_INVALID;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

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
      SIF_LOG_ERROR(TAG, "%s: %s: %s", path, obj_path(g, key), sif__h5_kept());
  }

  attr_target_close(file, obj);
  sif__h5_quiet_end(&quiet);
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
    SIF_LOG_ERROR(TAG, "metadata: NULL path or key");
    return SIF_ERR_INVALID;
  }

  const char* g = sif__hdf5_group(group);
  int status = attr_target_open(path, g, 0, file, obj);
  if (status != SIF_OK)
    return status;

  if (H5Aexists(*obj, key) <= 0) {
    SIF_LOG_ERROR(TAG, "%s: %s: missing", path, obj_path(g, key));
    return SIF_ERR_INVALID;
  }

  hid_t attr = H5Aopen(*obj, key, H5P_DEFAULT);
  if (attr < 0) {
    SIF_LOG_ERROR(TAG, "%s: %s: %s", path, obj_path(g, key), sif__h5_cause());
    return SIF_ERR_IO;
  }
  *kind = attr_kind_of(attr);
  close_id(attr);
  return SIF_OK;
}

static int null_output(void) {
  SIF_LOG_ERROR(TAG, "metadata: NULL output");
  return SIF_ERR_INVALID;
}

static int kind_mismatch(const char* path, const char* group, const char* key,
  sif_hdf5_attr_kind_t kind, const char* want) {
  static const char* const KIND[] = {"an integer", "a number", "a string"};
  const char* got = (kind == SIF_HDF5_ATTR_INT)      ? KIND[0]
                    : (kind == SIF_HDF5_ATTR_REAL)   ? KIND[1]
                    : (kind == SIF_HDF5_ATTR_STRING) ? KIND[2]
                                                     : "not a scalar";
  SIF_LOG_ERROR(TAG, "%s: %s: %s, expected %s", path,
    obj_path(sif__hdf5_group(group), key), got, want);
  return SIF_ERR_INVALID;
}

/* A getter's read, once the kind is known to fit. */
static int meta_read_value(hid_t obj, const char* path, const char* group,
  const char* key, hid_t mem_type, void* out) {
  const int status = attr_read(obj, key, mem_type, out);
  if (status != SIF_OK)
    SIF_LOG_ERROR(TAG, "%s: %s: %s", path,
      obj_path(sif__hdf5_group(group), key), sif__h5_cause());
  return status;
}

int sif_hdf5_get_attr_int(
  const char* filepath, const char* group, const char* key, int64_t* out) {

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  sif_hdf5_attr_kind_t kind;
  int status =
    out ? meta_open(filepath, group, key, &file, &obj, &kind) : null_output();
  if (status == SIF_OK) {
    status =
      kind == SIF_HDF5_ATTR_INT
        ? meta_read_value(obj, filepath, group, key, H5T_NATIVE_INT64, out)
        : kind_mismatch(filepath, group, key, kind, "an integer");
  }

  attr_target_close(file, obj);
  sif__h5_quiet_end(&quiet);
  return status;
}

int sif_hdf5_get_attr_real(
  const char* filepath, const char* group, const char* key, double* out) {

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  sif_hdf5_attr_kind_t kind;
  int status =
    out ? meta_open(filepath, group, key, &file, &obj, &kind) : null_output();
  if (status == SIF_OK) {
    /* An integer is a number too; HDF5 converts it on the way in. */
    status =
      (kind == SIF_HDF5_ATTR_REAL || kind == SIF_HDF5_ATTR_INT)
        ? meta_read_value(obj, filepath, group, key, H5T_NATIVE_DOUBLE, out)
        : kind_mismatch(filepath, group, key, kind, "a number");
  }

  attr_target_close(file, obj);
  sif__h5_quiet_end(&quiet);
  return status;
}

int sif_hdf5_get_attr_string(const char* filepath, const char* group,
  const char* key, char* buf, size_t len) {

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  sif_hdf5_attr_kind_t kind;
  int status = (buf && len)
                 ? meta_open(filepath, group, key, &file, &obj, &kind)
                 : null_output();
  if (status == SIF_OK) {
    if (kind != SIF_HDF5_ATTR_STRING) {
      status = kind_mismatch(filepath, group, key, kind, "a string");
    } else {
      status = attr_read_str(obj, key, buf, len);
      if (status == SIF_ERR_IO)
        SIF_LOG_ERROR(TAG, "%s: %s: unreadable string", filepath,
          obj_path(sif__hdf5_group(group), key));
    }
  }

  attr_target_close(file, obj);
  sif__h5_quiet_end(&quiet);
  return status;
}

int sif_hdf5_attr_kind(const char* filepath, const char* group, const char* key,
  sif_hdf5_attr_kind_t* out) {

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  sif_hdf5_attr_kind_t kind = SIF_HDF5_ATTR_OTHER;
  const int status =
    out ? meta_open(filepath, group, key, &file, &obj, &kind) : null_output();
  if (status == SIF_OK)
    *out = kind;

  attr_target_close(file, obj);
  sif__h5_quiet_end(&quiet);
  return status;
}

int sif_hdf5_attr_count(
  const char* filepath, const char* group, uint32_t* out) {

  if (!filepath || !out) {
    SIF_LOG_ERROR(TAG, "sif_hdf5_attr_count: NULL argument");
    return SIF_ERR_INVALID;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  int status =
    attr_target_open(filepath, sif__hdf5_group(group), 0, &file, &obj);
  if (status == SIF_OK) {
    /* Counted by iterating rather than asked of H5Oget_info, whose signature
     * has changed twice across the HDF5 versions sif builds against. */
    uint32_t n = 0;
    hsize_t idx = 0;
    if (H5Aiterate2(obj, H5_INDEX_NAME, H5_ITER_INC, &idx, count_attr, &n) <
        0) {
      SIF_LOG_ERROR(TAG, "%s: /%s: %s", filepath,
        sif__hdf5_group(group) ? sif__hdf5_group(group) : "", sif__h5_cause());
      status = SIF_ERR_IO;
    } else {
      *out = n;
    }
  }

  attr_target_close(file, obj);
  sif__h5_quiet_end(&quiet);
  return status;
}

int sif_hdf5_attr_name(const char* filepath, const char* group, uint32_t index,
  char* buf, size_t len) {

  if (!filepath || !buf || len == 0) {
    SIF_LOG_ERROR(TAG, "sif_hdf5_attr_name: NULL argument");
    return SIF_ERR_INVALID;
  }

  sif__h5_quiet_t quiet;
  sif__h5_quiet_begin(&quiet);

  hid_t file = H5I_INVALID_HID, obj = H5I_INVALID_HID;
  int status =
    attr_target_open(filepath, sif__hdf5_group(group), 0, &file, &obj);
  if (status == SIF_OK) {
    const ssize_t n = H5Aget_name_by_idx(obj, ".", H5_INDEX_NAME, H5_ITER_INC,
      (hsize_t)index, buf, len, H5P_DEFAULT);
    if (n < 0) {
      buf[0] = '\0';
      SIF_LOG_ERROR(TAG, "%s: /%s: no attribute at index %u", filepath,
        sif__hdf5_group(group) ? sif__hdf5_group(group) : "", index);
      status = SIF_ERR_INVALID;
    } else if ((size_t)n >= len) {
      status = SIF_ERR_RANGE;
    }
  }

  attr_target_close(file, obj);
  sif__h5_quiet_end(&quiet);
  return status;
}
