/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * The HDF5 snapshot format (SnapFormat 3), for gadget_io.c. Compiled only in
 * a build with HDF5; hdf5_off.c refuses in its place otherwise.
 *
 * The header is the attributes of `/Header`, under the names the GADGET-4
 * manual gives. Per-type data is in `/PartType<N>` -- which the manual calls
 * `ParticleType<N>`, so both are accepted. GADGET-4 moved the cosmology and
 * the unit system to `/Parameters`; older writers keep them in `/Header` or a
 * `/Units` group. Each is looked for in all of those places.
 */

#include "io/gadget_internal.h"

#include "sif/utils/logger.h"

#include <hdf5.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG "gadget"

struct sif_gadget_h5_file {
  hid_t file;
  /* "PartType" or "ParticleType". */
  const char* group_prefix;
  uint32_t n_types;
  uint64_t n_part_file[SIF_GADGET_MAX_TYPES];
};

/* HDF5 prints its error stack for every failed call, including the ones that
 * only ask whether something exists. Silenced for the duration of each entry
 * point, and put back as it was. */
typedef struct {
  H5E_auto2_t func;
  void* data;
} quiet_t;

static void quiet_begin(quiet_t* q) {
  H5Eget_auto2(H5E_DEFAULT, &q->func, &q->data);
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
}

static void quiet_end(const quiet_t* q) {
  H5Eset_auto2(H5E_DEFAULT, q->func, q->data);
}

static int link_exists(hid_t loc, const char* name) {
  return H5Lexists(loc, name, H5P_DEFAULT) > 0;
}

/*
 * A numeric attribute of up to @p cap elements, converted by HDF5 to
 * @p mem_type. A scalar and a one-element array read the same. Returns the
 * element count, or 0 if the attribute is missing, too long or not numeric.
 */
static size_t attr_array(
  hid_t loc, const char* name, hid_t mem_type, void* out, size_t cap) {
  if (loc < 0 || H5Aexists(loc, name) <= 0)
    return 0;

  hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
  hid_t space = attr >= 0 ? H5Aget_space(attr) : H5I_INVALID_HID;
  const hssize_t n = space >= 0 ? H5Sget_simple_extent_npoints(space) : -1;

  size_t got = 0;
  if (n >= 1 && (size_t)n <= cap && H5Aread(attr, mem_type, out) >= 0)
    got = (size_t)n;

  if (space >= 0)
    H5Sclose(space);
  if (attr >= 0)
    H5Aclose(attr);
  return got;
}

/* A scalar double, from the first group that has it. BoxSize may be a
 * three-vector in some writers; its first component is taken. */
static int attr_double_any(
  const hid_t* locs, int n_locs, const char* name, double* out) {
  double v[3];
  for (int i = 0; i < n_locs; i++) {
    if (attr_array(locs[i], name, H5T_NATIVE_DOUBLE, v, 3) > 0) {
      *out = v[0];
      return 1;
    }
  }
  return 0;
}

/* Adds every dataset in @p group to the header's block list, once. Walked
 * by index and tested by opening, rather than with H5Literate or
 * H5Oget_info, whose signatures changed between the HDF5 versions sif builds
 * against. */
static void add_datasets(hid_t group, sif_gadget_header_t* h) {
  H5G_info_t info;
  if (H5Gget_info(group, &info) < 0)
    return;

  for (hsize_t i = 0; i < info.nlinks; i++) {
    char name[32];
    const ssize_t len = H5Lget_name_by_idx(group, ".", H5_INDEX_NAME,
      H5_ITER_INC, i, name, sizeof(name), H5P_DEFAULT);
    if (len <= 0 || (size_t)len >= sizeof(name))
      continue;

    hid_t ds = H5Dopen2(group, name, H5P_DEFAULT);
    if (ds < 0)
      continue;
    H5Dclose(ds);

    const uint32_t shown =
      h->n_blocks < SIF_GADGET_MAX_BLOCKS ? h->n_blocks : SIF_GADGET_MAX_BLOCKS;
    bool seen = false;
    for (uint32_t b = 0; b < shown && !seen; b++)
      seen = strcmp(h->blocks[b], name) == 0;
    if (seen)
      continue;

    if (h->n_blocks < SIF_GADGET_MAX_BLOCKS)
      snprintf(h->blocks[h->n_blocks], sizeof(h->blocks[0]), "%s", name);
    h->n_blocks++;
  }
}

static void group_name(
  const sif_gadget_h5_file_t* f, uint32_t t, char* buf, size_t len) {
  snprintf(buf, len, "%s%u", f->group_prefix, t);
}

int sif__gadget_h5_open(
  const char* path, sif_gadget_h5_file_t** out_file, sif_gadget_header_t* h) {

  *out_file = NULL;
  quiet_t q;
  quiet_begin(&q);

  int status = SIF_ERR_IO;
  sif_gadget_h5_file_t* f = calloc(1, sizeof(*f));
  hid_t header = H5I_INVALID_HID, params = H5I_INVALID_HID;
  hid_t units = H5I_INVALID_HID;

  if (!f) {
    status = SIF_ERR_ALLOC;
    goto done;
  }
  f->file = H5I_INVALID_HID;

  f->file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f->file < 0) {
    SIF_LOG_ERROR(TAG, "could not open %s as an HDF5 file", path);
    goto done;
  }

  if (!link_exists(f->file, "Header")) {
    SIF_LOG_ERROR(TAG, "%s has no /Header: not a GADGET snapshot", path);
    goto done;
  }
  header = H5Gopen2(f->file, "Header", H5P_DEFAULT);
  if (link_exists(f->file, "Parameters"))
    params = H5Gopen2(f->file, "Parameters", H5P_DEFAULT);
  if (link_exists(f->file, "Units"))
    units = H5Gopen2(f->file, "Units", H5P_DEFAULT);

  /* --- counts --- */

  uint64_t high[SIF_GADGET_MAX_TYPES] = {0};
  const size_t n_types = attr_array(header, "NumPart_ThisFile",
    H5T_NATIVE_UINT64, h->n_part_file, SIF_GADGET_MAX_TYPES);
  if (n_types == 0 ||
      attr_array(header, "NumPart_Total", H5T_NATIVE_UINT64, h->n_part_total,
        SIF_GADGET_MAX_TYPES) != n_types ||
      attr_array(header, "MassTable", H5T_NATIVE_DOUBLE, h->mass_table,
        SIF_GADGET_MAX_TYPES) != n_types) {
    SIF_LOG_ERROR(TAG,
      "%s: /Header lacks NumPart_ThisFile, NumPart_Total or MassTable, or they "
      "disagree on the number of types",
      path);
    goto done;
  }

  /* Pre-GADGET-4 writers keep the totals in 32 bits and the high words
   * apart. */
  if (attr_array(header, "NumPart_Total_HighWord", H5T_NATIVE_UINT64, high,
        SIF_GADGET_MAX_TYPES) == n_types) {
    for (size_t t = 0; t < n_types; t++)
      h->n_part_total[t] += high[t] << 32;
  }
  h->n_types = (uint32_t)n_types;

  /* --- scalars --- */

  uint32_t n_files = 0;
  if (attr_array(
        header, "NumFilesPerSnapshot", H5T_NATIVE_UINT32, &n_files, 1) != 1 ||
      !attr_double_any(&header, 1, "Time", &h->time)) {
    SIF_LOG_ERROR(TAG, "%s: /Header lacks NumFilesPerSnapshot or Time", path);
    goto done;
  }
  h->n_files = n_files;
  attr_double_any(&header, 1, "Redshift", &h->redshift);
  attr_double_any(&header, 1, "BoxSize", &h->box_size);

  const hid_t cosmo_locs[2] = {header, params};
  h->has_cosmology =
    attr_double_any(cosmo_locs, 2, "Omega0", &h->omega0) &&
    attr_double_any(cosmo_locs, 2, "OmegaLambda", &h->omega_lambda) &&
    attr_double_any(cosmo_locs, 2, "HubbleParam", &h->hubble_param);

  const hid_t unit_locs[3] = {params, units, header};
  attr_double_any(unit_locs, 3, "UnitLength_in_cm", &h->unit_length_in_cm);

  /* --- particle groups --- */

  f->n_types = h->n_types;
  memcpy(f->n_part_file, h->n_part_file, sizeof(f->n_part_file));
  f->group_prefix = "PartType";
  for (uint32_t t = 0; t < h->n_types; t++) {
    char name[32];
    snprintf(name, sizeof(name), "ParticleType%u", t);
    if (link_exists(f->file, name)) {
      f->group_prefix = "ParticleType";
      break;
    }
  }

  for (uint32_t t = 0; t < h->n_types; t++) {
    char name[32];
    group_name(f, t, name, sizeof(name));
    if (!link_exists(f->file, name))
      continue;

    hid_t g = H5Gopen2(f->file, name, H5P_DEFAULT);
    if (g < 0)
      continue;
    add_datasets(g, h);

    if (h->precision == 0 && link_exists(g, "Coordinates")) {
      hid_t ds = H5Dopen2(g, "Coordinates", H5P_DEFAULT);
      hid_t ty = ds >= 0 ? H5Dget_type(ds) : H5I_INVALID_HID;
      if (ty >= 0)
        h->precision = (uint32_t)H5Tget_size(ty);
      if (ty >= 0)
        H5Tclose(ty);
      if (ds >= 0)
        H5Dclose(ds);
    }
    H5Gclose(g);
  }

  h->format = SIF_GADGET_FORMAT_HDF5;
  status = SIF_OK;

done:
  if (units >= 0)
    H5Gclose(units);
  if (params >= 0)
    H5Gclose(params);
  if (header >= 0)
    H5Gclose(header);
  quiet_end(&q);

  if (status == SIF_OK) {
    *out_file = f;
  } else {
    sif__gadget_h5_close(f);
  }
  return status;
}

int sif__gadget_h5_read(sif_gadget_h5_file_t* f, uint32_t ptype,
  sif_gadget_block_t block, uint64_t start, uint64_t count, double* out) {

  const char* dataset = block == SIF__GADGET_BLOCK_POS   ? "Coordinates"
                        : block == SIF__GADGET_BLOCK_VEL ? "Velocities"
                                                         : "Masses";
  const int comps = block == SIF__GADGET_BLOCK_MASS ? 1 : 3;

  char name[64];
  char gname[32];
  group_name(f, ptype, gname, sizeof(gname));
  snprintf(name, sizeof(name), "%s/%s", gname, dataset);

  quiet_t q;
  quiet_begin(&q);

  int status = SIF_ERR_IO;
  hid_t ds = H5I_INVALID_HID, fspace = H5I_INVALID_HID;
  hid_t mspace = H5I_INVALID_HID;

  if (!link_exists(f->file, gname) || !link_exists(f->file, name)) {
    SIF_LOG_ERROR(TAG, "the snapshot has no /%s", name);
    goto done;
  }

  ds = H5Dopen2(f->file, name, H5P_DEFAULT);
  fspace = ds >= 0 ? H5Dget_space(ds) : H5I_INVALID_HID;
  if (fspace < 0)
    goto done;

  hsize_t dims[2] = {0, 0};
  const int rank = H5Sget_simple_extent_ndims(fspace);
  if (rank < 1 || rank > 2 || H5Sget_simple_extent_dims(fspace, dims, NULL) < 0)
    goto done;

  const hsize_t width = rank == 2 ? dims[1] : 1;
  if ((int)width != comps || dims[0] != f->n_part_file[ptype] ||
      start + count > dims[0]) {
    SIF_LOG_ERROR(TAG, "/%s has shape (%llu, %llu), expected (%" PRIu64 ", %d)",
      name, (unsigned long long)dims[0], (unsigned long long)width,
      f->n_part_file[ptype], comps);
    goto done;
  }

  hsize_t off[2] = {start, 0};
  hsize_t cnt[2] = {count, width};
  hsize_t n_vals = count * width;
  mspace = H5Screate_simple(1, &n_vals, NULL);
  if (mspace < 0 ||
      H5Sselect_hyperslab(fspace, H5S_SELECT_SET, off, NULL, cnt, NULL) < 0 ||
      H5Dread(ds, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, out) < 0)
    goto done;

  status = SIF_OK;

done:
  if (mspace >= 0)
    H5Sclose(mspace);
  if (fspace >= 0)
    H5Sclose(fspace);
  if (ds >= 0)
    H5Dclose(ds);
  quiet_end(&q);
  return status;
}

void sif__gadget_h5_close(sif_gadget_h5_file_t* f) {
  if (!f)
    return;
  if (f->file >= 0)
    H5Fclose(f->file);
  free(f);
}

#undef TAG
