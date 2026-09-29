/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Stands in for fits_io.c in a build without cfitsio (SIF_FITS_SUPPORT=OFF,
 * or AUTO that found none). The reader exists, so code calling it compiles
 * and links the same either way, and refuses: it is called at the start of a
 * run, where failing costs nothing, and says why in the log.
 */

#include "sif/io/fits_io.h"

#include "io/fits_internal.h"
#include "sif/utils/logger.h"

#include <stdio.h>

int sif__fits_describe(char* buf, size_t len) {
  snprintf(buf, len, "no cfitsio");
  return 0;
}

static int refuse(const char* path) {
  SIF_LOG_ERROR("fits",
    "this build of sif has no FITS support, so %s cannot be used. Rebuild "
    "with -DSIF_FITS_SUPPORT=ON",
    path ? path : "(null)");
  return SIF_ERR_UNSUPPORTED;
}

sif_field_t* sif_field_read_fits(const char* const* paths, uint32_t n_paths,
  const char* hdu, const sif_field_columns_t* columns, const char* where,
  double fraction, uint64_t seed) {
  (void)hdu;
  (void)columns;
  (void)where;
  (void)fraction;
  (void)seed;
  refuse(paths && n_paths ? paths[0] : NULL);
  return NULL;
}

int sif_fits_print_summary(const char* path, FILE* stream) {
  (void)stream;
  return refuse(path);
}

int sif_fits_inspect(const char* path) { return refuse(path); }

int sif_catalogue_write_fits(const char* filepath, const sif_catalogue_t* catalogue) {
  (void)catalogue;
  return refuse(filepath);
}

sif_catalogue_t* sif_catalogue_read_fits(const char* filepath) {
  refuse(filepath);
  return NULL;
}

sif_fits_key_kind_t sif_fits_key_kind(
  const char* path, const char* hdu, const char* key) {
  (void)hdu;
  (void)key;
  refuse(path);
  return SIF_FITS_KEY_MISSING;
}

double sif_fits_get_key_real(
  const char* path, const char* hdu, const char* key) {
  (void)hdu;
  (void)key;
  refuse(path);
  return 0.0;
}

void sif_fits_get_key_string(
  const char* path, const char* hdu, const char* key, char* buf, size_t len) {
  (void)hdu;
  (void)key;
  if (buf && len)
    buf[0] = '\0';
  refuse(path);
}

int sif_fits_set_key_int(
  const char* path, const char* hdu, const char* key, int64_t value) {
  (void)hdu;
  (void)key;
  (void)value;
  return refuse(path);
}

int sif_fits_set_key_real(
  const char* path, const char* hdu, const char* key, double value) {
  (void)hdu;
  (void)key;
  (void)value;
  return refuse(path);
}

int sif_fits_set_key_string(
  const char* path, const char* hdu, const char* key, const char* value) {
  (void)hdu;
  (void)key;
  (void)value;
  return refuse(path);
}

int sif_profiles_write_fits(const char* filepath,
  const sif_density_profiles_t* dens, const sif_velocity_profiles_t* vel) {
  (void)dens;
  (void)vel;
  return refuse(filepath);
}

int sif_profiles_read_header_fits(
  const char* filepath, int* out_has_density, int* out_has_velocity) {
  (void)out_has_density;
  (void)out_has_velocity;
  return refuse(filepath);
}

int sif_profiles_read_fits(const char* filepath,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {
  if (out_dens)
    *out_dens = NULL;
  if (out_vel)
    *out_vel = NULL;
  return refuse(filepath);
}

int sif_size_function_write_fits(
  const char* filepath, const sif_size_function_t* vsf) {
  (void)vsf;
  return refuse(filepath);
}

sif_size_function_t* sif_size_function_read_fits(const char* filepath) {
  refuse(filepath);
  return NULL;
}
