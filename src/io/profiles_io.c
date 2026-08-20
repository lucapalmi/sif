/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/profiles_io.h"

#include "measure/profiles_internal.h"
#include "sif/utils/logger.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * The shape two sets have to agree on before they can share a file: they are
 * written as one row, so a row would otherwise mean two different things on
 * its two halves.
 */
static int shapes_agree(
  const sif_density_profiles_t* dens, const sif_velocity_profiles_t* vel) {

  if (!dens || !vel)
    return 1;

  return dens->n_voids == vel->n_voids && dens->n_bins == vel->n_bins &&
         dens->ext == vel->ext;
}

/*
 * The first line, which says what the rest of the file is. Left positioned at
 * the bin edges, so the reader below can carry straight on.
 */
static int read_header(FILE* file, const char* filepath, uint64_t* n_voids,
  uint32_t* n_bins, sif_real* ext, int* has_dens, int* has_vel,
  int* differential) {

  if (fscanf(file, "%" SCNu64 " %" SCNu32 " " SIF_SCN_REAL " %d %d %d", n_voids,
        n_bins, ext, has_dens, has_vel, differential) != 6) {
    SIF_LOG_ERROR("io", "failed to read the profile header from %s", filepath);
    return SIF_ERR_INVALID;
  }

  if (*n_voids == 0 || *n_bins == 0 || (!*has_dens && !*has_vel)) {
    SIF_LOG_ERROR("io",
      "%s describes an empty profile set (%" PRIu64 " voids, %" PRIu32
      " bins, density=%d velocity=%d)",
      filepath, *n_voids, *n_bins, *has_dens, *has_vel);
    return SIF_ERR_INVALID;
  }

  return SIF_OK;
}

int sif_profiles_read_header_ascii(const char* filepath, uint64_t* out_n_voids,
  uint32_t* out_n_bins, sif_real* out_ext, int* out_has_density,
  int* out_has_velocity, int* out_differential) {

  if (!filepath) {
    SIF_LOG_ERROR("io", "invalid filepath for sif_profiles_read_header_ascii");
    return SIF_ERR_INVALID;
  }

  FILE* file = fopen(filepath, "r");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for reading", filepath);
    return SIF_ERR_INVALID;
  }

  uint64_t n_voids = 0;
  uint32_t n_bins = 0;
  sif_real ext = 0;
  int has_dens = 0, has_vel = 0, differential = 0;

  const int status = read_header(file, filepath, &n_voids, &n_bins, &ext,
    &has_dens, &has_vel, &differential);
  fclose(file);

  if (status != SIF_OK)
    return status;

  if (out_n_voids)
    *out_n_voids = n_voids;
  if (out_n_bins)
    *out_n_bins = n_bins;
  if (out_ext)
    *out_ext = ext;
  if (out_has_density)
    *out_has_density = has_dens;
  if (out_has_velocity)
    *out_has_velocity = has_vel;
  if (out_differential)
    *out_differential = differential;

  return SIF_OK;
}

int sif_profiles_write_ascii(const sif_density_profiles_t* dens,
  const sif_velocity_profiles_t* vel, const sif_catalog_t* cat,
  const char* filepath) {

  if ((!dens && !vel) || !cat || !filepath) {
    SIF_LOG_ERROR("io",
      "sif_profiles_write_ascii needs a catalogue, a path and at least one "
      "profile set");
    return SIF_ERR_INVALID;
  }

  if (!shapes_agree(dens, vel)) {
    SIF_LOG_ERROR("io",
      "the density and velocity sets disagree on their shape and cannot share "
      "a file");
    return SIF_ERR_INVALID;
  }

  const uint64_t n_voids = dens ? dens->n_voids : vel->n_voids;
  const uint32_t n_bins = dens ? dens->n_bins : vel->n_bins;
  const sif_real ext = dens ? dens->ext : vel->ext;
  const sif_real* r_edges = dens ? dens->r_edges : vel->r_edges;

  /* Row i is void i. A catalogue of a different length is a different
   * catalogue, and writing it would label every row with the wrong void. */
  if (cat->n_voids != n_voids) {
    SIF_LOG_ERROR("io",
      "the catalogue holds %" PRIu64 " voids but the profiles hold %" PRIu64
      "; they are not the same measurement",
      cat->n_voids, n_voids);
    return SIF_ERR_INVALID;
  }

  FILE* file = fopen(filepath, "w");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for writing", filepath);
    return SIF_ERR_IO;
  }

  /* The shape goes first so the reader can size both the catalogue and the
   * rows once, instead of growing them or scanning the file twice. */
  fprintf(file, "%" PRIu64 " %" PRIu32 " " SIF_PRI_REAL " %d %d %d\n", n_voids,
    n_bins, ext, dens ? 1 : 0, vel ? 1 : 0,
    (dens && dens->differential) ? 1 : 0);

  /* Shared by every row, so written once. */
  for (uint32_t j = 0; j <= n_bins; j++)
    fprintf(file, "%s" SIF_PRI_REAL, j ? " " : "", r_edges[j]);
  fprintf(file, "\n");

  for (uint64_t i = 0; i < n_voids; i++) {
    fprintf(file,
      SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL,
      cat->cx[i], cat->cy[i], cat->cz[i], cat->radii[i]);

    if (dens) {
      const sif_real* row = sif_density_profiles_get(dens, i);
      for (uint32_t j = 0; j < n_bins; j++)
        fprintf(file, " " SIF_PRI_REAL, row[j]);
    }

    if (vel) {
      const sif_real* row = sif_velocity_profiles_get(vel, i);
      for (uint32_t j = 0; j < n_bins; j++)
        fprintf(file, " " SIF_PRI_REAL, row[j]);
    }

    fprintf(file, "\n");
  }

  /* fprintf() reports nothing useful per call, so the stream's error flag and
   * the flush inside fclose() are what say whether the file actually reached
   * the disk. Without this a full quota reads back as a short file. */
  const bool ok = (ferror(file) == 0);
  if (fclose(file) != 0 || !ok) {
    SIF_LOG_ERROR("io", "failed to flush %s to disk", filepath);
    return SIF_ERR_IO;
  }

  SIF_LOG_INFO("io", "saved %" PRIu64 " profiles (%s) to %s (ASCII)", n_voids,
    dens && vel ? "density and velocity" : (dens ? "density" : "velocity"),
    filepath);
  return SIF_OK;
}

int sif_profiles_read_ascii(const char* filepath, sif_catalog_t** out_cat,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {

  if (!filepath) {
    SIF_LOG_ERROR("io", "invalid filepath for sif_profiles_read_ascii");
    return SIF_ERR_INVALID;
  }

  /* Cleared up front so that every path out of here, including the ones that
   * give up before anything is allocated, leaves the caller with NULL rather
   * than with whatever the pointers held. */
  if (out_cat)
    *out_cat = NULL;
  if (out_dens)
    *out_dens = NULL;
  if (out_vel)
    *out_vel = NULL;

  FILE* file = fopen(filepath, "r");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for reading", filepath);
    return SIF_ERR_INVALID;
  }

  uint64_t n_voids = 0;
  uint32_t n_bins = 0;
  sif_real ext = 0;
  int has_dens = 0, has_vel = 0, differential = 0;

  if (read_header(file, filepath, &n_voids, &n_bins, &ext, &has_dens, &has_vel,
        &differential) != SIF_OK) {
    fclose(file);
    return SIF_ERR_INVALID;
  }

  /* Asked for something the file does not carry. Handing back an empty set
   * would be indistinguishable from a measurement of zeros. */
  if ((out_dens && !has_dens) || (out_vel && !has_vel)) {
    SIF_LOG_ERROR("io", "%s carries no %s profiles", filepath,
      (out_dens && !has_dens) ? "density" : "velocity");
    fclose(file);
    return SIF_ERR_INVALID;
  }

  int status = SIF_ERR_ALLOC;

  sif_catalog_t* cat = NULL;
  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;

  if (out_cat) {
    cat = sif_catalog_alloc(n_voids);
    if (!cat) {
      SIF_LOG_ERROR("io", "OOM allocating the catalogue for %s", filepath);
      goto fail;
    }
  }

  if (out_dens) {
    dens = sif__density_profiles_alloc(n_voids, n_bins, ext, differential != 0);
    if (!dens) {
      SIF_LOG_ERROR("io", "OOM allocating density profiles for %s", filepath);
      goto fail;
    }
  }

  if (out_vel) {
    vel = sif__velocity_profiles_alloc(n_voids, n_bins, ext);
    if (!vel) {
      SIF_LOG_ERROR("io", "OOM allocating velocity profiles for %s", filepath);
      goto fail;
    }
  }

  status = SIF_ERR_INVALID;

  /* The edges the file carries rather than ones recomputed from ext and
   * n_bins: what was measured is what the file says. */
  for (uint32_t j = 0; j <= n_bins; j++) {
    sif_real edge;
    if (fscanf(file, SIF_SCN_REAL, &edge) != 1) {
      SIF_LOG_ERROR(
        "io", "failed reading bin edge %" PRIu32 " from %s", j, filepath);
      goto fail;
    }
    if (dens)
      dens->r_edges[j] = edge;
    if (vel)
      vel->r_edges[j] = edge;
  }

  /* Rows are required to be there: the header said how many, and a file that
   * stops short is truncated rather than merely small. */
  for (uint64_t i = 0; i < n_voids; i++) {
    sif_real cx, cy, cz, radius;

    if (fscanf(file,
          SIF_SCN_REAL " " SIF_SCN_REAL " " SIF_SCN_REAL " " SIF_SCN_REAL, &cx,
          &cy, &cz, &radius) != 4) {
      SIF_LOG_ERROR(
        "io", "failed reading void %" PRIu64 " from %s", i, filepath);
      goto fail;
    }

    if (cat) {
      cat->cx[i] = cx;
      cat->cy[i] = cy;
      cat->cz[i] = cz;
      cat->radii[i] = radius;
    }

    /* Both blocks are consumed whether or not they were asked for: the
     * columns are there either way, and skipping them by parsing is what lets
     * a caller take only the half it wants. */
    for (int block = 0; block < 2; block++) {
      const int present = block == 0 ? has_dens : has_vel;
      if (!present)
        continue;

      sif_real* row = NULL;
      if (block == 0 && dens)
        row = &dens->profiles[i * n_bins];
      else if (block == 1 && vel)
        row = &vel->v_rad[i * n_bins];

      for (uint32_t j = 0; j < n_bins; j++) {
        sif_real value;
        if (fscanf(file, SIF_SCN_REAL, &value) != 1) {
          SIF_LOG_ERROR("io",
            "failed reading %s bin %" PRIu32 " of void %" PRIu64 " from %s",
            block == 0 ? "density" : "velocity", j, i, filepath);
          goto fail;
        }
        if (row)
          row[j] = value;
      }
    }
  }

  if (cat) {
    /* Filled in directly rather than through sif_catalog_append(), so the size
     * has to be set by hand. */
    cat->n_voids = n_voids;
    *out_cat = cat;
  }
  if (dens)
    *out_dens = dens;
  if (vel)
    *out_vel = vel;

  fclose(file);

  SIF_LOG_INFO(
    "io", "loaded %" PRIu64 " profiles from %s (ASCII)", n_voids, filepath);
  return SIF_OK;

fail:
  /* Nothing built here survives a failure -- a half-read set the caller cannot
   * tell from a complete one is worse than none at all. The outputs are
   * already NULL from the top of the call. */
  sif_catalog_free(cat);
  sif_density_profiles_free(dens);
  sif_velocity_profiles_free(vel);

  fclose(file);
  return status;
}
