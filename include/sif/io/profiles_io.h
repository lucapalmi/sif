/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file profiles_io.h
 * @brief Reading and writing stacked profiles, in plain text.
 *
 * Text for the same reason a catalogue is: a profile set is small next to the
 * field it was measured from, and it is the thing that gets plotted and handed
 * to other tools.
 *
 * One row per void, holding the void it belongs to and then its profile, so a
 * row is self-contained and the file needs nothing else to be read:
 *
 *     n_voids n_bins ext has_density has_velocity differential
 *     r_edge[0] r_edge[1] ... r_edge[n_bins]
 *     cx cy cz radius  density[0..n_bins-1]  v_rad[0..n_bins-1]
 *     ...
 *
 * The two leading lines are the only ragged ones, so the table proper loads
 * with `numpy.loadtxt(path, skiprows=2)` and comes out as one row per void.
 * Bin edges are in units of each void's own radius, which is the axis the
 * profiles are binned on; multiply by the radius in the row for physical
 * units. Either profile block is absent when the file was written without it,
 * and `differential` says whether a density bin holds its own shell or
 * everything enclosed -- the values do not say which.
 *
 * That flag also decides which radius a density column belongs at: a
 * cumulative bin is everything within `r_edge[i + 1]`, a differential bin is a
 * shell and belongs at the midpoint of its two edges. Reading a cumulative
 * profile at bin centres shifts it by half a bin, which is enough to move a
 * feature at r = R_v off that mark.
 */

#ifndef SIF_IO_PROFILES_IO_H
#define SIF_IO_PROFILES_IO_H

#include "sif/core/macros.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalog.h"

/**
 * @brief Write one or both profile sets to a text file.
 *
 * The void metadata comes from @p cat, which is why it is needed here: the
 * sets themselves hold rows, not the voids the rows belong to. Row i of a set
 * is void i of the catalogue it was measured from, so it has to be that
 * catalogue.
 *
 * @param dens Density set, or NULL to leave densities out.
 * @param vel Velocity set, or NULL to leave velocities out. At least one of
 * the two is required, and two given together must agree on their shape.
 * @param cat The catalogue the sets were measured from.
 * @param filepath Path to the output file, truncated if it exists.
 * @return SIF_OK, SIF_ERR_INVALID on a NULL or mismatched argument, or
 * SIF_ERR_IO if the file could not be written.
 *
 * @note Values are written with #SIF_PRI_REAL, which carries enough
 * significant digits to recover the stored sif_real exactly, so a profile
 * survives a write/read round trip unchanged.
 */
int sif_profiles_write_ascii(const sif_density_profiles_t* dens,
  const sif_velocity_profiles_t* vel, const sif_catalog_t* cat,
  const char* filepath);

/**
 * @brief Read the leading line of a profile file, without the rows.
 *
 * What the file holds, so a caller can size its own work or ask
 * sif_profiles_read_ascii() only for blocks that are actually there.
 *
 * @param filepath Path to the input file.
 * @param out_n_voids,out_n_bins,out_ext Shape of the set, or NULL to skip
 * any of them.
 * @param out_has_density,out_has_velocity Non-zero if the file carries that
 * block, or NULL to skip.
 * @param out_differential Non-zero if the density bins hold shells rather than
 * enclosed volumes, or NULL to skip.
 * @return SIF_OK, or SIF_ERR_INVALID if the file cannot be opened or its
 * first line does not parse.
 */
SIF_NODISCARD int sif_profiles_read_header_ascii(const char* filepath,
  uint64_t* out_n_voids, uint32_t* out_n_bins, sif_real* out_ext,
  int* out_has_density, int* out_has_velocity, int* out_differential);

/**
 * @brief Read what sif_profiles_write_ascii() wrote.
 *
 * Every output is optional and follows the same convention as
 * sif_profiles(): pass NULL for anything not wanted, and it is skipped rather
 * than read and thrown away. Asking for a set the file does not carry is an
 * error, not an empty result.
 *
 * @param filepath Path to the input file.
 * @param out_cat Address of a catalogue pointer for the voids the rows
 * describe, or NULL to skip.
 * @param out_dens Address of a density set pointer, or NULL to skip.
 * @param out_vel Address of a velocity set pointer, or NULL to skip.
 * @return SIF_OK, SIF_ERR_INVALID for a bad or truncated file or a request
 * the file cannot satisfy, or SIF_ERR_ALLOC.
 *
 * @note Every output is allocated fresh -- unlike sif_profiles(), this does
 * not refill a set the caller already owns, so a pointer to a live set is
 * overwritten rather than reused.
 *
 * @note On any failure every output is left NULL, including one this call
 * built before a later row failed to parse.
 */
SIF_NODISCARD int sif_profiles_read_ascii(const char* filepath,
  sif_catalog_t** out_cat, sif_density_profiles_t** out_dens,
  sif_velocity_profiles_t** out_vel);

#endif /* SIF_IO_PROFILES_IO_H */
