/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file hdf5_io.h
 * @brief A catalogue and its products, in one HDF5 file; and particles out of
 * any HDF5 file.
 *
 * HDF5 because it is a standard: the file reads back with h5py, or any other
 * HDF5 reader, without sif installed. One file holds one catalogue and what
 * was measured from it, each in a group of its own:
 *
 * @code
 * /                     sif_format = "sif", sif_format_version, sif_version
 * /catalogue              n_voids, coordinates ("cartesian" or "sky");
 *                       centres (N, 3), its columns named by its
 *                       attribute columns ("cx cy cz" or "ra dec z");
 *                       radii (N); footprint and footprint_shell (N)
 *                       when present
 * /density_profiles     n_voids, n_bins, ext, differential;
 *                       r_edges (n_bins + 1), profiles (N, n_bins)
 * /velocity_profiles    n_voids, n_bins, ext; r_edges, v_rad (N, n_bins)
 * /size_function        n_bins, r_min, r_max, binning ("ln" or "linear"),
 *                       options; r_edges, r_centres, counts, vsf, err
 * @endcode
 *
 * Scalars describing a product are attributes of its group, so the file says
 * what it holds: `f["size_function"].attrs["binning"]` in h5py. Arrays are
 * stored at the precision sif was built with -- each dataset carries its own
 * type -- and read back at whatever precision the reading build uses.
 *
 * **Every writer creates the file if it is missing and replaces only its own
 * group** -- so a file can hold any subset of the products, a catalogue can be
 * rewritten without losing what was measured from it, and a product can be
 * saved on its own. Keeping the products consistent with the catalogue is the
 * caller's business: a row count that disagrees with the catalogue already in
 * the file is warned about, never refused. An existing file that is not a sif
 * HDF5 file is never written into.
 *
 * A replaced group's old data is not reclaimed -- HDF5 does not shrink a file
 * -- so a file rewritten many times grows. `h5repack` compacts it.
 *
 * **In a build without HDF5** (see SIF_HDF5_SUPPORT) every function here still
 * exists. A writer then dumps the product as plain text next to the requested
 * path, at `<path>.<group>.txt`, logs a warning naming that file and returns
 * SIF_OK: the run that produced the data does not lose it to a build option.
 * A reader returns #SIF_ERR_UNSUPPORTED (or NULL), since there is nothing to
 * fall back to. sif_init() logs which kind of build this is.
 */

#ifndef SIF_IO_HDF5_IO_H
#define SIF_IO_HDF5_IO_H

#include "sif/core/macros.h"
#include "sif/io/field_io.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/size_function.h"

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Write the catalogue into `/catalogue`, replacing it if the file has
 * one.
 *
 * The footprint columns are written when the catalogue carries them, and an
 * old footprint is removed when it does not, so the group always matches the
 * catalogue just written. The `coordinates` attribute says what the centres
 * are: for a sky catalogue (sif_catalogue_to_sky()) each row of `centres` is
 * right ascension, declination and redshift.
 *
 * @param filepath The file, created if missing.
 * @param catalogue Catalogue to write.
 * @return SIF_OK; SIF_ERR_INVALID on a NULL argument; SIF_ERR_IO if the file
 * could not be written, or exists and is not a sif HDF5 file.
 */
int sif_catalogue_write_hdf5(const char* filepath, const sif_catalogue_t* catalogue);

/**
 * @brief Read `/catalogue`, footprint included when the file has one, and
 * Cartesian or on the sky as the file says -- Cartesian for a file that does
 * not say, written before catalogues could be anything else.
 *
 * @param filepath The file.
 * @return The catalogue, owned by the caller and released with
 * sif_catalogue_free(); NULL if the file, or the group, is missing or malformed,
 * or this build has no HDF5.
 */
SIF_NODISCARD sif_catalogue_t* sif_catalogue_read_hdf5(const char* filepath);

/**
 * @brief Write stacked profiles into `/density_profiles` and
 * `/velocity_profiles`.
 *
 * Either set may be NULL, and a NULL one leaves its group in the file as it
 * was: writing the densities does not remove velocities written earlier.
 *
 * @param filepath The file, created if missing.
 * @param dens Density profiles, or NULL.
 * @param vel Velocity profiles, or NULL.
 * @return SIF_OK; SIF_ERR_INVALID if both are NULL; SIF_ERR_IO as for
 * sif_catalogue_write_hdf5().
 */
int sif_profiles_write_hdf5(const char* filepath,
  const sif_density_profiles_t* dens, const sif_velocity_profiles_t* vel);

/**
 * @brief Which profile sets a file holds, without reading them.
 *
 * So a caller can ask sif_profiles_read_hdf5() only for what is there, the
 * way sif_profiles_read_header_ascii() does for a text file.
 *
 * @param filepath The file.
 * @param out_has_density,out_has_velocity Set to 1 if the file holds that
 * set and 0 if not, or NULL to skip.
 * @return SIF_OK; SIF_ERR_IO if the file is missing or not a sif HDF5 file;
 * SIF_ERR_UNSUPPORTED in a build without HDF5.
 */
SIF_NODISCARD int sif_profiles_read_header_hdf5(
  const char* filepath, int* out_has_density, int* out_has_velocity);

/**
 * @brief Read stacked profiles.
 *
 * Every output is optional -- pass NULL for a set that is not wanted -- and
 * asking for one the file does not hold is an error, not an empty result.
 *
 * @param filepath The file.
 * @param out_dens Address of a density set pointer, or NULL to skip.
 * @param out_vel Address of a velocity set pointer, or NULL to skip.
 * @return SIF_OK; SIF_ERR_INVALID for a request the file cannot satisfy;
 * SIF_ERR_IO for a missing or malformed file; SIF_ERR_ALLOC;
 * SIF_ERR_UNSUPPORTED in a build without HDF5.
 *
 * @note Every output is allocated fresh, and on any failure every output is
 * left NULL.
 */
SIF_NODISCARD int sif_profiles_read_hdf5(const char* filepath,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel);

/**
 * @brief Write a size function into `/size_function`, replacing it if the
 * file has one.
 *
 * @param filepath The file, created if missing.
 * @param vsf Size function to write, measured or modelled.
 * @return SIF_OK; SIF_ERR_INVALID on a NULL argument; SIF_ERR_IO as for
 * sif_catalogue_write_hdf5().
 */
int sif_size_function_write_hdf5(
  const char* filepath, const sif_size_function_t* vsf);

/**
 * @brief Read `/size_function`.
 *
 * @param filepath The file.
 * @return The size function, released with sif_size_function_free(); NULL if
 * the file or the group is missing or malformed, or this build has no HDF5.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_read_hdf5(
  const char* filepath);

/**
 * @brief Read particles out of any HDF5 file -- a simulation snapshot, a halo
 * catalogue, a mock -- into a new field.
 *
 * Each part of the field is a dataset, named by its path from the root:
 * `"Coordinates"`, or `"PartType1/Masses"`. A dataset of two dimensions, a
 * row per particle, is read a column at a time: `"PartType1/Coordinates[0]"`
 * is its first column, counted from 0 as h5py and numpy count. Any numeric
 * type is read, converted by HDF5. So the dark matter of an IllustrisTNG
 * snapshot, in Mpc/h:
 *
 * @code
 * const sif_field_columns_t cols = {.x = "PartType1/Coordinates[0]",
 *   .y = "PartType1/Coordinates[1]", .z = "PartType1/Coordinates[2]"};
 * const char* files[] = {"snap_099.0.hdf5", "snap_099.1.hdf5"};
 * sif_field_t* f = sif_field_read_hdf5(files, 2, &cols, 1e-3, 1.0, 0);
 * @endcode
 *
 * Several files read as one, their rows one after another in the order
 * given; every dataset has to be in every file, with the same row count as
 * the others there, and each file is checked for that before anything is
 * read. A GADGET snapshot has its own reader, sif_field_read_gadget(), which
 * also finds its files and units by itself.
 *
 * **Subsampling** keeps exactly round(fraction * N) of the N rows, over all
 * the files together, as sif_field_read_fits() does. A value that is not
 * finite in any row fails the read, naming the dataset and the row.
 *
 * Opens any HDF5 file, not only one sif wrote; needs a build with HDF5.
 *
 * @param paths The files, in the order their rows are wanted.
 * @param n_paths How many; at least 1.
 * @param columns Which dataset fills which part of the field, and by their
 * names whether it holds positions or sky coordinates.
 * @param length_scale What Cartesian positions are multiplied by, to bring
 * them to the units wanted: 1e-3 from kpc/h to Mpc/h. Velocities and weights
 * are read as they are. 1 for sky coordinates.
 * @param fraction Share of the rows to keep, in (0, 1].
 * @param seed Seed for the subsample; ignored when @p fraction is 1.
 * @return The field, owned by the caller and released with sif_field_free().
 * NULL on any failure, with the reason in the log: an argument out of range, a
 * dataset a file does not have or cannot give (not numeric, more than two
 * dimensions, a column it does not have), row counts that disagree, a value
 * not finite, a file that cannot be read, or a build without HDF5.
 */
SIF_NODISCARD sif_field_t* sif_field_read_hdf5(const char* const* paths,
  uint32_t n_paths, const sif_field_columns_t* columns, double length_scale,
  double fraction, uint64_t seed);

/**
 * @defgroup hdf5_attrs Free-form metadata
 * @brief Named scalar values attached to the file or to one of its products.
 *
 * What the arrays cannot say -- how the catalogue was made, which simulation
 * it came from, what the run was for -- as attributes, the way a FITS header
 * carries keywords. On the root (`group` NULL, "" or "/") they describe the
 * file and stay put whatever is rewritten. On a product's group ("catalogue",
 * "size_function", ...) they describe that product, and go with it when it is
 * rewritten: a new catalogue is not described by the old one's threshold.
 *
 * Values are 64-bit integers, doubles or strings. The attributes sif itself
 * reads (`n_voids`, `n_bins`, `sif_format`, ...) cannot be set here, and on
 * the root every name beginning `sif_` is kept for the library. Anything else
 * may be overwritten freely. Everything reads back with h5py as
 * `f.attrs[key]` or `f[group].attrs[key]`.
 *
 * In a build without HDF5 the setters append `group key = value` lines to
 * `<path>.attributes.txt` and warn, as the product writers do; the getters
 * return #SIF_ERR_UNSUPPORTED.
 * @{
 */

/** @brief What kind of value a metadata entry holds. */
typedef enum {
  /** A signed or unsigned integer, read with sif_hdf5_get_attr_int(). */
  SIF_HDF5_ATTR_INT = 1,
  /** A floating-point number, read with sif_hdf5_get_attr_real(). */
  SIF_HDF5_ATTR_REAL,
  /** A string, read with sif_hdf5_get_attr_string(). */
  SIF_HDF5_ATTR_STRING,
  /** Anything else -- an array, a compound -- that another tool put there
   * and these functions do not read. */
  SIF_HDF5_ATTR_OTHER
} sif_hdf5_attr_kind_t;

/**
 * @brief Set an integer entry.
 *
 * @param filepath The file. For the root it is created if missing; a product
 * group has to exist already -- write the product first.
 * @param group NULL, "" or "/" for the file itself, or a product's group.
 * @param key The entry's name.
 * @param value The value.
 * @return SIF_OK; SIF_ERR_INVALID for a missing group or a reserved key;
 * SIF_ERR_IO if the file could not be written, or is not a sif HDF5 file.
 */
int sif_hdf5_set_attr_int(
  const char* filepath, const char* group, const char* key, int64_t value);

/** @brief Set a real entry; see sif_hdf5_set_attr_int(). */
int sif_hdf5_set_attr_real(
  const char* filepath, const char* group, const char* key, double value);

/** @brief Set a string entry; see sif_hdf5_set_attr_int(). */
int sif_hdf5_set_attr_string(
  const char* filepath, const char* group, const char* key, const char* value);

/**
 * @brief Read an integer entry.
 * @return SIF_OK; SIF_ERR_INVALID if the entry is missing or not an integer;
 * SIF_ERR_IO for a missing or foreign file; SIF_ERR_UNSUPPORTED without HDF5.
 */
SIF_NODISCARD int sif_hdf5_get_attr_int(
  const char* filepath, const char* group, const char* key, int64_t* out);

/**
 * @brief Read a real entry. An integer entry is read too, as its value.
 * @return As sif_hdf5_get_attr_int().
 */
SIF_NODISCARD int sif_hdf5_get_attr_real(
  const char* filepath, const char* group, const char* key, double* out);

/**
 * @brief Read a string entry into @p buf.
 * @return As sif_hdf5_get_attr_int(), and SIF_ERR_RANGE if the string did not
 * fit -- in which case @p buf holds as much of it as did.
 */
SIF_NODISCARD int sif_hdf5_get_attr_string(const char* filepath,
  const char* group, const char* key, char* buf, size_t len);

/**
 * @brief What kind of value an entry holds.
 * @return SIF_OK; SIF_ERR_INVALID if there is no such entry; SIF_ERR_IO;
 * SIF_ERR_UNSUPPORTED.
 */
SIF_NODISCARD int sif_hdf5_attr_kind(const char* filepath, const char* group,
  const char* key, sif_hdf5_attr_kind_t* out);

/**
 * @brief How many entries a group carries, sif's own included.
 * @return SIF_OK; SIF_ERR_INVALID for a missing group; SIF_ERR_IO;
 * SIF_ERR_UNSUPPORTED.
 */
SIF_NODISCARD int sif_hdf5_attr_count(
  const char* filepath, const char* group, uint32_t* out);

/**
 * @brief The name of entry @p index, in alphabetical order, so that
 * sif_hdf5_attr_count() and this list a whole header.
 * @return As sif_hdf5_get_attr_string(), with SIF_ERR_INVALID for an index
 * past the end.
 */
SIF_NODISCARD int sif_hdf5_attr_name(const char* filepath, const char* group,
  uint32_t index, char* buf, size_t len);

/** @} */

#endif /* SIF_IO_HDF5_IO_H */
