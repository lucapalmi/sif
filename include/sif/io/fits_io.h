/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file fits_io.h
 * @brief Reading FITS tables -- survey catalogues and their randoms -- into a
 * particle field.
 *
 * A FITS table names its columns, so the reader is told which column plays
 * which part: a sif_field_columns_t maps the field's positions -- x, y and z,
 * or on the sky ra, dec and z -- velocities and weight onto names in the
 * file. Each entry is a column name or, when a
 * single column will not do, an arithmetic expression over columns in
 * cfitsio's row-filter syntax -- which is how a weight built from several
 * columns is read:
 *
 * @code
 * const sif_field_columns_t cols = {
 *   .ra = "RA", .dec = "DEC", .z = "Z",
 *   .w = "WEIGHT_SYSTOT * (WEIGHT_NOZ + WEIGHT_CP - 1)",
 * };
 * const char* files[] = {"galaxies_NGC.fits", "galaxies_SGC.fits"};
 * sif_field_t* field =
 *   sif_field_read_fits(files, 2, NULL, &cols, "Z > 0.43 && Z < 0.7", 1.0, 0);
 * @endcode
 *
 * Column types are converted on the way in, so a table of 32-bit floats,
 * 64-bit doubles or integers reads the same; string, logical and complex
 * columns are refused. A vector column -- a `POS` holding three values a row
 * -- is read an element at a time, by expression: `.x = "POS[1]"`, `.y =
 * "POS[2]"`, `.z = "POS[3]"`. Undefined values (TNULL, or a NaN in a float
 * column) in any row the filter keeps fail the read, naming the column and
 * the row -- the filter is where such rows are dropped, with `!ISNULL(Z)`.
 *
 * Paths are taken literally: cfitsio's extended filename syntax
 * (`file.fits[1][Z > 0.4]`) is not interpreted, since what it does -- copy the
 * filtered table into memory -- is what the @p hdu and @p where arguments
 * exist to avoid. Compressed files (`.fits.gz`) are read, uncompressed into
 * memory as they are opened; for a large catalogue that costs its size in
 * memory, and an uncompressed copy reads faster.
 *
 * sif_fits_inspect() lists what a file holds -- its HDUs, and each table's
 * columns with their types and units -- for finding the names to read.
 *
 * Needs a build with cfitsio (SIF_FITS_SUPPORT); without one the reader
 * returns NULL, and says so in the log.
 */

#ifndef SIF_IO_FITS_IO_H
#define SIF_IO_FITS_IO_H

#include "sif/core/macros.h"
#include "sif/io/field_io.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/field.h"
#include "sif/structures/size_function.h"

#include <stdint.h>
#include <stdio.h>

/* The columns are named with sif_field_columns_t (field_io.h): here each
 * entry is a column name, matched without regard to case, or an arithmetic
 * expression over columns in cfitsio's syntax. */

/**
 * @brief Read a catalogue of one or more FITS tables into a new field.
 *
 * A catalogue split over several files -- the two galactic caps, randoms in
 * numbered parts -- reads as one table, the files' rows one after another in
 * the order given. Every file must have the table @p hdu names and every
 * column the sources use, and is checked for them before anything is read.
 *
 * The tables are streamed in chunks of rows, so memory is the field plus a
 * buffer of a few MB, and one bit per row when there is a filter. Row order
 * is kept.
 *
 * **Filtering** keeps the rows for which @p where, a boolean expression in
 * cfitsio's row-filter syntax, is true. It is evaluated once, before
 * anything is read, so the field is allocated once at its final size.
 *
 * **Subsampling** keeps exactly round(fraction * N) of the N rows the filter
 * passes, over all the files together, as sif_field_read_gadget() does: every
 * subset of that size equally likely, and the same seed and files giving the
 * same rows. Filtering first and then subsampling is what keeps the data and
 * its randoms cut identically, whatever fraction of each is kept.
 *
 * @param paths The files, taken literally, in the order their rows are
 * wanted.
 * @param n_paths How many; at least 1.
 * @param hdu The table to read in every file: its EXTNAME, or its extension
 * number in decimal (1 for the first extension, the primary HDU being 0).
 * NULL reads the first table in each file.
 * @param columns Which columns fill the field, and by their names whether it
 * holds positions or sky coordinates.
 * @param where Row filter, or NULL to keep every row.
 * @param fraction Share of the filtered rows to keep, in (0, 1]; 1 keeps all
 * and draws nothing.
 * @param seed Seed for the subsample; ignored when @p fraction is 1.
 * @return The field, owned by the caller and released with sif_field_free().
 * NULL on any failure, with the reason in the log: a NULL argument or no
 * paths, a missing position, x y named with ra dec, a velocity named only in
 * part, a fraction outside (0, 1] or one that keeps nothing, a column or HDU
 * the file does not have, an expression that does not parse or gives the wrong
 * type, an undefined value in a kept row, a file that cannot be opened or read,
 * an allocation failure, or a build without cfitsio.
 */
SIF_NODISCARD sif_field_t* sif_field_read_fits(const char* const* paths,
  uint32_t n_paths, const char* hdu, const sif_field_columns_t* columns,
  const char* where, double fraction, uint64_t seed);

/**
 * @brief Print what a FITS file holds: every HDU, and for each table its row
 * count and its columns, with their FITS type (TFORM), their unit, and a note
 * for a vector column or one the reader cannot use.
 *
 * Written to a stream rather than through the logger, as
 * sif_gadget_print_header() is: it is the answer to a question, not a
 * diagnostic.
 *
 * @param path The file, taken literally.
 * @param stream Where to print, e.g. stdout.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument; SIF_ERR_IO for a file
 * that cannot be opened or read; SIF_ERR_UNSUPPORTED in a build without
 * cfitsio.
 */
int sif_fits_print_summary(const char* path, FILE* stream);

/**
 * @brief sif_fits_print_summary() to stdout, for a look at a file before
 * deciding how to read it.
 *
 * @return As sif_fits_print_summary().
 */
int sif_fits_inspect(const char* path);

/**
 * @defgroup fits_products Catalogues and their products
 * @brief A catalogue and what was measured from it, in one FITS file, as the
 * HDF5 file holds them:
 *
 * @code
 * [0] primary           SIFFMT = 'sif', SIFVER; nothing else sif writes
 * [.] VOIDS             a row per void; COORDS, and the catalogue's metadata
 * [.] DENSITY_PROFILES  a row per void, DENSITY[n_bins]; N_BINS, EXT,
 *                       DIFFERENTIAL, EDGE0 ... EDGE<n_bins>
 * [.] VELOCITY_PROFILES the same, V_RAD[n_bins]
 * [.] SIZE_FUNCTION     a row per bin: R_LOW, R_HIGH, R_CENTRE, COUNT, VSF,
 *                       ERR; N_BINS, R_MIN, R_MAX, OPTIONS, BINNING
 * @endcode
 *
 * **Every writer creates the file if it is missing and replaces only its own
 * HDU**, so a file can hold any subset of the products, and a catalogue can
 * be rewritten without losing what was measured from it. A product whose row
 * count disagrees with the catalogue's is warned about, never refused. An
 * existing file sif did not write -- no SIFFMT = 'sif' in its primary header
 * -- is never written into. The primary header is otherwise the caller's:
 * sif_fits_set_key_string() and its kin write there by default.
 * @{
 */

/**
 * @brief Write a catalogue into the VOIDS table, a row per void.
 *
 * The columns follow what the catalogue holds, with the names the ASCII
 * format gives them: CX, CY, CZ and R for Cartesian centres; RA and DEC (in
 * degrees), Z (the redshift) and R (comoving, Mpc/h) for a catalogue on the
 * sky (sif_catalogue_to_sky()). The footprint columns, FOOTPRINT and
 * FOOTPRINT_SHELL, follow when the catalogue carries them. Values are written
 * at the precision sif was built with, and the table's COORDS keyword says
 * "cartesian" or "sky". The catalogue's metadata (the catalogue_meta group)
 * are keywords of the same header, and travel with it.
 *
 * @param filepath The file: created if missing, its VOIDS table replaced if
 * it has one.
 * @param catalogue Catalogue to write.
 * @return SIF_OK; SIF_ERR_INVALID on a NULL argument; SIF_ERR_IO if the file
 * could not be written, or is a FITS file sif did not write;
 * SIF_ERR_UNSUPPORTED in a build without cfitsio.
 */
int sif_catalogue_write_fits(const char* filepath, const sif_catalogue_t* catalogue);

/**
 * @brief Read a catalogue written by sif_catalogue_write_fits(), or any table
 * named VOIDS with the same columns.
 *
 * The centres' column names say what they are: RA, DEC and Z give a
 * catalogue on the sky, CX, CY and CZ (or X, Y and Z) a Cartesian one; R or
 * RADIUS is the radius. The footprint comes back when the table has both of
 * its columns, and every keyword of the table's header that is not FITS
 * structure comes back as the catalogue's metadata, in lower case.
 *
 * @param filepath The file.
 * @return The catalogue, owned by the caller and released with
 * sif_catalogue_free(); NULL on failure, with the reason in the log.
 */
SIF_NODISCARD sif_catalogue_t* sif_catalogue_read_fits(const char* filepath);

/**
 * @brief Write stacked profiles into DENSITY_PROFILES and VELOCITY_PROFILES.
 *
 * Either set may be NULL, and a NULL one leaves its table in the file as it
 * was. Bin edges are keywords of each table, so a set holds at most 9999
 * bins.
 *
 * @return SIF_OK; SIF_ERR_INVALID if both are NULL, or for a set of too many
 * bins; SIF_ERR_IO as for sif_catalogue_write_fits().
 */
int sif_profiles_write_fits(const char* filepath,
  const sif_density_profiles_t* dens, const sif_velocity_profiles_t* vel);

/**
 * @brief Which profile sets a file holds, without reading them.
 * @return SIF_OK; SIF_ERR_IO if the file cannot be opened.
 */
SIF_NODISCARD int sif_profiles_read_header_fits(
  const char* filepath, int* out_has_density, int* out_has_velocity);

/**
 * @brief Read stacked profiles, as sif_profiles_read_hdf5() does: every
 * output optional, a set the file does not hold an error.
 * @return SIF_OK; SIF_ERR_INVALID for a set the file does not hold;
 * SIF_ERR_IO for a missing or malformed file; SIF_ERR_ALLOC.
 */
SIF_NODISCARD int sif_profiles_read_fits(const char* filepath,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel);

/**
 * @brief Write a size function into SIZE_FUNCTION, replacing it if the file
 * has one.
 * @return As sif_catalogue_write_fits().
 */
int sif_size_function_write_fits(
  const char* filepath, const sif_size_function_t* vsf);

/**
 * @brief Read SIZE_FUNCTION.
 * @return The size function, released with sif_size_function_free(); NULL if
 * the file or the table is missing or malformed, or this build has no
 * cfitsio.
 */
SIF_NODISCARD sif_size_function_t* sif_size_function_read_fits(
  const char* filepath);

/** @} */

/**
 * @defgroup fits_keys Header keywords
 * @brief Single values from a FITS header, read and written by name.
 *
 * A convenience, for what the tables do not hold: a mock's box size, the
 * cosmology a catalogue was made in, the settings of the run that wrote it.
 * Each call opens the file, does its one thing and closes it.
 *
 * `hdu` is NULL (or "") for the **primary header**, where a file keeps what
 * describes it as a whole; otherwise an extension number in decimal, or an
 * EXTNAME. Note the difference from sif_field_read_fits(), for which NULL is
 * the first table.
 *
 * Names are matched without regard to case. A name longer than eight
 * characters is a HIERARCH keyword, which cfitsio writes and finds on its own.
 *
 * The getters return a value and nothing else: 0 or an empty string for a
 * keyword that is not there, as for a file that cannot be read -- which is
 * logged, so the two can be told apart. sif_fits_key_kind() says whether a
 * keyword is there at all.
 * @{
 */

/** @brief What kind of value a keyword holds. */
typedef enum {
  /** No such keyword, or one with no value. */
  SIF_FITS_KEY_MISSING = 0,
  SIF_FITS_KEY_INT,
  SIF_FITS_KEY_REAL,
  SIF_FITS_KEY_STRING,
  /** T or F. */
  SIF_FITS_KEY_LOGICAL
} sif_fits_key_kind_t;

/** @brief What kind of value a keyword holds, or #SIF_FITS_KEY_MISSING. */
sif_fits_key_kind_t sif_fits_key_kind(
  const char* path, const char* hdu, const char* key);

/**
 * @brief A numeric keyword's value: an integer, a real, or a logical as 1 or
 * 0. 0 for a keyword that is not there, and for one that holds text, which
 * is warned about.
 */
double sif_fits_get_key_real(
  const char* path, const char* hdu, const char* key);

/**
 * @brief A keyword's value as text, into @p buf: a string without its quotes,
 * and a number or a logical as it is written ("1.5", "T"). An empty string
 * for a keyword that is not there. Truncated to fit @p len.
 */
void sif_fits_get_key_string(
  const char* path, const char* hdu, const char* key, char* buf, size_t len);

/**
 * @brief Write an integer keyword, replacing one of the same name.
 * @param path The file, which has to exist.
 * @param hdu As for the getters: NULL for the primary header.
 * @param key The keyword's name.
 * @param value The value.
 * @return SIF_OK; SIF_ERR_INVALID for a missing HDU or a name FITS does not
 * allow; SIF_ERR_IO if the file could not be written; SIF_ERR_UNSUPPORTED in a
 * build without cfitsio.
 */
int sif_fits_set_key_int(
  const char* path, const char* hdu, const char* key, int64_t value);

/** @brief Write a real keyword; see sif_fits_set_key_int(). */
int sif_fits_set_key_real(
  const char* path, const char* hdu, const char* key, double value);

/** @brief Write a string keyword; see sif_fits_set_key_int(). */
int sif_fits_set_key_string(
  const char* path, const char* hdu, const char* key, const char* value);

/** @} */

#endif /* SIF_IO_FITS_IO_H */
