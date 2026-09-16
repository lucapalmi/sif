/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file sdf.h
 * @brief The sif data format: opening and closing a `.sdf` file.
 *
 * An `.sdf` file holds the products of a run -- a void catalogue and the
 * things measured from it -- as a chain of self-describing blocks that can be
 * added to later without rewriting the file. The on-disk layout is specified
 * in the Formats section of the documentation; this header is the handle side
 * of it, and says nothing about what a block contains.
 *
 * A file is used through an opaque #sif_sdf_t, obtained from sif_sdf_open() or
 * sif_sdf_create() and released with sif_sdf_close(). The handle carries the
 * stream and the validated file header, so that reading a block is a call on
 * an already-validated file rather than a path and a fresh set of checks every
 * time.
 *
 * **Errors travel in a #sif_sdf_status_t the caller owns**, passed as the last
 * argument of every operation that can fail, so that the return value is
 * always the thing the call produces. Three rules go with it:
 *
 * - a call entered with an error already set does nothing and returns
 *   immediately, so a sequence needs one check at the end rather than one
 *   after each call;
 * - the first error wins, so the status that reaches that check is the one
 *   that says what went wrong;
 * - sif_sdf_close() runs regardless, because a file still has to be closed.
 *
 * .. code-block:: c
 *
 *    sif_sdf_status_t status = SIF_SDF_OK;
 *
 *    sif_sdf_t* file = sif_sdf_create("run.sdf", 1000.0, cat, 0, &status);
 *    // ... write blocks, none of which runs if an earlier one failed ...
 *    sif_sdf_close(file, &status);
 *
 *    if (status != SIF_SDF_OK)
 *      fprintf(stderr, "run.sdf: %s", sif_sdf_strerror(status));
 */

#ifndef SIF_IO_SDF_H
#define SIF_IO_SDF_H

#include "sif/core/macros.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalog.h"
#include "sif/structures/size_function.h"

#include <stdint.h>

/** @brief Magic number at the start of every `.sdf` file. */
#define SIF_SDF_MAGIC "SIFD"

/** @brief Version of the container layout this build reads and writes.
 *
 * 2 since the headers carry checksums of their own. Version 1 is not read:
 * its block headers are laid out differently and nothing stands behind
 * either header, so a version-1 file is refused rather than read on trust.
 */
#define SIF_SDF_VERSION 2

/**
 * @brief Byte-order sentinel, written natively and compared as an integer.
 *
 * A file whose sentinel does not read back as this was written on a machine of
 * the other endianness. The reader refuses it rather than swapping it: the
 * payloads are raw arrays, and swapping them would be a second format.
 */
#define SIF_SDF_BYTE_ORDER 0x01020304u

/**
 * @defgroup sdf_status Status codes for the sif data format
 * @brief What an `.sdf` operation reports through its status argument.
 *
 * The format has more distinguishable failures than the library-wide
 * SIF_ERR_* set can name -- "this is not an `.sdf` file" and "this file was
 * written by a newer sif" lead a caller to do different things, and both would
 * otherwise be SIF_ERR_IO -- so the format carries its own codes. See the
 * error handling section of STYLE.md for when that is allowed.
 *
 * They stay compatible with the rest of the library rather than replacing it.
 * Success is 0 and every failure is negative, so `!= SIF_SDF_OK` and `< 0` mean
 * what they mean everywhere else; the four generic conditions alias the codes
 * from `macros.h` so that a caller mixing the two vocabularies can never see a
 * value that belongs to neither; and the format's own codes start well below
 * them, at -100, leaving room for the generic set to grow.
 *
 * Pass any of them to sif_sdf_strerror() for a message.
 * @{
 */
typedef enum {
  SIF_SDF_OK = SIF_OK,
  /** A NULL or out-of-range argument. */
  SIF_SDF_ERR_INVALID = SIF_ERR_INVALID,
  SIF_SDF_ERR_ALLOC = SIF_ERR_ALLOC,
  /** The underlying read, write or seek failed. */
  SIF_SDF_ERR_IO = SIF_ERR_IO,

  /** No such file. */
  SIF_SDF_ERR_NOT_FOUND = -100,
  /** The file exists but the process may not open it that way. */
  SIF_SDF_ERR_PERMISSION = -101,
  /** sif_sdf_create() was asked not to overwrite, and the path is taken. */
  SIF_SDF_ERR_EXISTS = -102,
  /** The magic number is not #SIF_SDF_MAGIC. */
  SIF_SDF_ERR_NOT_SDF = -103,
  /** Written on a machine of the other endianness. */
  SIF_SDF_ERR_BYTE_ORDER = -104,
  /** Container version beyond what this build knows. */
  SIF_SDF_ERR_VERSION = -105,
  /** The file ends in the middle of something it declared. */
  SIF_SDF_ERR_TRUNCATED = -106,
  /** Internally inconsistent: a length, a checksum or a value that cannot be
   * what it claims to be.
   */
  SIF_SDF_ERR_CORRUPT = -107,
  /** The operation is not allowed by the mode the handle was opened with. */
  SIF_SDF_ERR_MODE = -108,
  /** The file holds no catalogue, so it holds nothing at all. */
  SIF_SDF_ERR_NO_CATALOG = -109,
  /** Measured from a different catalogue than the one the file holds. */
  SIF_SDF_ERR_CATALOG_MISMATCH = -110,
  /** The file already holds a block of that kind under that name. */
  SIF_SDF_ERR_NAME_TAKEN = -111,
  /** No block of the kind and name asked for. */
  SIF_SDF_ERR_ABSENT = -112,
  /** The value is there and is not of the type that was asked for. */
  SIF_SDF_ERR_TYPE = -113
} sif_sdf_status_t;
/** @} */

/**
 * @brief How a file is opened.
 *
 * There is no read-write mode. A file grows by appending whole blocks and
 * nothing already written is ever modified, so the only two things a caller
 * can want are to look at what is there and to add to it.
 */
typedef enum {
  /** Read what is there. The file is never modified. */
  SIF_SDF_READ = 0,
  /** Read what is there, and append new blocks to the end. */
  SIF_SDF_APPEND = 1
} sif_sdf_mode_t;

/**
 * @brief What a block holds.
 *
 * Only a catalogue and the things measured from it. A modelled size function
 * is not on the list and is not storable: it comes from parameters rather than
 * from the file's catalogue, and a file whose blocks are not all derived from
 * that catalogue is the thing this format exists to prevent.
 *
 * Values are the on-disk numbers and never change. Anything from
 * #SIF_SDF_BLOCK_PRIVATE up is left permanently unassigned, for a caller that
 * wants to put something of its own in a file: sif skips such a block whole,
 * neither reading it nor checking it against the file's catalogue.
 */
typedef enum {
  /** Voids: centres and radii. Always the first block, exactly one. */
  SIF_SDF_BLOCK_CATALOG = 1,
  /** Metadata alone, with no data section. */
  SIF_SDF_BLOCK_META = 2,
  /** Stacked radial density profiles, one row per void. */
  SIF_SDF_BLOCK_DENSITY_PROFILES = 3,
  /** Stacked radial velocity profiles, one row per void. */
  SIF_SDF_BLOCK_VELOCITY_PROFILES = 4,
  /** A measured void size function. */
  SIF_SDF_BLOCK_SIZE_FUNCTION = 5,
  /** First of the range sif never assigns. A block at or above this is
   * skipped whole: sif does not read it, and does not check it against the
   * file's catalogue either.
   */
  SIF_SDF_BLOCK_PRIVATE = 0x8000
} sif_sdf_block_type_t;

/**
 * @brief Width of the reals in a block's payload.
 *
 * Recorded per block and honoured on read: a build whose `sif_real` is the
 * other width converts rather than refusing, which is what lets a catalogue
 * written by a double build be read by a single-precision one. Fields and
 * grids make the opposite trade, because they are large enough that reading
 * has to be a copy and nothing else.
 */
typedef enum { SIF_SDF_F32 = 1, SIF_SDF_F64 = 2 } sif_sdf_dtype_t;

/**
 * @defgroup sdf_create_flags Creation flags
 * @brief Options for sif_sdf_create(), OR-ed together.
 * @{
 */
/** @brief Truncate the file if it already exists.
 *
 * Off by default, and deliberately so: an `.sdf` file is the output of a run
 * that may have taken hours, and the usual way to lose one is a script that
 * creates before it checks. Without this flag an existing path is
 * #SIF_SDF_ERR_EXISTS.
 */
#define SIF_SDF_OVERWRITE (1u << 0)
/** @} */

/**
 * @brief An open `.sdf` file.
 *
 * Opaque: it holds a stream and a file descriptor, and keeping their types out
 * of the public headers is what lets those headers stay free of `<stdio.h>`
 * and the POSIX surface. Everything a caller needs from it is reachable
 * through the accessors below.
 */
typedef struct sif_sdf sif_sdf_t;

/**
 * @brief Open an existing `.sdf` file.
 *
 * The file header is read and validated here, once, so that every later
 * operation on the handle can assume it is looking at an `.sdf` file this
 * build understands.
 *
 * @param filepath Path to the file.
 * @param mode Read, or read and append.
 * @param status Caller's status, which must not be NULL. Left untouched on
 * success. If it already holds an error this call does nothing and returns
 * NULL.
 * @return The handle, owned by the caller and released with sif_sdf_close().
 * NULL on failure.
 */
SIF_NODISCARD sif_sdf_t* sif_sdf_open(
  const char* filepath, sif_sdf_mode_t mode, sif_sdf_status_t* status);

/**
 * @brief Create a new `.sdf` file around a catalogue.
 *
 * The catalogue is not optional and there is no call that adds one later. A
 * file is a catalogue and the measurements made from it, so writing the header
 * and block 0 in one operation is what makes a file without a catalogue
 * impossible to produce rather than merely forbidden.
 *
 * The handle comes back in #SIF_SDF_APPEND mode, ready for those measurements.
 * Every one of them is checked against @p cat -- see
 * #SIF_SDF_ERR_CATALOG_MISMATCH -- so a set measured from a different
 * catalogue cannot end up in this file.
 *
 * @param filepath Path to the file.
 * @param box_length Simulation box size. Recorded once here rather than per
 * block, since every product in a file shares it.
 * @param cat The catalogue the file is built around. Written as block 0 and
 * not retained: the handle keeps its identity and its void count, not the
 * catalogue itself.
 * @param flags OR of the creation flags above, or 0.
 * @param status As sif_sdf_open(). #SIF_SDF_ERR_EXISTS when the path is taken
 * and #SIF_SDF_OVERWRITE was not given.
 * @return The handle, owned by the caller and released with sif_sdf_close().
 * NULL on failure.
 *
 * @note An empty catalogue is written like any other. A run that found no
 * voids has a result, and it is not the same thing as a run that was never
 * made.
 */
SIF_NODISCARD sif_sdf_t* sif_sdf_create(const char* filepath, double box_length,
  const sif_catalog_t* cat, sif_option flags, sif_sdf_status_t* status);

/**
 * @brief Flush, close and release a file.
 *
 * The one operation that ignores an inherited error: a file that failed
 * halfway through still has to be closed, and a caller unwinding after a
 * failure would otherwise leak it.
 *
 * @param file File to close. NULL is accepted and ignored.
 * @param status Caller's status, or NULL to close without reporting. Set to
 * #SIF_SDF_ERR_IO if the file could not be flushed *and* no earlier error is
 * recorded, since that earlier one is the more informative of the two.
 *
 * @note Worth passing a status on an append handle. Writing a block only means
 * the bytes reached the stdio buffer; a full disk surfaces here, and a file
 * whose last block never landed is a file with a torn block at the end.
 *
 * @warning The handle is released whatever the status says. There is nothing
 * to retry and nothing left to close.
 */
void sif_sdf_close(sif_sdf_t* file, sif_sdf_status_t* status);

/**
 * @brief Read the catalogue the file is built around.
 *
 * @param file The file.
 * @param status As sif_sdf_open().
 * @return The catalogue, owned by the caller and released with
 * sif_catalog_free(). NULL on failure.
 *
 * @note A fresh catalogue every call: the file does not keep one to hand out,
 * so two calls give two catalogues and both must be freed. They carry the same
 * sif_catalog_t::id, which is the one recorded in the file, so either can be
 * measured from and the results appended here.
 *
 * @note Read at whatever precision the file was written with and converted if
 * this build differs, so the values are the file's to within the narrower of
 * the two.
 */
SIF_NODISCARD sif_catalog_t* sif_sdf_catalog(
  sif_sdf_t* file, sif_sdf_status_t* status);

/**
 * @defgroup sdf_products Writing and reading the measured products
 * @brief Appending a measurement to a file, and taking one back out.
 *
 * Every append is checked against the file's catalogue before anything is
 * written: the measurement carries the identity of the catalogue it was made
 * from, and a file holds one catalogue, so a set measured from anything else
 * is #SIF_SDF_ERR_CATALOG_MISMATCH rather than a block whose rows line up with
 * nothing. A measurement that names no catalogue at all -- a modelled or
 * stitched size function -- is refused for the same reason.
 *
 * The name tells apart several blocks of one kind: two size functions binned
 * differently, two profile sets measured to different extents. It may be NULL
 * or empty for the one block of its kind, must fit in 16 bytes, and must not
 * already be in use by a block of the same kind -- a density set and a
 * velocity set measured together are free to share one. Reading takes the same
 * name, or NULL for the first block of the kind.
 *
 * An append either lands whole or leaves the file exactly as it was: a write
 * that fails cuts the file back, so one failed append does not cost a caller
 * everything already in the file. The handle refuses every later write once
 * that has happened.
 * @{
 */

/**
 * @brief Append a stacked density profile set.
 *
 * @param file File to append to, open for appending.
 * @param profs The set, measured from the catalogue @p file holds.
 * @param name Tag for the block, or NULL.
 * @param status As sif_sdf_open().
 */
void sif_sdf_append_density_profiles(sif_sdf_t* file,
  const sif_density_profiles_t* profs, const char* name,
  sif_sdf_status_t* status);

/**
 * @brief Append a stacked radial velocity profile set.
 * @param file File to append to, open for appending.
 * @param profs The set, measured from the catalogue @p file holds.
 * @param name Tag for the block, or NULL.
 * @param status As sif_sdf_open().
 */
void sif_sdf_append_velocity_profiles(sif_sdf_t* file,
  const sif_velocity_profiles_t* profs, const char* name,
  sif_sdf_status_t* status);

/**
 * @brief Append a measured size function.
 *
 * @param file File to append to, open for appending.
 * @param vsf The size function, binned from the catalogue @p file holds.
 * @param name Tag for the block, or NULL.
 * @param status As sif_sdf_open(). #SIF_SDF_ERR_CATALOG_MISMATCH for a model
 * or a stitched size function, neither of which came from this catalogue.
 */
void sif_sdf_append_size_function(sif_sdf_t* file,
  const sif_size_function_t* vsf, const char* name, sif_sdf_status_t* status);

/**
 * @brief Read a density profile set back.
 *
 * @param file The file.
 * @param name Tag to look for, or NULL for the first set in the file.
 * @param status As sif_sdf_open(). #SIF_SDF_ERR_ABSENT when there is no such
 * block, which is an error rather than an empty result: a caller handed
 * nothing would plot it.
 * @return The set, owned by the caller and released with
 * sif_density_profiles_free(). NULL on failure.
 *
 * @note Comes back carrying the file's catalogue, so a set read out of a file
 * can be written back to it and to no other.
 */
SIF_NODISCARD sif_density_profiles_t* sif_sdf_density_profiles(
  sif_sdf_t* file, const char* name, sif_sdf_status_t* status);

/**
 * @brief Read a velocity profile set back.
 * @param file The file.
 * @param name Tag to look for, or NULL for the first set in the file.
 * @param status As sif_sdf_density_profiles().
 * @return The set, owned by the caller and released with
 * sif_velocity_profiles_free(). NULL on failure.
 */
SIF_NODISCARD sif_velocity_profiles_t* sif_sdf_velocity_profiles(
  sif_sdf_t* file, const char* name, sif_sdf_status_t* status);

/**
 * @brief Read a size function back.
 * @param file The file.
 * @param name Tag to look for, or NULL for the first one in the file.
 * @param status As sif_sdf_density_profiles().
 * @return The size function, owned by the caller and released with
 * sif_size_function_free(). NULL on failure.
 */
SIF_NODISCARD sif_size_function_t* sif_sdf_size_function(
  sif_sdf_t* file, const char* name, sif_sdf_status_t* status);

/** @} */

/**
 * @defgroup sdf_meta Metadata
 * @brief Typed notes attached to a file: what the run was, and under what.
 *
 * A file's metadata is a table of typed key/value pairs -- integers, doubles,
 * UTF-8 strings, singly or as arrays. It is what makes a file still mean
 * something a year later: the redshift, the cosmology, which simulation, which
 * settings. Nothing requires a file to carry any, and a file with none reads
 * as one whose notes are empty rather than as one that is missing something.
 *
 * The table is a #sif_sdf_meta_t, and it is the same type going in and coming
 * out, so the natural thing to do is the thing that works:
 *
 * .. code-block:: c
 *
 *    sif_sdf_meta_t* meta = sif_sdf_meta_read(file, &status);
 *    sif_sdf_meta_put_str(meta, "validated", "2026-08-22");
 *    sif_sdf_meta_write(file, meta, &status);
 *    sif_sdf_meta_free(meta);
 *
 * **On disk a file may hold several metadata blocks, and in the API it has one
 * table.** An append-only file cannot rewrite what it wrote, so correcting a
 * value means appending another block that carries it; sif_sdf_meta_read()
 * merges every block in file order and later values win. The consequence is
 * that a key can be overwritten but not removed: a table written without it
 * leaves the earlier block, and the earlier block still says what it said.
 *
 * Keys beginning `sif.` are reserved for the parameters the library stores
 * about a block's own shape, and sif_sdf_meta_put_i64() and its siblings
 * refuse them.
 *
 * Types are not converted. Asking for a double and finding an integer is
 * #SIF_SDF_ERR_TYPE rather than a quiet promotion -- if the two were
 * interchangeable there would be no reason to have both, and a value that
 * comes back as something the caller did not store is how a plot ends up wrong
 * without anyone noticing.
 *
 * There is no boolean type. A flag is an integer of 0 or 1, which is what the
 * library's own `sif.differential` is.
 * @{
 */

/** @brief What a metadata value holds. */
typedef enum {
  /** No such key. */
  SIF_SDF_META_NONE = 0,
  SIF_SDF_META_I64 = 1,
  SIF_SDF_META_F64 = 2,
  /** UTF-8 text. */
  SIF_SDF_META_STR = 3
} sif_sdf_meta_type_t;

/**
 * @brief A table of metadata.
 *
 * Opaque, and small: a few tens of keys is what one is for, and lookup is a
 * scan rather than a hash.
 */
typedef struct sif_sdf_meta sif_sdf_meta_t;

/**
 * @brief Allocate an empty table.
 * @return The table, owned by the caller and released with
 * sif_sdf_meta_free(). NULL on allocation failure.
 */
SIF_NODISCARD sif_sdf_meta_t* sif_sdf_meta_alloc(void);

/**
 * @brief Release a table.
 * @param meta Table to free. NULL is accepted and ignored.
 */
void sif_sdf_meta_free(sif_sdf_meta_t* meta);

/**
 * @defgroup sdf_meta_put Setting values
 * @brief Add a value to a table, replacing whatever the key held.
 *
 * These report nothing. A table is built by a run of calls, and a failure --
 * out of memory, or a key under the reserved `sif.` prefix -- is logged, marks
 * the table, and surfaces at sif_sdf_meta_write(), which refuses to write a
 * table that did not come out as the caller asked. One check, at the call
 * where it matters.
 *
 * @param meta Table to add to.
 * @param key Key, which must not begin `sif.`. An existing one is replaced.
 * @{
 */
void sif_sdf_meta_put_i64(sif_sdf_meta_t* meta, const char* key, int64_t value);
void sif_sdf_meta_put_f64(sif_sdf_meta_t* meta, const char* key, double value);
/** @param value UTF-8, copied. */
void sif_sdf_meta_put_str(
  sif_sdf_meta_t* meta, const char* key, const char* value);
/** @param values,n_values The array, copied. */
void sif_sdf_meta_put_i64v(sif_sdf_meta_t* meta, const char* key,
  const int64_t* values, uint32_t n_values);
/** @param values,n_values The array, copied. */
void sif_sdf_meta_put_f64v(sif_sdf_meta_t* meta, const char* key,
  const double* values, uint32_t n_values);
/** @} */

/**
 * @defgroup sdf_meta_get Reading values
 * @brief Take a value out of a table.
 *
 * These return their status rather than taking one, which is the rule for
 * everything here that does not touch the file. A key that is not there is an
 * ordinary answer to an ordinary question, and reporting it through a caller's
 * status would stop every file operation that followed it.
 *
 * @param meta Table to read.
 * @param key Key to look for.
 * @return #SIF_SDF_OK, #SIF_SDF_ERR_ABSENT if the key is not in the table, or
 * #SIF_SDF_ERR_TYPE if it is and holds something else.
 * @{
 */
SIF_NODISCARD sif_sdf_status_t sif_sdf_meta_get_i64(
  const sif_sdf_meta_t* meta, const char* key, int64_t* out);
SIF_NODISCARD sif_sdf_status_t sif_sdf_meta_get_f64(
  const sif_sdf_meta_t* meta, const char* key, double* out);
/**
 * @param out Receives a borrowed, NUL-terminated pointer, valid until the key
 * is replaced or the table is freed.
 */
SIF_NODISCARD sif_sdf_status_t sif_sdf_meta_get_str(
  const sif_sdf_meta_t* meta, const char* key, const char** out);
/**
 * @param out,out_n Receive a borrowed pointer and its length, valid until the
 * key is replaced or the table is freed.
 */
SIF_NODISCARD sif_sdf_status_t sif_sdf_meta_get_i64v(const sif_sdf_meta_t* meta,
  const char* key, const int64_t** out, uint32_t* out_n);
/**
 * @param out,out_n Receive a borrowed pointer and its length, valid until the
 * key is replaced or the table is freed.
 */
SIF_NODISCARD sif_sdf_status_t sif_sdf_meta_get_f64v(const sif_sdf_meta_t* meta,
  const char* key, const double** out, uint32_t* out_n);
/** @} */

/**
 * @brief Whether a table holds a key.
 * @return Non-zero if it does.
 */
int sif_sdf_meta_has(const sif_sdf_meta_t* meta, const char* key);

/**
 * @brief What a key holds.
 * @return Its type, or #SIF_SDF_META_NONE if the table has no such key.
 */
sif_sdf_meta_type_t sif_sdf_meta_type(
  const sif_sdf_meta_t* meta, const char* key);

/**
 * @brief How many elements a key holds.
 * @return The count -- 1 for a scalar, the length for an array, the bytes for
 * a string, and 0 for a key that is not there.
 */
uint32_t sif_sdf_meta_length(const sif_sdf_meta_t* meta, const char* key);

/**
 * @brief Keys in a table.
 * @return The count, or 0 for a NULL table.
 */
uint32_t sif_sdf_meta_count(const sif_sdf_meta_t* meta);

/**
 * @brief One key, by position, so a table can be walked without knowing what
 * is in it.
 *
 * @param meta The table.
 * @param index Below sif_sdf_meta_count().
 * @return Borrowed, NUL-terminated, valid until the table is freed. NULL if
 * @p index names no key.
 *
 * @note The order is the order the keys were added, which for a table that was
 * read is the order they appear in the file.
 */
const char* sif_sdf_meta_key(const sif_sdf_meta_t* meta, uint32_t index);

/**
 * @brief Read a file's metadata: every block of it, merged.
 *
 * @param file The file.
 * @param status As sif_sdf_open().
 * @return The table, owned by the caller and released with
 * sif_sdf_meta_free(). Empty rather than NULL when the file carries no
 * metadata, since that is a file without notes and not a failure. NULL only on
 * a real failure.
 */
SIF_NODISCARD sif_sdf_meta_t* sif_sdf_meta_read(
  sif_sdf_t* file, sif_sdf_status_t* status);

/**
 * @brief Append a table to a file, as a new metadata block.
 *
 * Everything in @p meta is written, whether or not the file already says it.
 * Duplicates are the mechanism rather than a mistake: the copy written now is
 * the one a later read will see.
 *
 * @param file File to append to, open for appending.
 * @param meta Table to write. An empty one writes nothing and is not an error.
 * @param status As sif_sdf_open(). #SIF_SDF_ERR_INVALID if any
 * sif_sdf_meta_put_i64() or sibling on @p meta had failed.
 */
void sif_sdf_meta_write(
  sif_sdf_t* file, const sif_sdf_meta_t* meta, sif_sdf_status_t* status);

/** @} */

/**
 * @brief What one block of a file is.
 *
 * Filled by sif_sdf_info(), which is how a caller finds out what a file it did
 * not write has in it.
 */
typedef struct {
  /** The block's kind, or 0 if there is no such block. */
  sif_sdf_block_type_t type;
  /** Voids, or bins, depending on the kind. */
  uint64_t n_items;
  /** The block's tag, terminated, empty if it has none. */
  char name[17];
} sif_sdf_block_info_t;

/**
 * @brief Describe one block, by position in the file.
 *
 * @param file The file.
 * @param index Block to describe, below sif_sdf_n_blocks().
 * @param out Filled with the description, or zeroed if @p index names no
 * block -- so a `type` of 0 is what an out-of-range index reads as. No status:
 * the only way to fail is to ask for a block the caller was already told the
 * file does not have.
 */
void sif_sdf_info(
  const sif_sdf_t* file, uint32_t index, sif_sdf_block_info_t* out);

/**
 * @brief Whether a file holds a particular block.
 *
 * The way to avoid #SIF_SDF_ERR_ABSENT, for a caller who would rather ask than
 * be told.
 *
 * @param file The file.
 * @param type Kind of block to look for.
 * @param name Tag to look for, or NULL for any block of the kind.
 * @return Non-zero if it is there.
 */
int sif_sdf_contains(
  const sif_sdf_t* file, sif_sdf_block_type_t type, const char* name);

/**
 * @brief Identity of the catalogue the file holds.
 *
 * What every append is checked against: a measurement whose `source_id` is not
 * this was made from a different catalogue.
 *
 * @param file The file.
 * @return The identity, or 0 for a NULL handle or a file whose catalogue has
 * not been written yet.
 */
uint64_t sif_sdf_catalog_id(const sif_sdf_t* file);

/**
 * @brief Voids in the file's catalogue, without reading it.
 * @param file The file.
 * @return The count, or 0 for a NULL handle.
 */
uint64_t sif_sdf_n_voids(const sif_sdf_t* file);

/**
 * @brief Blocks in the file, the catalogue included.
 * @param file The file.
 * @return The count, or 0 for a NULL handle.
 */
uint32_t sif_sdf_n_blocks(const sif_sdf_t* file);

/**
 * @brief The mode a file was opened with.
 * @param file The file.
 * @return The mode, or #SIF_SDF_READ for a NULL handle -- the mode that
 * permits least.
 */
sif_sdf_mode_t sif_sdf_mode(const sif_sdf_t* file);

/**
 * @brief The box size the file records.
 * @param file The file.
 * @return The box length, or 0 for a NULL handle.
 */
double sif_sdf_box_length(const sif_sdf_t* file);

/**
 * @brief The path the file was opened from.
 * @param file The file.
 * @return Borrowed pointer, valid until sif_sdf_close(). NULL for a NULL
 * handle.
 */
const char* sif_sdf_path(const sif_sdf_t* file);

/**
 * @brief When the file was created, in seconds since the epoch.
 * @param file The file.
 * @return The timestamp, or 0 for a NULL handle.
 */
uint64_t sif_sdf_created(const sif_sdf_t* file);

/**
 * @brief What wrote the file: the library version, as text.
 *
 * @param file The file.
 * @return Borrowed, NUL-terminated, valid until sif_sdf_close(). NULL for a
 * NULL handle. Empty if the file records nothing.
 */
const char* sif_sdf_writer(const sif_sdf_t* file);

/**
 * @brief How much of a file is not a whole, sound block.
 *
 * Reads every payload and verifies every checksum, which opening does not:
 * opening asks whether a file can be used, and this asks how much of it is
 * real. It costs a full pass over the file and it changes nothing.
 *
 * @param filepath Path to the file.
 * @param status As sif_sdf_open(). #SIF_SDF_ERR_NO_CATALOG when not even the
 * first block survives, which is a file with nothing to recover rather than a
 * file with a damaged tail.
 * @return Bytes at the end of the file that are not part of a sound block, or
 * 0 for a file that is entirely sound.
 */
SIF_NODISCARD uint64_t sif_sdf_verify(
  const char* filepath, sif_sdf_status_t* status);

/**
 * @brief Cut a damaged file back to the part of it that is sound.
 *
 * What finishes the sentence the append rule starts. A run that dies partway
 * through an append leaves a block that is not whole, and a file ending in one
 * does not open at all -- so without this, one interrupted write would put
 * every good block in the file out of reach.
 *
 * The scan stops at the **first** block that does not hold, and everything
 * from there on is dropped. Past a damaged block there is no way to know where
 * the next one begins: the chain is held together by each block's own length.
 * So a file damaged in the middle loses its tail as well, and that is honest
 * -- hunting for the next block header would be guessing, and a guess that
 * lands inside a payload is how a recovery tool invents data.
 *
 * @param filepath Path to the file, which must be writable.
 * @param status As sif_sdf_verify().
 * @return Bytes dropped, or 0 if there was nothing to drop.
 *
 * @warning This truncates the file. Run sif_sdf_verify() first if the caller
 * wants to know what it would cost before paying it.
 */
uint64_t sif_sdf_repair(const char* filepath, sif_sdf_status_t* status);

/**
 * @brief A message for a status code.
 *
 * @param status Any #sif_sdf_status_t, including the aliased generic codes.
 * @return Borrowed, NUL-terminated, static storage. Never NULL: a code this
 * build does not know reads as an unknown error rather than returning nothing
 * for a caller to print.
 */
const char* sif_sdf_strerror(sif_sdf_status_t status);

#endif /* SIF_IO_SDF_H */
