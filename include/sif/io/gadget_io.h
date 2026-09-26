/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file gadget_io.h
 * @brief Reading GADGET snapshots into a particle field.
 *
 * All three snapshot formats are read: the two legacy binaries (SnapFormat 1,
 * and SnapFormat 2 with its 4-character block labels) and HDF5 (SnapFormat
 * 3). A binary file may carry either the 256-byte GADGET-2/3 header or the
 * shorter GADGET-4 one, and be in either byte order; the reader tells them
 * apart from the file itself. Positions and velocities may be single or
 * double precision, which is inferred from the block sizes.
 *
 * A snapshot split over several files is read as one: name any one of its
 * files, or the base name, and the rest are found -- `snap_010.N`,
 * `snap_010.N.hdf5`, and both inside `snapdir_010/`.
 *
 * **Arguments are enumerations, one type per slot, each with its own range of
 * values.** A call spells out every choice it makes, and one that passes a
 * value in the wrong slot -- a velocity option where the mass option goes --
 * is refused with SIF_ERR_INVALID naming the slot, rather than read as
 * whatever integer it happens to be.
 *
 * HDF5 files need a build with HDF5 (SIF_HDF5_SUPPORT); without one they are
 * refused with SIF_ERR_UNSUPPORTED, and the binary formats still work.
 */

#ifndef SIF_IO_GADGET_IO_H
#define SIF_IO_GADGET_IO_H

#include "sif/core/macros.h"
#include "sif/structures/field.h"

#include <stdint.h>
#include <stdio.h>

/**
 * @brief Particle types a header can describe. GADGET-4 makes the count a
 * compile-time choice (NTYPES, 6 by default); a file with more is refused.
 */
#define SIF_GADGET_MAX_TYPES 16

/** @brief Block names a header records, for sif_gadget_print_header(). */
#define SIF_GADGET_MAX_BLOCKS 32

/**
 * @defgroup gadget_args Reader arguments
 * @brief The choices sif_field_read_gadget() takes, one enumeration per
 * argument.
 *
 * Each enumeration starts at its own multiple of 0x100, so the reader can tell
 * which slot a value was meant for and refuse one passed in the wrong place.
 * @{
 */

/** @brief Which on-disk format to expect. AUTO reads it from the file. */
typedef enum {
  SIF_GADGET_FORMAT_AUTO = 0x100,
  /** Legacy binary, SnapFormat 1: unlabelled Fortran records. */
  SIF_GADGET_FORMAT_1,
  /** Legacy binary, SnapFormat 2: each record preceded by a 4-char label. */
  SIF_GADGET_FORMAT_2,
  /** HDF5, SnapFormat 3. */
  SIF_GADGET_FORMAT_HDF5
} sif_gadget_format_t;

/**
 * @brief Which particle type to read. One per call: a field holds a single
 * population, so mixing gas with dark matter is left to the caller.
 */
typedef enum {
  SIF_GADGET_PTYPE_0 = 0x200,
  SIF_GADGET_PTYPE_1,
  SIF_GADGET_PTYPE_2,
  SIF_GADGET_PTYPE_3,
  SIF_GADGET_PTYPE_4,
  SIF_GADGET_PTYPE_5
} sif_gadget_ptype_t;

/**
 * @brief Whether to read velocities, and in which convention.
 *
 * GADGET stores u = v / sqrt(a) for a cosmological run, v being the peculiar
 * velocity. RAW keeps u as stored; PECULIAR multiplies by sqrt(a), taking a
 * from the header's Time -- which is only the scale factor in a cosmological
 * run, so PECULIAR is wrong for any other.
 */
typedef enum {
  SIF_GADGET_VELOCITY_SKIP = 0x300,
  SIF_GADGET_VELOCITY_RAW,
  SIF_GADGET_VELOCITY_PECULIAR
} sif_gadget_velocity_t;

/**
 * @brief Whether to read particle masses into the field's weights.
 *
 * A type whose mass-table entry is non-zero has one mass for every particle,
 * and READ then leaves the weights NULL -- an unweighted field -- rather than
 * filling an array with a constant. The mass is in the header.
 */
typedef enum {
  SIF_GADGET_MASS_SKIP = 0x400,
  SIF_GADGET_MASS_READ
} sif_gadget_mass_t;

/**
 * @brief The length unit the snapshot is written in. The field always comes
 * out in Mpc/h, and the box length with it.
 *
 * A binary snapshot does not record its unit, so it has to be named: KPC for
 * GADGET's default kpc/h, MPC for a run already in Mpc/h. AUTO reads
 * UnitLength_in_cm from an HDF5 file (from `Parameters`, `Units` or `Header`,
 * in that order) and is refused for a binary one.
 */
typedef enum {
  SIF_GADGET_LENGTH_KPC = 0x500,
  SIF_GADGET_LENGTH_MPC,
  SIF_GADGET_LENGTH_AUTO
} sif_gadget_length_t;

/** @} */

/**
 * @brief What one snapshot file says about itself.
 *
 * Filled by sif_gadget_read_header() from a single file. Counts in
 * `n_part_file` are that file's; `n_part_total` is the whole snapshot's.
 */
typedef struct {
  /** The format found: never SIF_GADGET_FORMAT_AUTO. */
  sif_gadget_format_t format;
  /** Binary only: 1 if the file is in the other byte order from this
   * machine's, and is swapped on the way in.
   */
  int is_swapped;
  /** Binary only: 1 for the 256-byte GADGET-2/3 header, 0 for GADGET-4's. */
  int is_legacy_header;
  /** Bytes per position component: 4 or 8, or 0 if the file holds no
   * particles to tell from.
   */
  uint32_t precision;

  /** Particle types the header describes. */
  uint32_t n_types;
  uint64_t n_part_file[SIF_GADGET_MAX_TYPES];
  uint64_t n_part_total[SIF_GADGET_MAX_TYPES];
  /** Mass of every particle of a type, or 0 for a type whose particles carry
   * their own. In GADGET mass units, 1e10 Msun/h by default.
   */
  double mass_table[SIF_GADGET_MAX_TYPES];

  /** Scale factor for a cosmological run, time otherwise. */
  double time;
  double redshift;
  /** In the file's own length unit, unconverted. */
  double box_size;
  uint32_t n_files;

  /** 1 if the cosmology below was found: the legacy binary header and most
   * HDF5 files carry it, the GADGET-4 binary header does not.
   */
  int has_cosmology;
  double omega0;
  double omega_lambda;
  double hubble_param;

  /** UnitLength_in_cm, or 0 if the file does not record it (every binary
   * file).
   */
  double unit_length_in_cm;

  /** Blocks the file names: the labels of a SnapFormat 2 file, the datasets
   * of an HDF5 one (across all particle groups). A SnapFormat 1 file labels
   * nothing, and records only the count.
   */
  uint32_t n_blocks;
  char blocks[SIF_GADGET_MAX_BLOCKS][32];
} sif_gadget_header_t;

/**
 * @brief Read the header of one snapshot file.
 *
 * Reads one file only, even from a multi-file snapshot; resolving the path
 * works as in sif_field_read_gadget(), and the first file is the one read.
 *
 * @param path A snapshot file, or a multi-file snapshot's base name.
 * @param format Expected format, or SIF_GADGET_FORMAT_AUTO.
 * @param out The header, filled on success and zeroed on failure.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument or a bad format value;
 * SIF_ERR_IO for a missing file, one that is not a GADGET snapshot, or one in
 * a different format from @p format; SIF_ERR_UNSUPPORTED for HDF5 in a build
 * without it.
 */
SIF_NODISCARD int sif_gadget_read_header(
  const char* path, sif_gadget_format_t format, sif_gadget_header_t* out);

/**
 * @brief Print a header in human-readable form.
 *
 * Written to a stream rather than through the logger, so it appears whatever
 * the log level: it is the answer to a question, not a diagnostic.
 *
 * @param header The header.
 * @param stream Where to print, e.g. stdout.
 */
void sif_gadget_print_header(const sif_gadget_header_t* header, FILE* stream);

/**
 * @brief Read and print a snapshot's header to stdout.
 *
 * sif_gadget_read_header() then sif_gadget_print_header(), for a look at a
 * file before deciding how to read it.
 *
 * @return As sif_gadget_read_header().
 */
int sif_gadget_inspect(const char* path);

/**
 * @brief Read one particle type of a GADGET snapshot into a new field.
 *
 * Every file of the snapshot is read in order. The headers are read first, so
 * the field is allocated once at its final size and each file is streamed
 * into it in chunks: memory is the field plus a buffer of a few MB, whatever
 * the precision on disk. The file order is kept, so a subsample comes out in
 * the order the particles appear in the snapshot.
 *
 * **Subsampling** keeps exactly round(fraction * N) of the N particles, each
 * subset of that size equally likely (selection sampling, Knuth's Algorithm
 * S), drawn from a sif_prng_state_t seeded with @p seed: the same seed and
 * the same files give the same particles.
 *
 * @param path Any one file of the snapshot, or its base name (`snap_010` for
 * `snap_010.0` ... or `snapdir_010/snap_010.0.hdf5` ...).
 * @param format Expected format, or SIF_GADGET_FORMAT_AUTO.
 * @param ptype The particle type to read.
 * @param velocity Whether to read velocities, and how.
 * @param mass Whether to read masses into the weights.
 * @param length The snapshot's length unit; positions and box come out in
 * Mpc/h.
 * @param fraction Share of the particles to keep, in (0, 1]; 1 keeps all and
 * draws nothing.
 * @param seed Seed for the subsample; ignored when @p fraction is 1.
 * @param out_field The field, owned by the caller and released with
 * sif_field_free(). NULL on any failure.
 * @param out_box_length Optional; the box size, in Mpc/h.
 * @return SIF_OK; SIF_ERR_INVALID for a NULL argument, an argument in the
 * wrong slot or out of range, a type the snapshot has no particles of, a
 * fraction that keeps none, or LENGTH_AUTO on a file that records no unit;
 * SIF_ERR_IO for a missing, truncated or inconsistent file -- including files
 * whose counts do not add up to the header's total; SIF_ERR_ALLOC;
 * SIF_ERR_UNSUPPORTED for HDF5 in a build without it.
 */
SIF_NODISCARD int sif_field_read_gadget(const char* path,
  sif_gadget_format_t format, sif_gadget_ptype_t ptype,
  sif_gadget_velocity_t velocity, sif_gadget_mass_t mass,
  sif_gadget_length_t length, double fraction, uint64_t seed,
  sif_field_t** out_field, double* out_box_length);

#endif /* SIF_IO_GADGET_IO_H */
