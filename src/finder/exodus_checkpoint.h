/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file exodus_checkpoint.h
 * @brief Writing, reading and matching exodus checkpoints. Private to the
 * library; the format itself is public, in sif/finder/exodus_finder.h.
 */

#ifndef SIF__FINDER_EXODUS_CHECKPOINT_H
#define SIF__FINDER_EXODUS_CHECKPOINT_H

#include <stddef.h>
#include <stdint.h>

#include "sif/core/macros.h"
#include "sif/finder/exodus_finder.h"
#include "sif/structures/catalogue.h"

/** @brief Room for a checkpoint path, a temporary suffix included. */
#define SIF__CHECKPOINT_PATH_MAX 4096

/**
 * @brief A checkpoint name no earlier run has used, `exodus-<16 hex>.ckpt`,
 * in the working directory.
 *
 * The digits mix the wall clock to the nanosecond, the process id and, where
 * the system has it, /dev/urandom, so two runs started together on one node
 * -- or on two nodes sharing a working directory -- do not collide.
 */
void sif__exodus_checkpoint_name(char* out, size_t size);

/**
 * @brief Write a checkpoint, atomically: to `<path>.tmp.<pid>`, flushed to
 * the disk, then renamed over @p path.
 *
 * The run parameters come from @p header; the format fields, the void count
 * and the checksum are filled in here.
 *
 * @param consequence NULL to make a failure an error, ending the run -- only
 * the first checkpoint at a path the caller named is that -- or what a
 * failure means for the run, appended to the warning it is logged as.
 * @return SIF_OK, or SIF_ERR_IO with the reason logged and any temporary
 * file removed.
 */
SIF_NODISCARD int sif__exodus_checkpoint_write(const char* path,
  sif_finder_exodus_checkpoint_t* header, const sif_real* radii,
  const sif_catalogue_t* cat, const char* consequence);

/**
 * @brief Read a checkpoint whole, voids in the frame they were saved in.
 *
 * @param[out] radii The ladder, n_radii long, released with
 * sif_free_aligned().
 * @param[out] cat The voids, released with sif_catalogue_free().
 * @return SIF_OK; SIF_ERR_IO, logged, for a file that cannot be read, is not
 * a checkpoint this build reads, is not the size its header says, or fails
 * its checksum; SIF_ERR_ALLOC.
 */
SIF_NODISCARD int sif__exodus_checkpoint_load(const char* path,
  sif_finder_exodus_checkpoint_t* header, sif_real** radii,
  sif_catalogue_t** cat);

/**
 * @brief Whether a checkpoint describes the run about to resume it.
 *
 * @param want The header this run would write, ladder sorted as it saves it.
 * @return SIF_OK, or SIF_ERR_INVALID with the first difference logged.
 */
SIF_NODISCARD int sif__exodus_checkpoint_match(const char* path,
  const sif_finder_exodus_checkpoint_t* want, const sif_real* want_radii,
  const sif_finder_exodus_checkpoint_t* got, const sif_real* got_radii);

/**
 * @brief Make the runs that follow stop after @p n_rungs rungs of their
 * ladder, checkpoint written, as though the job had been killed there; 0 lets
 * them finish. For the tests of resuming, which have no other way to
 * interrupt a run; not thread-safe, and not for anything else.
 */
void sif__exodus_halt_after_set(uint32_t n_rungs);

/** @brief CRC32 of a catalogue's four columns, chained in order. */
SIF_NODISCARD uint32_t sif__exodus_checkpoint_crc(const sif_catalogue_t* cat);

#endif /* SIF__FINDER_EXODUS_CHECKPOINT_H */
