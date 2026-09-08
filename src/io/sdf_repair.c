/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Checking a file, and cutting a damaged one back to the part that is sound.
 *
 * The format's whole safety argument is that a run which dies mid-append can
 * only ever leave a torn block at the end -- nothing already written is
 * touched. That argument is worth having only if there is something that
 * finishes the sentence: a file with a torn tail does not open, and without
 * this it would stay unopenable with every good block in it out of reach.
 *
 * The scan is stricter than the one at open. Opening reads block headers and
 * trusts the checksums until something asks for a block; this reads every
 * payload and verifies every checksum, because the question here is not "can
 * this file be used" but "how much of it is real".
 */

#include "sdf_internal.h"

#include "sif/utils/crc32.h"
#include "sif/utils/logger.h"

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Read at a time while checksumming a payload. Large enough that the syscall
 * is lost in the copy, small enough to sit anywhere. */
#define SCAN_CHUNK 65536

/*
 * Verifies one block's payload against the checksum in its header.
 *
 * Streamed rather than read whole: a profile set can be hundreds of megabytes,
 * and checking it should not need as much memory again.
 */
static sif_sdf_status_t block_verify(
  sif_sdf_t* file, const sif__sdf_entry_t* entry, char* buffer) {

  uint32_t crc = SIF_CRC32_INIT;
  uint64_t left = entry->header.meta_bytes + entry->header.data_bytes;
  uint64_t offset = sif__sdf_meta_offset(entry);

  while (left > 0) {
    const size_t take = (left < SCAN_CHUNK) ? (size_t)left : SCAN_CHUNK;

    const sif_sdf_status_t status =
      sif__sdf_read_raw(file, offset, buffer, take, &crc);
    if (status != SIF_SDF_OK)
      return status;

    offset += take;
    left -= take;
  }

  return (sif_crc32_final(crc) == entry->header.crc32) ? SIF_SDF_OK
                                                       : SIF_SDF_ERR_CORRUPT;
}

/*
 * Walks the chain until something does not hold, and reports where the sound
 * part of the file ends.
 *
 * Stops at the first block that fails rather than looking for a good one
 * further on. The chain is a chain: a block's length is what says where the
 * next one starts, so past a bad block there is no way to know where anything
 * is. Hunting for the next block magic would be guessing, and a guess that
 * lands inside a payload full of plausible bytes is how a recovery tool
 * invents data.
 */
static sif_sdf_status_t scan(sif_sdf_t* file, uint64_t file_size,
  uint64_t* out_good_end, uint32_t* out_blocks) {

  char* buffer = malloc(SCAN_CHUNK);
  if (!buffer) {
    SIF_LOG_ERROR("sdf", "failed to allocate the scan buffer");
    return SIF_SDF_ERR_ALLOC;
  }

  uint64_t pos = SIF__SDF_HEADER_BYTES;
  *out_good_end = pos;
  *out_blocks = 0;

  while (pos < file_size) {
    const uint64_t left = file_size - pos;
    if (left < SIF__SDF_BLOCK_BYTES)
      break;

    sif__sdf_entry_t entry;
    entry.offset = pos;

    if (sif__sdf_read_at(file, pos, &entry.header, SIF__SDF_BLOCK_BYTES) !=
        SIF_SDF_OK)
      break;

    if (memcmp(entry.header.magic, SIF__SDF_BLOCK_MAGIC, 4) != 0)
      break;

    const uint64_t room = left - SIF__SDF_BLOCK_BYTES;
    if (entry.header.meta_bytes > room ||
        entry.header.data_bytes > room - entry.header.meta_bytes)
      break;

    const uint64_t total = sif__sdf_block_bytes(&entry);
    if (total - SIF__SDF_BLOCK_BYTES > room)
      break;

    if (block_verify(file, &entry, buffer) != SIF_SDF_OK)
      break;

    pos += total;
    *out_good_end = pos;
    (*out_blocks)++;
  }

  free(buffer);

  /* Nothing to recover: without its catalogue a file holds nothing that means
   * anything, and cutting it back to a bare header would turn a damaged file
   * into an empty one. */
  if (*out_blocks == 0) {
    SIF_LOG_ERROR("sdf", "%s has no sound catalogue block", file->path);
    return SIF_SDF_ERR_NO_CATALOG;
  }

  return SIF_SDF_OK;
}

/* Opens, scans, and reports. The two public calls differ only in what they do
 * with the answer. */
static uint64_t scan_path(const char* filepath, sif_sdf_mode_t mode,
  uint64_t* out_good_end, sif_sdf_status_t* status) {

  sif_sdf_t* file = sif__sdf_open_raw(filepath, mode, status);
  if (!file)
    return 0;

  struct stat info;
  if (fstat(file->fd, &info) != 0) {
    SIF_LOG_ERROR("sdf", "could not stat %s", filepath);
    sif_sdf_close(file, NULL);
    *status = SIF_SDF_ERR_IO;
    return 0;
  }
  const uint64_t file_size = (uint64_t)info.st_size;

  uint64_t good_end = 0;
  uint32_t blocks = 0;
  const sif_sdf_status_t reason = scan(file, file_size, &good_end, &blocks);

  if (reason != SIF_SDF_OK) {
    sif_sdf_close(file, NULL);
    *status = reason;
    return 0;
  }

  SIF_LOG_INFO("sdf", "%s holds %u sound block%s in %llu bytes", filepath,
    (unsigned)blocks, blocks == 1 ? "" : "s", (unsigned long long)good_end);

  sif_sdf_close(file, NULL);

  *out_good_end = good_end;
  return file_size - good_end;
}

uint64_t sif_sdf_verify(const char* filepath, sif_sdf_status_t* status) {
  if (!sif__sdf_status_accepts(status, "verify"))
    return 0;

  uint64_t good_end = 0;
  const uint64_t damaged = scan_path(filepath, SIF_SDF_READ, &good_end, status);

  if (*status == SIF_SDF_OK && damaged > 0)
    SIF_LOG_WARNING("sdf", "%s ends in %llu bytes that are not a whole block",
      filepath, (unsigned long long)damaged);

  return damaged;
}

uint64_t sif_sdf_repair(const char* filepath, sif_sdf_status_t* status) {
  if (!sif__sdf_status_accepts(status, "repair"))
    return 0;

  /* Opened for appending because the fix is a write, even though the write
   * only ever makes the file shorter. */
  uint64_t good_end = 0;
  const uint64_t damaged =
    scan_path(filepath, SIF_SDF_APPEND, &good_end, status);

  if (*status != SIF_SDF_OK || damaged == 0)
    return 0;

  /* Reopened rather than kept open through the scan: truncate() takes a path,
   * and doing it on a closed file means there is no buffered state anywhere
   * that could still be flushed over the part being removed. */
  if (truncate(filepath, (off_t)good_end) != 0) {
    SIF_LOG_ERROR("sdf", "could not cut %s back to %llu bytes", filepath,
      (unsigned long long)good_end);
    *status = SIF_SDF_ERR_IO;
    return 0;
  }

  SIF_LOG_WARNING("sdf", "dropped %llu bytes from the end of %s",
    (unsigned long long)damaged, filepath);

  return damaged;
}
