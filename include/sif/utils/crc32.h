/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file crc32.h
 * @brief CRC32 checksum, used to validate the .xfield and .xgrid formats.
 *
 * Two ways in. sif_crc32() checksums one buffer and is the whole story for a
 * caller with contiguous data. The running form -- #SIF_CRC32_INIT,
 * sif_crc32_update(), sif_crc32_final() -- chains several buffers into one
 * checksum, which is what the binary writers need: a field's coordinates,
 * velocities and weights live in separate allocations but are one payload.
 */

#ifndef SIF_UTILS_CRC32_H
#define SIF_UTILS_CRC32_H

#include "sif/core/macros.h"

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Compute the CRC32 checksum of a data buffer.
 *
 * Uses the standard IEEE 802.3 polynomial (0xEDB88320), table-driven, one byte
 * per iteration.
 *
 * @param data Buffer to checksum.
 * @param length Size of the buffer, in bytes.
 * @return The 32-bit checksum. A zero-length buffer checksums to 0.
 *
 * @note This is an error-detecting code, not a cryptographic hash: it says
 * whether bytes changed in transit, not whether someone changed them on
 * purpose.
 */
SIF_NODISCARD uint32_t sif_crc32(const void* data, size_t length);

/** @brief Starting value for a running checksum. */
#define SIF_CRC32_INIT 0xFFFFFFFFu

/**
 * @brief Fold another buffer into a running checksum.
 *
 * Order matters: the same buffers folded in a different order give a different
 * checksum, so a reader has to walk the payload exactly as the writer did.
 *
 * @param crc Running value, starting from #SIF_CRC32_INIT.
 * @param data Buffer to fold in.
 * @param length Size of the buffer, in bytes.
 * @return The updated running value, which is not yet a checksum -- pass it to
 * sif_crc32_final().
 */
SIF_NODISCARD uint32_t sif_crc32_update(
  uint32_t crc, const void* data, size_t length);

/**
 * @brief Turn a running value into the checksum.
 * @param crc Running value from sif_crc32_update().
 * @return The 32-bit checksum.
 */
SIF_NODISCARD uint32_t sif_crc32_final(uint32_t crc);

#endif /* SIF_UTILS_CRC32_H */
