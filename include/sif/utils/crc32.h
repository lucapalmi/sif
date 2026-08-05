#ifndef __SIF_UTILS_CRC32_H__
#define __SIF_UTILS_CRC32_H__

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Computes the CRC32 checksum of a data buffer.
 * 
 * Uses the standard IEEE 802.3 polynomial (0xEDB88320).
 * 
 * @param data Pointer to the data buffer
 * @param length Size of the data buffer in bytes
 * @return The computed 32-bit CRC checksum
 */
uint32_t sif_crc32(const void* data, size_t length);

#endif /* __SIF_UTILS_CRC32_H__ */
