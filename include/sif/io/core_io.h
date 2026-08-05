#ifndef __SIF_CORE_IO_H__
#define __SIF_CORE_IO_H__

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>

/*
 * @brief Reads a large block of data from disk using multiple threads.
 * * @param fd Open file descriptor
 * @param dest Destination memory buffer
 * @param total_bytes Total bytes to read
 * @param base_offset The starting byte offset in the file
 * @return 1 on success, 0 on failure
 */
int sif_parallel_pread(int fd, void* dest, size_t total_bytes, off_t base_offset);

/*
 * @brief Blazingly fast line counter for huge ASCII files.
 * @return The number of data rows (total lines minus skip_header).
 */
uint64_t sif_count_ascii_rows(const char* filepath, uint32_t skip_header);

#endif /* __SIF_CORE_IO_H__ */