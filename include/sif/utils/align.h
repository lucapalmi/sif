#ifndef __SIF_ALIGN_H__
#define __SIF_ALIGN_H__

#include "sif/core/macros.h"
#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#  define SIF_ASSUME_ALIGNED(ptr)                                              \
    __builtin_assume_aligned((ptr), __SIF_CACHE_LINE)
#  define SIF_ALIGNED_FN                                                       \
    __attribute__((malloc, assume_aligned(__SIF_CACHE_LINE)))
#else
#  define SIF_ASSUME_ALIGNED(ptr) (ptr)
#  define SIF_ALIGNED_FN
#endif

/**
 * @brief Allocate aligned memory
 *
 * @param size The number of bytes required
 *
 * @return Pointer to the aligned memory (NULL on allocation failure)
 */
SIF_ALIGNED_FN void* sif_malloc_aligned(size_t size);

/**
 * @brief Allocate and zero-initialize aligned memory
 *
 * @param count Number of elements
 * @param size Size of each element
 *
 * @return Pointer to the zero-initialized, aligned memory (NULL on allocation
 * failure)
 */
SIF_ALIGNED_FN void* sif_calloc_aligned(size_t count, size_t size);

/*
 * @brief Re-allocate aligned memory with a different buffer size
 *
 * @param ptr Pointer to the memory to re-allocate
 * @param old_size Current size of the memory block
 * @param new_size New size of the memory block
 *
 * @return Pointer to the new memory block (NULL on allocation failure)
 *
 * @note If the allocation is successful, the old memory is freed automatically
 */
void* sif_realloc_aligned(void* ptr, size_t old_size, size_t new_size);

/**
 * @brief Frees an aligned pointer
 *
 * @param ptr The pointer to free
 */
void sif_free_aligned(void* ptr);

#endif /* __SIF_ALIGN_H__ */