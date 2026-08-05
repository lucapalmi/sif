#include "sif/utils/align.h"

#include <stdlib.h>
#include <string.h>

void* sif_malloc_aligned(size_t size) {
  if (size == 0)
    return NULL;

  /* Pad size to a multiple of __SIF_CACHE_LINE to prevent false sharing
   * between OpenMP threads in adjacent memory blocks. */
  size_t padded_size = (size + __SIF_CACHE_LINE - 1) & ~(__SIF_CACHE_LINE - 1);
  void* ptr = NULL;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L &&                \
  !defined(__APPLE__)
  /* C11 Standard Aligned Allocator */
  ptr = aligned_alloc(__SIF_CACHE_LINE, padded_size);
#else
  /* POSIX fallback (macOS and older Linux glibc) */
  if (posix_memalign(&ptr, __SIF_CACHE_LINE, padded_size) != 0) {
    return NULL;
  }
#endif

  return ptr;
}

void* sif_calloc_aligned(size_t count, size_t size) {
  size_t total_size = count * size;
  void* ptr = sif_malloc_aligned(total_size);

  if (ptr) {
    /* We padded the allocation, so it is safe to memset the padded size.
     * This maxes out memory bandwidth because it's cache-aligned. */
    size_t padded_size =
      (total_size + __SIF_CACHE_LINE - 1) & ~(__SIF_CACHE_LINE - 1);
    memset(ptr, 0, padded_size);
  }

  return ptr;
}

void* sif_realloc_aligned(void* ptr, size_t old_size, size_t new_size) {
  /* Mimic standard realloc behavior: size 0 means free */
  if (new_size == 0) {
    sif_free_aligned(ptr);
    return NULL;
  }

  /* Mimic standard realloc behavior: NULL ptr means malloc */
  if (!ptr) {
    return sif_malloc_aligned(new_size);
  }

  /* Allocate the new aligned block */
  void* new_ptr = sif_malloc_aligned(new_size);

  if (new_ptr) {
    /* Safely copy the existing data, truncating if the new size is smaller */
    size_t copy_size = (old_size < new_size) ? old_size : new_size;
    memcpy(new_ptr, ptr, copy_size);

    /* Free the old block */
    sif_free_aligned(ptr);
  }

  /* If allocation fails, it returns NULL and leaves the old ptr untouched */
  return new_ptr;
}

void sif_free_aligned(void* ptr) {
  if (ptr) {
    free(ptr);
  }
}