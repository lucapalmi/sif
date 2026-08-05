#ifndef __SIF_VOID_CATALOG_H__
#define __SIF_VOID_CATALOG_H__

#include "sif/core/macros.h"
#include <stdint.h>

/*
 * @brief Dynamic void catalogue created by the void finder
 *
 * @note cx/cy/cz/radii are views into a single cache-line aligned arena
 * (_block). They move whenever the capacity changes, so never cache them
 * across an append or a trim.
 */
typedef struct {
  real_t* _block;

  real_t* cx;
  real_t* cy;
  real_t* cz;
  real_t* radii;

  uint64_t n_voids;
  uint64_t capacity;
} sif_catalog_t;

/*
 * @brief Create a void catalog
 *
 * @param initial_capacity The capacity at which the catalog is initialized.
 * Clamped up to 1, so a successfully returned catalog is always usable.
 *
 * @return Pointer to the catalog, NULL on failure
 */
NODISCARD sif_catalog_t* sif_catalog_alloc(uint64_t initial_capacity);

/*
 * @brief Frees a void catalog, including the catalog struct itself
 *
 * @param catalog The catalog to free (NULL is a no-op)
 */
void sif_catalog_free(sif_catalog_t* catalog);

/*
 * @brief Appends a void to a catalog, doubling its capacity if necessary
 *
 * @param catalog The catalog where the data are appended
 * @param x The x-coordinate of the void center
 * @param y The y-coordinate of the void center
 * @param z The z-coordinate of the void center
 * @param r The radius of the void
 *
 * @return SIF_OK on success. SIF_ERR_ALLOC if the catalog could not grow, in
 * which case the catalog is left untouched and the void is NOT stored.
 * SIF_ERR_INVALID on a NULL catalog.
 */
int sif_catalog_append(
  sif_catalog_t* catalog, real_t x, real_t y, real_t z, real_t r);

/*
 * @brief Trims the excess memory from a catalog
 *
 * @param catalog The catalog to trim
 *
 * @return SIF_OK on success, including when there is nothing to trim.
 * SIF_ERR_ALLOC if the smaller buffer could not be allocated, in which case
 * the catalog keeps its current larger allocation and stays fully valid.
 * SIF_ERR_INVALID on a NULL catalog.
 */
int sif_catalog_trim(sif_catalog_t* catalog);

#endif /* __SIF_VOID_CATALOG_H__ */
