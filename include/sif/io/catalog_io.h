#ifndef __SIF_CATALOG_IO_H__
#define __SIF_CATALOG_IO_H__

#include "sif/core/macros.h"
#include "sif/structures/catalog.h"

/*
 * @brief Saves a void catalog to an ASCII file
 *
 * @param catalog The catalog to save
 * @param filepath The path to the output file
 *
 * @return 0 on success, non-zero on failure
 */
int sif_catalog_write_ascii(const sif_catalog_t* catalog, const char* filepath);

/*
 * @brief Loads a void catalog from an ASCII file
 *
 * @param filepath The path to the input file
 *
 * @return The loaded catalog (NULL on failure)
 */
NODISCARD sif_catalog_t* sif_catalog_read_ascii(const char* filepath);

#endif /* __SIF_CATALOG_IO_H__ */