#include "sif/io/catalog_io.h"
#include "sif/utils/logger.h"

/* scanf needs a different conversion for float vs double. */
#ifdef __SIF_USE_DOUBLE
#  define __SIF_SCN_REAL "%lf"
#else
#  define __SIF_SCN_REAL "%f"
#endif

#include <stdio.h>
#include <stdlib.h>

int sif_catalog_write_ascii(const sif_catalog_t* catalog, const char* filepath) {
  if (!catalog || !filepath) {
    SIF_LOG_ERROR("io", "invalid arguments for write_catalog_ascii");
    return 1;
  }

  FILE* file = fopen(filepath, "w");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for writing", filepath);
    return 1;
  }

  /* Write the total number of voids as a header for easier loading */
  fprintf(file, "%" PRIu64 "\n", catalog->n_voids);

  /* Write the void data */
  for (uint64_t i = 0; i < catalog->n_voids; i++) {
    fprintf(file, "%f %f %f %f\n", catalog->cx[i], catalog->cy[i],
      catalog->cz[i], catalog->radii[i]);
  }

  fclose(file);

  SIF_LOG_INFO("io", "saved %" PRIu64 " voids to %s (ASCII)", catalog->n_voids,
    filepath);
  return 0;
}

sif_catalog_t* sif_catalog_read_ascii(const char* filepath) {
  if (!filepath) {
    SIF_LOG_ERROR("io", "invalid filepath for read_catalog_ascii");
    return NULL;
  }

  FILE* file = fopen(filepath, "r");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for reading", filepath);
    return NULL;
  }

  uint64_t n_voids;
  if (fscanf(file, "%" SCNu64, &n_voids) != 1) {
    SIF_LOG_ERROR("io", "failed to read void count from %s", filepath);
    fclose(file);
    return NULL;
  }

  sif_catalog_t* catalog = sif_catalog_alloc(n_voids);
  if (!catalog) {
    SIF_LOG_ERROR("io", "failed to allocate catalog for loading");
    fclose(file);
    return NULL;
  }

  for (uint64_t i = 0; i < n_voids; i++) {
    real_t x, y, z, r;
    if (fscanf(file,
          __SIF_SCN_REAL " " __SIF_SCN_REAL " " __SIF_SCN_REAL " " __SIF_SCN_REAL,
          &x, &y, &z, &r) != 4) {
      SIF_LOG_ERROR("io", "failed reading void %" PRIu64 " from %s", i, filepath);
      sif_catalog_free(catalog);
      fclose(file);
      return NULL;
    }

    catalog->cx[i] = x;
    catalog->cy[i] = y;
    catalog->cz[i] = z;
    catalog->radii[i] = r;
  }

  /* Update the size tracking variable since we bypassed the append function */
  catalog->n_voids = n_voids;

  fclose(file);
  SIF_LOG_INFO("io", "loaded %" PRIu64 " voids from %s (ASCII)", n_voids, filepath);

  return catalog;
}