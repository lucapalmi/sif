#ifndef __SIF_PY_CATALOG_H__
#define __SIF_PY_CATALOG_H__

#include "py_common.h"
#include "sif/structures/catalog.h"

/*
 * @brief Python object wrapping the C sif_catalog_t struct
 */
typedef struct {
  PyObject_HEAD sif_catalog_t* catalog;
} sifCatalogObject;

/*
 * @brief Expose the Type Object for module registration and type-checking
 */
extern PyTypeObject sifCatalogType;

#endif /* __SIF_PY_CATALOG_H__ */