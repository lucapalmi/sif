#ifndef __SIF_PY_OCTREE_H__
#define __SIF_PY_OCTREE_H__

#include "py_common.h"
#include "sif/structures/octree.h"

/*
 * @brief Python object wrapping the C sif_octree_t struct
 */
typedef struct {
  PyObject_HEAD sif_octree_t* tree;
} sifOctreeObject;

/*
 * @brief Expose the Type Object for module registration and type-checking
 */
extern PyTypeObject sifOctreeType;

#endif /* __SIF_PY_OCTREE_H__ */
