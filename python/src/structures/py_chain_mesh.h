#ifndef __SIF_PY_CHAIN_MESH_H__
#define __SIF_PY_CHAIN_MESH_H__

#include "py_common.h"
#include "sif/structures/chain_mesh.h"

/*
 * @brief Python object wrapping the C sif_chain_mesh_t struct
 */
typedef struct {
  PyObject_HEAD sif_chain_mesh_t* mesh;
} sifChainMeshObject;

/*
 * @brief Expose the Type Object for module registration and type-checking
 */
extern PyTypeObject sifChainMeshType;

#endif /* __SIF_PY_CHAIN_MESH_H__ */
