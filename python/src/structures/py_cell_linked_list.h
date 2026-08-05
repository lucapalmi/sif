#ifndef __SIF_PY_CELL_LINKED_LIST_H__
#define __SIF_PY_CELL_LINKED_LIST_H__

#include "py_common.h"
#include "sif/structures/cell_linked_list.h"

/*
 * @brief Python object wrapping the C sif_cell_linked_list_t struct
 */
typedef struct {
  PyObject_HEAD sif_cell_linked_list_t* cll;
} sifCellLinkedListObject;

/*
 * @brief Expose the Type Object for module registration and type-checking
 */
extern PyTypeObject sifCellLinkedListType;

#endif /* __SIF_PY_CELL_LINKED_LIST_H__ */
