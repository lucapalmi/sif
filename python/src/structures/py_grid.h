#ifndef __SIF_PY_GRID_H__
#define __SIF_PY_GRID_H__

#include "py_common.h"
#include "sif/structures/grid.h"

typedef struct {
  PyObject_HEAD sif_grid_t* grid;
} sifGridObject;

extern PyTypeObject sifGridType;

#endif /* __SIF_PY_GRID_H__ */