#ifndef __SIF_PY_PROFILES_H__
#define __SIF_PY_PROFILES_H__

#include "py_common.h"
#include "sif/measure/profiles.h"

/*
 * @brief Unified Python container wrapping density and/or velocity structures
 */
typedef struct {
  PyObject_HEAD 
  sif_density_profiles_t* dens;
  sif_velocity_profiles_t* vel;
} sifProfilesObject;

extern PyTypeObject sifProfilesType;

/* Functional API Entry Point */
PyObject* py_sif_profiles(PyObject* self, PyObject* args, PyObject* kwds);

#endif /* __SIF_PY_PROFILES_H__ */