#ifndef PREDICATES_H
#define PREDICATES_H

#include "sif/core/macros.h"

void exactinit();

#ifdef __PREDICATES_USE_FLOAT
float orient2d(float pa[2], float pb[2], float pc[2]);
float orient3d(float pa[3], float pb[3], float pc[3], float pd[3]);
float orient3dfast(float pa[3], float pb[3], float pc[3], float pd[3]);
float incircle(float pa[2], float pb[2], float pc[2], float pd[2]);
float insphere(float pa[3], float pb[3], float pc[3], float pd[3], float pe[3]);
#else
double orient2d(double pa[2], double pb[2], double pc[2]);
double orient3d(double pa[3], double pb[3], double pc[3], double pd[3]);
double incircle(double pa[2], double pb[2], double pc[2], double pd[2]);
double insphere(
  double pa[3], double pb[3], double pc[3], double pd[3], double pe[3]);
#endif

#endif
