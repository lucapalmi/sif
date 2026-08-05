#ifndef __SIF_SYSTEM_H__
#define __SIF_SYSTEM_H__

#include <stdbool.h>
#include <stdint.h>

#include "sif/core/macros.h"

/*
 * @brief FFTW3 configuration
 */
typedef struct {
  bool skip_tuning;
} sif_fft_config_t;

/*
 *@brief OpenMP configuration
 */
typedef struct {
  uint32_t n_threads;
} sif_omp_config_t;

/*
 * @brief Genral library configuration
 */
typedef struct {
  sif_fft_config_t* fft_config;
  sif_omp_config_t* omp_config;
  bool verbose;
  uint8_t log_level;
} sif_config_t;

/*
 *@brief Initializes the library
 *
 * @param config A configuration struct for the library
 */
void sif_init(sif_config_t* config);

/*
 *@brief Finalizes the library
 */
void sif_finalize();

#define SIF_VERBOSE                                                            \
  (&(sif_config_t){.fft_config = NULL,                                         \
    .omp_config = NULL,                                                        \
    .verbose = true,                                                           \
    .log_level = SIF_LOG_LEVEL_TRACE})

#define SIF_QUIET                                                              \
  (&(sif_config_t){.fft_config = NULL,                                         \
    .omp_config = NULL,                                                        \
    .verbose = false,                                                          \
    .log_level = SIF_LOG_LEVEL_WARNING})

#define SIF_STANDARD                                                           \
  (&(sif_config_t){.fft_config = NULL,                                         \
    .omp_config = NULL,                                                        \
    .verbose = false,                                                          \
    .log_level = SIF_LOG_LEVEL_INFO})

#endif /* __SIF_SYSTEM_H__ */
