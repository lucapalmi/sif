#ifndef __SIF_GET_SYSTEM_H__
#define __SIF_GET_SYSTEM_H__

#include <stdbool.h>
#include <stdint.h>

#include "math/fft.h"
#include "sif/utils/timer.h"

/* --- Internal Logger State --- */
typedef struct {
  uint8_t level;
  bool verbose;
} logger_state_t;

typedef struct {
  logger_state_t logger;

  uint8_t save_memory;
  uint32_t max_threads;
  fft_manager_t* fft_mgr;

  sif_timer_t total_runtime_timer;
  sif_timer_t scratch_timer;
} system_state_t;

system_state_t* get_system_state();
system_state_t* get_system_state_safe();

/* --- OpenMP Wrappers --- */
int system_get_max_threads(void);
int system_get_num_threads(void);
int system_get_thread_num(void);

#endif /* __SIF_GET_SYSTEM_H__ */
