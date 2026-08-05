#ifndef __SIF_TIMER_H__
#define __SIF_TIMER_H__

#include <time.h>

/*
 * @brief Simple monotonic wall-clock timer
 */
typedef struct {
  struct timespec start_time;
  struct timespec stop_time;
} sif_timer_t;

/*
 * @brief Start the timer
 *
 * @param timer The timer
 */
void sif_timer_start(sif_timer_t* timer);

/*
 * @brief Stop the timer
 *
 * @param timer The timer
 */
void sif_timer_stop(sif_timer_t* timer);

/*
 * @brief Get the elapsed time in milliseconds
 *        Must be called after finder_timer_stop
 *
 * @param timer The timer
 *
 * @return Elapsed time in ms
 */
double sif_timer_elapsed_ms(const sif_timer_t* timer);

#endif /* __SIF__TIMER_H__ */
