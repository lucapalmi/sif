#ifndef __SIF_LOGGER_H__
#define __SIF_LOGGER_H__

#include <stdint.h>
#include <stdio.h>

#include "sif/core/macros.h"

/* Minimum COMPILE-TIME level. Defaults to TRACE so runtime handles it. */
#ifndef SIF_LOG_LEVEL
#  define SIF_LOG_LEVEL SIF_LOG_LEVEL_TRACE
#endif

void __sif_log_impl(uint8_t level, const char* tag, const char* fmt, ...);

void __sif_log_flush(void);

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_TRACE
#  define SIF_LOG_TRACE(tag, fmt, ...)                                         \
    __sif_log_impl(SIF_LOG_LEVEL_TRACE, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_TRACE(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_DEBUG
#  define SIF_LOG_DEBUG(tag, fmt, ...)                                         \
    __sif_log_impl(SIF_LOG_LEVEL_DEBUG, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_DEBUG(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_INFO
#  define SIF_LOG_INFO(tag, fmt, ...)                                          \
    __sif_log_impl(SIF_LOG_LEVEL_INFO, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_INFO(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_WARNING
#  define SIF_LOG_WARNING(tag, fmt, ...)                                       \
    __sif_log_impl(SIF_LOG_LEVEL_WARNING, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_WARNING(tag, fmt, ...) ((void)0)
#endif

#if SIF_LOG_LEVEL <= SIF_LOG_LEVEL_ERROR
#  define SIF_LOG_ERROR(tag, fmt, ...)                                         \
    __sif_log_impl(SIF_LOG_LEVEL_ERROR, tag, fmt, ##__VA_ARGS__)
#else
#  define SIF_LOG_ERROR(tag, fmt, ...) ((void)0)
#endif

#define SIF_LOG_FLUSH() __sif_log_flush()

#endif /* __SIF_LOGGER_H__ */
