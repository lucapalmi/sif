#ifndef __SIF_SORT_H__
#define __SIF_SORT_H__

#include <stdint.h>

#define SIF_DEFINE_QUICKSORT(function_name, type, less_than_expr)              \
  static void function_name##_recursive(                                       \
    type* arr, int64_t low, int64_t high) {                                    \
    if (low >= high)                                                           \
      return;                                                                  \
    int64_t lt = low, gt = high;                                               \
    type pivot = arr[low + (high - low) / 2];                                  \
    int64_t i = low;                                                           \
    while (i <= gt) {                                                          \
      type a = arr[i], b = pivot;                                              \
      uint8_t is_less = (less_than_expr);                                      \
      a = pivot;                                                               \
      b = arr[i];                                                              \
      uint8_t is_greater = (less_than_expr);                                   \
      if (is_less) {                                                           \
        type temp = arr[lt];                                                   \
        arr[lt] = arr[i];                                                      \
        arr[i] = temp;                                                         \
        lt++;                                                                  \
        i++;                                                                   \
      } else if (is_greater) {                                                 \
        type temp = arr[i];                                                    \
        arr[i] = arr[gt];                                                      \
        arr[gt] = temp;                                                        \
        gt--;                                                                  \
      } else {                                                                 \
        i++;                                                                   \
      }                                                                        \
    }                                                                          \
    function_name##_recursive(arr, low, lt - 1);                               \
    function_name##_recursive(arr, gt + 1, high);                              \
  }                                                                            \
  static inline void function_name(type* arr, uint64_t count) {                \
    if (count > 1) {                                                           \
      function_name##_recursive(arr, 0, (int64_t)count - 1);                   \
    }                                                                          \
  }

#endif /* __SIF_SORT_H__ */
