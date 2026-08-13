/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#ifndef SIF_TESTS_TEST_UTIL_H
#define SIF_TESTS_TEST_UTIL_H

/*
 * Sanitizer builds run these tests 50-100x slower, and several of them carry
 * an O(n_queries * n_particles) brute-force reference. Scale the problem sizes
 * down when instrumented so the suite stays runnable under ASan/UBSan/TSan
 * while keeping full size for normal builds.
 */

#if defined(__has_feature)
#  if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) ||   \
    __has_feature(memory_sanitizer)
#    define SIF_TEST_INSTRUMENTED 1
#  endif
#endif

#if !defined(SIF_TEST_INSTRUMENTED) &&                                         \
  (defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__))
#  define SIF_TEST_INSTRUMENTED 1
#endif

#ifndef SIF_TEST_INSTRUMENTED
#  define SIF_TEST_INSTRUMENTED 0
#endif

#if SIF_TEST_INSTRUMENTED
#  define SIF_TEST_SCALE(n) (((n) / 20) > 1 ? ((n) / 20) : 2)
#else
#  define SIF_TEST_SCALE(n) (n)
#endif

#endif /* SIF_TESTS_TEST_UTIL_H */
