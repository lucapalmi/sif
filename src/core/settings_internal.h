/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file settings_internal.h
 * @brief How the library reads its own settings: typed, validated, and with
 * the default coming from the registry in settings.c rather than the call
 * site.
 *
 * Each reader asserts that the key is registered as its kind. Before
 * sif_system_init() they read the registry's default.
 */

#ifndef SIF__CORE_SETTINGS_INTERNAL_H
#define SIF__CORE_SETTINGS_INTERNAL_H

#include <stdbool.h>

#include "sif/core/macros.h"

/**
 * @brief A numeric setting. A value that is not a number, or is below the
 * setting's minimum, reads as the default, with a warning naming where the
 * value came from.
 */
double sif__setting_number(const char* key);

/** @brief A flag setting, with the same fallback as sif__setting_number(). */
bool sif__setting_flag(const char* key);

/**
 * @brief A directory setting, created along with any missing parents.
 *
 * Unlike the other readers there is no fallback: a path naming an unset
 * variable, an empty path or one that cannot be created fails, having logged
 * where the value came from, rather than putting data somewhere the user did
 * not say.
 *
 * @param[out] path The expanded path, borrowed from the table and valid until
 * the setting is next written; NULL on failure.
 * @return SIF_OK; SIF_ERR_INVALID for an unusable value; SIF_ERR_IO if the
 * directory cannot be created; SIF_ERR_UNINITIALISED before
 * sif_system_init().
 */
SIF_NODISCARD int sif__setting_directory(const char* key, const char** path);

/**
 * @brief Create @p path and every missing component above it, as `mkdir -p`
 * does. Permissions are left to the process umask.
 *
 * @return SIF_OK if the directory exists afterwards; SIF_ERR_INVALID for an
 * empty or over-long path; SIF_ERR_IO, having logged, otherwise.
 */
SIF_NODISCARD int sif__ensure_dir(const char* path);

#endif /* SIF__CORE_SETTINGS_INTERNAL_H */
