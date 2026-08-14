/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file settings.h
 * @brief Persistent key/value runtime settings.
 *
 * The table is process-global, populated by sif_init() from the settings file
 * and written back on sif_finalize() if anything changed. Values may reference
 * environment variables (`$HOME`, `${USER}`), which are expanded on write and
 * stored alongside the raw text.
 *
 * @warning The table is not synchronized. Reads are safe once the settings are
 * populated, but calling sif_setting_set() concurrently with any other access
 * is a data race -- and so is sif_setting_get() with a fallback, which writes
 * the fallback into the table when the key is absent.
 */

#ifndef SIF_CORE_SETTINGS_H
#define SIF_CORE_SETTINGS_H

#include "sif/core/macros.h"

/**
 * @brief Set a runtime setting, creating it if absent and overwriting if not.
 *
 * The value takes effect immediately and the table is flagged for saving at
 * finalization. Environment references in @p value are expanded now, not at
 * read time, so a later change to the environment is not picked up.
 *
 * Keys are truncated at 63 characters, values at 255, with a warning.
 *
 * @param key Setting name. Ignored if NULL, or if called before sif_init().
 * @param value Raw value, possibly containing environment references.
 *
 * @note The file format is one `key = value` per line with no escaping, so a
 * key containing `=` or a newline, or a value containing a newline, would not
 * read back as what was written and is rejected with a warning.
 *
 * @warning Adding a new key may reallocate the table, which invalidates every
 * pointer previously returned by sif_setting_get().
 */
void sif_setting_set(const char* key, const char* value);

/**
 * @brief Look up a runtime setting, with an optional default.
 *
 * If @p key is missing and @p fallback is non-NULL, the fallback is inserted
 * into the table under @p key and then returned, so the next call finds it.
 *
 * @param key Setting name.
 * @param fallback Value to install and return if the key is absent. May be
 * NULL to query without creating.
 * @return The expanded value, owned by the settings table -- the caller must
 * neither free nor modify it. Returns @p fallback unexpanded if the table is
 * not yet populated or the entry could not be created, and NULL if the key is
 * absent and no fallback was given.
 *
 * @warning The returned pointer is valid only until the next
 * sif_setting_set(), including the implicit one this function performs when it
 * installs a fallback. Copy the string if it has to outlive that.
 */
const char* sif_setting_get(const char* key, const char* fallback);

#endif /* SIF_CORE_SETTINGS_H */
