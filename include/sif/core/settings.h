#ifndef __SIF_SETTINGS_H__
#define __SIF_SETTINGS_H__

#include "sif/core/macros.h"

/*
 * @brief Set a runtime setting (creates it if missing, overwrites if exists).
 * The change is immediately applied in memory and flagged for auto-saving.
 * You can use environment variables (e.g. $HOME or ${USER}) in the value.
 */
void sif_setting_set(const char* key, const char* value);

/*
 * @brief Get a runtime setting. Returns fallback_value if the key doesn't
 * exist. If the key is missing but a fallback is provided, the fallback is
 * safely injected into the settings and returned. The returned pointer is
 * strictly memory-safe and fully expanded.
 */
const char* sif_setting_get(const char* key, const char* fallback_value);

#endif /* __SIF_SETTINGS_H__ */