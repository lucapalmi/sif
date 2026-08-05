#include "sif/core/settings.h"
#include "sif/utils/logger.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <process.h>
#  define getpid _getpid
#else
#  include <unistd.h>
#endif

/* The Dual-State Dictionary Struct for C99-compliant Thread Safety */
typedef struct {
  char key[64];
  char raw_value[256];
  char expanded_value[256];
} sif_kv_pair_t;

static sif_kv_pair_t* _settings = NULL;
static uint32_t _settings_count = 0;
static uint32_t _settings_capacity = 0;
static char _settings_filepath[512] = {0};

/* Tracks if we actually need to save to disk */
static uint8_t _settings_modified = 0;

static char* _trim_whitespace(char* str) {
  while (isspace((unsigned char)*str))
    str++;
  if (*str == 0)
    return str;
  char* end = str + strlen(str) - 1;
  while (end > str && isspace((unsigned char)*end))
    end--;
  end[1] = '\0';
  return str;
}

/* Dynamically translates $HOME or ${USER} into absolute paths */
static void _expand_env_vars(const char* input, char* output, size_t max_len) {
  const char* src = input;
  char* dst = output;
  size_t len = 0;

  while (*src && len < max_len - 1) {
    if (*src == '$') {
      src++;
      char var_name[64] = {0};
      int v_idx = 0;
      int has_braces = (*src == '{');

      if (has_braces)
        src++;

      while (
        *src && (isalnum((unsigned char)*src) || *src == '_') && v_idx < 63) {
        var_name[v_idx++] = *src++;
      }

      if (has_braces && *src == '}')
        src++;

      if (v_idx > 0) {
        char* env_val = getenv(var_name);
        if (env_val) {
          size_t env_len = strlen(env_val);
          if (len + env_len < max_len - 1) {
            strcpy(dst, env_val);
            dst += env_len;
            len += env_len;
          }
        }
      }
    } else {
      *dst++ = *src++;
      len++;
    }
  }
  *dst = '\0';
}

static void __sif_settings_save(void) {
  if (!_settings || _settings_filepath[0] == '\0' || !_settings_modified)
    return;

  char tmp_path[1024];
  snprintf(
    tmp_path, sizeof(tmp_path), "%s.tmp.%d", _settings_filepath, getpid());

  FILE* f = fopen(tmp_path, "w");
  if (!f) {
    SIF_LOG_WARNING("settings", "failed to open temporary config file");
    return;
  }

  fprintf(f, "# SIF Runtime Configuration\n");
  fprintf(
    f, "# This file is auto-generated. Paths support $ENV variables.\n\n");

  for (uint32_t i = 0; i < _settings_count; i++) {
    /* Write the RAW value (e.g. $HOME/.sif) to disk, not the expanded one */
    fprintf(f, "%s = %s\n", _settings[i].key, _settings[i].raw_value);
  }
  fclose(f);

  /* Atomic rename prevents parallel job array corruption */
  if (rename(tmp_path, _settings_filepath) == 0) {
    SIF_LOG_TRACE("settings", "auto-saved configuration to disk");
    _settings_modified = 0;
  } else {
    remove(tmp_path);
  }
}

void __sif_settings_init(const char* default_dir) {
  snprintf(
    _settings_filepath, sizeof(_settings_filepath), "%s/config", default_dir);
  _settings_capacity = 16;
  _settings_count = 0;
  _settings = malloc(_settings_capacity * sizeof(sif_kv_pair_t));

  FILE* f = fopen(_settings_filepath, "r");
  if (f) {
    char line[512];
    while (fgets(line, sizeof(line), f)) {
      char* trimmed = _trim_whitespace(line);
      if (trimmed[0] == '\0' || trimmed[0] == '#' || trimmed[0] == ';')
        continue;

      char* eq = strchr(trimmed, '=');
      if (eq) {
        *eq = '\0';
        char* key = _trim_whitespace(trimmed);
        char* val = _trim_whitespace(eq + 1);
        if (strlen(key) > 0)
          sif_setting_set(key, val);
      }
    }
    fclose(f);
    SIF_LOG_TRACE(
      "settings", "loaded runtime config from %s", _settings_filepath);
  }

  /* Reset dirty flag so we don't save just because we loaded a file */
  _settings_modified = 0;

  /* Inject Smart Defaults! (Because they use fallbacks, they trigger
   * auto-injection if missing) */
  sif_setting_get("cache_directory", "$HOME/.sif/grid_cache");
  sif_setting_get("fft_wisdom_dir", "$HOME/.sif/wisdoms");
}

void __sif_settings_finalize(void) {
  /* Auto-save if anything was touched during execution! */
  if (_settings_modified) {
    __sif_settings_save();
  }

  if (_settings) {
    free(_settings);
    _settings = NULL;
  }
  _settings_count = 0;
  _settings_capacity = 0;
}

void sif_setting_set(const char* key, const char* value) {
  if (!_settings || !key || !value)
    return;

  /* Check if it exists */
  for (uint32_t i = 0; i < _settings_count; i++) {
    if (strcmp(_settings[i].key, key) == 0) {
      if (strcmp(_settings[i].raw_value, value) == 0)
        return; /* Value hasn't changed, no need to dirty the flag */

      strncpy(_settings[i].raw_value, value, 255);
      _settings[i].raw_value[255] = '\0';
      _expand_env_vars(value, _settings[i].expanded_value, 256);

      _settings_modified = 1;
      return;
    }
  }

  /* Expand array if full */
  if (_settings_count == _settings_capacity) {
    _settings_capacity *= 2;
    _settings = realloc(_settings, _settings_capacity * sizeof(sif_kv_pair_t));
  }

  /* Add new key */
  strncpy(_settings[_settings_count].key, key, 63);
  _settings[_settings_count].key[63] = '\0';

  strncpy(_settings[_settings_count].raw_value, value, 255);
  _settings[_settings_count].raw_value[255] = '\0';
  _expand_env_vars(value, _settings[_settings_count].expanded_value, 256);

  _settings_count++;
  _settings_modified = 1;
}

const char* sif_setting_get(const char* key, const char* fallback) {
  if (!_settings || !key)
    return fallback;

  for (uint32_t i = 0; i < _settings_count; i++) {
    if (strcmp(_settings[i].key, key) == 0) {
      return _settings[i]
        .expanded_value; /* Safely return the expanded pointer */
    }
  }

  /* Get-or-Create pattern: If key is missing but fallback exists, inject it! */
  if (fallback) {
    sif_setting_set(key, fallback);
    return _settings[_settings_count - 1].expanded_value;
  }

  return NULL;
}