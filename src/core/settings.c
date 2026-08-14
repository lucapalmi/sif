/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/core/settings.h"

#include "sif/utils/logger.h"

#include "core/system_internal.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Fixed-size entries, so the table is one flat allocation and a lookup is a
 * linear scan over contiguous memory. There are a handful of settings, read a
 * handful of times per run; anything cleverer would be harder to reason about
 * for no measurable gain. */
#define KEY_CAP   64
#define VALUE_CAP 256
#define PATH_CAP  512
#define LINE_CAP  512

#define INITIAL_CAPACITY 16

/*
 * One entry, stored twice.
 *
 * The raw text is what goes back to disk, so that "$HOME/.sif" stays portable
 * across machines and users; the expansion is what callers get, because
 * sif_setting_get() returns a borrowed pointer and has nowhere to expand into.
 * Keeping both is what lets the file round-trip without the reader ever seeing
 * an unexpanded path.
 */
typedef struct {
  char key[KEY_CAP];
  char raw_value[VALUE_CAP];
  char expanded_value[VALUE_CAP];
} kv_pair_t;

static kv_pair_t* settings = NULL;
static uint32_t settings_count = 0;
static uint32_t settings_capacity = 0;
static char settings_filepath[PATH_CAP] = {0};

/* Whether the table differs from what is on disk. Saving is skipped otherwise,
 * so a read-only run never rewrites the file -- which also keeps a job array
 * of hundreds of ranks from all racing to save the same unchanged table. */
static bool settings_modified = false;

/* --- string helpers --- */

/* Trims in place at the tail and by returning a later pointer at the head, so
 * the caller keeps ownership of the original buffer. */
static char* trim_whitespace(char* str) {
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

/* Copies into a fixed field, reporting truncation rather than letting a
 * half-written path through: a truncated directory still names something the
 * run would happily write into. */
static void copy_bounded(char* dst, size_t cap, const char* src) {
  size_t len = strlen(src);
  if (len >= cap) {
    SIF_LOG_WARNING(
      "settings", "%zu-character entry truncated to %zu", len, cap - 1);
    len = cap - 1;
  }
  memcpy(dst, src, len);
  dst[len] = '\0';
}

/*
 * Substitutes $VAR and ${VAR} from the environment.
 *
 * Expansion happens once, when the value is written, so a setting resolves
 * against the environment of the process that set it and not against whatever
 * it becomes later -- the alternative would have the same key mean different
 * things at different points in one run.
 *
 * An unset variable expands to nothing, as in a shell. Output is truncated to
 * fit rather than dropping the offending piece, so the result is at least a
 * prefix of the intended path.
 */
static void expand_env_vars(const char* src, char* out, size_t cap) {
  size_t len = 0;

  while (*src && len + 1 < cap) {
    if (*src != '$') {
      out[len++] = *src++;
      continue;
    }

    src++;
    const bool braced = (*src == '{');
    if (braced)
      src++;

    char name[KEY_CAP];
    size_t n = 0;
    while (*src && (isalnum((unsigned char)*src) || *src == '_') &&
           n + 1 < sizeof(name)) {
      name[n++] = *src++;
    }
    name[n] = '\0';

    if (braced && *src == '}')
      src++;

    /* A '$' with no name behind it is far more likely a literal than a
     * mistake, so it is passed through untouched. */
    if (n == 0) {
      out[len++] = '$';
      if (braced && len + 1 < cap)
        out[len++] = '{';
      continue;
    }

    const char* value = getenv(name);
    if (!value)
      continue;

    while (*value && len + 1 < cap)
      out[len++] = *value++;
  }

  out[len] = '\0';
}

/* --- table --- */

/* Index of @p key, or -1 if absent. */
static int32_t settings_find(const char* key) {
  for (uint32_t i = 0; i < settings_count; i++) {
    if (strcmp(settings[i].key, key) == 0)
      return (int32_t)i;
  }
  return -1;
}

/*
 * Inserts or overwrites, returning the entry's index or -1 if it could not be
 * stored. Both public entry points go through here, so validation and the
 * dirty flag are decided in exactly one place.
 */
static int32_t settings_put(const char* key, const char* value) {
  /* The file format is one `key = value` per line with no escaping, so a
   * newline anywhere, or an '=' in a key, would produce a file that no longer
   * reads back as what was written. Refusing beats corrupting. */
  if (strchr(key, '\n') || strchr(key, '=') || strchr(value, '\n')) {
    SIF_LOG_WARNING("settings", "rejected unrepresentable setting '%s'", key);
    return -1;
  }

  int32_t idx = settings_find(key);
  if (idx >= 0) {
    if (strcmp(settings[idx].raw_value, value) == 0)
      return idx; /* unchanged: leave the table clean */

    copy_bounded(settings[idx].raw_value, VALUE_CAP, value);
    expand_env_vars(value, settings[idx].expanded_value, VALUE_CAP);
    settings_modified = true;
    return idx;
  }

  if (settings_count == settings_capacity) {
    uint32_t new_capacity =
      settings_capacity ? settings_capacity * 2 : INITIAL_CAPACITY;
    kv_pair_t* grown = realloc(settings, new_capacity * sizeof(kv_pair_t));
    if (!grown) {
      SIF_LOG_ERROR("settings",
        "cannot grow the table to %u entries; '%s' "
        "not stored",
        new_capacity, key);
      return -1;
    }
    settings = grown;
    settings_capacity = new_capacity;
  }

  idx = (int32_t)settings_count++;
  copy_bounded(settings[idx].key, KEY_CAP, key);
  copy_bounded(settings[idx].raw_value, VALUE_CAP, value);
  expand_env_vars(value, settings[idx].expanded_value, VALUE_CAP);
  settings_modified = true;

  return idx;
}

/*
 * Writes the table out through a temporary file and a rename.
 *
 * A job array pointed at one $HOME has every rank finalizing at once, into a
 * file the others may still be reading. Rename is atomic within a filesystem,
 * so each reader sees one complete table or the other, never a half-written
 * one; the pid in the temporary name keeps the ranks from colliding before
 * they get there.
 */
static void settings_save(void) {
  if (!settings || settings_filepath[0] == '\0' || !settings_modified)
    return;

  char tmp_path[PATH_CAP + 32];
  int written = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld",
    settings_filepath, (long)getpid());
  if (written < 0 || (size_t)written >= sizeof(tmp_path))
    return;

  FILE* f = fopen(tmp_path, "w");
  if (!f) {
    SIF_LOG_WARNING("settings", "failed to open temporary config file");
    return;
  }

  fprintf(f, "# SIF Runtime Configuration\n");
  fprintf(
    f, "# This file is auto-generated. Paths support $ENV variables.\n\n");

  for (uint32_t i = 0; i < settings_count; i++) {
    fprintf(f, "%s = %s\n", settings[i].key, settings[i].raw_value);
  }

  /* Check the writes before the rename. A full disk or an exceeded quota
   * fails at fclose(), and renaming regardless would replace a good table
   * with a truncated one -- the exact failure the rename exists to prevent. */
  bool ok = (ferror(f) == 0);
  if (fclose(f) != 0)
    ok = false;

  if (ok && rename(tmp_path, settings_filepath) == 0) {
    SIF_LOG_TRACE("settings", "auto-saved configuration to disk");
    settings_modified = false;
  } else {
    SIF_LOG_WARNING(
      "settings", "failed to save configuration to %s", settings_filepath);
    remove(tmp_path);
  }
}

void sif__settings_init(const char* default_dir) {
  int written = snprintf(
    settings_filepath, sizeof(settings_filepath), "%s/config", default_dir);
  if (written < 0 || (size_t)written >= sizeof(settings_filepath)) {
    SIF_LOG_WARNING(
      "settings", "config path too long; settings will not persist");
    settings_filepath[0] = '\0';
  }

  settings_count = 0;
  settings_capacity = INITIAL_CAPACITY;
  settings = malloc(settings_capacity * sizeof(kv_pair_t));
  if (!settings) {
    SIF_LOG_ERROR("settings", "failed to allocate the settings table");
    settings_capacity = 0;
    return;
  }

  FILE* f = settings_filepath[0] ? fopen(settings_filepath, "r") : NULL;
  if (f) {
    char line[LINE_CAP];
    while (fgets(line, sizeof(line), f)) {
      /* An over-long line comes back in pieces, and the tail of one would
       * parse as an entry of its own -- so discard it whole. Reaching EOF
       * without a newline is the ordinary last line, not an over-long one. */
      if (!strchr(line, '\n') && !feof(f)) {
        SIF_LOG_WARNING(
          "settings", "ignoring over-long line in %s", settings_filepath);
        int c;
        while ((c = fgetc(f)) != EOF && c != '\n') {
        }
        continue;
      }

      char* trimmed = trim_whitespace(line);
      if (trimmed[0] == '\0' || trimmed[0] == '#' || trimmed[0] == ';')
        continue;

      char* eq = strchr(trimmed, '=');
      if (!eq)
        continue;

      *eq = '\0';
      char* key = trim_whitespace(trimmed);
      char* val = trim_whitespace(eq + 1);
      if (key[0] != '\0')
        settings_put(key, val);
    }
    fclose(f);
    SIF_LOG_TRACE(
      "settings", "loaded runtime config from %s", settings_filepath);
  }

  /* Loading is not a modification: without this the first finalize would
   * rewrite a file identical to the one just read. */
  settings_modified = false;

  /* Defaults go in through the get-or-create path, so the first run writes a
   * file that lists them. That file is the documentation for these knobs --
   * a user who wants the cache elsewhere has something concrete to edit.
   *
   * They are expressed against $HOME where there is one, so the same file
   * still resolves on a machine that mounts the home directory elsewhere.
   * Without HOME the caller has already fallen back to a real directory, and
   * recording that beats recording a "$HOME/..." that expands to "/...". */
  const char* root = getenv("HOME") ? "$HOME/.sif" : default_dir;
  char path[VALUE_CAP];

  snprintf(path, sizeof(path), "%s/grid_cache", root);
  sif_setting_get("cache_directory", path);

  snprintf(path, sizeof(path), "%s/wisdoms", root);
  sif_setting_get("fft_wisdom_dir", path);
}

void sif__settings_finalize(void) {
  settings_save();

  free(settings);
  settings = NULL;
  settings_count = 0;
  settings_capacity = 0;
  settings_filepath[0] = '\0';
  settings_modified = false;
}

/* --- public accessors --- */

void sif_setting_set(const char* key, const char* value) {
  if (!settings || !key || !value)
    return;

  settings_put(key, value);
}

const char* sif_setting_get(const char* key, const char* fallback) {
  if (!settings || !key)
    return fallback;

  int32_t idx = settings_find(key);
  if (idx < 0) {
    if (!fallback)
      return NULL;

    /* Get-or-create: a caller that had to supply a default has just told us
     * what the setting should be, and recording it is what makes the setting
     * visible in the file instead of living only in the call site. */
    idx = settings_put(key, fallback);
    if (idx < 0)
      return fallback;
  }

  return settings[idx].expanded_value;
}

#undef KEY_CAP
#undef VALUE_CAP
#undef PATH_CAP
#undef LINE_CAP
#undef INITIAL_CAPACITY
