/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/core/settings.h"

#include "sif/utils/logger.h"

#include "core/settings_internal.h"
#include "core/system_internal.h"

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
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

/* The largest number a numeric setting takes. Exact as a double and as a
 * uint64_t, so a value read as a count converts without overflow -- which
 * "1e30" or "inf" would not. */
#define NUMBER_MAX 1e15

/* --- the settings sif knows --- */

typedef enum {
  SETTING_PATH,   /* a directory, created on first use */
  SETTING_FLAG,   /* 1/0, true/false, yes/no, on/off */
  SETTING_NUMBER, /* a number no smaller than setting_def_t::min */
} setting_kind_t;

/*
 * One setting: its name, what it holds, its default as it would be written in
 * the file, and the comment the file carries above it.
 *
 * The list is the single source of truth. Every key in it is in the table from
 * sif__settings_init() on, with its default, so the library reads a setting
 * without supplying one and no default lives at a call site; sif_setting_set()
 * refuses any key not in it, so a misspelt key is an error instead of a
 * setting nothing reads; and the file is written from it, so every setting is
 * listed there, documented, whether or not it was ever changed.
 */
typedef struct {
  const char* key;
  setting_kind_t kind;
  const char* fallback;
  double min;
  const char* doc;
} setting_def_t;

static const setting_def_t registry[] = {
  {"cache_directory", SETTING_PATH, "$SIF_HOME/grid_cache", 0.0,
    "Where grid_cache_enabled keeps the grids it computes."},
  {"grid_cache_enabled", SETTING_FLAG, "0", 0.0,
    "Cache CIC grids on disk. Never evicted: a 2048^3 grid is ~34 GB."},
  {"fft_wisdom_dir", SETTING_PATH, "$SIF_HOME/wisdoms", 0.0,
    "Where FFTW's tuned plans are kept between runs."},
  {"fft_tuning_max_gib", SETTING_NUMBER, SIF__FFT_TUNING_MAX_GIB_DEFAULT, 0.0,
    "Largest spectrum, in GiB, whose transforms are tuned; 0 never tunes."},
  {"cic_tile_particles", SETTING_NUMBER, SIF__CIC_TILE_PARTICLES_DEFAULT, 1.0,
    "Particles per tile of the CIC deposit's sort; bounds its scratch."},
  {"scratch_directory", SETTING_PATH, "$SIF_HOME/scratch", 0.0,
    "Where save_memory runs keep the grid they move out of memory: as large "
    "as the grid, read once per smoothing, so a fast filesystem with room."},
};

#define N_REGISTRY (sizeof(registry) / sizeof(registry[0]))

/* --- the table --- */

/*
 * Where a value comes from, lowest precedence first: a setting reads as its
 * highest layer that is set. The default is always set for a known key, so a
 * known key always reads as something.
 */
enum {
  LAYER_DEFAULT,
  LAYER_FILE,
  LAYER_ENV,
  LAYER_SESSION,
  N_LAYERS
};

/*
 * One value, stored twice.
 *
 * The raw text is what goes back to disk, so that "$HOME/.sif" stays portable
 * across machines and users; the expansion is what callers get, because
 * sif_setting_get() returns a borrowed pointer and has nowhere to expand into.
 * Keeping both is what lets the file round-trip without the reader ever seeing
 * an unexpanded path.
 */
typedef struct {
  bool set;
  uint32_t line; /* LAYER_FILE: the line it was read from, 0 if set since */
  char raw[VALUE_CAP];
  char value[VALUE_CAP];
  /* The first variable `raw` names that the environment does not set, or
   * empty. It expands to nothing, as in a shell, and the typed readers refuse
   * the value: "$TMPDIR/sif" on a node without TMPDIR is "/sif". */
  char unset_var[KEY_CAP];
  /* `raw` or its expansion did not fit, and `value` holds a prefix of it. The
   * typed readers refuse it like an unset variable: a truncated path names
   * another directory. */
  bool truncated;
} layer_t;

typedef struct {
  char key[KEY_CAP];
  const setting_def_t* def; /* NULL for a key this version does not know */
  bool warned;              /* a malformed value has been reported */
  layer_t layers[N_LAYERS];
} entry_t;

/* The known keys come first, in registry order, then whatever unknown keys
 * the file held -- kept so that saving does not drop what a newer sif wrote.
 * The table grows only while the file is read, never after, so the pointers
 * sif_setting_get() hands out never move. */
static entry_t* settings = NULL;
static uint32_t settings_count = 0;
static uint32_t settings_capacity = 0;
static char settings_filepath[PATH_CAP] = {0};
static char settings_root[PATH_CAP] = {0};

/* Whether the file differs from what the table would write. Saving is skipped
 * otherwise, so a read-only run never rewrites the file -- which also keeps a
 * job array of hundreds of ranks from all racing to save the same unchanged
 * table. */
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
 * Substitutes $VAR and ${VAR} from the environment, and $SIF_HOME with the
 * directory holding the settings file whether or not the environment sets it.
 *
 * Expansion happens once, when the value is loaded or set, so a setting
 * resolves against the environment of the process that reads it and not
 * against whatever it becomes later -- the alternative would have the same key
 * mean different things at different points in one run.
 *
 * An unset variable expands to nothing, as in a shell, and the first one is
 * named in @p unset_var for the readers to refuse. Output is truncated to fit,
 * and the return says so: the readers refuse that too.
 */
static bool expand_env_vars(
  const char* src, char* out, size_t cap, char* unset_var) {
  size_t len = 0;
  unset_var[0] = '\0';

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

    /* $SIF_HOME is always the directory sif_system_init() settled on, which
     * is SIF_HOME itself when that is set and not empty -- and something
     * usable when it is empty, where getenv() would give "/scratch". */
    const char* value =
      strcmp(name, "SIF_HOME") == 0 ? settings_root : getenv(name);
    if (!value) {
      if (unset_var[0] == '\0')
        copy_bounded(unset_var, KEY_CAP, name);
      continue;
    }

    while (*value && len + 1 < cap)
      out[len++] = *value++;
    if (*value) {
      out[len] = '\0';
      return true;
    }
  }

  out[len] = '\0';
  return *src != '\0';
}

static void layer_store(layer_t* l, const char* raw, uint32_t line) {
  l->set = true;
  l->line = line;
  copy_bounded(l->raw, VALUE_CAP, raw);
  l->truncated = strlen(raw) >= VALUE_CAP;
  if (expand_env_vars(raw, l->value, VALUE_CAP, l->unset_var))
    l->truncated = true;
}

/* --- lookup --- */

static const setting_def_t* registry_find(const char* key) {
  for (size_t i = 0; i < N_REGISTRY; i++)
    if (strcmp(registry[i].key, key) == 0)
      return &registry[i];
  return NULL;
}

static entry_t* settings_find(const char* key) {
  for (uint32_t i = 0; i < settings_count; i++)
    if (strcmp(settings[i].key, key) == 0)
      return &settings[i];
  return NULL;
}

/* The layer a setting reads as, and which one that is. */
static const layer_t* entry_effective(const entry_t* e, int* which) {
  for (int l = N_LAYERS - 1; l >= 0; l--) {
    if (e->layers[l].set) {
      if (which)
        *which = l;
      return &e->layers[l];
    }
  }
  return NULL;
}

/* The environment variable that overrides @p key: SIF_ and the key in upper
 * case, SIF_FFT_TUNING_MAX_GIB for fft_tuning_max_gib. */
static void env_name(const char* key, char* out, size_t cap) {
  size_t n = (size_t)snprintf(out, cap, "SIF_%s", key);
  for (size_t i = 4; i < n && i < cap; i++)
    out[i] = (char)toupper((unsigned char)out[i]);
}

/*
 * Where the value a setting reads as came from, as the head of a log line:
 * the file and line, the variable, or what it is otherwise. Messages about a
 * value start here, so they point at the place to fix it.
 */
static void describe_origin(
  const entry_t* e, int which, char* out, size_t cap) {
  switch (which) {
  case LAYER_SESSION:
    snprintf(out, cap, "%s (set for this process)", e->key);
    break;
  case LAYER_ENV: {
    char var[KEY_CAP + 4];
    env_name(e->key, var, sizeof(var));
    snprintf(out, cap, "%s", var);
    break;
  }
  case LAYER_FILE:
    if (e->layers[LAYER_FILE].line)
      snprintf(out, cap, "%s:%u: %s", settings_filepath,
        e->layers[LAYER_FILE].line, e->key);
    else
      snprintf(out, cap, "%s (set for %s)", e->key, settings_filepath);
    break;
  default:
    snprintf(out, cap, "%s (default)", e->key);
  }
}

/* --- validation --- */

static bool parse_flag(const char* s, bool* out) {
  static const char* yes[] = {"1", "true", "yes", "on"};
  static const char* no[] = {"0", "false", "no", "off"};
  for (size_t i = 0; i < 4; i++) {
    if (strcasecmp(s, yes[i]) == 0) {
      *out = true;
      return true;
    }
    if (strcasecmp(s, no[i]) == 0) {
      *out = false;
      return true;
    }
  }
  return false;
}

static bool parse_number(const char* s, double min, double* out) {
  char* end = NULL;
  errno = 0;
  const double v = strtod(s, &end);
  /* The range test also turns away NaN, and with NUMBER_MAX infinity. */
  if (end == s || errno != 0 || !(v >= min && v <= NUMBER_MAX))
    return false;
  while (isspace((unsigned char)*end))
    end++;
  if (*end != '\0')
    return false;
  *out = v;
  return true;
}

/* What a value of this kind has to be, for the messages that refuse one. */
static void describe_expected(
  const setting_def_t* def, char* out, size_t cap) {
  switch (def->kind) {
  case SETTING_FLAG:
    snprintf(out, cap, "1/0, true/false, yes/no or on/off");
    break;
  case SETTING_NUMBER:
    snprintf(out, cap, "a number from %g to %g", def->min, NUMBER_MAX);
    break;
  default:
    snprintf(out, cap, "a directory");
  }
}

/* Whether @p value is one @p def can hold. Paths are checked when they are
 * read, since an unset variable may well be set in the process that does. */
static bool value_valid(const setting_def_t* def, const char* value) {
  double d;
  bool b;
  switch (def->kind) {
  case SETTING_FLAG:
    return parse_flag(value, &b);
  case SETTING_NUMBER:
    return parse_number(value, def->min, &d);
  default:
    return value[0] != '\0';
  }
}

/* --- file --- */

/*
 * Writes the table out through a temporary file and a rename.
 *
 * A job array pointed at one $HOME has every rank finalising at once, into a
 * file the others may still be reading. Rename is atomic within a filesystem,
 * so each reader sees one complete table or the other, never a half-written
 * one; the pid in the temporary name keeps the ranks from colliding before
 * they get there.
 *
 * Only what the file itself said, or a persistent sif_setting_set(), is
 * written as a value; a setting at its default is written commented out, so a
 * later sif that changes the default is not held to the old one.
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
    SIF_LOG_WARNING(
      "settings", "%s: cannot open for writing: %s", tmp_path, strerror(errno));
    return;
  }

  fprintf(f,
    "# sif settings, one `key = value` per line. A commented-out key is at its\n"
    "# default. Values may use $VAR and ${VAR}, expanded when sif starts, and\n"
    "# $SIF_HOME, the directory holding this file. Each setting can also be\n"
    "# given for one process by the environment variable SIF_<KEY>, e.g.\n"
    "# SIF_FFT_TUNING_MAX_GIB=0, which takes precedence over this file.\n");

  for (uint32_t i = 0; i < settings_count; i++) {
    const entry_t* e = &settings[i];
    if (!e->def)
      continue;
    fprintf(f, "\n# %s\n", e->def->doc);
    if (e->layers[LAYER_FILE].set)
      fprintf(f, "%s = %s\n", e->key, e->layers[LAYER_FILE].raw);
    else
      fprintf(f, "# %s = %s\n", e->key, e->def->fallback);
  }

  bool header = false;
  for (uint32_t i = 0; i < settings_count; i++) {
    const entry_t* e = &settings[i];
    if (e->def)
      continue;
    if (!header) {
      fprintf(f, "\n# Not settings of this version of sif, kept as found.\n");
      header = true;
    }
    fprintf(f, "%s = %s\n", e->key, e->layers[LAYER_FILE].raw);
  }

  /* Check the writes before the rename. A full disk or an exceeded quota
   * fails at fclose(), and renaming regardless would replace a good table
   * with a truncated one -- the exact failure the rename exists to prevent. */
  bool ok = (ferror(f) == 0);
  if (fclose(f) != 0)
    ok = false;

  if (ok && rename(tmp_path, settings_filepath) == 0) {
    SIF_LOG_TRACE("settings", "%s: saved", settings_filepath);
    settings_modified = false;
  } else {
    SIF_LOG_WARNING("settings", "%s: not saved", settings_filepath);
    remove(tmp_path);
  }
}

/* A line of the file, into the file layer of its key. Unknown keys get an
 * entry of their own, which is the only way the table grows. */
static void settings_load_line(const char* key, const char* raw, uint32_t line) {
  entry_t* e = settings_find(key);
  if (!e) {
    SIF_LOG_WARNING("settings", "%s:%u: %s: not a setting (kept, not used)",
      settings_filepath, line, key);

    if (settings_count == settings_capacity) {
      const uint32_t grown_cap = settings_capacity * 2;
      entry_t* grown = realloc(settings, grown_cap * sizeof(entry_t));
      if (!grown) {
        SIF_LOG_ERROR("settings", "%s:%u: %s: dropped: out of memory",
          settings_filepath, line, key);
        return;
      }
      settings = grown;
      settings_capacity = grown_cap;
    }
    e = &settings[settings_count++];
    memset(e, 0, sizeof(*e));
    copy_bounded(e->key, KEY_CAP, key);
  }

  layer_store(&e->layers[LAYER_FILE], raw, line);
}

/* Returns whether the file exists -- or exists and must not be written. */
static bool settings_load(void) {
  FILE* f = settings_filepath[0] ? fopen(settings_filepath, "r") : NULL;
  if (!f && settings_filepath[0] && errno != ENOENT) {
    /* There, but not ours to read: saving would replace it with the defaults
     * and lose whatever it holds. */
    SIF_LOG_WARNING("settings",
      "%s: cannot read (%s); running on defaults, and nothing will be saved "
      "to it",
      settings_filepath, strerror(errno));
    settings_filepath[0] = '\0';
    return true;
  }
  if (!f)
    return false;

  char line[LINE_CAP];
  uint32_t lineno = 0;
  while (fgets(line, sizeof(line), f)) {
    lineno++;
    /* An over-long line comes back in pieces, and the tail of one would
     * parse as an entry of its own -- so discard it whole. Reaching EOF
     * without a newline is the ordinary last line, not an over-long one. */
    if (!strchr(line, '\n') && !feof(f)) {
      SIF_LOG_WARNING("settings", "%s:%u: longer than %d characters (ignored)",
        settings_filepath, lineno, LINE_CAP - 1);
      int c;
      while ((c = fgetc(f)) != EOF && c != '\n') {
      }
      continue;
    }

    char* trimmed = trim_whitespace(line);
    if (trimmed[0] == '\0' || trimmed[0] == '#' || trimmed[0] == ';')
      continue;

    char* eq = strchr(trimmed, '=');
    if (!eq) {
      SIF_LOG_WARNING("settings", "%s:%u: no '=' (ignored)", settings_filepath,
        lineno);
      continue;
    }

    *eq = '\0';
    char* key = trim_whitespace(trimmed);
    char* val = trim_whitespace(eq + 1);
    if (key[0] != '\0')
      settings_load_line(key, val, lineno);
  }
  fclose(f);
  SIF_LOG_TRACE("settings", "%s: loaded", settings_filepath);
  return true;
}

/* --- lifetime --- */

void sif__settings_init(const char* root_dir) {
  copy_bounded(settings_root, sizeof(settings_root), root_dir);

  int written = snprintf(
    settings_filepath, sizeof(settings_filepath), "%s/config", root_dir);
  if (written < 0 || (size_t)written >= sizeof(settings_filepath)) {
    SIF_LOG_WARNING("settings", "%s/config: path too long: settings will not "
      "persist", root_dir);
    settings_filepath[0] = '\0';
  }

  settings_count = 0;
  settings_capacity = INITIAL_CAPACITY > N_REGISTRY ? INITIAL_CAPACITY
                                                    : (uint32_t)N_REGISTRY;
  settings = calloc(settings_capacity, sizeof(entry_t));
  if (!settings) {
    SIF_LOG_ERROR("settings", "failed to allocate the settings table");
    settings_capacity = 0;
    return;
  }

  /* The defaults and the environment are fixed for the process, so both are
   * resolved now, against the environment the process started with. */
  for (size_t i = 0; i < N_REGISTRY; i++) {
    entry_t* e = &settings[settings_count++];
    copy_bounded(e->key, KEY_CAP, registry[i].key);
    e->def = &registry[i];
    layer_store(&e->layers[LAYER_DEFAULT], registry[i].fallback, 0);

    char var[KEY_CAP + 4];
    env_name(e->key, var, sizeof(var));
    const char* env = getenv(var);
    if (env) {
      layer_store(&e->layers[LAYER_ENV], env, 0);
      SIF_LOG_DEBUG("settings", "%s: %s = %s", var, e->key, env);
    }
  }

  /* A first run writes the file, so that there is one listing every setting
   * for the user to edit. Loading an existing one is not a modification:
   * without this the first finalise would rewrite the file just read. */
  settings_modified = !settings_load();
}

void sif__settings_finalise(void) {
  settings_save();

  free(settings);
  settings = NULL;
  settings_count = 0;
  settings_capacity = 0;
  settings_filepath[0] = '\0';
  settings_root[0] = '\0';
  settings_modified = false;
}

/* --- public accessors --- */

/*
 * Why @p value is no value for @p def, into @p reason; 0 if it is one. The
 * one place a write is judged, for sif_setting_check() and the writers.
 */
static int value_problem(
  const setting_def_t* def, const char* value, char* reason, size_t cap) {
  /* The file format is one `key = value` per line with no escaping. */
  if (strchr(value, '\n')) {
    snprintf(reason, cap,
      "value contains a newline, which the config file cannot hold");
    return 1;
  }
  if (strlen(value) >= VALUE_CAP) {
    snprintf(
      reason, cap, "%zu characters (at most %d)", strlen(value), VALUE_CAP - 1);
    return 1;
  }

  /* Expanded first: a number may well be given as "$SLURM_CPUS". A path with
   * an unset variable is let through, for the process that sets it. */
  char expanded[VALUE_CAP];
  char unset[KEY_CAP];
  expand_env_vars(value, expanded, sizeof(expanded), unset);
  const char* checked = def->kind == SETTING_PATH ? value : expanded;
  if (!value_valid(def, checked)) {
    char expected[64];
    describe_expected(def, expected, sizeof(expected));
    snprintf(reason, cap, "'%s' (expected %s)", value, expected);
    return 1;
  }
  return 0;
}

/* "not a setting (known: ...)", for a key the registry does not hold. */
static void describe_unknown(char* reason, size_t cap) {
  size_t len = (size_t)snprintf(reason, cap, "not a setting (known: ");
  for (size_t i = 0; i < N_REGISTRY && len < cap; i++)
    len += (size_t)snprintf(
      reason + len, cap - len, "%s%s", i ? ", " : "", registry[i].key);
  if (len < cap)
    snprintf(reason + len, cap - len, ")");
}

int sif_setting_check(
  const char* key, const char* value, char* reason, size_t reason_size) {
  char local[512];
  char* why = reason && reason_size ? reason : local;
  const size_t cap = reason && reason_size ? reason_size : sizeof(local);

  if (!key || !value) {
    snprintf(why, cap, "%s: NULL", !key ? "key" : "value");
    return SIF_ERR_INVALID;
  }
  const setting_def_t* def = registry_find(key);
  if (!def) {
    describe_unknown(why, cap);
    return SIF_ERR_INVALID;
  }
  if (value_problem(def, value, why, cap))
    return SIF_ERR_INVALID;
  why[0] = '\0';
  return SIF_OK;
}

/*
 * What sif_setting_set() and sif_setting_override() share: the checks, which
 * refuse a write the table could not use, and the entry to write into.
 * @p value is NULL only when an override is being cleared.
 */
static int setting_write_check(
  const char* fn, const char* key, const char* value, entry_t** out) {
  if (!key) {
    SIF_LOG_ERROR("settings", "%s: key: NULL", fn);
    return SIF_ERR_INVALID;
  }

  /* The table is NULL either before sif_system_init() or after an init whose
   * allocation of it failed, and only the first is the caller's to fix. */
  if (!settings) {
    if (!sif__system_state_safe()) {
      (void)sif__system_state(); /* logs, and records the status */
      return SIF_ERR_UNINITIALISED;
    }
    SIF_LOG_ERROR("settings",
      "%s: not stored: no settings table (its allocation failed at init)", key);
    return SIF_ERR_ALLOC;
  }

  entry_t* e = settings_find(key);
  char reason[512];
  if (!e || !e->def) {
    describe_unknown(reason, sizeof(reason));
    SIF_LOG_ERROR("settings", "%s: %s", key, reason);
    return SIF_ERR_INVALID;
  }

  if (value && value_problem(e->def, value, reason, sizeof(reason))) {
    SIF_LOG_ERROR("settings", "%s: not stored: %s", key, reason);
    return SIF_ERR_INVALID;
  }

  *out = e;
  return SIF_OK;
}

int sif_setting_set(const char* key, const char* value) {
  entry_t* e = NULL;
  if (!value) {
    SIF_LOG_ERROR("settings", "%s: value: NULL", key ? key : "(NULL key)");
    return SIF_ERR_INVALID;
  }
  const int status = setting_write_check("sif_setting_set", key, value, &e);
  if (status != SIF_OK)
    return status;

  if (settings_filepath[0] == '\0')
    SIF_LOG_WARNING("settings",
      "%s: set for this process only: there is no settings file to save it to",
      key);

  layer_t* file = &e->layers[LAYER_FILE];
  if (!file->set || strcmp(file->raw, value) != 0) {
    layer_store(file, value, 0);
    settings_modified = true;
  }
  e->warned = false;

  /* Saved, but not what this process reads: say so, or the set looks like it
   * did nothing. */
  int which;
  entry_effective(e, &which);
  if (which == LAYER_SESSION) {
    SIF_LOG_WARNING("settings",
      "%s: saved, but the value set for this process takes precedence", key);
  } else if (which == LAYER_ENV) {
    char var[KEY_CAP + 4];
    env_name(key, var, sizeof(var));
    SIF_LOG_WARNING(
      "settings", "%s: saved, but %s takes precedence in this process", key,
      var);
  }
  return SIF_OK;
}

int sif_setting_override(const char* key, const char* value) {
  entry_t* e = NULL;
  const int status =
    setting_write_check("sif_setting_override", key, value, &e);
  if (status != SIF_OK)
    return status;

  if (value)
    layer_store(&e->layers[LAYER_SESSION], value, 0);
  else
    memset(&e->layers[LAYER_SESSION], 0, sizeof(layer_t));
  e->warned = false;
  return SIF_OK;
}

const char* sif_setting_get(const char* key, const char* fallback) {
  if (!settings || !key)
    return fallback;

  /* An unknown key the file held is kept for saving, not for reading. */
  const entry_t* e = settings_find(key);
  const layer_t* l = (e && e->def) ? entry_effective(e, NULL) : NULL;
  return l ? l->value : fallback;
}

/* --- typed readers --- */

/*
 * The entry for a key the library reads. Asking for a key that is not in the
 * registry, or as the wrong kind, is a bug in sif rather than a bad value, so
 * it asserts.
 */
static entry_t* setting_typed(const char* key, setting_kind_t kind) {
  entry_t* e = settings ? settings_find(key) : NULL;
  SIF_ASSERT(!settings || (e && e->def && e->def->kind == kind));
  return e;
}

/*
 * The value of @p e as its kind reads it, or the default with a warning when
 * the value cannot be read so. The warning is given once per value: these are
 * read on every transform and every deposit.
 */
static const char* setting_value_or_default(entry_t* e) {
  int which;
  const layer_t* l = entry_effective(e, &which);
  if (l->unset_var[0] == '\0' && !l->truncated && value_valid(e->def, l->value))
    return l->value;

  if (!e->warned) {
    char origin[PATH_CAP];
    char expected[64];
    describe_origin(e, which, origin, sizeof(origin));
    describe_expected(e->def, expected, sizeof(expected));
    if (l->unset_var[0])
      SIF_LOG_WARNING("settings", "%s: $%s is not set ('%s'); using %s",
        origin, l->unset_var, l->raw, e->layers[LAYER_DEFAULT].value);
    else if (l->truncated)
      SIF_LOG_WARNING("settings",
        "%s: longer than %d characters once expanded; using %s", origin,
        VALUE_CAP - 1, e->layers[LAYER_DEFAULT].value);
    else
      SIF_LOG_WARNING("settings", "%s: '%s' (expected %s); using %s", origin,
        l->raw, expected, e->layers[LAYER_DEFAULT].value);
    e->warned = true;
  }
  return e->layers[LAYER_DEFAULT].value;
}

double sif__setting_number(const char* key) {
  const setting_def_t* def = registry_find(key);
  SIF_ASSERT(def && def->kind == SETTING_NUMBER);

  entry_t* e = setting_typed(key, SETTING_NUMBER);
  const char* s = e ? setting_value_or_default(e) : def->fallback;

  double v = def->min;
  parse_number(s, def->min, &v);
  return v;
}

bool sif__setting_flag(const char* key) {
  const setting_def_t* def = registry_find(key);
  SIF_ASSERT(def && def->kind == SETTING_FLAG);

  entry_t* e = setting_typed(key, SETTING_FLAG);
  const char* s = e ? setting_value_or_default(e) : def->fallback;

  bool v = false;
  parse_flag(s, &v);
  return v;
}

int sif__ensure_dir(const char* path) {
  if (!path || path[0] == '\0')
    return SIF_ERR_INVALID;

  char tmp[PATH_CAP];
  int written = snprintf(tmp, sizeof(tmp), "%s", path);
  if (written < 0 || (size_t)written >= sizeof(tmp)) {
    SIF_LOG_ERROR("settings", "%s: path too long (at most %d characters)",
      path, PATH_CAP - 1);
    return SIF_ERR_INVALID;
  }

  size_t len = strlen(tmp);
  while (len > 1 && tmp[len - 1] == '/')
    tmp[--len] = '\0';

  /* Walk the separators, creating each ancestor in turn: mkdir() creates one
   * level only. EEXIST is the ordinary case here, not a failure -- the whole
   * point is that we do not know how much of the chain is already there. */
  for (char* p = tmp + 1; *p; p++) {
    if (*p != '/')
      continue;

    *p = '\0';
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
      SIF_LOG_ERROR("settings", "%s: cannot create: %s", tmp, strerror(errno));
      return SIF_ERR_IO;
    }
    *p = '/';
  }

  if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
    SIF_LOG_ERROR("settings", "%s: cannot create: %s", tmp, strerror(errno));
    return SIF_ERR_IO;
  }

  struct stat st;
  if (stat(tmp, &st) != 0 || !S_ISDIR(st.st_mode)) {
    SIF_LOG_ERROR("settings", "%s: exists, but is not a directory", tmp);
    return SIF_ERR_IO;
  }

  return SIF_OK;
}

int sif__setting_directory(const char* key, const char** path) {
  *path = NULL;
  entry_t* e = setting_typed(key, SETTING_PATH);
  if (!e) {
    (void)sif__system_state(); /* logs, and records the status */
    return SIF_ERR_UNINITIALISED;
  }

  /* No falling back to the default here: a directory is where data goes, and
   * writing it somewhere other than where the user said is worse than not
   * writing it. */
  int which;
  const layer_t* l = entry_effective(e, &which);
  char origin[PATH_CAP];
  describe_origin(e, which, origin, sizeof(origin));

  if (l->unset_var[0]) {
    SIF_LOG_ERROR("settings", "%s: $%s is not set ('%s')", origin,
      l->unset_var, l->raw);
    return SIF_ERR_INVALID;
  }
  if (l->truncated) {
    SIF_LOG_ERROR("settings", "%s: longer than %d characters once expanded",
      origin, VALUE_CAP - 1);
    return SIF_ERR_INVALID;
  }
  if (l->value[0] == '\0') {
    SIF_LOG_ERROR("settings", "%s: empty path", origin);
    return SIF_ERR_INVALID;
  }

  const int status = sif__ensure_dir(l->value);
  if (status != SIF_OK)
    return status;

  *path = l->value;
  return SIF_OK;
}

#undef KEY_CAP
#undef VALUE_CAP
#undef PATH_CAP
#undef LINE_CAP
#undef INITIAL_CAPACITY
#undef NUMBER_MAX
#undef N_REGISTRY
