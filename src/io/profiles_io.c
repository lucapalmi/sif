/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/profiles_io.h"

#include "measure/profiles_internal.h"
#include "sif/utils/logger.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>

/*
 * The shape two sets have to agree on before they can share a file: they are
 * written as one row, so a row would otherwise mean two different things on
 * its two halves.
 */
static int shapes_agree(
  const sif_density_profiles_t* dens, const sif_velocity_profiles_t* vel) {

  if (!dens || !vel)
    return 1;

  return dens->n_voids == vel->n_voids && dens->n_bins == vel->n_bins &&
         dens->ext == vel->ext;
}

int sif_profiles_write_ascii(const char* filepath,
  const sif_density_profiles_t* dens, const sif_velocity_profiles_t* vel,
  const sif_catalogue_t* cat) {

  if ((!dens && !vel) || !cat || !filepath) {
    SIF_LOG_ERROR("io",
      "sif_profiles_write_ascii needs a path, a catalogue and at least one "
      "profile set");
    return SIF_ERR_INVALID;
  }

  if (!shapes_agree(dens, vel)) {
    SIF_LOG_ERROR("io",
      "the density and velocity sets disagree on their shape and cannot share "
      "a file");
    return SIF_ERR_INVALID;
  }

  const uint64_t n_voids = dens ? dens->n_voids : vel->n_voids;
  const uint32_t n_bins = dens ? dens->n_bins : vel->n_bins;
  const sif_real ext = dens ? dens->ext : vel->ext;
  const sif_real* r_edges = dens ? dens->r_edges : vel->r_edges;

  /* Row i is void i. A catalogue of a different length is a different
   * catalogue, and writing it would label every row with the wrong void. */
  if (cat->n_voids != n_voids) {
    SIF_LOG_ERROR("io",
      "the catalogue holds %" PRIu64 " voids but the profiles hold %" PRIu64
      "; they are not the same measurement",
      cat->n_voids, n_voids);
    return SIF_ERR_INVALID;
  }

  FILE* file = fopen(filepath, "w");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for writing", filepath);
    return SIF_ERR_IO;
  }

  /* What the file is, then what each column is, all behind '#': the shape
   * first, so the reader can size everything once, then the bin edges every
   * row shares, then the names. numpy.loadtxt reads the rows as they are. */
  fprintf(file, "#n=%" PRIu64 "\n#n_bins=%" PRIu32 "\n#ext=" SIF_PRI_REAL "\n",
    n_voids, n_bins, ext);
  fprintf(file, "#differential=%d\n#r_edges=", (dens && dens->differential));
  for (uint32_t j = 0; j <= n_bins; j++)
    fprintf(file, "%s" SIF_PRI_REAL, j ? " " : "", r_edges[j]);
  fputs(cat->units == SIF_COORDINATES_SKY ? "\n#ra dec z r" : "\n#cx cy cz r",
    file);
  for (int block = 0; block < 2; block++) {
    if (block == 0 ? !dens : !vel)
      continue;
    for (uint32_t j = 0; j < n_bins; j++)
      fprintf(file, " %s_%" PRIu32, block == 0 ? "density" : "v_rad", j);
  }
  fputc('\n', file);

  for (uint64_t i = 0; i < n_voids; i++) {
    fprintf(file,
      SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL " " SIF_PRI_REAL,
      cat->cx[i], cat->cy[i], cat->cz[i], cat->radii[i]);

    if (dens) {
      const sif_real* row = sif_density_profiles_get(dens, i);
      for (uint32_t j = 0; j < n_bins; j++)
        fprintf(file, " " SIF_PRI_REAL, row[j]);
    }

    if (vel) {
      const sif_real* row = sif_velocity_profiles_get(vel, i);
      for (uint32_t j = 0; j < n_bins; j++)
        fprintf(file, " " SIF_PRI_REAL, row[j]);
    }

    fputc('\n', file);
  }

  /* fprintf() reports nothing useful per call, so the stream's error flag and
   * the flush inside fclose() are what say whether the file actually reached
   * the disk. Without this a full quota reads back as a short file. */
  const bool ok = (ferror(file) == 0);
  if (fclose(file) != 0 || !ok) {
    SIF_LOG_ERROR("io", "failed to flush %s to disk", filepath);
    return SIF_ERR_IO;
  }

  SIF_LOG_INFO("io", "saved %" PRIu64 " profiles (%s) to %s (ASCII)", n_voids,
    dens && vel ? "density and velocity" : (dens ? "density" : "velocity"),
    filepath);
  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* the header                                                                */
/* ------------------------------------------------------------------------ */

/* What the header says the rest of the file is. */
typedef struct {
  uint64_t n_voids;
  uint32_t n_bins;
  sif_real ext;
  int has_dens, has_vel, differential;
  bool sky;
  /* The bin edges, n_bins + 1 of them, from a file with the '#' header; NULL
   * for one from before it, where they are the line after the shape. */
  sif_real* edges;
} header_t;

/* The header of a file from before the '#' one: a single line holding the
 * shape and the flags. Left at the bin edges, which follow it. */
static int read_legacy_header(FILE* file, const char* filepath, header_t* h) {
  if (fscanf(file, "%" SCNu64 " %" SCNu32 " " SIF_SCN_REAL " %d %d %d",
        &h->n_voids, &h->n_bins, &h->ext, &h->has_dens, &h->has_vel,
        &h->differential) != 6) {
    SIF_LOG_ERROR("io", "failed to read the profile header from %s", filepath);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/* The values after "r_edges=", n + 1 of them, into a new array. */
static sif_real* parse_edges(const char* text, uint32_t n_bins) {
  sif_real* edges = malloc(((size_t)n_bins + 1) * sizeof(sif_real));
  if (!edges)
    return NULL;
  const char* p = text;
  for (uint32_t j = 0; j <= n_bins; j++) {
    char* end;
    edges[j] = sizeof(sif_real) == 4 ? (sif_real)strtof(p, &end)
                                     : (sif_real)strtod(p, &end);
    if (end == p) {
      free(edges);
      return NULL;
    }
    p = end;
  }
  p += strspn(p, " \t\r\n");
  if (*p) {
    free(edges);
    return NULL;
  }
  return edges;
}

/*
 * The names line: the centres (cx cy cz, or ra dec z), r, then density_0 ...
 * and v_rad_0 ... for the blocks the file holds, in that order. The blocks
 * are what it says; the rest has to be exactly this, since a profile file is
 * sif's own and its rows are read in this order.
 */
static int parse_names(char* text, const char* filepath, header_t* h) {
  static const char* const CART[] = {"cx", "cy", "cz", "r"};
  static const char* const SKY[] = {"ra", "dec", "z", "r"};

  char* save = NULL;
  char* tok = strtok_r(text, " \t", &save);
  h->sky = tok && strcasecmp(tok, "ra") == 0;
  const char* const* centre = h->sky ? SKY : CART;
  for (int c = 0; c < 4; c++, tok = strtok_r(NULL, " \t", &save)) {
    if (!tok || strcasecmp(tok, centre[c]) != 0) {
      SIF_LOG_ERROR("io",
        "%s: the columns have to begin cx cy cz r, or ra dec z r", filepath);
      return SIF_ERR_INVALID;
    }
  }

  h->has_dens = h->has_vel = 0;
  for (int block = 0; block < 2; block++) {
    const char* prefix = block == 0 ? "density_" : "v_rad_";
    if (!tok || strncasecmp(tok, prefix, strlen(prefix)) != 0)
      continue;
    for (uint32_t j = 0; j < h->n_bins;
      j++, tok = strtok_r(NULL, " \t", &save)) {
      char want[32];
      snprintf(want, sizeof want, "%s%" PRIu32, prefix, j);
      if (!tok || strcasecmp(tok, want) != 0) {
        SIF_LOG_ERROR("io", "%s: expected column %s after %s", filepath, want,
          j ? "the one before" : "the centres");
        return SIF_ERR_INVALID;
      }
    }
    *(block == 0 ? &h->has_dens : &h->has_vel) = 1;
  }
  if (tok) {
    SIF_LOG_ERROR("io", "%s: column %s is not one sif writes", filepath, tok);
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

/* "key=value" lines, and the names line, until the first row -- which is
 * left to be read. */
static int read_named_header(FILE* file, const char* filepath, header_t* h) {
  bool have_n = false, have_bins = false, have_ext = false, have_names = false;
  char* line = NULL;
  size_t cap = 0;
  char* edges_text = NULL;
  int status = SIF_OK;

  for (;;) {
    const long at = ftell(file);
    const ssize_t len = getline(&line, &cap, file);
    if (len < 0)
      break;
    char* p = line + strspn(line, " \t\r\n");
    if (!*p)
      continue;
    if (*p != '#') {
      fseek(file, at, SEEK_SET); /* the first row, for the caller */
      break;
    }
    p++;
    p += strspn(p, " \t");
    char* end = p + strlen(p);
    while (end > p && isspace((unsigned char)end[-1]))
      *--end = '\0';

    char* eq = strchr(p, '=');
    if (eq) {
      *eq = '\0';
      const char* key = p;
      const char* value = eq + 1;
      char* stop;
      if (strcmp(key, "n") == 0) {
        h->n_voids = strtoull(value, &stop, 10);
        have_n = stop != value;
      } else if (strcmp(key, "n_bins") == 0) {
        h->n_bins = (uint32_t)strtoul(value, &stop, 10);
        have_bins = stop != value;
      } else if (strcmp(key, "ext") == 0) {
        h->ext = (sif_real)strtod(value, &stop);
        have_ext = stop != value;
      } else if (strcmp(key, "differential") == 0) {
        h->differential = atoi(value) != 0;
      } else if (strcmp(key, "r_edges") == 0) {
        free(edges_text);
        edges_text = strdup(value);
      }
    } else if (!have_names && have_bins) {
      status = parse_names(p, filepath, h);
      if (status != SIF_OK)
        break;
      have_names = true;
    }
  }

  if (status == SIF_OK &&
      !(have_n && have_bins && have_ext && edges_text && have_names)) {
    SIF_LOG_ERROR("io",
      "%s: the header needs n, n_bins, ext, r_edges and the column names",
      filepath);
    status = SIF_ERR_INVALID;
  }
  if (status == SIF_OK && h->n_bins > 0) {
    h->edges = parse_edges(edges_text, h->n_bins);
    if (!h->edges) {
      SIF_LOG_ERROR("io", "%s: r_edges is not %" PRIu32 " numbers", filepath,
        h->n_bins + 1);
      status = SIF_ERR_INVALID;
    }
  }

  free(edges_text);
  free(line);
  return status;
}

/*
 * The header, of either kind: a file that starts with '#' has the named one,
 * anything else is from before it. Left positioned at the bin edges for the
 * old kind, at the first row for the new.
 */
static int read_header(FILE* file, const char* filepath, header_t* h) {
  memset(h, 0, sizeof *h);

  int c;
  while ((c = fgetc(file)) != EOF && isspace(c))
    ;
  if (c != EOF)
    ungetc(c, file);

  const int status = c == '#' ? read_named_header(file, filepath, h)
                              : read_legacy_header(file, filepath, h);
  if (status != SIF_OK) {
    free(h->edges);
    h->edges = NULL;
    return status;
  }

  if (h->n_voids == 0 || h->n_bins == 0 || (!h->has_dens && !h->has_vel)) {
    SIF_LOG_ERROR("io",
      "%s describes an empty profile set (%" PRIu64 " voids, %" PRIu32
      " bins, density=%d velocity=%d)",
      filepath, h->n_voids, h->n_bins, h->has_dens, h->has_vel);
    free(h->edges);
    h->edges = NULL;
    return SIF_ERR_INVALID;
  }
  return SIF_OK;
}

int sif_profiles_read_header_ascii(const char* filepath, uint64_t* out_n_voids,
  uint32_t* out_n_bins, sif_real* out_ext, int* out_has_density,
  int* out_has_velocity, int* out_differential) {

  if (!filepath) {
    SIF_LOG_ERROR("io", "invalid filepath for sif_profiles_read_header_ascii");
    return SIF_ERR_INVALID;
  }

  FILE* file = fopen(filepath, "r");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for reading", filepath);
    return SIF_ERR_INVALID;
  }

  header_t h;
  const int status = read_header(file, filepath, &h);
  fclose(file);
  if (status != SIF_OK)
    return status;
  free(h.edges);

  if (out_n_voids)
    *out_n_voids = h.n_voids;
  if (out_n_bins)
    *out_n_bins = h.n_bins;
  if (out_ext)
    *out_ext = h.ext;
  if (out_has_density)
    *out_has_density = h.has_dens;
  if (out_has_velocity)
    *out_has_velocity = h.has_vel;
  if (out_differential)
    *out_differential = h.differential;

  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* the rows                                                                  */
/* ------------------------------------------------------------------------ */

int sif_profiles_read_ascii(const char* filepath, sif_catalogue_t** out_cat,
  sif_density_profiles_t** out_dens, sif_velocity_profiles_t** out_vel) {

  if (!filepath) {
    SIF_LOG_ERROR("io", "invalid filepath for sif_profiles_read_ascii");
    return SIF_ERR_INVALID;
  }

  /* Cleared up front so that every path out of here, including the ones that
   * give up before anything is allocated, leaves the caller with NULL rather
   * than with whatever the pointers held. */
  if (out_cat)
    *out_cat = NULL;
  if (out_dens)
    *out_dens = NULL;
  if (out_vel)
    *out_vel = NULL;

  FILE* file = fopen(filepath, "r");
  if (!file) {
    SIF_LOG_ERROR("io", "failed to open %s for reading", filepath);
    return SIF_ERR_INVALID;
  }

  header_t h;
  if (read_header(file, filepath, &h) != SIF_OK) {
    fclose(file);
    return SIF_ERR_INVALID;
  }
  const uint64_t n_voids = h.n_voids;
  const uint32_t n_bins = h.n_bins;

  int status = SIF_ERR_INVALID;
  sif_catalogue_t* cat = NULL;
  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;

  /* Asked for something the file does not carry. Handing back an empty set
   * would be indistinguishable from a measurement of zeros. */
  if ((out_dens && !h.has_dens) || (out_vel && !h.has_vel)) {
    SIF_LOG_ERROR("io", "%s carries no %s profiles", filepath,
      (out_dens && !h.has_dens) ? "density" : "velocity");
    goto fail;
  }

  status = SIF_ERR_ALLOC;
  if (out_cat) {
    cat = sif_catalogue_alloc(n_voids);
    if (!cat) {
      SIF_LOG_ERROR("io", "OOM allocating the catalogue for %s", filepath);
      goto fail;
    }
    if (h.sky)
      cat->units = SIF_COORDINATES_SKY;
  }

  if (out_dens) {
    dens =
      sif__density_profiles_alloc(n_voids, n_bins, h.ext, h.differential != 0);
    if (!dens) {
      SIF_LOG_ERROR("io", "OOM allocating density profiles for %s", filepath);
      goto fail;
    }
  }

  if (out_vel) {
    vel = sif__velocity_profiles_alloc(n_voids, n_bins, h.ext);
    if (!vel) {
      SIF_LOG_ERROR("io", "OOM allocating velocity profiles for %s", filepath);
      goto fail;
    }
  }

  status = SIF_ERR_INVALID;

  /* The edges the file carries rather than ones recomputed from ext and
   * n_bins: what was measured is what the file says. */
  for (uint32_t j = 0; j <= n_bins; j++) {
    sif_real edge;
    if (h.edges) {
      edge = h.edges[j];
    } else if (fscanf(file, SIF_SCN_REAL, &edge) != 1) {
      SIF_LOG_ERROR(
        "io", "failed reading bin edge %" PRIu32 " from %s", j, filepath);
      goto fail;
    }
    if (dens)
      dens->r_edges[j] = edge;
    if (vel)
      vel->r_edges[j] = edge;
  }

  /* Rows are required to be there: the header said how many, and a file that
   * stops short is truncated rather than merely small. */
  for (uint64_t i = 0; i < n_voids; i++) {
    sif_real cx, cy, cz, radius;

    if (fscanf(file,
          SIF_SCN_REAL " " SIF_SCN_REAL " " SIF_SCN_REAL " " SIF_SCN_REAL, &cx,
          &cy, &cz, &radius) != 4) {
      SIF_LOG_ERROR(
        "io", "failed reading void %" PRIu64 " from %s", i, filepath);
      goto fail;
    }

    if (cat) {
      cat->cx[i] = cx;
      cat->cy[i] = cy;
      cat->cz[i] = cz;
      cat->radii[i] = radius;
    }

    /* Both blocks are consumed whether or not they were asked for: the
     * columns are there either way, and skipping them by parsing is what lets
     * a caller take only the half it wants. */
    for (int block = 0; block < 2; block++) {
      const int present = block == 0 ? h.has_dens : h.has_vel;
      if (!present)
        continue;

      sif_real* row = NULL;
      if (block == 0 && dens)
        row = &dens->profiles[i * n_bins];
      else if (block == 1 && vel)
        row = &vel->v_rad[i * n_bins];

      for (uint32_t j = 0; j < n_bins; j++) {
        sif_real value;
        if (fscanf(file, SIF_SCN_REAL, &value) != 1) {
          SIF_LOG_ERROR("io",
            "failed reading %s bin %" PRIu32 " of void %" PRIu64 " from %s",
            block == 0 ? "density" : "velocity", j, i, filepath);
          goto fail;
        }
        if (row)
          row[j] = value;
      }
    }
  }

  if (cat) {
    /* Filled in directly rather than through sif_catalogue_append(), so the size
     * has to be set by hand. */
    cat->n_voids = n_voids;
    *out_cat = cat;
  }
  if (dens)
    *out_dens = dens;
  if (vel)
    *out_vel = vel;

  free(h.edges);
  fclose(file);

  SIF_LOG_INFO(
    "io", "loaded %" PRIu64 " profiles from %s (ASCII)", n_voids, filepath);
  return SIF_OK;

fail:
  /* Nothing built here survives a failure -- a half-read set the caller cannot
   * tell from a complete one is worse than none at all. The outputs are
   * already NULL from the top of the call. */
  sif_catalogue_free(cat);
  sif_density_profiles_free(dens);
  sif_velocity_profiles_free(vel);
  free(h.edges);

  fclose(file);
  return status;
}
