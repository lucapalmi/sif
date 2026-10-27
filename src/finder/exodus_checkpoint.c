/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Exodus checkpoints.
 *
 * What a run carries from one rung to the next is the catalogue and nothing
 * else: the overlap mask, the void index and the largest radius so far are
 * all rebuilt from it by accept_void(), each rung re-smooths from the stored
 * transform rather than from the rung before, and the speculation windows
 * are re-learned per rung and change only the speed. So the voids and the
 * number of rungs they cover are a complete state, and a run resumed from
 * them commits the same voids in the same order as one never stopped.
 *
 * The header stores the run's parameters so that a resume can refuse a run
 * that is not the one saved. The grid and the meshes are only fingerprinted
 * by their sizes and total weights: anything finer would cost a pass over
 * them, and the point of the check is to catch the wrong configuration, not
 * a deliberately different field.
 */

#include "exodus_checkpoint.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "sif/utils/align.h"
#include "sif/utils/crc32.h"
#include "sif/utils/logger.h"
#include "sif/utils/random.h"

#include "io/internal.h"
#include "utils/logger_internal.h"

#define TAG "exodus_finder"

#define BYTE_ORDER_MARK 0x01020304u

/* The header is part of the file format, so its size is too. */
typedef char
  header_is_192_bytes[(sizeof(sif_finder_exodus_checkpoint_t) == 192) ? 1 : -1];

void sif__exodus_checkpoint_name(char* out, size_t size) {
  snprintf(out, size, "exodus-%016" PRIx64 ".ckpt", sif__unique_id());
}

uint32_t sif__exodus_checkpoint_crc(const sif_catalogue_t* cat) {
  const size_t bytes = (size_t)cat->n_voids * sizeof(sif_real);

  uint32_t crc = SIF_CRC32_INIT;
  crc = sif_crc32_update(crc, cat->cx, bytes);
  crc = sif_crc32_update(crc, cat->cy, bytes);
  crc = sif_crc32_update(crc, cat->cz, bytes);
  crc = sif_crc32_update(crc, cat->radii, bytes);
  return sif_crc32_final(crc);
}

/* The checksum the header carries: itself with that field zero, then the
 * payload in file order. */
static uint32_t file_crc(const sif_finder_exodus_checkpoint_t* header,
  const sif_real* radii, const sif_catalogue_t* cat) {

  sif_finder_exodus_checkpoint_t h = *header;
  h.crc32 = 0;

  const size_t bytes = (size_t)cat->n_voids * sizeof(sif_real);

  uint32_t crc = SIF_CRC32_INIT;
  crc = sif_crc32_update(crc, &h, sizeof(h));
  crc = sif_crc32_update(crc, radii, (size_t)h.n_radii * sizeof(sif_real));
  crc = sif_crc32_update(crc, cat->cx, bytes);
  crc = sif_crc32_update(crc, cat->cy, bytes);
  crc = sif_crc32_update(crc, cat->cz, bytes);
  crc = sif_crc32_update(crc, cat->radii, bytes);
  return sif_crc32_final(crc);
}

/* --- writing --- */

/* One failure, at the level the caller asked for. */
static int write_failed(
  const char* path, const char* what, int err, const char* consequence) {

  if (!consequence) {
    if (err)
      return sif__io_os_error(TAG, path, what, err);
    SIF_LOG_ERROR(TAG, "%s: %s", path, what);
    return SIF_ERR_IO;
  }

  SIF_LOG_WARNING(TAG, "%s: %s%s%s (%s)", path, what, err ? ": " : "",
    err ? sif__io_strerror(err) : "", consequence);
  return SIF_ERR_IO;
}

int sif__exodus_checkpoint_write(const char* path,
  sif_finder_exodus_checkpoint_t* header, const sif_real* radii,
  const sif_catalogue_t* cat, const char* consequence) {

  memcpy(header->magic, SIF_FINDER_CHECKPOINT_MAGIC, 4);
  header->version = SIF_FINDER_CHECKPOINT_VERSION;
  header->byte_order = BYTE_ORDER_MARK;
  header->is_double = (sizeof(sif_real) == 8) ? 1u : 0u;
  header->n_voids = cat->n_voids;
  memset(header->reserved, 0, sizeof(header->reserved));
  header->crc32 = file_crc(header, radii, cat);

  char tmp[SIF__CHECKPOINT_PATH_MAX];
  const int n = snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid());
  if (n < 0 || (size_t)n >= sizeof(tmp))
    return write_failed(path, "path too long", 0, consequence);

  FILE* file = fopen(tmp, "wb");
  if (!file)
    return write_failed(tmp, "cannot create", errno, consequence);

  const size_t col = (size_t)cat->n_voids;
  int ok =
    fwrite(header, sizeof(*header), 1, file) == 1 &&
    fwrite(radii, sizeof(sif_real), header->n_radii, file) == header->n_radii &&
    fwrite(cat->cx, sizeof(sif_real), col, file) == col &&
    fwrite(cat->cy, sizeof(sif_real), col, file) == col &&
    fwrite(cat->cz, sizeof(sif_real), col, file) == col &&
    fwrite(cat->radii, sizeof(sif_real), col, file) == col;
  int err = ok ? 0 : errno;

  /* A rename only makes the file atomic if its bytes are on the disk first;
   * otherwise a node that dies right after can leave the new name pointing
   * at a file of zeros, with the old checkpoint already gone. */
  if (ok && (fflush(file) != 0 || fsync(fileno(file)) != 0)) {
    ok = 0;
    err = errno;
  }
  if (fclose(file) != 0 && ok) {
    ok = 0;
    err = errno;
  }

  if (!ok) {
    unlink(tmp);
    return write_failed(tmp, "cannot write", err, consequence);
  }

  if (rename(tmp, path) != 0) {
    err = errno;
    unlink(tmp);
    return write_failed(path, "cannot replace", err, consequence);
  }

  return SIF_OK;
}

/* --- reading --- */

/* What a header's seeds mean, for messages. */
static const char* centres_label(uint32_t options) {
  return (options & SIF_FINDER_SEEDS_MINIMA)
           ? "minima (smoothed at the rung's scale)"
           : "every cell, on its node";
}

/* The assignment scheme a header records, for messages. */
static const char* scheme_label(uint32_t assignment) {
  static const char* names[] = {"unknown", "NGP", "CIC", "TSC", "PCS"};
  const uint32_t s = assignment & 0xffu;
  return s <= 4 ? names[s] : "unknown";
}

static int read_failed(const char* path, const char* fmt, ...)
  SIF_PRINTF_LIKE(2, 3);

static int read_failed(const char* path, const char* fmt, ...) {
  char what[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(what, sizeof(what), fmt, ap);
  va_end(ap);

  SIF_LOG_ERROR(TAG, "%s: %s", path, what);
  return SIF_ERR_IO;
}

int sif__exodus_checkpoint_load(const char* path,
  sif_finder_exodus_checkpoint_t* header, sif_real** radii,
  sif_catalogue_t** cat) {

  *radii = NULL;
  *cat = NULL;

  FILE* file = fopen(path, "rb");
  if (!file)
    return sif__io_os_error(TAG, path, NULL, errno);

  struct stat st;
  if (fstat(fileno(file), &st) != 0) {
    const int err = errno;
    fclose(file);
    return sif__io_os_error(TAG, path, NULL, err);
  }
  const uint64_t file_bytes = (uint64_t)st.st_size;

  int status = SIF_ERR_IO;
  sif_finder_exodus_checkpoint_t h;

  if (fread(&h, sizeof(h), 1, file) != 1) {
    read_failed(path,
      "%" PRIu64 " bytes, shorter than a checkpoint header "
      "(%zu)",
      file_bytes, sizeof(h));
    goto done;
  }
  if (memcmp(h.magic, SIF_FINDER_CHECKPOINT_MAGIC, 4) != 0) {
    read_failed(path, "not an exodus checkpoint (no %s at byte 0)",
      SIF_FINDER_CHECKPOINT_MAGIC);
    goto done;
  }
  if (h.byte_order != BYTE_ORDER_MARK) {
    read_failed(path, "written on a machine of the other byte order");
    goto done;
  }
  if (h.version == 0 || h.version > SIF_FINDER_CHECKPOINT_VERSION) {
    read_failed(path, "version %u (expected 1 to %u)", h.version,
      (unsigned)SIF_FINDER_CHECKPOINT_VERSION);
    goto done;
  }
  const uint32_t is_double = (sizeof(sif_real) == 8) ? 1u : 0u;
  if (h.is_double != is_double) {
    read_failed(path, "%s values, this build is %s",
      h.is_double ? "FP64" : "FP32", is_double ? "FP64" : "FP32");
    goto done;
  }
  if (h.n_radii == 0 || h.next_rung > h.n_radii || h.n_prior > h.n_voids) {
    read_failed(path,
      "header: %u rungs, %u done, %" PRIu64 " prior voids of %" PRIu64
      " (inconsistent)",
      h.n_radii, h.next_rung, h.n_prior, h.n_voids);
    goto done;
  }

  /* Sized against the file before anything is allocated from the header:
   * a damaged count would otherwise ask for any amount of memory. */
  const uint64_t expect =
    sizeof(h) + ((uint64_t)h.n_radii + 4u * h.n_voids) * sizeof(sif_real);
  if (h.n_voids > UINT64_MAX / 64u || file_bytes != expect) {
    read_failed(path,
      "%" PRIu64 " bytes, header says %" PRIu64 " (%u rungs, %" PRIu64
      " voids)",
      file_bytes, expect, h.n_radii, h.n_voids);
    goto done;
  }

  *radii = sif_malloc_aligned((size_t)h.n_radii * sizeof(sif_real));
  *cat = sif_catalogue_alloc(h.n_voids);
  if (!*radii || !*cat) {
    status = SIF_ERR_ALLOC;
    if (!*radii)
      SIF_LOG_ERROR(TAG, "%s: cannot allocate %u radii", path, h.n_radii);
    goto done;
  }

  const size_t col = (size_t)h.n_voids;
  sif_catalogue_t* c = *cat;
  if (fread(*radii, sizeof(sif_real), h.n_radii, file) != h.n_radii ||
      fread(c->cx, sizeof(sif_real), col, file) != col ||
      fread(c->cy, sizeof(sif_real), col, file) != col ||
      fread(c->cz, sizeof(sif_real), col, file) != col ||
      fread(c->radii, sizeof(sif_real), col, file) != col) {
    status = sif__io_os_error(TAG, path, "read", ferror(file) ? errno : EIO);
    goto done;
  }
  c->n_voids = h.n_voids;

  const uint32_t got = file_crc(&h, *radii, c);
  if (got != h.crc32) {
    read_failed(path, "corrupt: checksum %08x (expected %08x)", got, h.crc32);
    goto done;
  }

  /* Brought up to date only now, the checksum being over the header as it
   * was written. Version 1 predates the assignment schemes: CIC, not
   * interlaced. */
  if (h.version < 2)
    h.assignment = SIF_ASSIGNMENT_CIC;

  *header = h;
  status = SIF_OK;

done:
  fclose(file);
  if (status != SIF_OK) {
    sif_free_aligned(*radii);
    sif_catalogue_free(*cat);
    *radii = NULL;
    *cat = NULL;
  }
  return status;
}

/* --- matching --- */

/* Both sides came from the same sif_real, widened, so equal means equal. A
 * total weight is the exception: it is a parallel reduction, and its last
 * bits move with the thread count, which a resumed job is free to change. */
#define WEIGHT_RTOL 1e-9

static int weight_matches(double a, double b) {
  return fabs(a - b) <= WEIGHT_RTOL * fmax(fabs(a), fabs(b));
}

int sif__exodus_checkpoint_match(const char* path,
  const sif_finder_exodus_checkpoint_t* want, const sif_real* want_radii,
  const sif_finder_exodus_checkpoint_t* got, const sif_real* got_radii) {

  const char* run_kind[2] = {"a periodic box", "a survey"};
  const char* weighting[2] = {"unweighted", "weighted"};

#define MISMATCH(...)                                                          \
  do {                                                                         \
    char what_[384];                                                           \
    snprintf(what_, sizeof(what_), __VA_ARGS__);                               \
    SIF_LOG_ERROR(TAG, "%s: not this run: %s", path, what_);                   \
    return SIF_ERR_INVALID;                                                    \
  } while (0)

  if (got->is_survey != want->is_survey)
    MISMATCH("saved from %s (this run is %s)", run_kind[got->is_survey & 1],
      run_kind[want->is_survey & 1]);
  if (got->is_weighted != want->is_weighted)
    MISMATCH("saved from %s tracers (this run's are %s)",
      weighting[got->is_weighted & 1], weighting[want->is_weighted & 1]);

  if (got->n_radii != want->n_radii)
    MISMATCH("ladder of %u radii (expected %u)", got->n_radii, want->n_radii);
  for (uint32_t i = 0; i < want->n_radii; i++) {
    if (got_radii[i] != want_radii[i])
      MISMATCH("radius %u of the ladder, largest first: %g (expected %g)", i,
        (double)got_radii[i], (double)want_radii[i]);
  }

  if (got->threshold != want->threshold)
    MISMATCH("threshold %g (expected %g)", got->threshold, want->threshold);
  if (got->overlap_fraction != want->overlap_fraction)
    MISMATCH("overlap fraction %g (expected %g)", got->overlap_fraction,
      want->overlap_fraction);

  const uint32_t search = SIF__FINDER_SEARCH_MASK;
  if ((got->options & search) != (want->options & search))
    MISMATCH("search factor %g (expected %g)",
      (double)sif__finder_search_factor(got->options),
      (double)sif__finder_search_factor(want->options));
  if ((got->options ^ want->options) & SIF_FINDER_KEEP_CIC_WINDOW)
    MISMATCH("CIC window %s (this run %s it)",
      (got->options & SIF_FINDER_KEEP_CIC_WINDOW) ? "kept" : "deconvolved",
      (want->options & SIF_FINDER_KEEP_CIC_WINDOW) ? "keeps" : "deconvolves");
  if ((got->options ^ want->options) & SIF__FINDER_SEEDS_MASK)
    MISMATCH("centres %s (this run's: %s)", centres_label(got->options),
      centres_label(want->options));

  if (got->grid_n_cells != want->grid_n_cells ||
      got->box_length != want->box_length)
    MISMATCH("grid of %u cells over %g (expected %u over %g)",
      got->grid_n_cells, got->box_length, want->grid_n_cells, want->box_length);
  if ((got->options ^ want->options) & SIF_FINDER_ALIASING_WINDOW)
    MISMATCH("window deconvolved %s (this run's: %s)",
      (got->options & SIF_FINDER_ALIASING_WINDOW) ? "allowing for aliasing"
                                                  : "in full",
      (want->options & SIF_FINDER_ALIASING_WINDOW) ? "allowing for aliasing"
                                                   : "in full");
  if (got->assignment != want->assignment)
    MISMATCH("grid assigned with %s%s (this run's: %s%s)",
      scheme_label(got->assignment),
      (got->assignment >> 8) ? ", interlaced" : "",
      scheme_label(want->assignment),
      (want->assignment >> 8) ? ", interlaced" : "");

  if (got->n_particles != want->n_particles ||
      !weight_matches(got->total_weight, want->total_weight))
    MISMATCH("%" PRIu64 " tracers of total weight %.17g (expected %" PRIu64
             " of %.17g)",
      got->n_particles, got->total_weight, want->n_particles,
      want->total_weight);

  if (want->is_survey) {
    if (got->random_n_particles != want->random_n_particles ||
        !weight_matches(got->random_total_weight, want->random_total_weight))
      MISMATCH("%" PRIu64 " randoms of total weight %.17g (expected %" PRIu64
               " of %.17g)",
        got->random_n_particles, got->random_total_weight,
        want->random_n_particles, want->random_total_weight);
    for (int k = 0; k < 3; k++) {
      if (got->offset[k] != want->offset[k])
        MISMATCH(
          "offset[%d] %g (expected %g)", k, got->offset[k], want->offset[k]);
    }
  }

  if (got->n_prior != want->n_prior || got->prior_crc32 != want->prior_crc32) {
    if (want->n_prior == 0)
      MISMATCH("started from %" PRIu64 " prior voids (this run has none)",
        got->n_prior);
    if (got->n_prior == 0)
      MISMATCH("started from no prior voids (this run has %" PRIu64 ")",
        want->n_prior);
    MISMATCH("started from %" PRIu64 " prior voids, checksum %08x (this "
             "run's: %" PRIu64 ", %08x)",
      got->n_prior, got->prior_crc32, want->n_prior, want->prior_crc32);
  }

#undef MISMATCH

  return SIF_OK;
}

/* --- public reader --- */

sif_catalogue_t* sif_finder_exodus_checkpoint_read(
  const char* filepath, sif_finder_exodus_checkpoint_t* header) {

  if (!filepath) {
    SIF_LOG_ERROR(TAG, "filepath: NULL");
    sif__error_status_set(SIF_ERR_INVALID);
    return NULL;
  }

  sif_finder_exodus_checkpoint_t h;
  sif_real* radii = NULL;
  sif_catalogue_t* cat = NULL;

  int status = sif__exodus_checkpoint_load(filepath, &h, &radii, &cat);
  sif_free_aligned(radii);
  if (status != SIF_OK) {
    sif__error_status_set(status);
    return NULL;
  }

  if (h.is_survey &&
      (h.offset[0] != 0.0 || h.offset[1] != 0.0 || h.offset[2] != 0.0)) {
    const sif_real back[3] = {
      (sif_real)-h.offset[0], (sif_real)-h.offset[1], (sif_real)-h.offset[2]};
    status = sif_catalogue_translate(cat, back);
    if (status != SIF_OK) {
      sif_catalogue_free(cat);
      sif__error_status_set(status);
      return NULL;
    }
  }

  if (header)
    *header = h;
  return cat;
}

int sif_finder_exodus_checkpoint_check(const char* filepath, FILE* report) {
  if (!filepath) {
    SIF_LOG_ERROR(TAG, "filepath: NULL");
    return SIF_ERR_INVALID;
  }

  sif_finder_exodus_checkpoint_t h;
  sif_real* radii = NULL;
  sif_catalogue_t* cat = NULL;
  const int status = sif__exodus_checkpoint_load(filepath, &h, &radii, &cat);
  if (status != SIF_OK)
    return status;

  if (report) {
    const sif_real r_hi = radii[0], r_lo = radii[h.n_radii - 1];
    sif_real v_lo = 0.0f, v_hi = 0.0f;
    for (uint64_t i = 0; i < cat->n_voids; i++) {
      v_lo = i ? SIF_REAL_MIN(v_lo, cat->radii[i]) : cat->radii[i];
      v_hi = i ? SIF_REAL_MAX(v_hi, cat->radii[i]) : cat->radii[i];
    }

    fprintf(report, "%s\n", filepath);
    fprintf(report,
      "format        exodus checkpoint version %u, %s precision\n", h.version,
      h.is_double ? "double" : "single");
    fprintf(report, "run           %s, %s\n", h.is_survey ? "survey" : "box",
      h.is_weighted ? "weighted" : "unweighted");
    fprintf(report, "ladder        %u radii, %g down to %g\n", h.n_radii,
      (double)r_hi, (double)r_lo);
    if (h.next_rung < h.n_radii)
      fprintf(report, "progress      %u of %u rungs done, next at %g\n",
        h.next_rung, h.n_radii, (double)radii[h.next_rung]);
    else
      fprintf(report, "progress      all %u rungs done\n", h.n_radii);
    fprintf(report, "voids         %" PRIu64, h.n_voids);
    if (h.n_voids)
      fprintf(report, ", radii %g to %g", (double)v_lo, (double)v_hi);
    if (h.n_prior)
      fprintf(report, " (%" PRIu64 " from a prior catalogue)", h.n_prior);
    fputc('\n', report);
    /* Stored widened from a float: printed at the float's precision, so a
     * threshold of -0.7 does not read as -0.69999999. */
    fprintf(report, "threshold     %.7g\n", h.threshold);
    fprintf(report, "overlap       %.7g\n", h.overlap_fraction);
    fprintf(report, "search factor %g\n",
      (double)sif__finder_search_factor(h.options));
    fprintf(report, "CIC window    %s\n",
      (h.options & SIF_FINDER_KEEP_CIC_WINDOW) ? "kept" : "deconvolved");
    fprintf(report, "centres       %s\n", centres_label(h.options));
    fprintf(report, "grid          %u^3 cells over %.8g, %s%s\n",
      h.grid_n_cells, h.box_length, scheme_label(h.assignment),
      (h.assignment >> 8) ? " interlaced" : "");
    fprintf(report, "window        deconvolved %s\n",
      (h.options & SIF_FINDER_ALIASING_WINDOW) ? "allowing for aliasing"
                                               : "in full");
    fprintf(report, "tracers       %" PRIu64 ", total weight %.10g\n",
      h.n_particles, h.total_weight);
    if (h.is_survey) {
      fprintf(report, "randoms       %" PRIu64 ", total weight %.10g\n",
        h.random_n_particles, h.random_total_weight);
      fprintf(report, "offset        (%.8g, %.8g, %.8g)\n", h.offset[0],
        h.offset[1], h.offset[2]);
    }
  }

  sif_free_aligned(radii);
  sif_catalogue_free(cat);
  return SIF_OK;
}
