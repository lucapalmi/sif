/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "io/hdf5_util.h"

#include "io/internal.h"
#include "sif/core/macros.h"
#include "sif/utils/logger.h"

#include <stdio.h>
#include <string.h>

#define CAUSE_CAP 256

/* What sif__h5_keep() saved; cleared at the start of every entry point, so a
 * failure path that saved nothing reads as "HDF5 error" rather than as an
 * older call's cause. */
static SIF_THREAD_LOCAL char kept[CAUSE_CAP];

void sif__h5_quiet_begin(sif__h5_quiet_t* q) {
  H5Eget_auto2(H5E_DEFAULT, &q->func, &q->data);
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
  kept[0] = '\0';
}

void sif__h5_quiet_end(const sif__h5_quiet_t* q) {
  H5Eset_auto2(H5E_DEFAULT, q->func, q->data);
}

/* Walked outermost first, so the last entry seen is the innermost. */
static herr_t keep_last(unsigned n, const H5E_error2_t* e, void* data) {
  (void)n;
  char* out = data;
  unsigned long long eof = 0, stored = 0;
  if (e->min_num == H5E_NOTHDF5) {
    snprintf(out, CAUSE_CAP, "not an HDF5 file");
  } else if (e->desc &&
             sscanf(e->desc,
               "truncated file: eof = %llu, sblock->base_addr = %*u, "
               "stored_eof = %llu",
               &eof, &stored) == 2) {
    /* HDF5's own wording names its internals; the sizes are what matter. */
    snprintf(out, CAUSE_CAP, "%llu bytes, superblock says %llu (truncated)",
      eof, stored);
  } else if (e->desc && e->desc[0]) {
    snprintf(out, CAUSE_CAP, "%s", e->desc);
  } else {
    char msg[CAUSE_CAP];
    if (H5Eget_msg(e->min_num, NULL, msg, sizeof(msg)) > 0)
      snprintf(out, CAUSE_CAP, "%s", msg);
  }
  return 0;
}

const char* sif__h5_cause(void) {
  static SIF_THREAD_LOCAL char buf[CAUSE_CAP];
  buf[0] = '\0';
  H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, keep_last, buf);
  if (!buf[0])
    snprintf(buf, sizeof(buf), "HDF5 error");
  return buf;
}

int sif__h5_keep(void) {
  snprintf(kept, sizeof(kept), "%s", sif__h5_cause());
  return SIF_ERR_IO;
}

const char* sif__h5_kept(void) { return kept[0] ? kept : "HDF5 error"; }

hid_t sif__h5_open_read(const char* tag, const char* path) {
  if (sif__io_check_readable(tag, path, NULL) != SIF_OK)
    return H5I_INVALID_HID;

  const hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0)
    SIF_LOG_ERROR(tag, "%s: %s", path, sif__h5_cause());
  return file;
}

#undef CAUSE_CAP
