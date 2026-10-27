/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * Scratch files: data a run moves out of memory and reads back, in the
 * directory the scratch_directory setting names.
 *
 * A scratch file is unlinked the moment it is created. The run keeps reading
 * and writing it through the descriptor, and the space goes back to the
 * filesystem when the descriptor is closed -- or when the process dies, which
 * is the case this is for: a job killed at its time limit would otherwise
 * leave a file the size of its grid behind, in a directory nobody looks at.
 * The cost is that the file does not show up in a listing while the run is
 * going, so its path and size are logged when it is made.
 */

#include "sif/utils/logger.h"
#include "sif/utils/random.h"

#include "core/settings_internal.h"
#include "io/internal.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#define TAG "scratch"

/* Tries at a name before giving up: a collision needs two 64-bit ids to
 * agree, so a second one failing too says something is wrong with the
 * directory, not with the luck. */
#define CREATE_TRIES 4

uint64_t sif__unique_id(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);

  uint64_t seed = ((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec) ^
                  ((uint64_t)getpid() << 32);

  /* The clock alone is enough on one machine; two nodes started by the same
   * scheduler tick, writing to one shared directory, are what the kernel's
   * entropy is for. Its absence only weakens that case. */
  FILE* urandom = fopen("/dev/urandom", "rb");
  if (urandom) {
    uint64_t noise = 0;
    if (fread(&noise, sizeof(noise), 1, urandom) == 1)
      seed ^= noise;
    fclose(urandom);
  }

  return sif__splitmix64(&seed);
}

int sif__scratch_create(const char* what, uint64_t bytes, sif__scratch_t* out) {
  out->fd = -1;
  out->path[0] = '\0';

  const char* dir = NULL;
  int status = sif__setting_directory("scratch_directory", &dir);
  if (status != SIF_OK)
    return status;

  /* Checked before anything is written, so a run that cannot fit fails at the
   * start and says where, instead of an ENOSPC some way into the write. The
   * filesystem can of course fill up after this; that ENOSPC is still
   * reported, just later. */
  struct statvfs fs;
  if (statvfs(dir, &fs) == 0) {
    const uint64_t avail = (uint64_t)fs.f_bavail * (uint64_t)fs.f_frsize;
    if (avail < bytes) {
      SIF_LOG_ERROR(TAG,
        "%s: %.1f GiB free, %s needs %.1f GiB (set scratch_directory, or "
        "SIF_SCRATCH_DIRECTORY, to a filesystem with room)",
        dir, (double)avail / (1u << 30), what, (double)bytes / (1u << 30));
      return SIF_ERR_IO;
    }
  }

  for (int attempt = 0; attempt < CREATE_TRIES; attempt++) {
    snprintf(out->path, sizeof(out->path), "%s/sif-scratch-%016" PRIx64 ".bin",
      dir, sif__unique_id());

    /* O_EXCL: a name another process holds is never shared, however unlikely
     * it was to come up. */
    out->fd = open(out->path, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (out->fd >= 0)
      break;
    if (errno != EEXIST)
      return sif__io_os_error(TAG, out->path, "cannot create", errno);
  }
  if (out->fd < 0) {
    SIF_LOG_ERROR(TAG, "%s: no free name after %d tries", dir, CREATE_TRIES);
    return SIF_ERR_IO;
  }

  if (unlink(out->path) != 0)
    SIF_LOG_WARNING(TAG,
      "%s: cannot unlink while open (%s); removed when the run ends instead",
      out->path, sif__io_strerror(errno));
  else
    out->unlinked = 1;

  SIF_LOG_INFO(TAG, "%s: %s, %.1f GiB, gone when the run ends", out->path, what,
    (double)bytes / (1u << 30));
  return SIF_OK;
}

int sif__scratch_write(
  const sif__scratch_t* s, const void* src, uint64_t bytes, uint64_t offset) {
  const char* p = src;
  uint64_t done = 0;
  while (done < bytes) {
    const ssize_t n =
      pwrite(s->fd, p + done, (size_t)(bytes - done), (off_t)(offset + done));
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return sif__io_os_error(TAG, s->path, "write", n < 0 ? errno : EIO);
    done += (uint64_t)n;
  }
  return SIF_OK;
}

int sif__scratch_read(
  const sif__scratch_t* s, void* dst, uint64_t bytes, uint64_t offset) {
  char* p = dst;
  uint64_t done = 0;
  while (done < bytes) {
    const ssize_t n =
      pread(s->fd, p + done, (size_t)(bytes - done), (off_t)(offset + done));
    if (n < 0 && errno == EINTR)
      continue;
    /* 0 is the end of a file shorter than what was written to it: something
     * else truncated it. */
    if (n <= 0)
      return sif__io_os_error(TAG, s->path, "read", n < 0 ? errno : EIO);
    done += (uint64_t)n;
  }
  return SIF_OK;
}

int sif__scratch_settle(const sif__scratch_t* s) {
  /* Written back and dropped from the page cache, so the memory the spill
   * was for is actually returned: a batch system charges dirty pages to the
   * job, and on a node short of memory they are what it would be killed
   * over. */
  if (fsync(s->fd) != 0)
    return sif__io_os_error(TAG, s->path, "sync", errno);
#ifdef POSIX_FADV_DONTNEED
  (void)posix_fadvise(s->fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
  return SIF_OK;
}

void sif__scratch_close(sif__scratch_t* s) {
  if (!s || s->fd < 0)
    return;
  close(s->fd);
  if (!s->unlinked && unlink(s->path) != 0)
    SIF_LOG_WARNING(TAG, "%s: cannot remove (%s); delete it by hand", s->path,
      sif__io_strerror(errno));
  s->fd = -1;
  s->unlinked = 0;
}
