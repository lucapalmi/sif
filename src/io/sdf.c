/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

#include "sif/io/sdf.h"

#include "sdf_internal.h"
#include "sif/utils/crc32.h"
#include "sif/utils/logger.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* --- status --- */

/*
 * The gate every entry point passes through.
 *
 * A NULL status is a mistake in the calling code rather than a fact about the
 * file: there is nowhere to report it, so it is logged and the call declines
 * to run. An inherited error means something earlier in the sequence already
 * failed, and this call does nothing -- which is what lets a caller write a
 * run of calls and check once at the end.
 */
int sif__sdf_status_accepts(const sif_sdf_status_t* status, const char* what) {
  if (!status) {
    SIF_LOG_ERROR("sdf", "%s called without a status", what);
    return 0;
  }
  return *status == SIF_SDF_OK;
}

/*
 * Translates the reason open() or fopen() gave into the format's vocabulary.
 *
 * Only the conditions a caller can act on are named: a missing file is a
 * different mistake from a file it may not read, and a script can retry one
 * and not the other. Everything else is a failure of the filesystem and stays
 * SIF_SDF_ERR_IO, since nothing useful follows from telling them apart.
 */
static sif_sdf_status_t status_from_errno(int err) {
  switch (err) {
  case ENOENT:
  case ENOTDIR:
    return SIF_SDF_ERR_NOT_FOUND;
  case EACCES:
  case EPERM:
  case EROFS:
    return SIF_SDF_ERR_PERMISSION;
  case EEXIST:
    return SIF_SDF_ERR_EXISTS;
  case EISDIR:
  case ENAMETOOLONG:
    return SIF_SDF_ERR_INVALID;
  default:
    return SIF_SDF_ERR_IO;
  }
}

const char* sif_sdf_strerror(sif_sdf_status_t status) {
  switch (status) {
  case SIF_SDF_OK:
    return "no error";
  case SIF_SDF_ERR_INVALID:
    return "invalid argument";
  case SIF_SDF_ERR_ALLOC:
    return "out of memory";
  case SIF_SDF_ERR_IO:
    return "read or write failed";
  case SIF_SDF_ERR_NOT_FOUND:
    return "no such file";
  case SIF_SDF_ERR_PERMISSION:
    return "permission denied";
  case SIF_SDF_ERR_EXISTS:
    return "file already exists";
  case SIF_SDF_ERR_NOT_SDF:
    return "not a sif data file";
  case SIF_SDF_ERR_BYTE_ORDER:
    return "written on a machine of the other endianness";
  case SIF_SDF_ERR_VERSION:
    return "written by a newer sif than this build";
  case SIF_SDF_ERR_TRUNCATED:
    return "file ends in the middle of what it declares";
  case SIF_SDF_ERR_CORRUPT:
    return "file is internally inconsistent";
  case SIF_SDF_ERR_MODE:
    return "not allowed by the mode the file was opened with";
  case SIF_SDF_ERR_NO_CATALOG:
    return "file holds no catalogue";
  case SIF_SDF_ERR_CATALOG_MISMATCH:
    return "measured from a different catalogue than the file holds";
  case SIF_SDF_ERR_NAME_TAKEN:
    return "the file already holds a block of that kind under that name";
  case SIF_SDF_ERR_ABSENT:
    return "no such block, or no such key";
  case SIF_SDF_ERR_TYPE:
    return "the value is not of the type that was asked for";
  }

  /* Not a `default:` label inside the switch, so that adding a code without a
   * message is a compiler warning rather than a file that prints "unknown"
   * at a user. */
  return "unknown error";
}

/* --- handle --- */

/*
 * Releases a handle at any stage of construction, so the failure paths below
 * do not each have to know how far they got.
 */
static void handle_free(sif_sdf_t* file) {
  if (!file)
    return;

  if (file->stream)
    fclose(file->stream);
  free(file->blocks);
  free(file->path);
  free(file);
}

/* Frees the handle, records the reason and returns NULL, which is what every
 * failure inside an open or a create wants to do. The status is known to be
 * non-NULL and clear: sif__sdf_status_accepts() ran first. */
static sif_sdf_t* handle_fail(
  sif_sdf_t* file, sif_sdf_status_t reason, sif_sdf_status_t* status) {
  handle_free(file);
  *status = reason;
  return NULL;
}

/* An empty handle carrying nothing but its path, ready for a stream. */
static sif_sdf_t* handle_alloc(const char* filepath, sif_sdf_mode_t mode) {
  sif_sdf_t* file = calloc(1, sizeof(sif_sdf_t));
  if (!file)
    return NULL;

  const size_t path_bytes = strlen(filepath) + 1;
  file->path = malloc(path_bytes);
  if (!file->path) {
    handle_free(file);
    return NULL;
  }
  memcpy(file->path, filepath, path_bytes);

  file->fd = -1;
  file->mode = mode;
  file->failure = SIF_SDF_OK;
  return file;
}

/* Binds an already-open stream to the handle and records its descriptor. The
 * descriptor belongs to the stream: it is there for the parallel block reads,
 * which need pread(), and is closed by fclose() like everything else. */
static sif_sdf_status_t handle_attach(sif_sdf_t* file, FILE* stream) {
  file->stream = stream;
  file->fd = fileno(stream);
  if (file->fd < 0) {
    SIF_LOG_ERROR("sdf", "no descriptor for %s", file->path);
    return SIF_SDF_ERR_IO;
  }
  return SIF_SDF_OK;
}

/* --- file header --- */

/*
 * Rejects a header this build cannot work with.
 *
 * Everything here is checked before a single byte past the header is trusted,
 * because every later operation reads lengths out of the file and allocates
 * from them: a file that is not an .sdf file has to be turned away while its
 * numbers are still only bytes.
 */
static sif_sdf_status_t header_validate(
  const sif_sdf_header_t* header, const char* filepath) {
  if (memcmp(header->magic, SIF_SDF_MAGIC, 4) != 0) {
    SIF_LOG_ERROR("sdf", "%s is not a sif data file", filepath);
    return SIF_SDF_ERR_NOT_SDF;
  }

  /* Checked before the version, since a byte-swapped file's version field is
   * a large meaningless number and "written by a newer sif" would be the
   * wrong thing to tell the caller. */
  if (header->byte_order != SIF_SDF_BYTE_ORDER) {
    SIF_LOG_ERROR(
      "sdf", "%s was written on a machine of the other endianness", filepath);
    return SIF_SDF_ERR_BYTE_ORDER;
  }

  if (header->version > SIF_SDF_VERSION) {
    SIF_LOG_ERROR("sdf",
      "%s is version %u, but this build reads up to version %u", filepath,
      header->version, (unsigned)SIF_SDF_VERSION);
    return SIF_SDF_ERR_VERSION;
  }

  /* Negated rather than written as <= 0, so a NaN fails it too: every
   * comparison against a NaN is false, and a box that is not a positive
   * number would otherwise reach the finders as a wrapping length. */
  if (!(header->box_length > 0.0)) {
    SIF_LOG_ERROR(
      "sdf", "%s declares a box length of %g", filepath, header->box_length);
    return SIF_SDF_ERR_CORRUPT;
  }

  return SIF_SDF_OK;
}

/* Caches the writer string as something that can be handed out: the field on
 * disk is zero-padded and may fill all 16 bytes without a terminator. */
static void writer_cache(sif_sdf_t* file) {
  memcpy(file->writer, file->header.writer, SIF__SDF_WRITER_BYTES);
  file->writer[SIF__SDF_WRITER_BYTES] = '\0';
}

/* --- raw i/o --- */

/*
 * Every read in the file goes through here: seek, then fill, looping over the
 * short reads fread() is entitled to return.
 *
 * Absolute offsets rather than a running position, because the block table
 * knows where everything is and a reader that tracked its own position would
 * have to be right about it at every step instead of at one.
 */
sif_sdf_status_t sif__sdf_read_at(
  sif_sdf_t* file, uint64_t offset, void* dest, size_t bytes) {
  if (bytes == 0)
    return SIF_SDF_OK;

  if (fseeko(file->stream, (off_t)offset, SEEK_SET) != 0) {
    SIF_LOG_ERROR("sdf", "could not seek to %llu in %s",
      (unsigned long long)offset, file->path);
    return SIF_SDF_ERR_IO;
  }

  char* out = (char*)dest;
  size_t done = 0;
  while (done < bytes) {
    const size_t got = fread(out + done, 1, bytes - done, file->stream);
    if (got == 0) {
      /* End of file where the file said there was more: the lengths in the
       * header are describing something that is not there. */
      const int at_end = feof(file->stream);
      SIF_LOG_ERROR("sdf", "%s ended %zu bytes into a read of %zu", file->path,
        done, bytes);
      return at_end ? SIF_SDF_ERR_TRUNCATED : SIF_SDF_ERR_IO;
    }
    done += got;
  }

  return SIF_SDF_OK;
}

sif_sdf_status_t sif__sdf_read_raw(
  sif_sdf_t* file, uint64_t offset, void* dest, size_t bytes, uint32_t* crc) {
  const sif_sdf_status_t status = sif__sdf_read_at(file, offset, dest, bytes);
  if (status != SIF_SDF_OK)
    return status;

  *crc = sif_crc32_update(*crc, dest, bytes);
  return SIF_SDF_OK;
}

/* Writes at wherever the stream is, which is where the last write left it. */
sif_sdf_status_t sif__sdf_write_now(
  sif_sdf_t* file, const void* src, size_t bytes) {
  if (bytes == 0)
    return SIF_SDF_OK;

  if (fwrite(src, 1, bytes, file->stream) != bytes) {
    SIF_LOG_ERROR("sdf", "could not write %zu bytes to %s", bytes, file->path);
    return SIF_SDF_ERR_IO;
  }
  return SIF_SDF_OK;
}

/* The zero padding that carries a block up to the next 64-byte boundary. */
sif_sdf_status_t sif__sdf_write_padding(sif_sdf_t* file, size_t bytes) {
  static const char zeros[SIF__SDF_BLOCK_ALIGN] = {0};
  SIF_ASSERT(bytes < sizeof(zeros));
  return sif__sdf_write_now(file, zeros, bytes);
}

/* Elements converted at a time when the file's width is not this build's.
 * Small enough to sit on the stack, large enough that the per-chunk seek is
 * lost in the read. */
#define CONVERT_CHUNK 1024

/*
 * Reads `count` reals stored at `offset` in the file's width, converting to
 * this build's sif_real if they differ, and folds the bytes *as they are on
 * disk* into the running checksum.
 *
 * The checksum has to see the file's bytes rather than the converted values,
 * or a build of the other precision would compute a different one for an
 * intact file and reject it.
 */
sif_sdf_status_t sif__sdf_read_reals(sif_sdf_t* file, uint64_t offset,
  uint64_t count, sif_sdf_dtype_t dtype, sif_real* dest, uint32_t* crc) {

  const size_t width = (dtype == SIF_SDF_F64) ? sizeof(double) : sizeof(float);

  if (width == sizeof(sif_real)) {
    /* Same width: straight into its final place, and the checksum is taken
     * from there rather than from a copy of it. */
    const size_t bytes = (size_t)count * width;
    const sif_sdf_status_t status = sif__sdf_read_at(file, offset, dest, bytes);
    if (status != SIF_SDF_OK)
      return status;

    *crc = sif_crc32_update(*crc, dest, bytes);
    return SIF_SDF_OK;
  }

  union {
    float f32[CONVERT_CHUNK];
    double f64[CONVERT_CHUNK];
  } scratch;

  uint64_t done = 0;
  while (done < count) {
    const uint64_t chunk =
      (count - done < CONVERT_CHUNK) ? (count - done) : CONVERT_CHUNK;
    const size_t bytes = (size_t)chunk * width;

    const sif_sdf_status_t status =
      sif__sdf_read_at(file, offset + done * width, &scratch, bytes);
    if (status != SIF_SDF_OK)
      return status;

    *crc = sif_crc32_update(*crc, &scratch, bytes);

    if (dtype == SIF_SDF_F64) {
      for (uint64_t i = 0; i < chunk; i++)
        dest[done + i] = (sif_real)scratch.f64[i];
    } else {
      for (uint64_t i = 0; i < chunk; i++)
        dest[done + i] = (sif_real)scratch.f32[i];
    }

    done += chunk;
  }

  return SIF_SDF_OK;
}

#undef CONVERT_CHUNK

/* --- the block table --- */

sif_sdf_status_t sif__sdf_blocks_push(
  sif_sdf_t* file, uint64_t offset, const sif_sdf_block_header_t* header) {

  if (file->n_blocks == file->block_capacity) {
    /* Doubling from a start that already covers a full run: a catalogue, two
     * profile sets, a size function and a note or two. */
    const uint32_t grown = file->block_capacity ? file->block_capacity * 2 : 8;
    sif__sdf_entry_t* table =
      realloc(file->blocks, (size_t)grown * sizeof(sif__sdf_entry_t));
    if (!table) {
      SIF_LOG_ERROR("sdf", "failed to grow the block table of %s", file->path);
      return SIF_SDF_ERR_ALLOC;
    }
    file->blocks = table;
    file->block_capacity = grown;
  }

  file->blocks[file->n_blocks].offset = offset;
  file->blocks[file->n_blocks].header = *header;
  file->n_blocks++;

  return SIF_SDF_OK;
}

/*
 * Walks the chain from the first block to the end of the file, recording what
 * is there.
 *
 * Done once, at open, so that finding a block afterwards costs no reads. Every
 * length is checked against what is left of the file before it is used, and
 * before anything is allocated from it: the numbers come out of the file, and
 * a file that is not what it claims can otherwise ask for an arbitrary amount
 * of memory or seek past its own end.
 */
static sif_sdf_status_t blocks_walk(sif_sdf_t* file) {
  struct stat info;
  if (fstat(file->fd, &info) != 0) {
    SIF_LOG_ERROR("sdf", "could not stat %s", file->path);
    return SIF_SDF_ERR_IO;
  }
  file->file_size = (uint64_t)info.st_size;

  uint64_t pos = SIF__SDF_HEADER_BYTES;

  while (pos < file->file_size) {
    const uint64_t left = file->file_size - pos;

    if (left < SIF__SDF_BLOCK_BYTES) {
      SIF_LOG_ERROR("sdf", "%s ends in %llu bytes that are not a block header",
        file->path, (unsigned long long)left);
      return SIF_SDF_ERR_TRUNCATED;
    }

    sif__sdf_entry_t entry;
    entry.offset = pos;

    sif_sdf_status_t status =
      sif__sdf_read_at(file, pos, &entry.header, SIF__SDF_BLOCK_BYTES);
    if (status != SIF_SDF_OK)
      return status;

    if (memcmp(entry.header.magic, SIF__SDF_BLOCK_MAGIC, 4) != 0) {
      SIF_LOG_ERROR("sdf", "%s has no block header at %llu", file->path,
        (unsigned long long)pos);
      return SIF_SDF_ERR_CORRUPT;
    }

    /* Written as two comparisons against what is left rather than as one sum,
     * because the sum is exactly what an unchecked file could overflow. */
    const uint64_t room = left - SIF__SDF_BLOCK_BYTES;
    if (entry.header.meta_bytes > room ||
        entry.header.data_bytes > room - entry.header.meta_bytes) {
      SIF_LOG_ERROR("sdf",
        "the block at %llu in %s declares more than the "
        "file holds",
        (unsigned long long)pos, file->path);
      return SIF_SDF_ERR_TRUNCATED;
    }

    const uint64_t total = sif__sdf_block_bytes(&entry);
    if (total - SIF__SDF_BLOCK_BYTES > room) {
      /* The payload fits and its padding does not, so the last block was cut
       * off mid-write rather than being short of data. */
      SIF_LOG_ERROR("sdf", "the block at %llu in %s is missing its padding",
        (unsigned long long)pos, file->path);
      return SIF_SDF_ERR_TRUNCATED;
    }

    status = sif__sdf_blocks_push(file, pos, &entry.header);
    if (status != SIF_SDF_OK)
      return status;

    pos += total;
  }

  file->end_offset = pos;
  return SIF_SDF_OK;
}

/*
 * The rules a file has to satisfy once its blocks are known: it is built
 * around one catalogue, and everything else in it came from that catalogue.
 *
 * This is the check the whole format exists for. Without it a file is a bag of
 * arrays, and a size function sitting next to a catalogue it was not measured
 * from looks exactly like one that was.
 */
static sif_sdf_status_t blocks_validate(sif_sdf_t* file) {
  if (file->n_blocks == 0) {
    /* Nothing but a file header. Legal only for an append handle, which is
     * how a create that died between its two writes gets finished rather than
     * mourned. */
    if (file->mode != SIF_SDF_APPEND) {
      SIF_LOG_ERROR("sdf", "%s holds no blocks at all", file->path);
      return SIF_SDF_ERR_NO_CATALOG;
    }
    return SIF_SDF_OK;
  }

  const sif_sdf_block_header_t* first = &file->blocks[0].header;

  if (first->type != SIF_SDF_BLOCK_CATALOG) {
    SIF_LOG_ERROR(
      "sdf", "%s does not begin with a catalogue block", file->path);
    return SIF_SDF_ERR_NO_CATALOG;
  }
  if (first->catalog_id == 0) {
    SIF_LOG_ERROR("sdf", "the catalogue in %s has no identity", file->path);
    return SIF_SDF_ERR_CORRUPT;
  }

  file->catalog_id = first->catalog_id;
  file->n_voids = first->n_items;

  for (uint32_t i = 1; i < file->n_blocks; i++) {
    const sif_sdf_block_header_t* header = &file->blocks[i].header;

    if (header->type == SIF_SDF_BLOCK_CATALOG) {
      SIF_LOG_ERROR("sdf", "%s holds a second catalogue, at block %u",
        file->path, (unsigned)i);
      return SIF_SDF_ERR_CORRUPT;
    }

    /* A note belongs to the file rather than to the catalogue, and says so by
     * carrying no identity. Everything else must carry this one. */
    const uint64_t expected =
      (header->type == SIF_SDF_BLOCK_META) ? 0 : file->catalog_id;

    if (header->catalog_id != expected) {
      SIF_LOG_ERROR("sdf", "block %u of %s was measured from another catalogue",
        (unsigned)i, file->path);
      return SIF_SDF_ERR_CATALOG_MISMATCH;
    }

    /* A set with a row per void has to have the catalogue's worth of rows.
     * The identity above already makes this nearly impossible to violate; it
     * costs a comparison and it is the assumption every reader of those rows
     * goes on to make. */
    const int rows_per_void = (header->type == SIF_SDF_BLOCK_DENSITY_PROFILES ||
                               header->type == SIF_SDF_BLOCK_VELOCITY_PROFILES);

    if (rows_per_void && header->n_items != file->n_voids) {
      SIF_LOG_ERROR("sdf", "block %u of %s holds %llu rows for %llu voids",
        (unsigned)i, file->path, (unsigned long long)header->n_items,
        (unsigned long long)file->n_voids);
      return SIF_SDF_ERR_CATALOG_MISMATCH;
    }
  }

  return SIF_SDF_OK;
}

/* --- the catalogue block --- */

/*
 * Writes the catalogue as block 0.
 *
 * Component-major, in the order the four views sit in the catalogue's own
 * arena, so that reading it back is four reads into four buffers rather than a
 * stride through interleaved values.
 */
/*
 * Cuts the file back to the length it had before a half-written block.
 *
 * An append that fails leaves bytes behind that are not a block, and a file
 * that ends in one of those does not open at all -- so the failure of one
 * append would cost the caller everything already in the file. Shrinking is
 * the one write that still works when the disk is full, so this is nearly
 * always able to put the file back.
 */
static void block_rollback(sif_sdf_t* file) {
  if (fflush(file->stream) != 0)
    clearerr(file->stream);

  if (ftruncate(file->fd, (off_t)file->end_offset) != 0) {
    SIF_LOG_WARNING("sdf",
      "%s could not be cut back after a failed append and now ends in a "
      "partial block",
      file->path);
    return;
  }

  file->file_size = file->end_offset;
  SIF_LOG_WARNING(
    "sdf", "%s was left as it was after a failed append", file->path);
}

sif_sdf_status_t sif__sdf_block_write(sif_sdf_t* file,
  sif_sdf_block_header_t* header, const void* meta, uint32_t meta_bytes,
  const sif__sdf_span_t* spans, uint32_t n_spans) {

  memcpy(header->magic, SIF__SDF_BLOCK_MAGIC, 4);
  header->type_version = SIF__SDF_TYPE_VERSION;
  header->meta_bytes = meta_bytes;

  header->data_bytes = 0;
  for (uint32_t i = 0; i < n_spans; i++)
    header->data_bytes += spans[i].bytes;

  /* Checksummed before anything is written, because it goes in the header
   * that precedes what it covers -- and in the same order a reader will walk
   * it, metadata first. */
  uint32_t crc = SIF_CRC32_INIT;
  crc = sif_crc32_update(crc, meta, meta_bytes);
  for (uint32_t i = 0; i < n_spans; i++)
    crc = sif_crc32_update(crc, spans[i].data, (size_t)spans[i].bytes);
  header->crc32 = sif_crc32_final(crc);

  const sif__sdf_entry_t entry = {file->end_offset, *header};
  const uint64_t total = sif__sdf_block_bytes(&entry);
  const uint64_t padding =
    total - SIF__SDF_BLOCK_BYTES - meta_bytes - header->data_bytes;

  if (fseeko(file->stream, (off_t)file->end_offset, SEEK_SET) != 0) {
    SIF_LOG_ERROR("sdf", "could not seek to the end of %s", file->path);
    file->failure = SIF_SDF_ERR_IO;
    return SIF_SDF_ERR_IO;
  }

  sif_sdf_status_t status =
    sif__sdf_write_now(file, header, SIF__SDF_BLOCK_BYTES);
  if (status == SIF_SDF_OK)
    status = sif__sdf_write_now(file, meta, meta_bytes);
  for (uint32_t i = 0; i < n_spans && status == SIF_SDF_OK; i++)
    status = sif__sdf_write_now(file, spans[i].data, (size_t)spans[i].bytes);
  if (status == SIF_SDF_OK)
    status = sif__sdf_write_padding(file, (size_t)padding);

  if (status != SIF_SDF_OK) {
    block_rollback(file);
    file->failure = status;
    return status;
  }

  /* Recorded only once the bytes are all out. Until then the table would be
   * describing something the file does not hold. */
  status = sif__sdf_blocks_push(file, file->end_offset, header);
  if (status != SIF_SDF_OK) {
    block_rollback(file);
    file->failure = status;
    return status;
  }

  file->end_offset += total;
  file->file_size = file->end_offset;

  return SIF_SDF_OK;
}

/* --- finding a block --- */

sif_sdf_status_t sif__sdf_name_pack(
  const char* name, char out[SIF__SDF_NAME_BYTES], const char* what) {
  memset(out, 0, SIF__SDF_NAME_BYTES);

  if (!name)
    return SIF_SDF_OK;

  const size_t bytes = strlen(name);
  if (bytes > SIF__SDF_NAME_BYTES) {
    /* Refused rather than cut down to fit: two names differing past the
     * sixteenth byte would land in the file as one, and the check that keeps
     * names unique would then be right to reject the second. */
    SIF_LOG_ERROR("sdf",
      "the name `%s` given to %s is %zu bytes, and %d is "
      "the most a block can carry",
      name, what, bytes, SIF__SDF_NAME_BYTES);
    return SIF_SDF_ERR_INVALID;
  }

  memcpy(out, name, bytes);
  return SIF_SDF_OK;
}

int32_t sif__sdf_find(
  const sif_sdf_t* file, sif_sdf_block_type_t type, const char* name) {

  char wanted[SIF__SDF_NAME_BYTES];
  if (sif__sdf_name_pack(name, wanted, "a lookup") != SIF_SDF_OK)
    return -1;

  /* No name asked for means the first block of the type, which is the whole
   * of what a caller who never named anything needs. */
  const int any = (name == NULL || name[0] == '\0');

  for (uint32_t i = 0; i < file->n_blocks; i++) {
    const sif_sdf_block_header_t* header = &file->blocks[i].header;
    if (header->type != (uint16_t)type)
      continue;
    if (any || memcmp(header->name, wanted, SIF__SDF_NAME_BYTES) == 0)
      return (int32_t)i;
  }

  return -1;
}

/* --- the gates --- */

sif_sdf_status_t sif__sdf_write_gate(sif_sdf_t* file, const char* what) {
  if (!file) {
    SIF_LOG_ERROR("sdf", "no file given to write %s to", what);
    return SIF_SDF_ERR_INVALID;
  }

  /* A handle that has already failed a write is not asked to try another: the
   * file was cut back to what it held, and piling a second attempt on top of
   * whatever went wrong is not how a caller finds out. */
  if (file->failure != SIF_SDF_OK) {
    SIF_LOG_ERROR("sdf", "%s already failed a write: %s", file->path,
      sif_sdf_strerror(file->failure));
    return file->failure;
  }

  if (file->mode != SIF_SDF_APPEND) {
    SIF_LOG_ERROR("sdf", "%s is open for reading", file->path);
    return SIF_SDF_ERR_MODE;
  }

  /* Even a note needs the catalogue to be there: a file is what was found and
   * what was measured about it, and neither is said by a file that has not
   * got as far as its first block. */
  if (file->n_blocks == 0 || file->catalog_id == 0) {
    SIF_LOG_ERROR("sdf", "%s holds no catalogue", file->path);
    return SIF_SDF_ERR_NO_CATALOG;
  }

  return SIF_SDF_OK;
}

sif_sdf_status_t sif__sdf_append_gate(sif_sdf_t* file,
  sif_sdf_block_type_t type, const char* name, uint64_t source_id) {

  const sif_sdf_status_t writable = sif__sdf_write_gate(file, "a measurement");
  if (writable != SIF_SDF_OK)
    return writable;

  /* The check the format exists for. A measurement that names no catalogue
   * cannot prove it came from this one, and one that names another did not:
   * either way its rows line up with nothing here. */
  if (source_id == 0) {
    SIF_LOG_ERROR("sdf",
      "the measurement offered to %s does not name the catalogue it came from",
      file->path);
    return SIF_SDF_ERR_CATALOG_MISMATCH;
  }
  if (source_id != file->catalog_id) {
    SIF_LOG_ERROR("sdf",
      "the measurement offered to %s was made from a different catalogue",
      file->path);
    return SIF_SDF_ERR_CATALOG_MISMATCH;
  }

  char packed[SIF__SDF_NAME_BYTES];
  const sif_sdf_status_t named = sif__sdf_name_pack(name, packed, "an append");
  if (named != SIF_SDF_OK)
    return named;

  /* Unique within a type rather than across the file, so a density set and a
   * velocity set measured together can share the name they were measured
   * under. */
  if (sif__sdf_find(file, type, name) >= 0) {
    SIF_LOG_ERROR("sdf", "%s already holds a block of this kind named `%s`",
      file->path, (name && name[0]) ? name : "");
    return SIF_SDF_ERR_NAME_TAKEN;
  }

  return SIF_SDF_OK;
}

sif_sdf_status_t sif__sdf_read_gate(const sif_sdf_t* file,
  const sif__sdf_entry_t* entry, uint64_t expected_bytes) {

  const sif_sdf_dtype_t dtype = (sif_sdf_dtype_t)entry->header.real_dtype;
  if (dtype != SIF_SDF_F32 && dtype != SIF_SDF_F64) {
    SIF_LOG_ERROR(
      "sdf", "a block in %s is stored as neither f32 nor f64", file->path);
    return SIF_SDF_ERR_CORRUPT;
  }

  if (entry->header.type_version > SIF__SDF_TYPE_VERSION) {
    SIF_LOG_ERROR("sdf",
      "a block in %s is laid out in a way version %u of this build does not "
      "know",
      file->path, (unsigned)SIF__SDF_TYPE_VERSION);
    return SIF_SDF_ERR_VERSION;
  }

  /* The length the block's own shape implies, against the length it declares.
   * Everything read afterwards is sized from that shape, so the two agreeing
   * is what makes the reads safe. */
  if (entry->header.data_bytes != expected_bytes) {
    SIF_LOG_ERROR("sdf",
      "a block in %s declares %llu bytes where its shape needs %llu",
      file->path, (unsigned long long)entry->header.data_bytes,
      (unsigned long long)expected_bytes);
    return SIF_SDF_ERR_CORRUPT;
  }

  return SIF_SDF_OK;
}

/* --- the catalogue block --- */

/*
 * Writes the catalogue as block 0.
 *
 * Component-major, in the order the four views sit in the catalogue's own
 * arena, so that reading it back is four reads into four buffers rather than a
 * stride through interleaved values.
 */
static sif_sdf_status_t catalog_write(
  sif_sdf_t* file, const sif_catalog_t* cat) {

  const uint64_t n_voids = cat->n_voids;
  const uint64_t view_bytes = n_voids * sizeof(sif_real);

  const sif__sdf_span_t spans[4] = {{cat->cx, view_bytes},
    {cat->cy, view_bytes}, {cat->cz, view_bytes}, {cat->radii, view_bytes}};

  sif_sdf_block_header_t header;
  memset(&header, 0, sizeof(sif_sdf_block_header_t));
  header.type = (uint16_t)SIF_SDF_BLOCK_CATALOG;
  header.real_dtype = (uint16_t)sif__sdf_native_dtype();
  header.n_items = n_voids;
  header.catalog_id = sif_catalog_id(cat);

  const sif_sdf_status_t status =
    sif__sdf_block_write(file, &header, NULL, 0, spans, 4);
  if (status != SIF_SDF_OK)
    return status;

  file->catalog_id = header.catalog_id;
  file->n_voids = n_voids;

  return SIF_SDF_OK;
}

sif_catalog_t* sif_sdf_catalog(sif_sdf_t* file, sif_sdf_status_t* status) {
  if (!sif__sdf_status_accepts(status, "catalog"))
    return NULL;

  if (!file) {
    SIF_LOG_ERROR("sdf", "no file given to read a catalogue from");
    *status = SIF_SDF_ERR_INVALID;
    return NULL;
  }
  if (file->n_blocks == 0) {
    SIF_LOG_ERROR("sdf", "%s holds no catalogue yet", file->path);
    *status = SIF_SDF_ERR_NO_CATALOG;
    return NULL;
  }

  const sif__sdf_entry_t* entry = &file->blocks[0];
  const uint64_t n_voids = entry->header.n_items;
  const sif_sdf_dtype_t dtype = (sif_sdf_dtype_t)entry->header.real_dtype;
  const uint64_t width = sif__sdf_dtype_bytes(dtype);

  /* Guarded before the multiplication rather than after it: a void count that
   * overflows the product could otherwise agree with a length that is nothing
   * of the sort. */
  if (n_voids > UINT64_MAX / (4 * width)) {
    SIF_LOG_ERROR("sdf", "the catalogue in %s claims %llu voids", file->path,
      (unsigned long long)n_voids);
    *status = SIF_SDF_ERR_CORRUPT;
    return NULL;
  }

  const sif_sdf_status_t gate =
    sif__sdf_read_gate(file, entry, n_voids * 4 * width);
  if (gate != SIF_SDF_OK) {
    *status = gate;
    return NULL;
  }

  sif_catalog_t* cat = sif_catalog_alloc(n_voids);
  if (!cat) {
    SIF_LOG_ERROR("sdf", "failed to allocate a catalogue of %llu voids",
      (unsigned long long)n_voids);
    *status = SIF_SDF_ERR_ALLOC;
    return NULL;
  }

  sif_real* views[4] = {cat->cx, cat->cy, cat->cz, cat->radii};
  uint64_t offset = sif__sdf_data_offset(entry);
  uint32_t crc = SIF_CRC32_INIT;

  for (int v = 0; v < 4; v++) {
    const sif_sdf_status_t reason =
      sif__sdf_read_reals(file, offset, n_voids, dtype, views[v], &crc);
    if (reason != SIF_SDF_OK) {
      sif_catalog_free(cat);
      *status = reason;
      return NULL;
    }
    offset += n_voids * width;
  }

  if (sif_crc32_final(crc) != entry->header.crc32) {
    SIF_LOG_ERROR(
      "sdf", "the catalogue in %s does not match its checksum", file->path);
    sif_catalog_free(cat);
    *status = SIF_SDF_ERR_CORRUPT;
    return NULL;
  }

  /* Filled in directly rather than through sif_catalog_append(), so the size
   * is set by hand -- and so is the identity, which is the file's rather than
   * the fresh one the allocator gave this catalogue a moment ago. */
  cat->n_voids = n_voids;
  cat->id = entry->header.catalog_id;

  SIF_LOG_DEBUG(
    "sdf", "read %llu voids from %s", (unsigned long long)n_voids, file->path);

  return cat;
}

/* --- open and create --- */

sif_sdf_t* sif_sdf_open(
  const char* filepath, sif_sdf_mode_t mode, sif_sdf_status_t* status) {
  if (!sif__sdf_status_accepts(status, "open"))
    return NULL;

  if (!filepath) {
    SIF_LOG_ERROR("sdf", "no path given to open");
    *status = SIF_SDF_ERR_INVALID;
    return NULL;
  }
  if (mode != SIF_SDF_READ && mode != SIF_SDF_APPEND) {
    SIF_LOG_ERROR("sdf", "unknown mode %d for %s", (int)mode, filepath);
    *status = SIF_SDF_ERR_INVALID;
    return NULL;
  }

  sif_sdf_t* file = handle_alloc(filepath, mode);
  if (!file) {
    SIF_LOG_ERROR("sdf", "failed to allocate a handle for %s", filepath);
    *status = SIF_SDF_ERR_ALLOC;
    return NULL;
  }

  /* "r+b" and not "ab": appending a block means seeking to the end and
   * writing there, while still being able to read what is already in the file
   * -- and an append-mode stream would force every write to the end whatever
   * the seek said, which is the one thing a format with an in-file back
   * reference cannot have. */
  FILE* stream = fopen(filepath, (mode == SIF_SDF_APPEND) ? "r+b" : "rb");
  if (!stream) {
    const sif_sdf_status_t reason = status_from_errno(errno);
    SIF_LOG_ERROR(
      "sdf", "could not open %s: %s", filepath, sif_sdf_strerror(reason));
    return handle_fail(file, reason, status);
  }

  sif_sdf_status_t reason = handle_attach(file, stream);
  if (reason != SIF_SDF_OK)
    return handle_fail(file, reason, status);

  if (fread(&file->header, 1, SIF__SDF_HEADER_BYTES, file->stream) !=
      SIF__SDF_HEADER_BYTES) {
    /* Short of a header is not merely a small file: an .sdf file is 64 bytes
     * before it holds anything at all. */
    reason = feof(file->stream) ? SIF_SDF_ERR_TRUNCATED : SIF_SDF_ERR_IO;
    SIF_LOG_ERROR("sdf", "could not read the header of %s: %s", filepath,
      sif_sdf_strerror(reason));
    return handle_fail(file, reason, status);
  }

  reason = header_validate(&file->header, filepath);
  if (reason != SIF_SDF_OK)
    return handle_fail(file, reason, status);

  writer_cache(file);

  /* Walked once here, so that every later call knows where everything is
   * without a read, and so that a file which is not internally consistent is
   * refused now rather than halfway through a caller's analysis. */
  reason = blocks_walk(file);
  if (reason != SIF_SDF_OK)
    return handle_fail(file, reason, status);

  reason = blocks_validate(file);
  if (reason != SIF_SDF_OK)
    return handle_fail(file, reason, status);

  SIF_LOG_DEBUG("sdf", "opened %s (version %u, box %g, written by %s)",
    filepath, file->header.version, file->header.box_length,
    file->writer[0] ? file->writer : "an unnamed build");
  SIF_LOG_DEBUG("sdf", "%s holds %u blocks over %llu voids", filepath,
    (unsigned)file->n_blocks, (unsigned long long)file->n_voids);

  return file;
}

sif_sdf_t* sif_sdf_create(const char* filepath, double box_length,
  const sif_catalog_t* cat, sif_option flags, sif_sdf_status_t* status) {
  if (!sif__sdf_status_accepts(status, "create"))
    return NULL;

  if (!filepath) {
    SIF_LOG_ERROR("sdf", "no path given to create");
    *status = SIF_SDF_ERR_INVALID;
    return NULL;
  }
  if (!(box_length > 0.0)) {
    SIF_LOG_ERROR(
      "sdf", "box length %g cannot be recorded in %s", box_length, filepath);
    *status = SIF_SDF_ERR_INVALID;
    return NULL;
  }
  if (!cat) {
    SIF_LOG_ERROR("sdf", "%s cannot be created without a catalogue", filepath);
    *status = SIF_SDF_ERR_INVALID;
    return NULL;
  }
  if (sif_catalog_id(cat) == 0) {
    /* Only reachable from a catalogue the library did not build, and there is
     * nothing to be done with one: every block written afterwards would claim
     * to belong to a catalogue that cannot be named. */
    SIF_LOG_ERROR(
      "sdf", "the catalogue offered to %s has no identity", filepath);
    *status = SIF_SDF_ERR_INVALID;
    return NULL;
  }

  /* A file is created through open() rather than fopen() for O_EXCL, which is
   * the only way to refuse an existing path without a race between the test
   * and the truncation. */
  const int open_flags =
    O_RDWR | O_CREAT | ((flags & SIF_SDF_OVERWRITE) ? O_TRUNC : O_EXCL);

  const int fd = open(filepath, open_flags, 0666);
  if (fd < 0) {
    const sif_sdf_status_t reason = status_from_errno(errno);
    SIF_LOG_ERROR(
      "sdf", "could not create %s: %s", filepath, sif_sdf_strerror(reason));
    *status = reason;
    return NULL;
  }

  /* Created in append mode: a fresh file has no blocks, and the only thing a
   * caller can do with one is write the catalogue it must start with. */
  sif_sdf_t* file = handle_alloc(filepath, SIF_SDF_APPEND);
  if (!file) {
    SIF_LOG_ERROR("sdf", "failed to allocate a handle for %s", filepath);
    close(fd);
    *status = SIF_SDF_ERR_ALLOC;
    return NULL;
  }

  FILE* stream = fdopen(fd, "r+b");
  if (!stream) {
    SIF_LOG_ERROR("sdf", "could not open a stream on %s", filepath);
    close(fd);
    return handle_fail(file, SIF_SDF_ERR_IO, status);
  }

  const sif_sdf_status_t reason = handle_attach(file, stream);
  if (reason != SIF_SDF_OK)
    return handle_fail(file, reason, status);

  /* Zeroed first, so every reserved byte in the header is zero on disk
   * whatever the fields below leave untouched. */
  memset(&file->header, 0, sizeof(sif_sdf_header_t));
  memcpy(file->header.magic, SIF_SDF_MAGIC, 4);
  file->header.version = SIF_SDF_VERSION;
  file->header.byte_order = SIF_SDF_BYTE_ORDER;
  file->header.box_length = box_length;

  const time_t now = time(NULL);
  file->header.created = (now == (time_t)-1) ? 0 : (uint64_t)now;

  const size_t writer_bytes = strlen(SIF__SDF_WRITER);
  memcpy(file->header.writer, SIF__SDF_WRITER,
    (writer_bytes < SIF__SDF_WRITER_BYTES) ? writer_bytes
                                           : SIF__SDF_WRITER_BYTES);

  if (fwrite(&file->header, SIF__SDF_HEADER_BYTES, 1, file->stream) != 1) {
    SIF_LOG_ERROR("sdf", "failed to write the header of %s", filepath);
    return handle_fail(file, SIF_SDF_ERR_IO, status);
  }

  writer_cache(file);
  file->end_offset = SIF__SDF_HEADER_BYTES;
  file->file_size = SIF__SDF_HEADER_BYTES;

  /* The catalogue goes in immediately, in the same call, so that no code path
   * anywhere can produce a file without one. A crash between the two writes
   * leaves a header over an empty chain, which is the one state
   * sif_sdf_open() accepts for appending and refuses for reading. */
  const sif_sdf_status_t written = catalog_write(file, cat);
  if (written != SIF_SDF_OK)
    return handle_fail(file, written, status);

  SIF_LOG_DEBUG("sdf", "created %s (box %g, %llu voids)", filepath, box_length,
    (unsigned long long)file->n_voids);

  return file;
}

void sif_sdf_close(sif_sdf_t* file, sif_sdf_status_t* status) {
  /* No sif__sdf_status_accepts() here. Closing is the one thing that has to
   * happen whatever went wrong before it, or a caller unwinding after a failure
   * leaks the file it was unwinding from. */
  if (!file)
    return;

  sif_sdf_status_t reason = SIF_SDF_OK;

  if (file->stream) {
    /* fwrite() succeeding only means the bytes reached the stdio buffer, and
     * a full disk surfaces at the flush inside fclose(). Unchecked, a run
     * whose last append never landed ends with a file that looks finished and
     * has a torn block at the end. */
    const int had_error = ferror(file->stream);
    if (fclose(file->stream) != 0 || had_error) {
      SIF_LOG_ERROR("sdf", "failed to flush %s to disk", file->path);
      reason = SIF_SDF_ERR_IO;
    }
    file->stream = NULL;
  }

  /* Reported only into a status that is still clear: whatever failed earlier
   * is what the caller needs to see, and a flush that failed because the run
   * was already being abandoned says less than the failure that abandoned
   * it. */
  if (status && *status == SIF_SDF_OK)
    *status = reason;

  /* The same release every failed open uses. Duplicating it here is how the
   * block table came to be freed on one path and not the other. */
  handle_free(file);
}

/* --- accessors --- */

sif_sdf_mode_t sif_sdf_mode(const sif_sdf_t* file) {
  return file ? file->mode : SIF_SDF_READ;
}

double sif_sdf_box_length(const sif_sdf_t* file) {
  return file ? file->header.box_length : 0.0;
}

const char* sif_sdf_path(const sif_sdf_t* file) {
  return file ? file->path : NULL;
}

uint64_t sif_sdf_created(const sif_sdf_t* file) {
  return file ? file->header.created : 0;
}

const char* sif_sdf_writer(const sif_sdf_t* file) {
  return file ? file->writer : NULL;
}

void sif_sdf_info(
  const sif_sdf_t* file, uint32_t index, sif_sdf_block_info_t* out) {
  if (!out)
    return;

  memset(out, 0, sizeof(sif_sdf_block_info_t));
  if (!file || index >= file->n_blocks)
    return;

  const sif_sdf_block_header_t* header = &file->blocks[index].header;
  out->type = (sif_sdf_block_type_t)header->type;
  out->n_items = header->n_items;

  /* The field on disk is padded rather than terminated, so the copy has to
   * carry its own terminator. */
  memcpy(out->name, header->name, SIF__SDF_NAME_BYTES);
  out->name[SIF__SDF_NAME_BYTES] = '\0';
}

int sif_sdf_contains(
  const sif_sdf_t* file, sif_sdf_block_type_t type, const char* name) {
  return file && sif__sdf_find(file, type, name) >= 0;
}

uint64_t sif_sdf_catalog_id(const sif_sdf_t* file) {
  return file ? file->catalog_id : 0;
}

uint64_t sif_sdf_n_voids(const sif_sdf_t* file) {
  return file ? file->n_voids : 0;
}

uint32_t sif_sdf_n_blocks(const sif_sdf_t* file) {
  return file ? file->n_blocks : 0;
}
