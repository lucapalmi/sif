/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/**
 * @file sdf_internal.h
 * @brief The on-disk `.sdf` header and the handle behind #sif_sdf_t. Private
 * to the library.
 *
 * Private for the reason `internal.h` is: this is where the stream, the file
 * descriptor and the raw header layout live, and keeping them out of
 * `include/` is what lets the public header stay free of `<stdio.h>` and the
 * POSIX surface. It is also what lets the handle grow -- a block table, a
 * cached file size -- without the public API moving.
 */

#ifndef SIF__IO_SDF_INTERNAL_H
#define SIF__IO_SDF_INTERNAL_H

#include <stdint.h>
#include <stdio.h>

#include "sif/io/sdf.h"

/** @brief Size of the on-disk file header, and the offset the first block
 * starts at. */
#define SIF__SDF_HEADER_BYTES 64

/** @brief Bytes every block is padded up to, so a block header always lands on
 * a cache line. */
#define SIF__SDF_BLOCK_ALIGN 64

/** @brief Size of a block header. */
#define SIF__SDF_BLOCK_BYTES 64

/** @brief Magic number at the start of every block header. */
#define SIF__SDF_BLOCK_MAGIC "SBLK"

/** @brief Payload layout version this build writes for every block type. */
#define SIF__SDF_TYPE_VERSION 1

/** @brief Length of the `writer` field, which is padded rather than
 * terminated. */
#define SIF__SDF_WRITER_BYTES 16

/** @brief Length of a block's optional tag, padded rather than terminated. */
#define SIF__SDF_NAME_BYTES 16

/**
 * @brief What sif_sdf_create() records in `writer`.
 *
 * The version comes from the build, which gets it from `project()`. The
 * fallback is for a translation unit compiled outside the library's own build
 * -- it says the build did not know rather than inventing a number.
 */
#ifndef SIF_VERSION_STRING
#  define SIF_VERSION_STRING "unknown"
#endif
#define SIF__SDF_WRITER "sif " SIF_VERSION_STRING

/**
 * @brief The 64-byte header at the start of an `.sdf` file.
 *
 * Fixed width, with explicit padding, so the layout does not move with the
 * compiler; `box_length` is a `double` however `sif_real` is configured, for
 * the same reason. Written and read as one block of bytes, so the field order
 * here is the field order on disk.
 */
typedef struct {
  /** #SIF_SDF_MAGIC. */
  char magic[4];
  /** #SIF_SDF_VERSION. */
  uint32_t version;
  /** #SIF_SDF_BYTE_ORDER, written natively. */
  uint32_t byte_order;
  /** Reserved, zero. */
  uint32_t flags;
  /** Simulation box size, shared by every product in the file. */
  double box_length;
  /** Creation time, in seconds since the epoch. */
  uint64_t created;
  /** Library version that created the file, zero-padded, not terminated. */
  char writer[SIF__SDF_WRITER_BYTES];
  /** Reserved, to hold the header at 64 bytes. */
  char padding[16];
} sif_sdf_header_t;

/* The header is written and read as 64 raw bytes, so a compiler that inserted
 * padding would silently produce files nothing else can read. C99 has no
 * _Static_assert, and a negative array bound is the portable way to fail at
 * compile time. */
typedef char sif__sdf_header_size_check
  [(sizeof(sif_sdf_header_t) == SIF__SDF_HEADER_BYTES) ? 1 : -1];

/**
 * @brief The 64-byte header in front of every block.
 *
 * Fixed width with explicit padding, like the file header, and written as one
 * block of bytes.
 */
typedef struct {
  /** #SIF__SDF_BLOCK_MAGIC. */
  char magic[4];
  /** What the block holds, as a #sif_sdf_block_type_t. */
  uint16_t type;
  /** Payload layout version for this type. */
  uint16_t type_version;
  /** Width of the payload's reals, as a #sif_sdf_dtype_t. */
  uint16_t real_dtype;
  /** Reserved, zero. */
  uint16_t reserved0;
  /** Reserved, zero. */
  uint32_t flags;
  /** Metadata table size, a multiple of 8. */
  uint32_t meta_bytes;
  /** CRC32 of the metadata table followed by the data section. */
  uint32_t crc32;
  /** Data section size. */
  uint64_t data_bytes;
  /** The block's primary count: voids, or bins. */
  uint64_t n_items;
  /** Identity of the catalogue this block belongs to. Zero only in a block
   * that is not derived from one. */
  uint64_t catalog_id;
  /** Optional tag telling blocks of one type apart, zero-padded and not
   * terminated. Empty when the block is unnamed. */
  char name[SIF__SDF_NAME_BYTES];
} sif_sdf_block_header_t;

typedef char sif__sdf_block_size_check
  [(sizeof(sif_sdf_block_header_t) == SIF__SDF_BLOCK_BYTES) ? 1 : -1];

/**
 * @brief One entry of the table an open file keeps: a block and where it is.
 *
 * Built by walking the chain once at open, so that finding a block afterwards
 * costs no reads at all.
 */
typedef struct {
  /** Offset of the block header in the file. */
  uint64_t offset;
  /** That header. */
  sif_sdf_block_header_t header;
} sif__sdf_entry_t;

/** @brief Where a block's metadata table begins. */
static inline uint64_t sif__sdf_meta_offset(const sif__sdf_entry_t* entry) {
  return entry->offset + SIF__SDF_BLOCK_BYTES;
}

/** @brief Where a block's data section begins. */
static inline uint64_t sif__sdf_data_offset(const sif__sdf_entry_t* entry) {
  return sif__sdf_meta_offset(entry) + entry->header.meta_bytes;
}

/**
 * @brief Bytes a whole block occupies, padding included.
 *
 * @warning Assumes the lengths have already been checked against the size of
 * the file, which is what keeps the sum from overflowing.
 */
static inline uint64_t sif__sdf_block_bytes(const sif__sdf_entry_t* entry) {
  const uint64_t payload = entry->header.meta_bytes + entry->header.data_bytes;
  return SIF__SDF_BLOCK_BYTES + ((payload + SIF__SDF_BLOCK_ALIGN - 1) &
                                  ~(uint64_t)(SIF__SDF_BLOCK_ALIGN - 1));
}

/**
 * @brief An open `.sdf` file.
 */
struct sif_sdf {
  /** The stream everything goes through. Never NULL on a live handle. */
  FILE* stream;
  /** #stream's descriptor, for the parallel block reads that come later.
   * Owned by #stream: closing that closes this. */
  int fd;
  /** What the handle is allowed to do. */
  sif_sdf_mode_t mode;
  /** The first error this handle suffered, or #SIF_SDF_OK.
   *
   * Not the caller's status -- that one is passed in and belongs to the
   * caller. This one is the handle's own memory of having gone wrong, so that
   * an append that failed halfway through a block can refuse every later
   * write instead of piling a second torn block on top of the first. Set by
   * the block writers; there is nothing yet that can set it here. */
  sif_sdf_status_t failure;
  /** The file header, already validated. */
  sif_sdf_header_t header;
  /** #sif_sdf_header_t::writer, terminated, so it can be handed out as a
   * string. */
  char writer[SIF__SDF_WRITER_BYTES + 1];
  /** Owned copy of the path, for messages. */
  char* path;

  /** Size of the file when it was opened, and after every append. */
  uint64_t file_size;
  /** Where the next block goes: the first byte past the last one. Equal to
   * #file_size unless the file ends in something that is not a block, which
   * the walk refuses to open. */
  uint64_t end_offset;

  /** The blocks, in file order. NULL only while a file has none. */
  sif__sdf_entry_t* blocks;
  /** Entries in use. */
  uint32_t n_blocks;
  /** Entries #blocks can hold before it must grow. */
  uint32_t block_capacity;

  /** Identity of the catalogue in block 0, or 0 while there is none. Lifted
   * out of the block table because every append is checked against it. */
  uint64_t catalog_id;
  /** Voids in block 0, for the same reason. */
  uint64_t n_voids;
};

/* --- the pieces the block writers share, implemented in sdf.c --- */

/**
 * @brief The gate every entry point passes through: a usable status, and no
 * error inherited from an earlier call.
 * @return Non-zero when the call should go ahead.
 */
int sif__sdf_status_accepts(const sif_sdf_status_t* status, const char* what);

/**
 * @brief Open a file and validate its header, without walking its blocks.
 *
 * What sif_sdf_open() is built on, and what the recovery pass needs: a file
 * whose chain is damaged cannot be opened the ordinary way, and repairing it
 * still starts from a valid header and a working stream.
 *
 * The handle comes back with an empty block table. Nothing that reads blocks
 * may be called on it until someone has filled that in.
 */
SIF_NODISCARD sif_sdf_t* sif__sdf_open_raw(
  const char* filepath, sif_sdf_mode_t mode, sif_sdf_status_t* status);

/**
 * @brief Read @p bytes at @p offset, looping over short reads.
 */
sif_sdf_status_t sif__sdf_read_at(
  sif_sdf_t* file, uint64_t offset, void* dest, size_t bytes);

/**
 * @brief As sif__sdf_read_at(), folding what was read into a running
 * checksum. For payload that is not made of reals and so needs no conversion.
 */
sif_sdf_status_t sif__sdf_read_raw(
  sif_sdf_t* file, uint64_t offset, void* dest, size_t bytes, uint32_t* crc);

/**
 * @brief Read @p count reals stored in @p dtype, converting to this build's
 * `sif_real` if they differ, and fold the bytes as stored into @p crc.
 */
sif_sdf_status_t sif__sdf_read_reals(sif_sdf_t* file, uint64_t offset,
  uint64_t count, sif_sdf_dtype_t dtype, sif_real* dest, uint32_t* crc);

/** @brief Write at the stream's current position. */
sif_sdf_status_t sif__sdf_write_now(
  sif_sdf_t* file, const void* src, size_t bytes);

/** @brief Write the zero padding that ends a block. */
sif_sdf_status_t sif__sdf_write_padding(sif_sdf_t* file, size_t bytes);

/** @brief Record a block in the open file's table. */
sif_sdf_status_t sif__sdf_blocks_push(
  sif_sdf_t* file, uint64_t offset, const sif_sdf_block_header_t* header);

/** @brief One array of a block's data section. */
typedef struct {
  const void* data;
  uint64_t bytes;
} sif__sdf_span_t;

/**
 * @brief Append one block: header, metadata, arrays, padding.
 *
 * Fills in @p header's `magic`, `meta_bytes`, `data_bytes` and `crc32` from
 * what it is given, so a caller sets only what describes its own product. On
 * success the block is in the file and in the table; on failure the file is
 * cut back to the length it had, so an append that could not finish leaves
 * nothing behind rather than a torn block at the end.
 *
 * @param header Everything but the four fields above.
 * @param meta Metadata table, or NULL for none.
 * @param spans The data section, in order.
 */
sif_sdf_status_t sif__sdf_block_write(sif_sdf_t* file,
  sif_sdf_block_header_t* header, const void* meta, uint32_t meta_bytes,
  const sif__sdf_span_t* spans, uint32_t n_spans);

/**
 * @brief Find a block by type and name.
 *
 * @param name The tag to match, or NULL/empty for the first block of the type.
 * @return Index into the file's table, or -1 if there is no such block.
 */
int32_t sif__sdf_find(
  const sif_sdf_t* file, sif_sdf_block_type_t type, const char* name);

/**
 * @brief Copy a caller's name into the fixed field, zero-padded.
 *
 * A name too long to fit is refused rather than cut down: two names that
 * differ past the sixteenth byte would otherwise land in the file as one, and
 * the duplicate check would be right to reject the second.
 */
sif_sdf_status_t sif__sdf_name_pack(
  const char* name, char out[SIF__SDF_NAME_BYTES], const char* what);

/**
 * @brief Everything that must hold before anything at all is written: the
 * handle is usable, open for appending, and the file has its catalogue.
 *
 * @param what Named in the message when it does not hold.
 */
sif_sdf_status_t sif__sdf_write_gate(sif_sdf_t* file, const char* what);

/**
 * @brief Everything that must hold before a product is appended.
 *
 * The file must be open for appending and not already broken, it must hold a
 * catalogue, the product must name that catalogue, and the name must be free.
 */
sif_sdf_status_t sif__sdf_append_gate(sif_sdf_t* file,
  sif_sdf_block_type_t type, const char* name, uint64_t source_id);

/**
 * @brief Everything that must hold before a block is read: it exists, and its
 * payload is the length its shape implies.
 *
 * @param expected_bytes What the caller works out from the block's shape.
 */
sif_sdf_status_t sif__sdf_read_gate(const sif_sdf_t* file,
  const sif__sdf_entry_t* entry, uint64_t expected_bytes);

/** @brief Width on disk of a block's reals. */
static inline uint64_t sif__sdf_dtype_bytes(sif_sdf_dtype_t dtype) {
  return (dtype == SIF_SDF_F64) ? 8u : 4u;
}

/** @brief The dtype this build writes. */
static inline sif_sdf_dtype_t sif__sdf_native_dtype(void) {
  return (sizeof(sif_real) == 8) ? SIF_SDF_F64 : SIF_SDF_F32;
}

/* --- the metadata table, implemented in sdf_meta.c --- */

/** @brief One key and its value. The value is owned, and a string is kept
 * terminated so it can be handed out as one. */
typedef struct {
  /** Owned, NUL-terminated. */
  char* key;
  /** A #sif_sdf_meta_type_t. */
  uint8_t type;
  /** Value length as it goes on the wire: 8 per element for a number, the
   * text alone for a string. */
  uint32_t bytes;
  /** Owned. A string has a terminator past #bytes that the wire does not. */
  void* value;
} sif__sdf_kv_t;

/**
 * @brief A table of metadata.
 *
 * A flat array scanned linearly. A table is a few tens of keys at the outside
 * -- it is a page of notes about a run, not an index -- so a hash would cost
 * more in code than it saves in comparisons.
 */
struct sif_sdf_meta {
  sif__sdf_kv_t* items;
  uint32_t count;
  uint32_t capacity;
  /** Non-zero once a put has failed. Every later put does nothing, and
   * sif_sdf_meta_write() refuses the table. */
  int failed;
};

/**
 * @brief Add a value, replacing whatever the key held, without the check that
 * keeps callers out of the `sif.` namespace.
 *
 * This is how the library writes the parameters that give a block its shape,
 * which live under exactly that prefix. The public sif_sdf_meta_put_i64() and
 * its siblings are this with the check in front.
 *
 * @param value Copied. @p bytes is the wire length: 8 per number, the text
 * alone for a string.
 */
void sif__sdf_meta_put(sif_sdf_meta_t* meta, const char* key,
  sif_sdf_meta_type_t type, const void* value, uint32_t bytes);

/** @brief sif__sdf_meta_put() for one integer. */
void sif__sdf_meta_put_i64(
  sif_sdf_meta_t* meta, const char* key, int64_t value);

/** @brief sif__sdf_meta_put() for one double. */
void sif__sdf_meta_put_f64(sif_sdf_meta_t* meta, const char* key, double value);

/**
 * @brief Encode a table into the bytes that go in a block.
 *
 * @param out Receives the buffer, owned by the caller and released with
 * free(). NULL for an empty table.
 * @param out_bytes Receives its length, a multiple of 8.
 */
sif_sdf_status_t sif__sdf_meta_encode(
  const sif_sdf_meta_t* meta, void** out, uint32_t* out_bytes);

/**
 * @brief Decode an encoded table into @p meta, which must be empty.
 *
 * Empty because that is what makes the duplicate check exact: a key already in
 * the table is a key this table declared twice, which is malformed. Merging
 * several blocks is done by decoding each on its own and folding the results
 * together, not by decoding them onto each other.
 */
sif_sdf_status_t sif__sdf_meta_decode(
  sif_sdf_meta_t* meta, const void* table, uint32_t bytes, const char* path);

/**
 * @brief Copy every key of @p src into @p dst, replacing what is there.
 *
 * The merge across metadata blocks: applied in file order, the last block to
 * carry a key is the one whose value survives.
 */
void sif__sdf_meta_merge(sif_sdf_meta_t* dst, const sif_sdf_meta_t* src);

/**
 * @brief Read one block's metadata into a fresh table, folding the bytes into
 * @p crc.
 *
 * The checksum covers the metadata and then the data, in that order, so a
 * reader takes the table first whether it needs the table or not.
 *
 * @param out Receives the table, owned by the caller and released with
 * sif_sdf_meta_free(). Empty when the block carries no metadata.
 */
sif_sdf_status_t sif__sdf_meta_read_block(sif_sdf_t* file,
  const sif__sdf_entry_t* entry, sif_sdf_meta_t** out, uint32_t* crc);

#endif /* SIF__IO_SDF_INTERNAL_H */
