/* Copyright (C) 2026 Luca Palmieri
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of sif. See COPYING for the full license text.
 */

/*
 * GADGET snapshots, in all three formats. The binary formats are read here;
 * HDF5 goes through gadget_internal.h, so this file compiles the same with or
 * without HDF5. The formats themselves are described in the GADGET-4 manual,
 * "Snapshot file format".
 *
 * Every file is opened behind one small interface -- open, read a range of
 * one type from one block as doubles, close -- and the driver at the bottom
 * does the rest once for all formats: finding the files, adding up the
 * counts, subsampling, converting units and writing into the field.
 */

#include "sif/io/gadget_io.h"

#include "io/gadget_internal.h"
#include "sif/utils/logger.h"
#include "sif/utils/random.h"

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define TAG "gadget"

/* Particles per read. Large enough that the per-read overhead vanishes, small
 * enough that the buffers (about 14 MB at this size) never matter next to the
 * field they fill. */
#define CHUNK ((uint64_t)1 << 18)

#define PATH_CAP 4096

/* One megaparsec in cm, the unit UnitLength_in_cm is converted against. */
#define MPC_IN_CM 3.085678e24

/* A header record longer than this is not a GADGET header; the largest real
 * one is the legacy 256 bytes. */
#define HEAD_CAP 1024

/* ------------------------------------------------------------------------ */
/* argument checking                                                         */
/* ------------------------------------------------------------------------ */

/* Which argument a value was meant for, from the 0x100 block it sits in. */
static const char* slot_of(int value) {
  switch (value & ~0xff) {
  case SIF_GADGET_FORMAT_AUTO:
    return "format";
  case SIF_GADGET_PTYPE_0:
    return "particle type";
  case SIF_GADGET_VELOCITY_SKIP:
    return "velocity";
  case SIF_GADGET_MASS_SKIP:
    return "mass";
  case SIF_GADGET_LENGTH_KPC:
    return "length";
  default:
    return NULL;
  }
}

static bool slot_ok(int value, int first, int n_values, const char* slot) {
  if (value >= first && value < first + n_values)
    return true;

  const char* meant = slot_of(value);
  if (meant && strcmp(meant, slot) != 0)
    SIF_LOG_ERROR(TAG,
      "the %s argument was given a %s option (0x%x); are two arguments "
      "swapped?",
      slot, meant, value);
  else
    SIF_LOG_ERROR(TAG, "0x%x is not a valid %s option", value, slot);
  return false;
}

static const char* format_name(sif_gadget_format_t format) {
  switch (format) {
  case SIF_GADGET_FORMAT_1:
    return "SnapFormat 1";
  case SIF_GADGET_FORMAT_2:
    return "SnapFormat 2";
  case SIF_GADGET_FORMAT_HDF5:
    return "HDF5";
  default:
    return "unknown";
  }
}

/* ------------------------------------------------------------------------ */
/* byte order                                                                */
/* ------------------------------------------------------------------------ */

static uint32_t bswap32(uint32_t v) {
  return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);
}

static uint64_t bswap64(uint64_t v) {
  return ((uint64_t)bswap32((uint32_t)v) << 32) | bswap32((uint32_t)(v >> 32));
}

static uint32_t get_u32(const unsigned char* p, bool swapped) {
  uint32_t v;
  memcpy(&v, p, 4);
  return swapped ? bswap32(v) : v;
}

static uint64_t get_u64(const unsigned char* p, bool swapped) {
  uint64_t v;
  memcpy(&v, p, 8);
  return swapped ? bswap64(v) : v;
}

static double get_f64(const unsigned char* p, bool swapped) {
  const uint64_t u = get_u64(p, swapped);
  double d;
  memcpy(&d, &u, 8);
  return d;
}

/* ------------------------------------------------------------------------ */
/* the binary formats                                                        */
/* ------------------------------------------------------------------------ */

/* A record's payload: where it starts and how long it is. */
typedef struct {
  off_t offset;
  uint64_t bytes;
  bool present;
} block_t;

typedef struct {
  FILE* f;
  bool swapped;
  block_t head, pos, vel, mass;
  /* Bytes per component in each block, 4 or 8, fixed at open. */
  uint32_t pos_elem, vel_elem, mass_elem;
} bin_file_t;

/*
 * The Fortran framing: a 4-byte length, the payload, the same length again.
 * Reads one record's framing from the current position and leaves the stream
 * after it. Returns 1 for a record, 0 at a clean end of file, -1 for a broken
 * one -- a length that runs off the end or trailing marker that disagrees.
 *
 * The length is 32 bits, which is why GADGET never writes a block past 4 GB:
 * the framing could not say how long it was.
 */
static int record_next(FILE* f, bool swapped, block_t* out) {
  unsigned char m[4];
  const size_t got = fread(m, 1, 4, f);
  if (got == 0 && feof(f))
    return 0;
  if (got != 4)
    return -1;

  const uint32_t len = get_u32(m, swapped);
  out->offset = ftello(f);
  out->bytes = len;
  out->present = true;

  if (fseeko(f, (off_t)len, SEEK_CUR) != 0 || fread(m, 1, 4, f) != 4)
    return -1;
  return get_u32(m, swapped) == len ? 1 : -1;
}

/* Everything in a header except the counts is a double or an int that has a
 * narrow range of sensible values. A layout guess that lands on the wrong
 * bytes almost never passes all of these. */
static bool header_sane(const sif_gadget_header_t* h) {
  if (h->n_files < 1 || h->n_files > (1u << 20))
    return false;
  if (!isfinite(h->time) || h->time < 0 || !isfinite(h->box_size) ||
      h->box_size < 0 || !isfinite(h->redshift))
    return false;

  for (uint32_t t = 0; t < h->n_types; t++) {
    if (!isfinite(h->mass_table[t]) || h->mass_table[t] < 0)
      return false;
    if (h->n_part_file[t] > h->n_part_total[t] ||
        h->n_part_total[t] > ((uint64_t)1 << 48))
      return false;
  }
  return true;
}

static uint64_t n_part_sum(const sif_gadget_header_t* h) {
  uint64_t n = 0;
  for (uint32_t t = 0; t < h->n_types; t++)
    n += h->n_part_file[t];
  return n;
}

/* Particles in the file that carry their own mass: the length of the MASS
 * block. */
static uint64_t n_mass_sum(const sif_gadget_header_t* h) {
  uint64_t n = 0;
  for (uint32_t t = 0; t < h->n_types; t++)
    if (h->mass_table[t] == 0)
      n += h->n_part_file[t];
  return n;
}

/* The positions block has to be 3 * N components of 4 or 8 bytes. With no
 * particles in the file there is nothing to check against. */
static bool pos_block_fits(const sif_gadget_header_t* h, const block_t* pos) {
  const uint64_t n = n_part_sum(h);
  if (n == 0)
    return !pos->present || pos->bytes == 0;
  return pos->present && (pos->bytes == 12 * n || pos->bytes == 24 * n);
}

/*
 * The GADGET-2/3 header: always 256 bytes, six types, 32-bit counts with the
 * high words of the totals in a separate array.
 */
static void parse_legacy(
  const unsigned char* p, bool sw, sif_gadget_header_t* h) {
  h->is_legacy_header = 1;
  h->n_types = 6;
  for (int t = 0; t < 6; t++) {
    h->n_part_file[t] = get_u32(p + 4 * t, sw);
    h->mass_table[t] = get_f64(p + 24 + 8 * t, sw);
    h->n_part_total[t] = (uint64_t)get_u32(p + 96 + 4 * t, sw) |
                         ((uint64_t)get_u32(p + 168 + 4 * t, sw) << 32);
  }
  h->time = get_f64(p + 72, sw);
  h->redshift = get_f64(p + 80, sw);
  h->n_files = get_u32(p + 124, sw);
  h->box_size = get_f64(p + 128, sw);
  h->omega0 = get_f64(p + 136, sw);
  h->omega_lambda = get_f64(p + 144, sw);
  h->hubble_param = get_f64(p + 152, sw);
  h->has_cosmology = 1;
}

/*
 * The GADGET-4 header, as the manual gives it: per-type counts in this file,
 * 64-bit totals, the mass table, then time, redshift, box size and the file
 * count -- with the struct padding a C compiler puts in when it is written
 * with one fwrite(). The width of the per-file counts and the number of types
 * are not recorded, so the caller tries each and keeps the one that fits.
 *
 * Returns false if a header of this shape cannot be @p size bytes long.
 */
static bool parse_g4(const unsigned char* p, uint64_t size, bool sw,
  uint32_t width, uint32_t n_types, sif_gadget_header_t* h) {

  const uint64_t total_off = ((uint64_t)width * n_types + 7) & ~(uint64_t)7;
  const uint64_t mass_off = total_off + 8 * n_types;
  const uint64_t time_off = mass_off + 8 * n_types;
  const uint64_t prefix = time_off + 28;

  /* What may follow the fields read here: padding to 8 bytes and, in some
   * configurations, two more 64-bit tree counts. */
  if (prefix > size || size - prefix > 24)
    return false;

  h->is_legacy_header = 0;
  h->has_cosmology = 0;
  h->n_types = n_types;
  for (uint32_t t = 0; t < n_types; t++) {
    h->n_part_file[t] =
      width == 8 ? get_u64(p + 8 * t, sw) : get_u32(p + 4 * t, sw);
    h->n_part_total[t] = get_u64(p + total_off + 8 * t, sw);
    h->mass_table[t] = get_f64(p + mass_off + 8 * t, sw);
  }
  h->time = get_f64(p + time_off, sw);
  h->redshift = get_f64(p + time_off + 8, sw);
  h->box_size = get_f64(p + time_off + 16, sw);
  h->n_files = get_u32(p + time_off + 24, sw);
  return true;
}

static int parse_header(const unsigned char* p, uint64_t size, bool sw,
  const block_t* pos, sif_gadget_header_t* h, const char* path) {

  if (size == 256) {
    parse_legacy(p, sw, h);
    if (header_sane(h) && pos_block_fits(h, pos))
      return SIF_OK;
    SIF_LOG_ERROR(TAG,
      "%s has a 256-byte header, but it does not read as a GADGET-2 header "
      "that matches the positions block",
      path);
    return SIF_ERR_IO;
  }

  /* Most types first within each width, so that when two type counts fit
   * the size the one leaving less unexplained padding wins. */
  const uint32_t widths[2] = {8, 4};
  for (int w = 0; w < 2; w++) {
    for (uint32_t n = SIF_GADGET_MAX_TYPES; n >= 1; n--) {
      sif_gadget_header_t trial = *h;
      if (!parse_g4(p, size, sw, widths[w], n, &trial))
        continue;
      if (header_sane(&trial) && pos_block_fits(&trial, pos)) {
        *h = trial;
        return SIF_OK;
      }
    }
  }

  SIF_LOG_ERROR(TAG,
    "%s: the %" PRIu64 "-byte header matches no GADGET header layout", path,
    size);
  return SIF_ERR_IO;
}

/* Bytes per component of a block of @p n particles with @p comps components,
 * or 0 if the size fits neither precision. */
static uint32_t elem_size(const block_t* b, uint64_t n, uint32_t comps) {
  if (n == 0)
    return 0;
  if (b->bytes == 4 * comps * n)
    return 4;
  if (b->bytes == 8 * comps * n)
    return 8;
  return 0;
}

static void blocks_add(sif_gadget_header_t* h, const char* name) {
  if (h->n_blocks < SIF_GADGET_MAX_BLOCKS)
    snprintf(h->blocks[h->n_blocks], sizeof(h->blocks[0]), "%s", name);
  h->n_blocks++;
}

/*
 * Walks every record, noting where the header, positions, velocities and
 * masses are. SnapFormat 2 says which is which with a label record before
 * each block. SnapFormat 1 does not, and relies on the fixed order: header,
 * positions, velocities, IDs, then masses if any particle carries its own.
 */
static int bin_open(const char* path, sif_gadget_format_t format, bool sw,
  bin_file_t* bf, sif_gadget_header_t* h) {

  memset(bf, 0, sizeof(*bf));
  bf->swapped = sw;
  bf->f = fopen(path, "rb");
  if (!bf->f) {
    SIF_LOG_ERROR(TAG, "could not open %s", path);
    return SIF_ERR_IO;
  }

  block_t rec;
  block_t f1_records[5] = {{0}};
  uint32_t n_records = 0;
  int r;

  if (format == SIF_GADGET_FORMAT_1) {
    while ((r = record_next(bf->f, sw, &rec)) == 1) {
      if (n_records < 5)
        f1_records[n_records] = rec;
      n_records++;
    }
    bf->head = f1_records[0];
    bf->pos = f1_records[1];
    h->n_blocks = n_records;
  } else {
    while ((r = record_next(bf->f, sw, &rec)) == 1) {
      unsigned char label[8];
      if (rec.bytes != 8 || fseeko(bf->f, rec.offset, SEEK_SET) != 0 ||
          fread(label, 1, 8, bf->f) != 8 || fseeko(bf->f, 4, SEEK_CUR) != 0) {
        r = -1;
        break;
      }

      block_t data;
      if (record_next(bf->f, sw, &data) != 1) {
        r = -1;
        break;
      }

      /* Four characters, padded with spaces. */
      char name[5];
      memcpy(name, label, 4);
      name[4] = '\0';
      for (int i = 3; i >= 0 && name[i] == ' '; i--)
        name[i] = '\0';
      blocks_add(h, name);

      if (strcmp(name, "HEAD") == 0)
        bf->head = data;
      else if (strcmp(name, "POS") == 0)
        bf->pos = data;
      else if (strcmp(name, "VEL") == 0)
        bf->vel = data;
      else if (strcmp(name, "MASS") == 0)
        bf->mass = data;
    }
  }

  if (r < 0) {
    SIF_LOG_ERROR(TAG,
      "%s is truncated, or its record markers do not match: not a %s file",
      path, format_name(format));
    return SIF_ERR_IO;
  }

  if (!bf->head.present || bf->head.bytes > HEAD_CAP || bf->head.bytes < 64) {
    SIF_LOG_ERROR(TAG, "%s has no GADGET header", path);
    return SIF_ERR_IO;
  }

  unsigned char head[HEAD_CAP];
  if (fseeko(bf->f, bf->head.offset, SEEK_SET) != 0 ||
      fread(head, 1, bf->head.bytes, bf->f) != bf->head.bytes) {
    SIF_LOG_ERROR(TAG, "failed to read the header of %s", path);
    return SIF_ERR_IO;
  }

  int status = parse_header(head, bf->head.bytes, sw, &bf->pos, h, path);
  if (status != SIF_OK)
    return status;

  const uint64_t n = n_part_sum(h);
  const uint64_t n_mass = n_mass_sum(h);

  if (format == SIF_GADGET_FORMAT_1) {
    if (n_records > 2)
      bf->vel = f1_records[2];
    /* Record 4 is the mass block only if some particle carries a mass;
     * otherwise it is whatever comes next, internal energy for gas. */
    if (n_records > 4 && n_mass > 0)
      bf->mass = f1_records[4];
  }

  bf->pos_elem = elem_size(&bf->pos, n, 3);
  h->precision = bf->pos_elem;

  if (bf->vel.present) {
    bf->vel_elem = elem_size(&bf->vel, n, 3);
    if (n > 0 && bf->vel_elem == 0) {
      SIF_LOG_ERROR(TAG,
        "%s: velocity block of %" PRIu64 " bytes does not fit %" PRIu64
        " particles",
        path, bf->vel.bytes, n);
      return SIF_ERR_IO;
    }
  }

  if (bf->mass.present) {
    bf->mass_elem = elem_size(&bf->mass, n_mass, 1);
    if (n_mass > 0 && bf->mass_elem == 0) {
      SIF_LOG_ERROR(TAG,
        "%s: mass block of %" PRIu64 " bytes does not fit %" PRIu64
        " particles with individual masses",
        path, bf->mass.bytes, n_mass);
      return SIF_ERR_IO;
    }
  }

  h->format = format;
  h->is_swapped = sw ? 1 : 0;
  return SIF_OK;
}

static int bin_read(bin_file_t* bf, const sif_gadget_header_t* h,
  uint32_t ptype, sif_gadget_block_t block, uint64_t start, uint64_t count,
  double* out, unsigned char* scratch) {

  const block_t* b;
  uint32_t elem, comps;
  uint64_t type_off = 0;

  /* Particles are ordered by type within every block; the mass block skips
   * the types whose mass is in the table. */
  if (block == SIF__GADGET_BLOCK_MASS) {
    b = &bf->mass;
    elem = bf->mass_elem;
    comps = 1;
    for (uint32_t t = 0; t < ptype; t++)
      if (h->mass_table[t] == 0)
        type_off += h->n_part_file[t];
  } else {
    b = block == SIF__GADGET_BLOCK_POS ? &bf->pos : &bf->vel;
    elem = block == SIF__GADGET_BLOCK_POS ? bf->pos_elem : bf->vel_elem;
    comps = 3;
    for (uint32_t t = 0; t < ptype; t++)
      type_off += h->n_part_file[t];
  }

  if (!b->present || elem == 0)
    return SIF_ERR_IO;

  const uint64_t n_vals = count * comps;
  const off_t at = b->offset + (off_t)((type_off + start) * comps * elem);
  if (fseeko(bf->f, at, SEEK_SET) != 0 ||
      fread(scratch, elem, n_vals, bf->f) != n_vals)
    return SIF_ERR_IO;

  if (elem == 4) {
    for (uint64_t i = 0; i < n_vals; i++) {
      const uint32_t u = get_u32(scratch + 4 * i, bf->swapped);
      float v;
      memcpy(&v, &u, 4);
      out[i] = v;
    }
  } else {
    for (uint64_t i = 0; i < n_vals; i++)
      out[i] = get_f64(scratch + 8 * i, bf->swapped);
  }
  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* one file, any format                                                      */
/* ------------------------------------------------------------------------ */

typedef struct {
  bool is_hdf5;
  bin_file_t bin;
  sif_gadget_h5_file_t* h5;
  sif_gadget_header_t header;
} snap_file_t;

/* Reads just enough of the file to say which format it is: the HDF5
 * signature, a SnapFormat 2 label, or a plausible header length. The byte
 * order comes out of the same look, since a length read the wrong way round
 * is not a plausible one. */
static int detect(const char* path, sif_gadget_format_t* format, bool* sw) {
  unsigned char buf[8];
  FILE* f = fopen(path, "rb");
  if (!f) {
    SIF_LOG_ERROR(TAG, "could not open %s", path);
    return SIF_ERR_IO;
  }
  const size_t got = fread(buf, 1, 8, f);
  fclose(f);

  static const unsigned char h5sig[8] = {
    0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
  if (got == 8 && memcmp(buf, h5sig, 8) == 0) {
    *format = SIF_GADGET_FORMAT_HDF5;
    *sw = false;
    return SIF_OK;
  }

  if (got == 8) {
    for (int s = 0; s < 2; s++) {
      const uint32_t m = get_u32(buf, s == 1);
      if (m == 8 && memcmp(buf + 4, "HEAD", 4) == 0) {
        *format = SIF_GADGET_FORMAT_2;
        *sw = s == 1;
        return SIF_OK;
      }
      if (m >= 64 && m <= HEAD_CAP) {
        *format = SIF_GADGET_FORMAT_1;
        *sw = s == 1;
        return SIF_OK;
      }
    }
  }

  SIF_LOG_ERROR(TAG, "%s is not a GADGET snapshot in any known format", path);
  return SIF_ERR_IO;
}

static void snap_close(snap_file_t* sf) {
  if (sf->is_hdf5)
    sif__gadget_h5_close(sf->h5);
  else if (sf->bin.f)
    fclose(sf->bin.f);
  sf->h5 = NULL;
  sf->bin.f = NULL;
}

static int snap_open(
  const char* path, sif_gadget_format_t want, snap_file_t* sf) {
  memset(sf, 0, sizeof(*sf));

  sif_gadget_format_t format;
  bool sw;
  int status = detect(path, &format, &sw);
  if (status != SIF_OK)
    return status;

  if (want != SIF_GADGET_FORMAT_AUTO && want != format) {
    SIF_LOG_ERROR(
      TAG, "%s is %s, not %s", path, format_name(format), format_name(want));
    return SIF_ERR_IO;
  }

  if (format == SIF_GADGET_FORMAT_HDF5) {
    sf->is_hdf5 = true;
    status = sif__gadget_h5_open(path, &sf->h5, &sf->header);
  } else {
    status = bin_open(path, format, sw, &sf->bin, &sf->header);
  }

  if (status != SIF_OK)
    snap_close(sf);
  return status;
}

static int snap_read(snap_file_t* sf, uint32_t ptype, sif_gadget_block_t block,
  uint64_t start, uint64_t count, double* out, unsigned char* scratch) {
  if (sf->is_hdf5)
    return sif__gadget_h5_read(sf->h5, ptype, block, start, count, out);
  return bin_read(
    &sf->bin, &sf->header, ptype, block, start, count, out, scratch);
}

/* ------------------------------------------------------------------------ */
/* finding the files                                                         */
/* ------------------------------------------------------------------------ */

typedef struct {
  /* The file to open first. */
  char first[PATH_CAP];
  /* For a numbered snapshot, file i is <base>.<i><suffix>. */
  bool numbered;
  char base[PATH_CAP];
  char suffix[8];
} snap_path_t;

static bool is_file(const char* p) {
  struct stat st;
  return stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

/* Whether @p p is <base>.<digits>, or the same followed by .hdf5. */
static bool split_numbered(const char* p, snap_path_t* sp) {
  size_t len = strlen(p);
  sp->suffix[0] = '\0';
  if (len > 5 && strcmp(p + len - 5, ".hdf5") == 0) {
    len -= 5;
    strcpy(sp->suffix, ".hdf5");
  }

  size_t d = len;
  while (d > 0 && isdigit((unsigned char)p[d - 1]))
    d--;
  if (d == len || d < 2 || p[d - 1] != '.')
    return false;

  memcpy(sp->base, p, d - 1);
  sp->base[d - 1] = '\0';
  return true;
}

static bool try_candidate(snap_path_t* sp, const char* base, const char* tail,
  bool numbered, const char* suffix) {
  const int n = snprintf(sp->first, PATH_CAP, "%s%s", base, tail);
  if (n <= 0 || n >= PATH_CAP || !is_file(sp->first))
    return false;

  sp->numbered = numbered;
  snprintf(sp->base, PATH_CAP, "%s", base);
  snprintf(sp->suffix, sizeof(sp->suffix), "%s", suffix);
  return true;
}

/*
 * A file that exists is taken as named, and if its name is numbered the rest
 * of the snapshot is found beside it. A name that does not exist is a base
 * name: `snap_010` finds `snap_010.hdf5`, `snap_010.0`, `snap_010.0.hdf5`,
 * and the same two inside `snapdir_010/`, which is where GADGET puts a
 * multi-file snapshot.
 */
static int path_resolve(const char* path, snap_path_t* sp) {
  memset(sp, 0, sizeof(*sp));

  if (strlen(path) >= PATH_CAP - 32) {
    SIF_LOG_ERROR(TAG, "path too long: %s", path);
    return SIF_ERR_INVALID;
  }

  if (is_file(path)) {
    snprintf(sp->first, PATH_CAP, "%s", path);
    sp->numbered = split_numbered(path, sp);
    return SIF_OK;
  }

  if (try_candidate(sp, path, ".hdf5", false, "") ||
      try_candidate(sp, path, ".0", true, "") ||
      try_candidate(sp, path, ".0.hdf5", true, ".hdf5"))
    return SIF_OK;

  /* snapdir_XXX, where XXX is what follows the last underscore of the name. */
  const char* slash = strrchr(path, '/');
  const char* name = slash ? slash + 1 : path;
  const char* us = strrchr(name, '_');
  if (us && us[1] != '\0') {
    bool digits = true;
    for (const char* c = us + 1; *c; c++)
      digits = digits && isdigit((unsigned char)*c);

    if (digits) {
      char dir_base[PATH_CAP];
      const int dir_len = slash ? (int)(slash - path + 1) : 0;
      snprintf(
        dir_base, PATH_CAP, "%.*ssnapdir_%s/%s", dir_len, path, us + 1, name);
      if (try_candidate(sp, dir_base, ".0", true, "") ||
          try_candidate(sp, dir_base, ".0.hdf5", true, ".hdf5"))
        return SIF_OK;
    }
  }

  SIF_LOG_ERROR(TAG,
    "no snapshot at %s: not a file, and no %s.0, %s.0.hdf5 or snapdir beside "
    "it",
    path, path, path);
  return SIF_ERR_IO;
}

/* The list of files, now that the first header has said how many there are. */
static int path_file(
  const snap_path_t* sp, uint32_t n_files, uint32_t i, char* buf) {
  if (n_files == 1) {
    snprintf(buf, PATH_CAP, "%s", sp->first);
    return SIF_OK;
  }

  if (!sp->numbered) {
    SIF_LOG_ERROR(TAG,
      "%s is one of %u files, but its name is not <base>.<n>, so the others "
      "cannot be found",
      sp->first, n_files);
    return SIF_ERR_IO;
  }

  const int n = snprintf(buf, PATH_CAP, "%s.%u%s", sp->base, i, sp->suffix);
  return (n > 0 && n < PATH_CAP) ? SIF_OK : SIF_ERR_INVALID;
}

/* ------------------------------------------------------------------------ */
/* header                                                                    */
/* ------------------------------------------------------------------------ */

int sif_gadget_read_header(
  const char* path, sif_gadget_format_t format, sif_gadget_header_t* out) {
  if (!path || !out)
    return SIF_ERR_INVALID;
  memset(out, 0, sizeof(*out));

  if (!slot_ok((int)format, SIF_GADGET_FORMAT_AUTO, 4, "format"))
    return SIF_ERR_INVALID;

  snap_path_t sp;
  int status = path_resolve(path, &sp);
  if (status != SIF_OK)
    return status;

  snap_file_t sf;
  status = snap_open(sp.first, format, &sf);
  if (status != SIF_OK)
    return status;

  *out = sf.header;
  snap_close(&sf);
  return SIF_OK;
}

void sif_gadget_print_header(const sif_gadget_header_t* h, FILE* stream) {
  if (!h || !stream)
    return;

  fprintf(stream, "format        %s", format_name(h->format));
  if (h->format != SIF_GADGET_FORMAT_HDF5)
    fprintf(stream, ", %s header, %s byte order",
      h->is_legacy_header ? "GADGET-2/3" : "GADGET-4",
      h->is_swapped ? "swapped" : "native");
  fprintf(stream, "\n");

  if (h->precision)
    fprintf(
      stream, "precision     %s\n", h->precision == 8 ? "double" : "single");
  fprintf(stream, "files         %u\n", h->n_files);
  fprintf(stream, "time          %.8g\n", h->time);
  fprintf(stream, "redshift      %.8g\n", h->redshift);
  fprintf(stream, "box size      %.8g", h->box_size);
  if (h->unit_length_in_cm > 0)
    fprintf(stream, "  (unit %.6g cm)", h->unit_length_in_cm);
  fprintf(stream, "\n");

  if (h->has_cosmology)
    fprintf(stream, "cosmology     Omega0 %.6g, OmegaLambda %.6g, h %.6g\n",
      h->omega0, h->omega_lambda, h->hubble_param);

  fprintf(stream, "type  %14s  %14s  %14s\n", "in file", "total", "mass");
  for (uint32_t t = 0; t < h->n_types; t++) {
    fprintf(stream, "%4u  %14" PRIu64 "  %14" PRIu64 "  ", t, h->n_part_file[t],
      h->n_part_total[t]);
    if (h->mass_table[t] > 0)
      fprintf(stream, "%14.8g\n", h->mass_table[t]);
    else
      fprintf(stream, "%14s\n", h->n_part_total[t] ? "per particle" : "-");
  }

  if (h->n_blocks > 0 && h->blocks[0][0] != '\0') {
    fprintf(stream, "blocks       ");
    const uint32_t shown =
      h->n_blocks < SIF_GADGET_MAX_BLOCKS ? h->n_blocks : SIF_GADGET_MAX_BLOCKS;
    for (uint32_t b = 0; b < shown; b++)
      fprintf(stream, " %s", h->blocks[b]);
    if (h->n_blocks > shown)
      fprintf(stream, " (+%u more)", h->n_blocks - shown);
    fprintf(stream, "\n");
  } else if (h->n_blocks > 0) {
    fprintf(stream, "blocks        %u, unlabelled\n", h->n_blocks);
  }
}

int sif_gadget_inspect(const char* path) {
  sif_gadget_header_t h;
  const int status = sif_gadget_read_header(path, SIF_GADGET_FORMAT_AUTO, &h);
  if (status != SIF_OK)
    return status;

  printf("%s\n", path);
  sif_gadget_print_header(&h, stdout);
  fflush(stdout);
  return SIF_OK;
}

/* ------------------------------------------------------------------------ */
/* the reader                                                                */
/* ------------------------------------------------------------------------ */

/* What is carried from file to file while the snapshot is streamed. */
typedef struct {
  sif_field_t* field;
  uint32_t ptype;
  bool read_vel, read_mass, subsample;
  double pos_scale, vel_scale;

  sif_prng_state_t prng;
  uint64_t n_total, n_keep, seen, kept;

  double *pos, *vel, *mass;
  unsigned char* scratch;
} stream_t;

static int stream_file(stream_t* s, snap_file_t* sf, const char* path) {
  const uint64_t n = sf->header.n_part_file[s->ptype];

  for (uint64_t start = 0; start < n; start += CHUNK) {
    const uint64_t c = (n - start < CHUNK) ? n - start : CHUNK;

    int status = snap_read(
      sf, s->ptype, SIF__GADGET_BLOCK_POS, start, c, s->pos, s->scratch);
    if (status == SIF_OK && s->read_vel)
      status = snap_read(
        sf, s->ptype, SIF__GADGET_BLOCK_VEL, start, c, s->vel, s->scratch);
    if (status == SIF_OK && s->read_mass)
      status = snap_read(
        sf, s->ptype, SIF__GADGET_BLOCK_MASS, start, c, s->mass, s->scratch);
    if (status != SIF_OK) {
      SIF_LOG_ERROR(TAG,
        "failed to read particles %" PRIu64 "-%" PRIu64 " of type %u from %s",
        start, start + c, s->ptype, path);
      return SIF_ERR_IO;
    }

    sif_field_t* fl = s->field;
    for (uint64_t j = 0; j < c; j++) {
      /* Algorithm S: keep this one with probability (still wanted) / (still
       * to see), which ends with exactly n_keep and makes every subset of
       * that size equally likely. */
      if (s->subsample) {
        const double u = sif_prng_next_double(&s->prng);
        const bool keep =
          (double)(s->n_total - s->seen) * u < (double)(s->n_keep - s->kept);
        s->seen++;
        if (!keep)
          continue;
      }

      const uint64_t k = s->kept++;
      fl->x[k] = (sif_real)(s->pos[3 * j] * s->pos_scale);
      fl->y[k] = (sif_real)(s->pos[3 * j + 1] * s->pos_scale);
      fl->z[k] = (sif_real)(s->pos[3 * j + 2] * s->pos_scale);
      if (s->read_vel) {
        fl->vx[k] = (sif_real)(s->vel[3 * j] * s->vel_scale);
        fl->vy[k] = (sif_real)(s->vel[3 * j + 1] * s->vel_scale);
        fl->vz[k] = (sif_real)(s->vel[3 * j + 2] * s->vel_scale);
      }
      if (s->read_mass)
        fl->weights[k] = (sif_real)s->mass[j];
    }
  }
  return SIF_OK;
}

int sif_field_read_gadget(const char* path, sif_gadget_format_t format,
  sif_gadget_ptype_t ptype, sif_gadget_velocity_t velocity,
  sif_gadget_mass_t mass, sif_gadget_length_t length, double fraction,
  uint64_t seed, sif_field_t** out_field, double* out_box_length) {

  if (out_field)
    *out_field = NULL;
  if (!path || !out_field) {
    SIF_LOG_ERROR(TAG, "path and out_field are required");
    return SIF_ERR_INVALID;
  }

  if (!slot_ok((int)format, SIF_GADGET_FORMAT_AUTO, 4, "format") ||
      !slot_ok((int)ptype, SIF_GADGET_PTYPE_0, 6, "particle type") ||
      !slot_ok((int)velocity, SIF_GADGET_VELOCITY_SKIP, 3, "velocity") ||
      !slot_ok((int)mass, SIF_GADGET_MASS_SKIP, 2, "mass") ||
      !slot_ok((int)length, SIF_GADGET_LENGTH_KPC, 3, "length"))
    return SIF_ERR_INVALID;

  if (!(fraction > 0 && fraction <= 1)) {
    SIF_LOG_ERROR(TAG, "fraction must be in (0, 1], not %g", fraction);
    return SIF_ERR_INVALID;
  }

  const uint32_t t = (uint32_t)(ptype - SIF_GADGET_PTYPE_0);

  snap_path_t sp;
  int status = path_resolve(path, &sp);
  if (status != SIF_OK)
    return status;

  /* --- pass 1: every header, to size the field before reading anything --- */

  snap_file_t sf;
  status = snap_open(sp.first, format, &sf);
  if (status != SIF_OK)
    return status;
  const sif_gadget_header_t h0 = sf.header;
  snap_close(&sf);

  /* Every file must be the same format as the first, whatever was asked. */
  const sif_gadget_format_t file_format = h0.format;

  if (t >= h0.n_types) {
    SIF_LOG_ERROR(TAG,
      "the snapshot has %u particle types; there is no type %u", h0.n_types, t);
    return SIF_ERR_INVALID;
  }

  uint64_t n_total = 0;
  char file[PATH_CAP];
  for (uint32_t i = 0; i < h0.n_files; i++) {
    status = path_file(&sp, h0.n_files, i, file);
    if (status != SIF_OK)
      return status;

    status = snap_open(file, file_format, &sf);
    if (status != SIF_OK)
      return status;
    const sif_gadget_header_t hi = sf.header;
    snap_close(&sf);

    if (hi.n_types != h0.n_types || hi.n_files != h0.n_files) {
      SIF_LOG_ERROR(
        TAG, "%s does not belong to the same snapshot as %s", file, sp.first);
      return SIF_ERR_IO;
    }
    n_total += hi.n_part_file[t];
  }

  if (n_total != h0.n_part_total[t]) {
    SIF_LOG_ERROR(TAG,
      "the files hold %" PRIu64 " particles of type %u, but the header says "
      "the snapshot has %" PRIu64 "; a file is missing or from another run",
      n_total, t, h0.n_part_total[t]);
    return SIF_ERR_IO;
  }
  if (n_total == 0) {
    SIF_LOG_ERROR(TAG, "the snapshot has no particles of type %u", t);
    return SIF_ERR_INVALID;
  }

  /* --- units and options --- */

  double pos_scale;
  if (length == SIF_GADGET_LENGTH_KPC) {
    pos_scale = 1e-3;
  } else if (length == SIF_GADGET_LENGTH_MPC) {
    pos_scale = 1.0;
  } else if (h0.unit_length_in_cm > 0) {
    pos_scale = h0.unit_length_in_cm / MPC_IN_CM;
  } else {
    SIF_LOG_ERROR(TAG,
      "%s does not record its length unit; name it with SIF_GADGET_LENGTH_KPC "
      "or SIF_GADGET_LENGTH_MPC",
      sp.first);
    return SIF_ERR_INVALID;
  }

  stream_t s;
  memset(&s, 0, sizeof(s));
  s.ptype = t;
  s.pos_scale = pos_scale;
  s.read_vel = velocity != SIF_GADGET_VELOCITY_SKIP;
  s.vel_scale = 1.0;
  if (velocity == SIF_GADGET_VELOCITY_PECULIAR) {
    if (!(h0.time > 0)) {
      SIF_LOG_ERROR(TAG,
        "peculiar velocities need a scale factor, and the "
        "header's time is %g",
        h0.time);
      return SIF_ERR_INVALID;
    }
    s.vel_scale = sqrt(h0.time);
  }

  if (mass == SIF_GADGET_MASS_READ) {
    if (h0.mass_table[t] > 0)
      SIF_LOG_INFO(TAG,
        "type %u has one mass for every particle (%g); the field is left "
        "unweighted",
        t, h0.mass_table[t]);
    else
      s.read_mass = true;
  }

  s.n_total = n_total;
  s.subsample = fraction < 1;
  s.n_keep =
    s.subsample ? (uint64_t)llround(fraction * (double)n_total) : n_total;
  if (s.n_keep == 0) {
    SIF_LOG_ERROR(TAG, "a fraction of %g keeps none of %" PRIu64 " particles",
      fraction, n_total);
    return SIF_ERR_INVALID;
  }
  sif_prng_init(&s.prng, seed);

  /* --- allocation --- */

  s.field = sif_field_alloc(s.n_keep);
  s.pos = malloc(3 * CHUNK * sizeof(double));
  s.vel = s.read_vel ? malloc(3 * CHUNK * sizeof(double)) : NULL;
  s.mass = s.read_mass ? malloc(CHUNK * sizeof(double)) : NULL;
  s.scratch = malloc(3 * CHUNK * 8);

  status = SIF_OK;
  if (!s.field || !s.pos || !s.scratch || (s.read_vel && !s.vel) ||
      (s.read_mass && !s.mass))
    status = SIF_ERR_ALLOC;
  if (status == SIF_OK)
    status = sif_field_reserve_positions(s.field);
  if (status == SIF_OK && s.read_vel)
    status = sif_field_reserve_velocities(s.field);
  if (status == SIF_OK && s.read_mass)
    status = sif_field_reserve_weights(s.field);

  /* --- pass 2: stream every file into the field --- */

  for (uint32_t i = 0; status == SIF_OK && i < h0.n_files; i++) {
    status = path_file(&sp, h0.n_files, i, file);
    if (status != SIF_OK)
      break;

    status = snap_open(file, file_format, &sf);
    if (status != SIF_OK)
      break;

    if (s.read_vel && sf.header.n_part_file[t] > 0 && !sf.is_hdf5 &&
        !sf.bin.vel.present) {
      SIF_LOG_ERROR(TAG, "%s has no velocity block", file);
      status = SIF_ERR_IO;
    } else if (s.read_mass && sf.header.n_part_file[t] > 0 && !sf.is_hdf5 &&
               !sf.bin.mass.present) {
      SIF_LOG_ERROR(TAG, "%s has no mass block", file);
      status = SIF_ERR_IO;
    } else {
      status = stream_file(&s, &sf, file);
    }
    snap_close(&sf);
  }

  free(s.pos);
  free(s.vel);
  free(s.mass);
  free(s.scratch);

  if (status == SIF_OK && s.kept != s.n_keep) {
    SIF_LOG_ERROR(
      TAG, "read %" PRIu64 " particles, expected %" PRIu64, s.kept, s.n_keep);
    status = SIF_ERR_IO;
  }

  if (status != SIF_OK) {
    sif_field_free(s.field);
    return status;
  }

  if (out_box_length)
    *out_box_length = h0.box_size * pos_scale;

  SIF_LOG_INFO(TAG,
    "loaded %" PRIu64 " of %" PRIu64 " type-%u particles from %u %s file%s",
    s.n_keep, n_total, t, h0.n_files, format_name(file_format),
    h0.n_files == 1 ? "" : "s");

  *out_field = s.field;
  return SIF_OK;
}

#undef TAG
#undef CHUNK
#undef PATH_CAP
#undef MPC_IN_CM
#undef HEAD_CAP
