# The sif data format (`.sdf`)

An `.sdf` file holds the products of a run -- a void catalogue and the things
measured from it -- in one place, and can be added to later without being
rewritten. It is the binary replacement for the ASCII catalogue and profile
files.

The design is deliberately narrow. Fields and grids are *not* products and stay
in `.xfield` and `.xgrid`, which exist to be read straight into memory at
speed; `.sdf` exists to keep a catalogue and its derived measurements together,
with enough metadata attached that a file still means something a year later.
Everything below is what that requires and nothing more, with reserved room in
both headers for whatever comes next.

## Anatomy

A file is a header followed by a chain of self-describing blocks:

```
+--------------------------------+  0
| file header              64 B  |
+--------------------------------+  64
| block 0  CATALOG               |     always first
|   block header           64 B  |
|   metadata        meta_bytes   |
|   data            data_bytes   |
|   zero padding      to 64 B    |
+--------------------------------+
| block 1  DENSITY_PROFILES      |
|   ...                          |
+--------------------------------+
| block 2  SIZE_FUNCTION         |
|   ...                          |
+--------------------------------+
                                    EOF
```

There is no index and no block count. A reader walks the chain from offset 64,
and each block header says how long its block is:

```
block_bytes = 64 + round_up_to_64(meta_bytes + data_bytes)
```

so skipping a block a reader does not want costs one seek, and listing a file's
contents reads only the 64-byte headers.

The absence of an index is the point rather than an omission. Appending is then
a single write at the end of the file -- nothing already on disk is touched,
not even a counter -- so a crash midway through an append can only ever leave a
torn *trailing* block. It can never leave a file whose header disagrees with
its contents.

## Conventions

**Byte order** is the writer's, recorded in the file header and checked by the
reader, which refuses a file it would have to swap rather than swapping it.
Cross-endian exchange is not a use this format serves.

**Integers** are fixed-width and unsigned unless stated. Counts in payload
arrays are always `uint64_t`, regardless of anything else in the block.

**Reals** in a payload are `float` or `double` as the block declares in
`real_dtype`, *not* as the reading build was compiled. A build whose `sif_real`
differs converts on read. This is the one place `.sdf` departs from `.xfield`,
which refuses a precision mismatch outright: a field is large enough that
reading it must be a copy into its final buffer and nothing else, while a
catalogue is small enough that a conversion pass costs nothing worth
protecting -- and a catalogue is the thing people hand to each other, so a
single-precision build must be able to read what a double build wrote.

Reals in the *metadata* are always `double`. A metadata table is small and
there is nothing to be gained by halving it.

**Strings** are UTF-8, without a terminator, with their length given
explicitly. Fixed-size character fields (`writer`, `name`) are zero-padded and
a reader must not assume a NUL is present.

**Padding** is zero-filled and never carries information. Metadata entries are
padded to a multiple of 8 bytes, so a block's data section always begins on an
8-byte boundary; a block as a whole is padded to a multiple of 64, so every
block header lands on a cache line.

**Checksums** are CRC32 (IEEE 802.3, the polynomial `sif_crc32()` uses),
computed over a block's metadata table followed by its data section, in that
order. The trailing block padding is not covered, since it is required to be
zero and any value there is a defect the length arithmetic already exposes.

## File header

64 bytes at offset 0.

| off | size | type | name | |
|---|---|---|---|---|
| 0 | 4 | `char[4]` | `magic` | `"SIFD"` |
| 4 | 4 | `uint32_t` | `version` | container version, currently 1 |
| 8 | 4 | `uint32_t` | `byte_order` | `0x01020304` written natively |
| 12 | 4 | `uint32_t` | `flags` | reserved, zero |
| 16 | 8 | `double` | `box_length` | the one global every product needs |
| 24 | 8 | `uint64_t` | `created` | seconds since the epoch |
| 32 | 16 | `char[16]` | `writer` | library version that created the file |
| 48 | 16 | `char[16]` | `padding` | reserved, zero |

```c
typedef struct {
  char     magic[4];
  uint32_t version;
  uint32_t byte_order;
  uint32_t flags;
  double   box_length;
  uint64_t created;
  char     writer[16];
  char     padding[16];
} sif_sdf_header_t;
```

`version` is the version of *this* structure and of the block framing around
it. Payload layouts version independently, per block; see `type_version`.

`box_length` is a `double` whatever the library was built with, for the same
reason it is in `.xfield`: a header must not change size with a build option.
It sits in the header rather than in metadata because every product in the file
is expressed in a box and a reader should not have to parse a metadata table to
find it.

## Blocks

### Block header

64 bytes, on a 64-byte boundary.

| off | size | type | name | |
|---|---|---|---|---|
| 0 | 4 | `char[4]` | `magic` | `"SBLK"` |
| 4 | 2 | `uint16_t` | `type` | see [Block types](#block-types) |
| 6 | 2 | `uint16_t` | `type_version` | payload layout version for this type |
| 8 | 2 | `uint16_t` | `real_dtype` | 1 = `float`, 2 = `double` |
| 10 | 2 | `uint16_t` | `reserved0` | reserved, zero |
| 12 | 4 | `uint32_t` | `flags` | reserved, zero |
| 16 | 4 | `uint32_t` | `meta_bytes` | metadata table size, multiple of 8 |
| 20 | 4 | `uint32_t` | `crc32` | over metadata + data |
| 24 | 8 | `uint64_t` | `data_bytes` | data section size |
| 32 | 8 | `uint64_t` | `n_items` | the block's primary count |
| 40 | 8 | `uint64_t` | `catalog_id` | which catalogue this block belongs to, or zero |
| 48 | 16 | `char[16]` | `name` | optional tag, zero-padded |

```c
typedef struct {
  char     magic[4];
  uint16_t type;
  uint16_t type_version;
  uint16_t real_dtype;
  uint16_t reserved0;
  uint32_t flags;
  uint32_t meta_bytes;
  uint32_t crc32;
  uint64_t data_bytes;
  uint64_t n_items;
  uint64_t catalog_id;
  char     name[16];
} sif_sdf_block_header_t;
```

Four of these fields carry more weight than their size suggests.

`data_bytes` is what makes the format extensible without a version bump: a
build that has never heard of `type == 12` still knows exactly how far to seek
past it, so an old library reads a new file minus the parts it cannot
interpret. The same argument applies to `type_version`: a block whose version
is higher than the reader knows is skippable, and only fails if something
actually asks for it.

`n_items` is the block's primary count -- voids for a catalogue or a profile
set, bins for a size function. It is in the header rather than in metadata so
that a listing can be produced without parsing anything, and so that the reader
has one number to cross-check the payload length against before it allocates
from a length the file supplied.

`catalog_id` is what makes a file's contents belong together. Block 0 declares
the identity of the catalogue it holds, and every block measured from that
catalogue repeats it, so "row *i* is void *i*" is checkable rather than a
convention maintained by hand across scripts. Only a `META` block may carry
zero, meaning it is not derived from the catalogue; a product block that does
is malformed.

The identity is a **token, not a hash of the values**. Hashing the catalogue
would seem to be the obvious way to do it and does not survive this format's
own rules: `real_dtype` is per block and a reader converts, so a catalogue
written as `float` and read into a `double` build hashes differently through no
fault of anyone's, and a `double` file read by a `float` build has genuinely
lost the bits the hash was over. So the identity is assigned once, when a
catalogue first exists, and travels with it -- in memory, and in this field on
disk. It is the same trick `.xgrid` plays with `source_key` for the CIC cache:
a file found in the right place still has to prove where it came from.

A catalogue that is modified stops being the catalogue that was written, and
its identity changes with it. That is a property of the library's catalogue
type rather than of this format, but it is the reason the check is worth
anything: appending profiles measured from an edited catalogue fails instead of
quietly producing a file whose rows line up with nothing.

`name` distinguishes several blocks of the same type. Two size functions with
different binnings, or profiles measured with two extents, are two blocks
tagged `"linear"` and `"lnbins"`, and a reader asks for the one it wants. An
empty name is allowed; a lookup that supplies no name takes the first block of
the type. Uniqueness of (`type`, `name`) is a writer's responsibility, not
something the format enforces.

### Metadata table

`meta_bytes` immediately after the block header, holding a sequence of entries.
The entry count is not stored: a reader consumes entries until `meta_bytes` is
exhausted.

Each entry is

| off | size | type | name | |
|---|---|---|---|---|
| 0 | 2 | `uint16_t` | `key_bytes` | length of the key, in bytes |
| 2 | 1 | `uint8_t` | `value_type` | 1 = `i64`, 2 = `f64`, 3 = UTF-8 |
| 3 | 1 | `uint8_t` | `reserved` | zero |
| 4 | 4 | `uint32_t` | `value_bytes` | length of the value, in bytes |
| 8 | `key_bytes` | | key | UTF-8, no terminator |
| | `value_bytes` | | value | |

padded with zeros to a multiple of 8 bytes. So

```
entry_bytes = round_up_to_8(8 + key_bytes + value_bytes)
```

An `i64` or `f64` entry holds `value_bytes / 8` elements and a string holds
`value_bytes` of UTF-8. The length is in bytes rather than in elements
deliberately: it is what makes an entry skippable **without knowing its type**,
which is the same property `data_bytes` gives a block. A reader that meets
`value_type == 9` steps over it and carries on with the rest of the table,
where a length counted in elements of an unknown width would leave it with
nowhere to resume.

Three types is the whole vocabulary. Several elements in one entry give
vectors for free -- a cosmology's parameters, a list of thresholds -- and a
value of any of the three covers everything a FITS card can express, with none
of a FITS card's limitations: no 80-column budget, no eight-character keys, no
parsing text back into a number and guessing whether it was meant to be one.

Keys are case-sensitive and must be unique within a table. **Keys beginning
`sif.` are reserved**: they are how the library stores a block's structural
parameters, and a caller writing its own must not use the prefix. Everything
else belongs to the caller.

Putting the structural parameters in the metadata rather than in a fixed
per-type struct is what keeps the number of moving parts down. One mechanism
serves both the shape of a product and the notes attached to it; a generic dump
tool prints a block's shape without knowing what the block is; the Python
bindings hand back a dictionary; and giving a product a new parameter later is
a new key with a default, rather than a new payload version. The cost is that
the reader must validate `data_bytes` against the size the metadata implies
before trusting either, which it should be doing regardless.

### Data section

`data_bytes` immediately after the metadata table, holding the block's arrays
back to back in the order its type prescribes, with no per-array header. The
layout is fixed by this specification, not self-described -- that is the line
between this format and a generic one.

Then zero padding to the next 64-byte boundary.

## Block types

| id | name | `n_items` | |
|---|---|---|---|
| 0 | | | invalid |
| 1 | `CATALOG` | voids | required, first |
| 2 | `META` | 0 | metadata only, no data section |
| 3 | `DENSITY_PROFILES` | voids | |
| 4 | `VELOCITY_PROFILES` | voids | |
| 5 | `SIZE_FUNCTION` | bins | measured from the catalogue |
| 6 -- 0x7FFF | | | reserved for sif |
| 0x8000 -- 0xFFFF | | | never assigned by sif; free for private use |

Field statistics -- the delta distribution and the delta moments -- are
deliberately absent. They describe a field rather than a catalogue, so they do
not belong under the rule that a catalogue is what a file is built around, and
they can have their own home if they ever need one.

All payload layouts below are `type_version` 1. `R` abbreviates a real of the
block's `real_dtype`.

### `CATALOG`

`n_items` is the number of voids. No structural metadata: the box is in the
file header and there is nothing else the container needs.

| array | elements | |
|---|---|---|
| `cx` | `n_items` R | centres, x |
| `cy` | `n_items` R | centres, y |
| `cz` | `n_items` R | centres, z |
| `radii` | `n_items` R | radii |

Component-major, matching the order the four views sit in a catalogue's arena,
so a reader fills each of them with one sequential read rather than striding
through interleaved values.

### `DENSITY_PROFILES`

`n_items` is the number of voids, which must equal the catalogue's.

| key | type | |
|---|---|---|
| `sif.n_bins` | `i64` | bins per row |
| `sif.ext` | `f64` | outer edge, in units of each void's radius |
| `sif.differential` | `i64` | 1 if a bin holds its own shell, 0 if it holds everything enclosed |

| array | elements | |
|---|---|---|
| `r_edges` | `n_bins + 1` R | bin edges, shared by every row |
| `profiles` | `n_items * n_bins` R | row-major, row *i* is void *i* |

`sif.differential` is not recoverable from the values and decides which radius
a bin belongs at: a cumulative bin is everything within its outer edge, a
differential bin is a shell and belongs at the midpoint of its edges. Reading a
cumulative profile at bin centres shifts it by half a bin, which is enough to
move a feature at `r = R_v` off that mark.

### `VELOCITY_PROFILES`

As `DENSITY_PROFILES`, without `sif.differential`, and with the second array
holding radial velocities.

| key | type | |
|---|---|---|
| `sif.n_bins` | `i64` | bins per row |
| `sif.ext` | `f64` | outer edge, in units of each void's radius |

| array | elements | |
|---|---|---|
| `r_edges` | `n_bins + 1` R | |
| `v_rad` | `n_items * n_bins` R | row-major |

### `SIZE_FUNCTION`

`n_items` is the number of bins.

| key | type | |
|---|---|---|
| `sif.options` | `i64` | the `sif_option` bit field the size function was produced with |
| `sif.r_min` | `f64` | lower edge of the first bin |
| `sif.r_max` | `f64` | upper edge of the last bin |

| array | elements | |
|---|---|---|
| `r_edges` | `n_items + 1` R | |
| `r_centers` | `n_items` R | |
| `counts` | `n_items` `uint64_t` | raw void count per bin |
| `vsf` | `n_items` R | the size function itself |
| `err` | `n_items` R | Poisson error on `vsf` |

`sif.options` has to travel with the values because it records whether the
size function is per unit `ln R` or per unit `R`, which nothing else in the
block reveals.

Only a *measured* size function is storable. A modelled one is evaluated
pointwise from parameters, counts nothing and comes from no catalogue, so it
has no identity to record and nothing in the file to belong to -- and a file
whose blocks are not all derived from its catalogue is exactly the thing this
format exists to prevent. Models are cheap to recompute; store the parameters
in a `META` block instead.

### `META`

`data_bytes` and `n_items` are zero; the block is its metadata table. This is
where file-level notes go -- redshift, cosmology, the parameters of the run,
whatever the caller wants to keep -- and because it is an ordinary block, more
of it can be appended later like anything else. `name` is unused and zero.

A file may hold several, and where two carry the same key **the last one in the
file wins**. That rule is what gives an append-only format a way to correct
itself: a value written today supersedes the one written last year without
touching it. A reader is expected to merge every `META` block in file order and
present the result as one table, so the several blocks on disk are one set of
notes to anything reading them.

The rule has a consequence worth stating: a key cannot be **removed**, only
overwritten. Writing a table that omits it leaves the earlier block, and the
earlier block still says what it said. There is deliberately no way to express
"unset" in version 1.

Nothing requires a file to carry metadata at all. It is a good habit and not an
obligation, and a file with no `META` block reads as a file whose notes are
empty rather than as one that is missing something.

## What a reader enforces

A reader that accepts a damaged file is worse than one that fails, because the
data flows onward and whatever it produces looks like a result. So:

1. `magic` and `byte_order` in the file header, and `magic` in every block
   header, must match.
2. `version` above what the build knows is a refusal, not a warning.
3. Block 0 must be a `CATALOG`, and a file holds exactly one. Its
   `catalog_id` must be non-zero.
4. `meta_bytes + data_bytes` must not run past the end of the file, and
   `data_bytes` must equal the size implied by `n_items` and the block's
   metadata. A length that fails either check is rejected before anything is
   allocated from it.
5. `crc32` must match the block's metadata and data. Checked when a block is
   read, not when the file is opened -- opening reads only the headers, and
   verifying every checksum up front would mean reading the whole file to
   answer a question about one block.
6. A profile block's `n_items` must equal the catalogue's.
7. Every block's `catalog_id` must equal block 0's, except a `META` block's,
   which is zero. Blocks in the private range are exempt: they belong to
   whoever wrote them, and a reader that skips a block has no business
   auditing it.
8. A metadata table must consume exactly `meta_bytes`, and **its** keys must be
   unique -- a table that declares one key twice does not say which value it
   means. Two *blocks* carrying the same key is the correction mechanism and
   not a duplicate; see `META`.
9. A trailing block that is short, or whose checksum fails, is an error rather
   than a truncation to be tolerated silently. Recovering the good prefix is a
   separate, explicit operation, and it stops at the first block that does not
   hold: past a damaged block the chain gives no way to know where the next one
   starts, and looking for one would be guessing.

An unknown `type`, an unknown `type_version`, and an unknown metadata
`value_type` are *not* errors: the first two are skipped by length, the third
is skipped within its table. Only a request for a block that cannot be
interpreted fails.

## Appending

Open the file for update, validate the file header, seek to the end, write one
block, flush. Nothing else is modified, and there is no index to keep in step.

A block that is derived from the catalogue needs `catalog_id`, which is read
from block 0's header without touching its payload -- 64 bytes, one seek.

Blocks are immutable once written. Rewriting a catalogue in place would leave
every dependent block silently wrong; a different catalogue is a different
file. An accidental edit is caught by block 0's own `crc32`, which covers the
payload that was rewritten. A deliberate one that recomputes the checksum is
forgery, and no field stored in the same file defends against that.

The format assumes **one writer at a time** and provides no locking. Two
processes appending to the same file concurrently will interleave and corrupt
it. Parallel runs write separate files.

## What this format does not do

Not by omission -- these are the trades that keep it small.

- **No fields or grids.** Those are `.xfield` and `.xgrid`, whose whole design
  is landing a multi-gigabyte payload in memory with no parse and no copy.
- **No compression.** Products are small next to the field they came from, and
  a compressed payload cannot be read in place.
- **No cross-endian reading.** Detected and refused, not converted.
- **No random access by name without a walk.** With blocks numbering in the
  handful, an index would cost more in write-time fragility than it saves in
  read-time seeks.
- **No self-describing payloads.** Array layouts are fixed per type and version
  by this document. Extensibility comes from new type ids and new metadata
  keys, both of which an old reader handles without understanding them.
- **No in-place edit, and no deletion.** A file only grows.

## Version history

**1** -- initial. Catalogue, density and velocity profiles, size function,
metadata blocks.
