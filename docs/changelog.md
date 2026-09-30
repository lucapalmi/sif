# Changelog

**sif** is at version 0.x: the API may change between releases, and breaking
changes are always listed here.

## Unreleased

### Breaking changes

- The ASCII particle reader is strict: a row with fewer columns than the
  format, or a read column that is not a number in full (`abc`, `1.5e`, an
  empty field), fails the read with its line number. Both used to be skipped
  or read as 0 without a word. Columns skipped with `*` may still hold
  anything
- An ASCII read into a field that already has a particle count fails if the
  file holds fewer rows, instead of shrinking the field
- Python: I/O exceptions carry the reason as their message. A missing file
  raises `FileNotFoundError` (or `PermissionError`, `IsADirectoryError`)
  everywhere; a malformed catalogue format string raises `ValueError`
- `sif_finder_exodus_survey_box()` takes the data before the randoms (NULL
  for the randoms alone)
- The finders refuse a radius that is not positive and finite, a threshold
  outside (-1, 0) and an overlap fraction outside [0, 1], as `sif-exodus`
  already did; they used to run on them
- Python: the finders raise `ValueError` (or `MemoryError`) with the reason,
  instead of a `RuntimeError` saying to check the log

### Added

- `sif_error_message()`, `sif_error_errno()`, `sif_error_status()` and
  `sif_error_clear()`: the first error logged on the thread since the last
  clear, whatever the log level

### Changed

- I/O errors name the file and the place in it (line, record and byte
  offset, HDU, dataset), and keep the system's, HDF5's and cfitsio's own
  diagnosis: a truncated GADGET, HDF5 or FITS file is reported as truncated
- Finder errors name the argument at fault

### Fixed

- The survey box is sized around the data as well as the randoms
  (`survey_box(..., data=data)` in Python). Sized on the randoms alone, a
  galaxy a few Mpc past the outermost random fell in the padding and
  `exodus_survey()` refused the run. Coordinates that are not finite are
  refused instead of passed over
- A finder run that failed partway (a search sphere wider than the mesh, out
  of memory) returned the voids found so far as if it had finished, and left
  the grid smoothed. It returns NULL, and restores the grid

## v0.2.0 -- 30/09/2026

### Breaking changes

- British spelling in the API:
  - `catalog` → `catalogue`, `center` → `centre` (C and Python)
  - `finalize` → `finalise`, `sif_field_quantize` → `sif_field_quantise`
  - `SIF_VECTORIZATION_REPORT` → `SIF_VECTORISATION_REPORT`
- HDF5 files from 0.1 are not readable: the group and dataset names changed
  with the renaming
- I/O argument order: readers return the field, `_into` readers take the
  object first, writers take the path first

### Added

- `sif-exodus`: stand-alone program to run the finder, for boxes
  and surveys (`--check`, `--template`), using Lua configuration files; 
  installed with the wheel
- GADGET reader: all formats, multi-file snapshots, subsampling
- FITS input and output (`SIF_FITS_SUPPORT`, needs cfitsio)
- Sky coordinates: conversion to comoving Cartesian and back (`to_sky()`)
- Named columns and metadata in every catalogue format
- ASCII and binary particle parsers
- `SIF_RECORD_RPATH`: builds from source run on clusters without the modules
- Docs: running exodus, void profiles, *Concepts* section

### Changed

- `pysif` is a package (`pysif/__init__.py` over `pysif._pysif`)

### Fixed

- Build against HDF5 1.10

## v0.1.0 -- 25/09/2026

- First release: exodus, profiles, size functions, excursion-peak model, HDF5,
  Python bindings
