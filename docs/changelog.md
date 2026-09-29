# Changelog

**sif** is at version 0.x: the API may change between releases, and breaking
changes are always listed here.

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
