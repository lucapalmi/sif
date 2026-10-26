# Changelog

Current version: v0.4 release 0


## v0.4.1 -- 26/10/2026

### Added

- Exodus variants, `"excursion-set"` (default) and `"excursion-peak"`:
  `sif_finder_exodus(..., variant, extras)`, `exodus(..., variant=)`,
  `finder.variant`
- `sif_finder_exodus_raw()`, `exodus_raw()`, `finder.exodus_params`: every
  setting explicit; `sif_finder_exodus_preset()` gives a variant's
- `sif_grid_assign()`: NGP, CIC, TSC and PCS, optionally interlaced
  (`Grid.assign()`, `grid.assignment` and `grid.interlace` in Lua)
- Aliasing-aware window deconvolution (`deconvolution="aliasing"`)

### Changed

- FFTs are bit-identical across thread counts, and 10-35% faster
- `sif_finder_exodus()` and `sif_finder_exodus_survey()` take a variant; the
  old signatures and the `_ex` functions are now the `_raw` ones
- Python `exodus()` takes a variant and run controls only; the finder settings
  and `checkpoint=False` moved to `exodus_raw()`. `keep_checkpoint=` added
- `sif-exodus`: `finder.variant` or `finder.exodus_params` is required;
  `overlap_fraction`, `search_factor` and `deconvolution` moved into the latter
- Exodus checkpoints are version 2; version 1 (0.4) ones still resume

### Fixed

- `sif_survey_volume()` (`pysif.measure.survey_volume()`) was biased low when
  the randoms' density varies across the sky (3-48%); now within a few
  tenths of a per cent
- With an overlap fraction above 0, a void could overlap a smaller one by
  more than the fraction allows
- Version 1 exodus checkpoints failed their own checksum
- `sif-exodus --check` reported a 0.4 checkpoint's grid assignment as unknown
- `sif_ep_multiplicity_function_emu()` with `collapse`: a grid started at a
  large nu was reported out of domain, and its void-in-cloud factor was off
  by up to 2.7%


## v0.4.0 -- 18/10/2026

### Added

- Exodus checkpoints: a run saves its voids after every rung and resumes
  from them bit for bit (`sif_finder_exodus_ex()`); off with
  `SIF_FINDER_NO_CHECKPOINT`
- Exodus can extend a prior catalogue, e.g. down to smaller radii;
  `sif_finder_exodus_checkpoint_read()` reads a checkpoint
- `sif-exodus`: `run.checkpoint`, `--checkpoint` / `--no-checkpoint`, a `prior`
  section, and `--check` on checkpoints
- Python: `checkpoint=` and `prior=` on the exodus finders, and
  `finders.read_checkpoint()`
- `sif_catalogue_to_cartesian()` (`Catalogue.to_cartesian()`), the inverse of
  `sif_catalogue_to_sky()`
- `sif_system_init()` warns when there are more threads than processors to
  run them
- Void-in-cloud in the excursion-set Monte Carlo and emulator (`collapse=`, and
  `cov=` for the emulator; about 0.09%)
- `SIF_SANITIZE` CMake option, for sanitizer builds

### Changed

- `sif_ep_first_crossing_counts()` and `sif_ep_multiplicity_function()` take
  the collapse barrier after the void barrier; the counts gain `collapse_counts`
- `sif_ep_multiplicity_function_emu()`: retrained out to 250 Mpc/h (about 0.06%
  below 80 Mpc/h); takes `collapse` and `cov` after `deriv_variance`
- `sif_ep_multiplicity_function_emu()` returns the density at the n radii;
  `SIF_EP_EMU_BIN_CENTRES` (`binned=True`) gives the old bin averages
- `sif_ep_multiplicity_function_emu()`: grids starting beyond the training
  walks are no longer out of domain; barriers rising with sigma are

### Fixed

- pip-installed `sif-exodus` ran on one core when `OMP_PROC_BIND` was set;
  reinstall the wheel to get the fixed launcher
- CIC grids, density contrasts, spectral moments and exodus catalogues no
  longer change with the thread count; cached `.xgrid` files are recomputed once

## v0.3.0 -- 13/10/2026

### Added

- `sif_survey_volume()` (`pysif.measure.survey_volume()`): compute survey 
   volumes from randoms
- `sif_finder_grid_cells_from_size()` (`pysif.finders.grid_cells_from_size()`):
  grid cells per side from a cell size, in physical units or mean separations
- `sif-exodus`: `input.observer`, for a Cartesian survey whose observer is
  not at the origin, and `input.footprint_volume`, to give the volume.

### Changed

- `sif_finder_suggest_mesh_cells()` takes the volume the tracers occupy (0,
  or `volume=None` in Python, for the box); pass a survey's footprint volume
- `survey_box(data, randoms, radii, search_factor=1.5)`: the data are
  required and the grid's cells are no longer needed. The box allows for any
  grid with cells up to the smallest radius; the finder refuses a coarser one
- `exodus_survey()` takes the `offset` from `survey_box()`, right after the
  meshes, and returns the voids in the caller's frame: no more
  `catalogue.translate(-offset)`
- `sif-exodus`: the footprint volume comes from `sif_survey_volume()`. The
  old measure was off by tens of per cent for sparse randoms or thin
  footprints
- `sif-exodus`: a configured grid coarser than the smallest radius is
  refused before the run starts
- Python: `Catalogue.radii`, `footprint` and `footprint_shell` are read-only
  views

### Removed

- `sif_finder_suggest_mesh_cells_survey()` (`suggest_mesh_cells_survey()`):
  `suggest_mesh_cells()` with the footprint volume replaces it
- `sif-exodus`: `finder.radii_units`. Radii are in the positions' units; a
  configuration that still sets it is told so

### Fixed

- The chain mesh's periodic nearest-neighbour search could skip the cell
  holding the nearest tracer when it sat across a box face

## v0.2.2 -- 9/10/2026

### Added

- `sif-exodus --check FILE` checks a data file as well as a configuration.
  It tells the kind from the contents (ASCII, FITS, HDF5, GADGET, `.xfield`),
  lists every problem with where it is, and prints a summary. The options:
  - `--deep` reads every value, for each column's range. On a configuration,
    it reads the inputs as the run would
  - `--format` overrides the guess
  - `--columns`, `--delimiter` and `--skip-header` check an ASCII file
    against a column format
- `sif_field_check_ascii()`, `sif_field_check_xfield()`, `sif_fits_check()`,
  `sif_hdf5_check()` and `sif_gadget_check()`: the same checks in the
  library, with a summary written to a stream
- `sif_gadget_is_snapshot()`: whether a path names a GADGET snapshot, asked
  without logging
- `SIF_ERR_UNINITIALISED`: what a call that needs `sif_system_init()` fails
  with without it (through `sif_error_status()` for those returning a
  pointer). Python raises `RuntimeError`; the finders used to raise
  `ValueError`

### Changed

- `sif_init()` / `sif_finalise()` are now `sif_system_init()` /
  `sif_system_finalise()` (Python: `pysif.system_init()` /
  `pysif.system_finalise()`). Only the finders,
  `sif_delta_moments_grid()`, `sif_delta_distribution_grid()` and the
  settings accessors need them.
- `sif_setting_set()` returns a status (`SIF_ERR_INVALID`, `SIF_ERR_ALLOC`,
  `SIF_ERR_UNINITIALISED`) instead of silently doing nothing, and
  `pysif.set_setting()` raises. 
- Without `sif_system_init()`, the library uses OpenMP's own thread count.
- Improved error messages for Field, Grid, ChainMesh and Catalog
- `sif_field_refresh_bounds()` and `sif_field_sort_morton()` refuse a field
  with a non-finite position, naming the first
- `sif_field_wrap_periodic()` takes a third counter, `n_nonfinite`, for the
  NaN and infinite coordinates it cannot repair (pass NULL to ignore it).
  `Field.wrap()` returns it as `'nonfinite'` and raises a `RuntimeWarning`
  when it is not zero, as does `read_field(..., wrap=True)`
- `sif_chain_mesh_alloc()` and `sif_grid_alloc()` report the reason they
  failed through `sif_error_status()`
- The CIC assignment and the chain mesh accept coordinates within a few ULPs
  of the box faces. The `--check` file checkers apply the same tolerance
- The field functions that return a status are `SIF_NODISCARD`

### Deprecated

- `sif_init()`, `sif_finalise()`, `pysif.init()` and `pysif.finalise()` still
  work, with a warning, and will be removed in the next release

### Removed

- Shewchuk's geometric predicates (`vendor/predicates`)
- The octree (`sif_octree_*`, `pysif.Octree`), which nothing used

### Fixed

- The survey finder's footprint read below 1 for voids well inside the
  survey. The footprint is now closed over 3^3 cells, or up to 9^3 where
  the randoms are too sparse for that, which also fills real gaps that
  narrow. A warning says when even 9^3 leaves chance holes
- Python exceptions that named the wrong cause.
- `sif_chain_mesh_alloc()` returned NULL without a message for an empty
  field or an allocation failure
- A catalogue metadata table that could not grow was reported as a bad key
- `sif_chain_mesh_alloc_consume()` documented that a refused field is
  emptied; it is left as it was. A failure after the handover now warns that
  the field was emptied

## v0.2.1 -- 30/09/2026

### Added

- `sif_error_message()`, `sif_error_errno()`, `sif_error_status()` and
  `sif_error_clear()`: the first error logged on the thread since the last
  clear, whatever the log level

### Changed

- I/O errors name the file and the place in it (line, record and byte
  offset, HDU, dataset), and keep the system's, HDF5's and cfitsio's own
  diagnosis: a truncated GADGET, HDF5 or FITS file is reported as truncated
- Finder errors name the argument at fault
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
