-- sif-exodus configuration: a periodic box.
--
-- A Lua script: settings are name = value, grouped in sections, and a
-- value can be anything Lua computes. Only mode, input, finder.radii,
-- finder.threshold, finder.variant (or finder.exodus_params) and output
-- are required; every other setting below is at its default and can be
-- left out.
--
-- Command-line variables arrive as strings: `sif-exodus -D snap=010`
-- makes `snap` here "010", and `snap = snap or "010"` gives it a
-- default. `sif-exodus --check` shows what a file resolves to.

-- box: a simulation, one tracer set in a periodic box. survey: data and
-- randoms, with edges (sif-exodus --template survey).
mode = "box"

input = {
  path = "tracers.xfield",
  -- xfield | ascii | binary | gadget | fits | hdf5; an .xfield or a .fits
  -- file needs none. FITS and HDF5 inputs may be a list of files, read as
  -- one: path = { "part_0.fits", "part_1.fits" }.
  format = "xfield",
  -- The periodic box. Recorded by .xfield and GADGET files, which a
  -- value here overrides; ascii, binary, FITS and HDF5 files need it.
  -- box_length = 1000,

  -- Beside path and format, the settings of that format, and only those.
  --
  -- ascii: columns are x y z, w for a weight, * for a column to skip.
  --   columns = "x y z",      -- e.g. "* x y z w"
  --   delimiter = " ",
  --   skip_header = 0,
  -- binary:
  --   columns = "x y z",
  --   precision = "float32",  -- float32 | float64: required
  --   layout = "rows",        -- rows (x0 y0 z0 x1 ...) | blocks (x0 x1 ... y0 ...)
  --   endian = "native",      -- native | little | big
  --   header_bytes = 0,
  -- gadget:
  --   snapformat = "auto",    -- 1 | 2 | 3 (HDF5) | "auto"
  --   ptype = 1,
  --   length = "auto",        -- kpc | mpc | auto (HDF5 files only)
  --   masses = false,         -- true: particle masses as weights
  --   fraction = 1,           -- a random subsample, in (0, 1]
  --   seed = 0,
  -- fits: a column, or an expression over columns, for each position --
  -- w = "W1 * W2", x = "POS[1]" for an element of a vector column.
  --   columns = { x = "X", y = "Y", z = "Z" },  -- and w, for a weight
  --   where = "Z > 0",        -- rows to keep; nil keeps every row
  --   hdu = 1,                -- the table: EXTNAME or number; nil: the first
  --   fraction = 1,           -- a random subsample of the kept rows
  --   seed = 0,
  -- hdf5: a dataset for each position, by its path; [k] is column k of a
  -- 2D one, from 0.
  --   columns = {
  --     x = "PartType1/Coordinates[0]",
  --     y = "PartType1/Coordinates[1]",
  --     z = "PartType1/Coordinates[2]",
  --   },                      -- and w, for a weight
  --   length_scale = 1,       -- multiplies positions: 1e-3 from kpc/h
  --   fraction = 1,
  --   seed = 0,
}

grid = {
  n_cells = "auto",  -- per side; auto: one cell per mean separation
  -- How each tracer is spread over the cells: ngp | cic | tsc | pcs, over
  -- 1, 2, 3 or 4 cells per axis. Higher orders alias less, and cost more.
  assignment = "cic",
  -- true: also deposit shifted by half a cell and average the two, which
  -- cancels the leading aliasing; a second deposit and two FFTs.
  interlace = false,
}

mesh = {
  n_cells = "auto",  -- per side; only speed and memory depend on it
}

finder = {
  -- In the box's units. ladder(r_min, r_max, step): log-spaced, 5% apart
  -- here. Also linspace(a, b, n) and geomspace(a, b, n), or any list of
  -- numbers. The smallest should span two grid cells or more.
  radii = ladder(10, 40, 0.05),
  threshold = -0.7,          -- the density contrast a void is grown to
  -- The finder: one of two. excursion-set grows voids from every cell
  -- below the threshold, with no overlap -- the voids of the excursion-set
  -- models, and the finder of sif-exodus 0.4. excursion-peak centres them
  -- on minima of the density smoothed at each rung's scale, a void refused
  -- only by a larger one it sits inside; its void size function depends on
  -- the grid less.
  variant = "excursion-set",
  -- The settings a variant fixes, to change one or to set them all
  -- without a variant (they then start from excursion-set's). Each one
  -- given overrides the variant's.
  -- exodus_params = {
  --   overlap_fraction = 0,   -- of the smaller radius; 0: no overlap
  --   search_factor = 1.5,    -- 1.25 | 1.5 | 1.75 | 2
  --   seeds = "cells",        -- cells | minima
  --   -- How the grid's assignment window is undone before smoothing: full
  --   -- divides it out (boosting aliasing and shot noise near the grid
  --   -- scale); aliasing divides out the window with its aliases (least
  --   -- squares); none keeps it.
  --   deconvolution = "full",
  -- },
}

-- Voids to start from: the run adds its own and writes both, these first.
-- To extend a catalogue to smaller radii, run the rungs below the ones it
-- was found with. Read like output, by its name; an ASCII catalogue from
-- another finder names its columns: "* x y z * r".
-- prior = { path = "large_voids.h5", format = "hdf5" }

-- .h5 or .hdf5: HDF5; .fits: FITS; anything else: ASCII. Each records
-- the run's settings alongside the voids.
output = "voids.h5"
-- output = { path = "voids.h5", format = "hdf5" }  -- hdf5 | ascii | fits

run = {
  threads = "auto",
  log_level = "info",  -- trace | debug | info | warning | error | none
  tune_fft = true,     -- false: estimated FFT plans, quicker to set up
  -- Progress, saved after every rung; a run that finds it resumes from it,
  -- so a job resubmitted as it was picks up where it stopped. Removed when
  -- the run completes. Default: the output's path + ".ckpt". false: none.
  -- checkpoint = "voids.h5.ckpt",
  -- Keep the grid on disk rather than its transform in memory: about half
  -- the memory, one more FFT and one read of the grid per smoothing, the
  -- same catalogue. The file goes in the sif setting scratch_directory,
  -- or $SIF_SCRATCH_DIRECTORY for one job; put it on a fast filesystem.
  save_memory = false,
}
