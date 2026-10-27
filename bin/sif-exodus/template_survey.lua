-- sif-exodus configuration: a survey.
--
-- A Lua script: settings are name = value, grouped in sections, and a
-- value can be anything Lua computes. Only mode, input (its data, random
-- and, on the sky, cosmology), finder.radii, finder.threshold and output
-- are required; every other setting below is at its default and can be
-- left out.
--
-- Command-line variables arrive as strings: `sif-exodus -D zbin=1`
-- makes `zbin` here "1", and `zbin = zbin or "1"` gives it a default.
-- `sif-exodus --check` shows what a file resolves to, and how large the
-- run will be.

-- survey: data and randoms, with edges. box: a simulation, one tracer set
-- in a periodic box (sif-exodus --template box).
mode = "survey"

input = {
  -- sky: the columns are ra and dec (in degrees) and the redshift z,
  -- taken to comoving Mpc/h with the cosmology, and the voids are written
  -- back on the sky. cartesian: they are x y z, comoving, with the
  -- observer at the origin unless observer says otherwise, and so are the
  -- voids.
  coordinates = "sky",
  -- Sky only. Flat unless omega_de is given.
  cosmology = { omega_m = 0.31, omega_r = 0, w0 = -1, wa = 0 },
  -- Cartesian only: where the observer is, if not at the origin.
  -- observer = { 0, 0, 0 },
  -- The footprint's volume, in (Mpc/h)^3, which sets the mean separation
  -- and so the default grid and meshes. Measured from the randoms unless
  -- given, which takes one angular mask seen over one range of distances;
  -- give it for a footprint of any other shape.
  -- footprint_volume = 1e9,

  -- The data and the randoms, each read as a box's input is: path,
  -- format, and the settings of that format (sif-exodus --template box
  -- lists them all). Either may carry a weight, w.
  data = {
    path = "galaxies.fits",  -- fits and hdf5: also a list of files
    format = "fits",         -- xfield | ascii | binary | gadget | fits | hdf5
    columns = { ra = "RA", dec = "DEC", z = "Z" },
    -- where = "Z > 0.43 && Z < 0.7",
  },
  random = {
    path = "randoms.fits",
    format = "fits",
    columns = { ra = "RA", dec = "DEC", z = "Z" },
    -- where = "Z > 0.43 && Z < 0.7",
    -- fraction = 0.5,       -- a random subsample, in (0, 1]
    -- seed = 0,
  },
}

grid = {
  -- Per side, over the box the survey is searched in, which is chosen
  -- around the footprint the randoms trace. auto: one cell per mean
  -- separation of the data inside the footprint.
  n_cells = "auto",
  -- How each tracer, data and random alike, is spread over the cells:
  -- ngp | cic | tsc | pcs, over 1, 2, 3 or 4 cells per axis. Higher orders
  -- alias less, and cost more.
  assignment = "cic",
  -- true: also deposit shifted by half a cell and average the two, which
  -- cancels the leading aliasing; a second deposit and two FFTs per grid.
  interlace = false,
}

mesh = {
  n_cells_data = "auto",    -- per side; only speed and memory depend
  n_cells_random = "auto",  -- on them
}

finder = {
  -- In comoving Mpc/h. ladder(r_min, r_max, step): log-spaced, 5% apart
  -- here. Also linspace(a, b, n) and geomspace(a, b, n), or any list of
  -- numbers. The smallest should span two grid cells or more, and must
  -- span one: the box is padded for cells no larger.
  radii = ladder(10, 40, 0.05),
  threshold = -0.7,          -- the density contrast a void is grown to
  -- The finder: one of two. excursion-set grows voids from every cell
  -- below the threshold, with no overlap -- the voids of the excursion-set
  -- models, and the finder of sif-exodus 0.4. excursion-peak centres them
  -- on minima of the density smoothed at each rung's scale, a void refused
  -- only by a larger one it sits inside; its void size function depends on
  -- the grid less.
  -- Only excursion-set runs on a survey, for now.
  variant = "excursion-set",
  -- The settings a variant fixes, to change one or to set them all
  -- without a variant (they then start from excursion-set's). Each one
  -- given overrides the variant's.
  -- exodus_params = {
  --   overlap_fraction = 0,   -- of the smaller radius; 0: no overlap
  --   search_factor = 1.5,    -- 1.25 | 1.5 | 1.75 | 2
  --   seeds = "cells",        -- cells | minima: periodic boxes only, for now
  --   -- How the grid's assignment window is undone before smoothing: full
  --   -- divides it out (boosting aliasing and shot noise near the grid
  --   -- scale); aliasing divides out the window with its aliases (least
  --   -- squares); none keeps it.
  --   deconvolution = "full",
  -- },
}

-- Voids to start from: the run adds its own and writes both, these first.
-- To extend a catalogue to smaller radii, run the rungs below the ones it
-- was found with (the search box follows the ladder, so this is close to,
-- not the same as, one run over both). Read like output, by its name; an
-- ASCII catalogue from another finder names its columns: "* x y z * r" (or
-- "* ra dec z * r"). On the sky, it is placed with input.cosmology.
-- prior = { path = "large_voids.h5", format = "hdf5" }

-- .h5 or .hdf5: HDF5; .fits: FITS; anything else: ASCII. Each records
-- the run's settings alongside the voids, and each void's footprint and
-- footprint_shell: the share of its sphere, and of the shell out to twice
-- its radius, inside the survey.
output = "voids.fits"
-- output = { path = "voids.fits", format = "fits" }  -- hdf5 | ascii | fits

run = {
  threads = "auto",
  log_level = "info",  -- trace | debug | info | warning | error | none
  tune_fft = true,     -- false: estimated FFT plans, quicker to set up
  -- Progress, saved after every rung; a run that finds it resumes from it,
  -- so a job resubmitted as it was picks up where it stopped. Removed when
  -- the run completes. Default: the output's path + ".ckpt". false: none.
  -- checkpoint = "voids.fits.ckpt",
  -- Keep the grid on disk rather than its transform in memory: about half
  -- the memory, one more FFT and one read of the grid per smoothing, the
  -- same catalogue. The file goes in the sif setting scratch_directory,
  -- or $SIF_SCRATCH_DIRECTORY for one job; put it on a fast filesystem.
  save_memory = false,
}
