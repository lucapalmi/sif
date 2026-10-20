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
  overlap_fraction = 0,      -- of the smaller radius; 0: no overlap
  search_factor = 1.5,       -- 1.25 | 1.5 | 1.75 | 2
  -- A centre at a minimum of the smoothed field goes where the minimum lies
  -- between the grid nodes; false leaves every centre on its node.
  centre_refinement_hessian = true,
  -- true: only those minima are candidates, so every void is centred on one
  -- (with overlap_fraction = 1, the excursion-set definition of a void).
  centre_is_minimum = false,
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
}
