-- sif-exodus configuration.
--
-- A Lua script: settings are name = value, grouped in sections, and a
-- value can be anything Lua computes. Only input, finder.radii,
-- finder.threshold and output are required; every other setting below
-- is at its default and can be left out.
--
-- Command-line variables arrive as strings: `sif-exodus -D snap=010`
-- makes `snap` here "010", and `snap = snap or "010"` gives it a
-- default. `sif-exodus --check` shows what a file resolves to.

input = {
  path = "tracers.xfield",
  -- xfield | ascii | binary | gadget | fits; an .xfield or a .fits file
  -- needs none. A FITS catalogue may be a list of files, read as one:
  -- path = { "part_0.fits", "part_1.fits" }.
  format = "xfield",
  -- The periodic box. Recorded by .xfield and GADGET files, which a
  -- value here overrides; ascii, binary and FITS files need it.
  -- box_length = 1000,

  -- One table for the format being read. Columns are x y z, w for a
  -- weight, * for a column to skip: "* x y z w".
  -- ascii = { columns = "x y z", delimiter = " ", skip_header = 0 },
  -- binary = {
  --   columns = "x y z",
  --   precision = "float32",  -- float32 | float64: required
  --   layout = "rows",        -- rows (x0 y0 z0 x1 ...) | blocks (x0 x1 ... y0 ...)
  --   endian = "native",      -- native | little | big
  --   header_bytes = 0,
  -- },
  -- gadget = {
  --   snapformat = "auto",    -- 1 | 2 | 3 (HDF5) | "auto"
  --   ptype = 1,
  --   length = "auto",        -- kpc | mpc | auto (HDF5 files only)
  --   masses = false,         -- true: particle masses as weights
  --   fraction = 1,           -- a random subsample, in (0, 1]
  --   seed = 0,
  -- },
  -- fits = {
  --   -- A column, or an expression over columns: w = "W1 * W2",
  --   -- x = "POS[1]" for an element of a vector column.
  --   columns = { x = "X", y = "Y", z = "Z" },  -- and w, for a weight
  --   where = "Z > 0",        -- rows to keep; nil keeps every row
  --   hdu = 1,                -- the table: EXTNAME or number; nil: the first
  --   fraction = 1,           -- a random subsample of the kept rows
  --   seed = 0,
  -- },
}

grid = {
  n_cells = "auto",  -- per side; auto: one cell per mean separation
}

mesh = {
  n_cells = "auto",  -- per side; only speed and memory depend on it
}

finder = {
  -- ladder(r_min, r_max, step): log-spaced, 5% apart here. Also
  -- linspace(a, b, n) and geomspace(a, b, n), or any list of numbers.
  radii = ladder(10, 40, 0.05),
  radii_units = "physical",  -- physical (the box's) | mps (mean separations)
  threshold = -0.7,          -- the density contrast a void is grown to
  overlap_fraction = 0,      -- of the smaller radius; 0: no overlap
  search_factor = 1.5,       -- 1.25 | 1.5 | 1.75 | 2
}

-- .h5 or .hdf5: HDF5, and .fits: FITS, both with the run's settings;
-- anything else: ASCII.
output = "voids.h5"
-- output = { path = "voids.h5", format = "hdf5" }  -- hdf5 | ascii | fits

run = {
  threads = "auto",
  log_level = "info",  -- trace | debug | info | warning | error | none
  tune_fft = true,     -- false: estimated FFT plans, quicker to set up
}
