# Sif void library

W.I.P

## License

Copyright (C) 2026 Luca Palmieri.

sif is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

sif is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

The full license text is in [COPYING](COPYING). Every source file carries an
`SPDX-License-Identifier: GPL-3.0-or-later` tag.

Note that this applies to the Python bindings as well: `pysif` links the same
GPL-licensed code, so anything distributed against it is bound by the same
terms.

## Third-party components

**Robust geometric predicates** — sif uses Jonathan
Richard Shewchuk's implementation of adaptive-precision floating-point
arithmetic and incircle/insphere tests (Carnegie Mellon University,
<https://www.cs.cmu.edu/~quake/robust.html>). The accompanying
`predicates.h` is not part of the original distribution.

**FFTW3** — sif links FFTW 3 (<https://www.fftw.org/>) for its transforms. FFTW
is not bundled; it is located at configure time and must be installed
separately. FFTW is licensed under the GNU General Public License, version 2 or
later.

**OpenMP** — sif is parallelized with OpenMP and links the runtime provided by
the compiler (LLVM's `libomp` or GCC's `libgomp`) under its
own license.
