#!/usr/bin/env bash
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The C libraries a pysif wheel carries -- FFTW and HDF5 -- built from pinned
# upstream sources into one prefix. cibuildwheel runs this once per platform
# (see [tool.cibuildwheel] in pyproject.toml); the wheel build then links
# against the prefix, and auditwheel/delocate copy the libraries into the
# wheel. A build from source (pip install .) never runs it: that one uses
# whatever FFTW and HDF5 the machine has.
#
# Built rather than taken from the system's packages because a wheel runs on
# machines other than the one that built it, and has to be:
#   - portable: no -march=native, and FFTW's SIMD kernels chosen at run time
#     (FFTW compiles each kernel set in files of its own and checks the CPU
#     before using one; its configure adds -mtune=native, which only tunes);
#   - old enough: on macOS, built for MACOSX_DEPLOYMENT_TARGET, where the
#     system's libraries target the machine that built them;
#   - lean: HDF5's C library and the deflate filter only, no tools, no
#     high-level or C++ libraries, nothing fetched at build time.
#
# And the OpenMP runtime. On Linux that is GCC's libgomp, which the system
# compiler already provides and auditwheel bundles; only its license text is
# collected here. On macOS, Apple's compiler has no runtime at all, and
# Homebrew's targets the macOS it was built on, so LLVM's libomp is built here
# too -- from the LLVM source release, of which only openmp/ is compiled.
#
# Usage: build-deps.sh PREFIX
# Leaves PREFIX/{include,lib} and PREFIX/licenses, the upstream license texts
# that go into the wheel next to sif's own.

set -euo pipefail

PREFIX=${1:?usage: build-deps.sh PREFIX}

FFTW_VERSION=3.3.10
FFTW_URL=https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz
FFTW_SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467

HDF5_VERSION=2.2.0
HDF5_URL=https://github.com/HDFGroup/hdf5/releases/download/${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz
HDF5_SHA256=1a1ab8209b35586fbc1aa279ba76d102130b95badcb20ca329587219112d8c16

LLVM_VERSION=23.1.2
LLVM_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz
LLVM_SHA256=c98bbef08a2b4c2613cd50e9aa9ae7b69b1fe6c16b2c40373bc0ab6116fdf78a

# The oldest macOS the wheel runs on; ignored elsewhere. arm64 Macs start at
# 11. cibuildwheel sets it too, from the same value in pyproject.toml.
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}

JOBS=$(getconf _NPROCESSORS_ONLN)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

fetch() { # url sha256 [paths...] -> unpacked in $WORK, only paths if given
  local url=$1 sha=$2 file="$WORK/${1##*/}"
  shift 2
  curl -fsSL --retry 3 -o "$file" "$url"
  if command -v sha256sum >/dev/null; then
    echo "$sha  $file" | sha256sum -c -
  else
    echo "$sha  $file" | shasum -a 256 -c -
  fi
  tar xf "$file" -C "$WORK" "$@"
}

# On macOS, a library installed with an @rpath install name is found by
# nothing that lacks the rpath -- delocate included, once the build's rpath
# is stripped from the installed extension. The absolute path always works,
# and delocate rewrites it when it copies the library into the wheel.
absolute_ids() {
  [ "$(uname -s)" = Darwin ] || return 0
  local lib
  for lib in "$PREFIX"/lib/*.dylib; do
    [ -L "$lib" ] || install_name_tool -id "$lib" "$lib"
  done
}

mkdir -p "$PREFIX/licenses"

# --- FFTW: double and single precision, each with its pthreads library ---
#
# The pthreads backend (SIF_FFTW_THREADING=threads) rather than OpenMP: it
# needs nothing from the compiler, and keeps FFTW out of the question of
# which OpenMP runtime the wheel carries.

case "$(uname -m)" in
  x86_64) FFTW_SIMD="--enable-sse2 --enable-avx --enable-avx2" ;;
  aarch64) FFTW_SIMD="--enable-neon" ;;
  # FFTW's config.guess predates Apple silicon and calls it 32-bit ARM, whose
  # NEON has no double precision; say what the machine is.
  arm64) FFTW_SIMD="--enable-neon --build=aarch64-apple-darwin" ;;
  *) FFTW_SIMD="" ;;
esac

fetch "$FFTW_URL" "$FFTW_SHA256"
for precision in double float; do
  flags=""
  [ "$precision" = float ] && flags="--enable-float"
  (
    cd "$WORK/fftw-$FFTW_VERSION"
    # shellcheck disable=SC2086 # the flag lists are meant to split
    ./configure --prefix="$PREFIX" --enable-shared --disable-static \
      --enable-threads --disable-fortran --disable-doc \
      $FFTW_SIMD $flags >/dev/null
    make -j "$JOBS" >/dev/null
    make install >/dev/null
    make distclean >/dev/null
  )
done
cp "$WORK/fftw-$FFTW_VERSION/COPYING" "$PREFIX/licenses/FFTW.txt"

# --- HDF5: the serial C library, with the deflate filter ---
#
# Deflate so that files other tools wrote compressed (h5py's
# compression="gzip") still read; zlib comes from the system, which every
# manylinux and macOS machine has. HDF5_ALLOW_EXTERNAL_SUPPORT=NO stops the
# build from downloading its own.

fetch "$HDF5_URL" "$HDF5_SHA256"
cmake -S "$WORK/hdf5-$HDF5_VERSION" -B "$WORK/hdf5-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DBUILD_TESTING=OFF \
  -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_HL_LIB=OFF \
  -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF -DHDF5_BUILD_JAVA=OFF \
  -DHDF5_BUILD_DOC=OFF -DHDF5_ENABLE_PARALLEL=OFF \
  -DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
  -DHDF5_ALLOW_EXTERNAL_SUPPORT=NO >/dev/null
cmake --build "$WORK/hdf5-build" -j "$JOBS" >/dev/null
cmake --install "$WORK/hdf5-build" >/dev/null
cp "$WORK/hdf5-$HDF5_VERSION/LICENSE" "$PREFIX/licenses/HDF5.txt"
absolute_ids

# --- the OpenMP runtime ---

if [ "$(uname -s)" = Darwin ]; then
  # The runtimes build is the only one LLVM supports for openmp/; it wants
  # the shared cmake/ modules, and the unit-test sources merely to configure.
  src="$WORK/llvm-project-$LLVM_VERSION.src"
  fetch "$LLVM_URL" "$LLVM_SHA256" \
    "${src##*/}/openmp" "${src##*/}/runtimes" "${src##*/}/cmake" \
    "${src##*/}/llvm/cmake" "${src##*/}/llvm/utils/llvm-lit" \
    "${src##*/}/third-party"
  cmake -S "$src/runtimes" -B "$WORK/omp-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DLLVM_ENABLE_RUNTIMES=openmp \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_DOCS=OFF \
    -DLIBOMP_INSTALL_ALIASES=OFF -DLIBOMP_OMPD_SUPPORT=OFF \
    -DLIBOMP_USE_HWLOC=OFF >/dev/null
  cmake --build "$WORK/omp-build" -j "$JOBS" >/dev/null
  cmake --install "$WORK/omp-build" >/dev/null
  cp "$src/openmp/LICENSE.TXT" "$PREFIX/licenses/LLVM-OpenMP.txt"
  absolute_ids
else
  # libgomp: GPL-3.0 with the GCC Runtime Library Exception, which is what
  # lets a program of any license ship it.
  cat /usr/share/licenses/libgcc/COPYING.RUNTIME \
    /usr/share/licenses/libgcc/COPYING3 > "$PREFIX/licenses/GCC-libgomp.txt"
fi

echo "build-deps: FFTW $FFTW_VERSION, HDF5 $HDF5_VERSION and the OpenMP runtime in $PREFIX"
