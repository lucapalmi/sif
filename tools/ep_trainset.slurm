#!/bin/bash
#
# Training-set generation for the excursion-set multiplicity emulator.
#
# One array task per cosmology; every barrier of that cosmology runs inside the
# task, so the covariance is factorized once and a failed task costs one
# cosmology rather than a slice of all of them.
#
# Throttled to 16 concurrent tasks (%16) to match the queue limit. With 64
# cosmologies that is four waves.
#
# Submit:
#   sbatch tools/ep_trainset.slurm
# Check the projected cost on one node first (STRONGLY recommended, see below):
#   sbatch --array=0 --export=ALL,BENCHMARK=1 tools/ep_trainset.slurm
#
#SBATCH --job-name=ep-trainset
#SBATCH --array=0-63%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=logs/ep_trainset_%A_%a.out
#SBATCH --error=logs/ep_trainset_%A_%a.err
#
# Uncomment and set for your site:
##SBATCH --partition=CHANGEME
##SBATCH --account=CHANGEME

set -euo pipefail

# --- knobs -------------------------------------------------------------------

# 72 physical cores. Set to 144 to use hyperthreading: the walk is a chain of
# short dependent dot products, so it does get something from SMT, but measure
# with BENCHMARK=1 before assuming it -- past the point where the factor rows
# stop fitting in L2, extra threads cost more in bandwidth than they return.
: "${OMP_NUM_THREADS:=${SLURM_CPUS_PER_TASK:-72}}"
export OMP_NUM_THREADS

# Keep threads on their own cores; the per-path buffers are hot and migrating
# them across sockets is pure loss.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

SPECTRA="${SPECTRA:-ep_spectra.npz}"
OUTDIR="${OUTDIR:-runs}"
NPATHS="${NPATHS:-1000000000}"

# --- environment -------------------------------------------------------------
# Replace with your site's module names. sif needs an OpenMP-capable compiler
# at BUILD time; at run time the extension only needs its libomp on the path.

source /u/lpalmieri/env/bin/activate

# --- run ---------------------------------------------------------------------

mkdir -p "$OUTDIR" logs

echo "host        $(hostname)"
echo "task        ${SLURM_ARRAY_TASK_ID:-0} of ${SLURM_ARRAY_TASK_COUNT:-?}"
echo "threads     $OMP_NUM_THREADS"
echo "paths/curve $NPATHS"
echo "started     $(date -Is)"

if [[ "${BENCHMARK:-0}" == "1" ]]; then
    # Measures throughput on this node and prints the projected cost of the
    # whole array. Costs a couple of seconds and writes nothing. Run this
    # before committing the full job: the paths/s below is the only number
    # that turns the design into a wall-clock estimate, and it is the one
    # thing that cannot be known off-cluster.
    python tools/ep_trainset_worker.py \
        --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
        --spectra "$SPECTRA" \
        --n-paths "$NPATHS" \
        --benchmark
    exit 0
fi

python tools/ep_trainset_worker.py \
    --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
    --spectra "$SPECTRA" \
    --out "$OUTDIR" \
    --n-paths "$NPATHS"

echo "finished    $(date -Is)"
