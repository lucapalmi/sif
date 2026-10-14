#!/bin/bash
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# Runs several single-node tasks inside one multi-node job, one per node.
#
# Why: on raven a user may have 16 jobs running at once
# (QOSMaxJobsPerUserLimit), whatever their size, which turns a 64-task array
# of single-node jobs into four waves. The cap counts jobs, not nodes, so
# packing four tasks per job runs four times as much at once while each task
# still owns a node.
#
# Array element a runs tasks FIRST + a * NNODES + k for k < NNODES. CMD is the
# command for one task, with $TASK substituted. Example (campaign 2, tasks
# 16..63 as 12 four-node jobs):
#
#   sbatch --nodes=4 --array=0-11 --export=ALL,FIRST=16,LAST=63,\
#   CMD='python tools/ep_trainset_worker.py --campaign 2 --task-id $TASK
#        --spectra tools/ep_spectra_v2.npz --out runs_v2' tools/ep_packed.slurm
#
#SBATCH --job-name=ep-packed
#SBATCH --partition=general
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=logs/ep_packed_%A_%a.out
#SBATCH --error=logs/ep_packed_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-72}" OMP_PROC_BIND=close OMP_PLACES=cores
source /u/lpalmieri/env/bin/activate
mkdir -p logs

: "${CMD:?set CMD to the per-task command, with \$TASK for the task id}"
FIRST="${FIRST:-0}"
NN="${SLURM_NNODES:-1}"

for k in $(seq 0 $((NN - 1))); do
    TASK=$((FIRST + ${SLURM_ARRAY_TASK_ID:-0} * NN + k))
    if [[ -n "${LAST:-}" && "$TASK" -gt "$LAST" ]]; then
        continue
    fi
    srun --nodes=1 --ntasks=1 --exact \
        --output="logs/ep_packed_%A_%a_task${TASK}.out" \
        bash -c "TASK=$TASK; $CMD" &
done
wait
