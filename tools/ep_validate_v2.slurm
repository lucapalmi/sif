#!/bin/bash
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# Fresh Monte Carlo for ep_validate_v2.py: tasks 0-7 the generic set V (one
# cosmology each), 8 the thesis-barrier set T, 9-12 the convergence set C,
# 13-20 the far-origin set W.
# Needs tools/ep_validation_spectra.npz, made locally with
#   python tools/ep_validate_v2.py spectra
#
#SBATCH --job-name=ep-valid2
#SBATCH --partition=general
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=logs/ep_valid2_%A_%a.out
#SBATCH --error=logs/ep_valid2_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-72}" OMP_PROC_BIND=close OMP_PLACES=cores
source /u/lpalmieri/env/bin/activate
mkdir -p logs
python tools/ep_validate_v2.py mc --task "${SLURM_ARRAY_TASK_ID:-0}" --out "${OUTDIR:-val_v2}"
