#!/usr/bin/env bash
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The emulator training pipeline, end to end.
#
# Stages 1 and 2 run locally; stage 3 runs on a cluster and is the only
# expensive one (~20 node-hours). Stages 4-6 run locally on its output.
#
# Each stage is idempotent and skips itself if its output already exists, so
# re-running after a failure resumes rather than restarting. Force a stage with
#   FORCE=1 ./run_training_pipeline.sh <stage>
#
# Usage:
#   ./run_training_pipeline.sh spectra    # 1. P(k) tables from CAMB
#   ./run_training_pipeline.sh submit     # 2. show the sbatch line
#   ./run_training_pipeline.sh qa         # 3. check the returned training set
#   ./run_training_pipeline.sh train      # 4. fit and export weights
#   ./run_training_pipeline.sh validate   # 5. fresh-ensemble validation
#   ./run_training_pipeline.sh studies    # 6. the supporting analyses
#   ./run_training_pipeline.sh all        # everything runnable locally

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
RUNS="${RUNS:-../runs}"
SPECTRA="${SPECTRA:-ep_spectra.npz}"
WEIGHTS="${WEIGHTS:-ep_emulator.npz}"
VALMC="${VALMC:-ep_validation_mc.npz}"
PLOTS="${PLOTS:-plots}"

have() { [[ -s "$1" && "${FORCE:-0}" != "1" ]]; }
say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }

stage_spectra() {
    say "1. P(k) tables (CAMB, local, ~4 min)"
    if have "$SPECTRA"; then
        echo "   $SPECTRA exists; skipping (FORCE=1 to regenerate)"
        return
    fi
    $PYTHON ep_make_spectra.py "$SPECTRA"
}

stage_submit() {
    say "2. the cluster array"
    cat <<EOF
   Copy this directory and $SPECTRA to the cluster, then:

     sbatch --array=0 --export=ALL,BENCHMARK=1 tools/ep_trainset.slurm   # check cost
     sbatch tools/ep_trainset.slurm                                      # the real run

   Fill in partition/account/modules in ep_trainset.slurm first. The array is
   64 tasks (one cosmology each, 24 barriers inside), throttled to 16 at a
   time, about 20 node-hours in total. Bring back the runs/ directory.
EOF
}

stage_qa() {
    say "3. quality assurance on the returned training set"
    [[ -d "$RUNS" ]] || { echo "   no $RUNS directory; run the array first"; exit 1; }
    mkdir -p "$PLOTS"
    $PYTHON ep_trainset_load.py "$RUNS"
    $PYTHON ep_qa.py "$RUNS" --plot "$PLOTS/qa.png"
}

stage_train() {
    say "4. training and weight export"
    if have "$WEIGHTS"; then
        echo "   $WEIGHTS exists; skipping (FORCE=1 to retrain)"
        return
    fi
    $PYTHON ep_train_final.py "$RUNS" -o "$WEIGHTS"
}

stage_validate() {
    say "5. validation against a fresh Monte Carlo"
    if ! have "$VALMC"; then
        echo "   generating a fresh ensemble (~26 min, 24 curves at 1e8 paths)"
        $PYTHON ep_validate.py --stage mc --mc "$VALMC"
    else
        echo "   $VALMC exists; reusing (FORCE=1 to regenerate)"
    fi
    $PYTHON ep_validate.py --stage analyse --mc "$VALMC" --weights "$WEIGHTS"
}

stage_studies() {
    say "6. the supporting studies (evidence, not required to train)"
    mkdir -p "$PLOTS"
    echo "-- feature ablation at full statistics"
    $PYTHON ep_ablation_full.py "$RUNS"
    echo "-- where the residual error lives"
    $PYTHON ep_lownu_diagnosis.py "$RUNS" --plot "$PLOTS/lownu.png"
}

case "${1:-all}" in
    spectra)  stage_spectra ;;
    submit)   stage_submit ;;
    qa)       stage_qa ;;
    train)    stage_train ;;
    validate) stage_validate ;;
    studies)  stage_studies ;;
    all)      stage_spectra; stage_submit; stage_qa; stage_train
              stage_validate ;;
    *)        echo "unknown stage: $1"; sed -n '12,22p' "$0"; exit 1 ;;
esac
