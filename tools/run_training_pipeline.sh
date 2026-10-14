#!/usr/bin/env bash
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# The emulator training pipeline, end to end.
#
# Four training campaigns share one set of 64 cosmologies: campaign 1 (runs/,
# walks placed by a nu window, origins below ~50 Mpc/h) and campaigns 2-4
# (runs_v2/ .. runs_v4/, walks placed by their origin: 2-700, 120-250 and
# 30-120 Mpc/h; training keeps origins below 300). Spectra are made locally;
# the Monte Carlo runs on a cluster; training and validation run anywhere.
# See training_README.md.
#
# Each stage is idempotent and skips itself if its output already exists, so
# re-running after a failure resumes rather than restarting. Force a stage with
#   FORCE=1 ./run_training_pipeline.sh <stage>
#
# Usage:
#   ./run_training_pipeline.sh spectra    # 1. P(k) tables from CAMB
#   ./run_training_pipeline.sh submit     # 2. show the sbatch lines
#   ./run_training_pipeline.sh qa         # 3. check the returned training set
#   ./run_training_pipeline.sh train      # 4. fit
#   ./run_training_pipeline.sh validate   # 5. fresh-ensemble validation
#   ./run_training_pipeline.sh export     # 6. C weights and test cases
#   ./run_training_pipeline.sh studies    # 7. the supporting analyses
#   ./run_training_pipeline.sh all        # everything runnable locally

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
RUNS="${RUNS:-../runs}"
RUNS_V2="${RUNS_V2:-../runs_v2}"
RUNS_V3="${RUNS_V3:-../runs_v3}"
RUNS_V4="${RUNS_V4:-../runs_v4}"
SPECTRA="${SPECTRA:-ep_spectra.npz}"
SPECTRA_V2="${SPECTRA_V2:-ep_spectra_v2.npz}"
WEIGHTS="${WEIGHTS:-ep_emulator.npz}"
HIDDEN="${HIDDEN:-48,48}"
VALMC="${VALMC:-ep_validation_mc.npz}"
VAL_V2="${VAL_V2:-../val_v2}"
PLOTS="${PLOTS:-plots}"

have() { [[ -s "$1" && "${FORCE:-0}" != "1" ]]; }
say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }

stage_spectra() {
    say "1. P(k) tables (CAMB, local, ~10 min)"
    # Campaign 1's table (N_K = 2400) is what runs/ was produced with; it is
    # kept, not regenerated, so the old curves stay reproducible.
    if have "$SPECTRA_V2"; then
        echo "   $SPECTRA_V2 exists; skipping (FORCE=1 to regenerate)"
    else
        $PYTHON ep_make_spectra.py "$SPECTRA_V2"
    fi
    if have ep_validation_spectra.npz; then
        echo "   ep_validation_spectra.npz exists; skipping"
    else
        $PYTHON ep_validate_v2.py spectra
    fi
}

stage_submit() {
    say "2. the cluster arrays"
    cat <<EOF
   Copy tools/ (with $SPECTRA_V2 and ep_validation_spectra.npz) to the
   cluster, and from the directory above it:

     mkdir -p logs
     sbatch --export=ALL,CAMPAIGN=2 tools/ep_trainset.slurm    # campaigns 2-4
     sbatch tools/ep_validate_v2.slurm                         # validation

   Each campaign is 64 tasks of ~25 min on a 72-core node. Where running jobs
   are capped per user (raven: 16), pack four tasks per job instead:

     sbatch --nodes=4 --array=0-15 --export=ALL,FIRST=0,LAST=63,\\
       CMD='python tools/ep_trainset_worker.py --campaign 2 --task-id \$TASK
            --spectra tools/ep_spectra_v2.npz --out runs_v2' tools/ep_packed.slurm

   and the same with --campaign 3 / --out runs_v3 and --campaign 4 / runs_v4.

   Campaign 1 (runs/) is reproduced with plain 'sbatch tools/ep_trainset.slurm'
   and $SPECTRA. Bring back runs_v2/ .. runs_v4/ and val_v2/.
EOF
}

stage_qa() {
    say "3. quality assurance on the returned training set"
    [[ -d "$RUNS" ]] || { echo "   no $RUNS directory; run the array first"; exit 1; }
    mkdir -p "$PLOTS"
    $PYTHON ep_trainset_load.py "$RUNS" "$RUNS_V2" "$RUNS_V3" "$RUNS_V4"
    $PYTHON ep_qa.py "$RUNS" --plot "$PLOTS/qa.png"
}

stage_train() {
    say "4. training (4-fold CV, then the final fit)"
    if have "$WEIGHTS"; then
        echo "   $WEIGHTS exists; skipping (FORCE=1 to retrain)"
        return
    fi
    $PYTHON ep_train_final.py "$RUNS" "$RUNS_V2" "$RUNS_V3" "$RUNS_V4" \
        --hidden "$HIDDEN" -o "$WEIGHTS" --json ep_train_cv.json
}

stage_validate() {
    say "5. validation against a fresh Monte Carlo"
    [[ -d "$VAL_V2" ]] || { echo "   no $VAL_V2; run ep_validate_v2.slurm first"; exit 1; }
    $PYTHON ep_validate_v2.py analyse --out "$VAL_V2" --weights "$WEIGHTS"
    $PYTHON ep_validate_v2.py invariance --weights "$WEIGHTS"
}

stage_export() {
    say "6. C weights and the stored test cases"
    $PYTHON ep_export_c.py "$WEIGHTS" -o ../src/model/ep_emu_weights.h
    $PYTHON ep_export_test_cases.py --weights "$WEIGHTS" --mc "$VALMC" \
        -o ../tests/ep_emu_cases.h
}

stage_studies() {
    say "7. the supporting studies (evidence, not required to train)"
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
    export)   stage_export ;;
    studies)  stage_studies ;;
    all)      stage_spectra; stage_submit; stage_qa; stage_train
              stage_validate; stage_export ;;
    *)        echo "unknown stage: $1"; sed -n '20,28p' "$0"; exit 1 ;;
esac
