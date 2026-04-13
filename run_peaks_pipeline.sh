#!/usr/bin/env bash
# =============================================================================
# nnUNet Peaks Pipeline — end-to-end run script for diffusion MRI peak data
#
# Peak data = 9-channel fiber orientation maps (3 peaks × x/y/z component),
# stored as separate NIfTI files _0000.nii.gz ... _0008.nii.gz per subject.
#
# Differences from the standard multilabel pipeline:
#   --peaks    sets channel_names→"nonorm" in dataset.json; auto-selects
#              nnUNetTrainerPeaks at training time (vector-aware augmentation:
#              rotation with R applied to vectors, PeakMirrorTransform, no
#              intensity augmentations). INDEPENDENT of --multilabel.
#   --multilabel sets multilabel=true in dataset.json (sigmoid loss, 4D labels).
#              INDEPENDENT of --peaks.
#   Both flags combined = peaks input + multilabel tractography targets (typical).
#   resampling order=1 set automatically when all channel names are "nonorm".
#
# Usage:
#   1. Adapt the CONFIG section below.
#   2. Make executable:  chmod +x run_peaks_pipeline.sh
#   3. Full pipeline:    ./run_peaks_pipeline.sh
#      Single step:      ./run_peaks_pipeline.sh preprocess
#
# Available steps:  preprocess | train | predict | evaluate | all (default)
#
# ─── PREREQUISITE ────────────────────────────────────────────────────────────
# The raw dataset.json must list ALL 9 channel_names before running preprocess.
# If it only has "0": "FA" (copied from the FA dataset), fix it first:
#
#   python - <<'EOF'
#   import json, sys
#   path = "<DATASET_ROOT>/nnunet_raw/Dataset005_MainTracts_Multilabel_Peaks/dataset.json"
#   with open(path) as f: dj = json.load(f)
#   dj["channel_names"] = {str(i): "nonorm" for i in range(9)}   # 9 peak channels
#   dj["multilabel"] = True
#   with open(path, "w") as f: json.dump(dj, f, indent=2)
#   print("Updated", path)
#   EOF
# =============================================================================
set -euo pipefail

# =============================================================================
# CONFIG — adapt these for every new dataset / experiment
# =============================================================================

# Use the FULL dataset name to avoid any ambiguity when multiple Dataset004_*
# folders exist in nnunet_raw. nnUNetv2 accepts -d <name> OR -d <int-id>.
DATASET_NAME="Dataset005_MainTracts_Multilabel_Peaks"  # rename dir if still 004!
CONFIGURATION="3d_fullres"            # nnUNet config: 2d | 3d_fullres | 3d_lowres
FOLD=0                                # cross-val fold (0–4 or "all")
TRAINER="nnUNetTrainerPeaks"          # peaks-aware trainer (auto-selected by --peaks)
PLANS="nnUNetPlans"                   # plans identifier
CHECKPOINT="checkpoint_best.pth"      # checkpoint for prediction / evaluation

DATASET_ROOT="/home/r948e/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/Robins_runs"

# nnUNet environment paths
RESULTS_DIR="${DATASET_ROOT}/nnunet_results"

# Prediction input / output
IMAGES_TS="${DATASET_ROOT}/nnunet_raw/${DATASET_NAME}/imagesTs"
PRED_DIR="${DATASET_ROOT}/results/${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIGURATION}/fold_${FOLD}/predictions_test"

# Evaluation
LABELS_TS="${DATASET_ROOT}/nnunet_raw/${DATASET_NAME}/labelsTs"
DATASET_JSON="${DATASET_ROOT}/nnunet_raw/${DATASET_NAME}/dataset.json"
PLANS_JSON="${DATASET_ROOT}/results/${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIGURATION}/plans.json"

# Hardware
DEVICE="cuda"
NUM_GPUS=1

# Preprocessing parallelism
NP_PREPROCESS="4"
PREPROCESS_CONFIGS="${CONFIGURATION}"

# =============================================================================
# Environment setup
# =============================================================================

source /home/r948e/dev/nnUNet/nnunet_env.sh
export nnUNet_raw="${DATASET_ROOT}/nnunet_raw"
export nnUNet_preprocessed="${DATASET_ROOT}/nnunet_preprocessed"
export nnUNet_results="${RESULTS_DIR}"
export nnUNet_compile=0   # disable torch.compile (Triton/gcc header mismatch)

echo "============================================================"
echo "  nnUNet Peaks Pipeline"
echo "  Dataset     : ${DATASET_NAME}"
echo "  Config      : ${CONFIGURATION}"
echo "  Fold        : ${FOLD}"
echo "  Trainer     : ${TRAINER}"
echo "  Results dir : ${RESULTS_DIR}"
echo "============================================================"

STEP="${1:-all}"

# =============================================================================
# Plan & Preprocess
# =============================================================================
run_preprocess() {
    echo ""
    echo ">>> [1/4] Plan & Preprocess (--peaks --multilabel)"
    echo "    --peaks      : channel_names->nonorm (NoNorm + linear resampling)"
    echo "    --multilabel : multilabel=true in dataset.json (sigmoid loss, 4D labels)"
    nnUNetv2_plan_and_preprocess \
        -d "${DATASET_NAME}" \
        -c ${PREPROCESS_CONFIGS} \
        -np "${NP_PREPROCESS}" \
        --peaks \
        --multilabel \
        --verify_dataset_integrity
    echo "    Done."
}

# =============================================================================
# Train
# =============================================================================
run_train() {
    echo ""
    echo ">>> [2/4] Train"
    echo "    Trainer : ${TRAINER}  Augmentation: rotation+scale+mirror (peak-aware)"
    nnUNetv2_train \
        "${DATASET_NAME}" \
        "${CONFIGURATION}" \
        "${FOLD}" \
        -tr "${TRAINER}" \
        -p "${PLANS}" \
        -num_gpus "${NUM_GPUS}" \
        -device "${DEVICE}" \
        --peaks \
        --multilabel
    echo "    Training done."
}

# =============================================================================
# Predict
# =============================================================================
run_predict() {
    echo ""
    echo ">>> [3/4] Predict"
    if [[ -z "${IMAGES_TS}" || -z "${PRED_DIR}" ]]; then
        echo "    ERROR: Set IMAGES_TS and PRED_DIR in the CONFIG section."
        exit 1
    fi
    mkdir -p "${PRED_DIR}"
    nnUNetv2_predict \
        -i "${IMAGES_TS}" \
        -o "${PRED_DIR}" \
        -d "${DATASET_NAME}" \
        -c "${CONFIGURATION}" \
        -tr "${TRAINER}" \
        -p "${PLANS}" \
        -f "${FOLD}" \
        -chk "${CHECKPOINT}" \
        -device "${DEVICE}" \
        --multilabel
    echo "    Predictions written to: ${PRED_DIR}"
}

# =============================================================================
# Evaluate
# =============================================================================
run_evaluate() {
    echo ""
    echo ">>> [4/4] Evaluate"
    if [[ -z "${LABELS_TS}" || -z "${PRED_DIR}" || -z "${DATASET_JSON}" || -z "${PLANS_JSON}" ]]; then
        echo "    ERROR: Set LABELS_TS, PRED_DIR, DATASET_JSON, PLANS_JSON in CONFIG."
        exit 1
    fi
    nnUNetv2_evaluate_folder \
        "${LABELS_TS}" \
        "${PRED_DIR}" \
        -djfile "${DATASET_JSON}" \
        -pfile  "${PLANS_JSON}" \
        --multilabel
    echo "    Summary written to: ${PRED_DIR}/summary.json"
}

# =============================================================================
# Dispatch
# =============================================================================
case "${STEP}" in
    preprocess) run_preprocess ;;
    train)      run_train      ;;
    predict)    run_predict    ;;
    evaluate)   run_evaluate   ;;
    all)
        run_preprocess
        run_train
        run_predict
        run_evaluate
        ;;
    *)
        echo "Unknown step '${STEP}'. Use: preprocess | train | predict | evaluate | all"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Pipeline step '${STEP}' finished."
echo "============================================================"
