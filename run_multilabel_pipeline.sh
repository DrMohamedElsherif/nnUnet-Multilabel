#!/usr/bin/env bash
# =============================================================================
# nnUNet Multilabel Pipeline — end-to-end run script
#
# Usage:
#   1. Adapt the CONFIG section below for your dataset.
#   2. Make executable:  chmod +x run_multilabel_pipeline.sh
#   3. Run full pipeline: ./run_multilabel_pipeline.sh
#      Or individual steps, e.g.: ./run_multilabel_pipeline.sh preprocess
#
# Available step names (pass as first argument, or omit to run all):
#   preprocess | train | predict | evaluate | all (default)
# =============================================================================
set -euo pipefail

# =============================================================================
# CONFIG — adapt these for every new dataset / experiment
# =============================================================================

DATASET_ID=4                          # integer dataset ID (e.g. 4 → Dataset004_...)
CONFIGURATION="3d_fullres"            # nnUNet config: 2d | 3d_fullres | 3d_lowres
FOLD=0                                # cross-val fold (0–4 or "all")
TRAINER="nnUNetTrainer"               # trainer class (default: nnUNetTrainer)
PLANS="nnUNetPlans"                   # plans identifier (default: nnUNetPlans)
CHECKPOINT="checkpoint_best.pth"      # checkpoint for prediction/evaluation

DATASET_ROOT="/home/r948e/E132-Projekte/Projects/2025_Peretzke_Elsherif_nnTractSeg/Robins_runs"

# Paths — nnUNet env variables must point to correct locations for this dataset
RESULTS_DIR="${DATASET_ROOT}/nnunet_results"

# Prediction input / output
IMAGES_TS="${DATASET_ROOT}/nnunet_raw/Dataset004_MainTracts_Multilabel/imagesTs"
PRED_DIR="${DATASET_ROOT}/results/Dataset004_MainTracts_Multilabel/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/predictions_test"

# Evaluation
LABELS_TS="${DATASET_ROOT}/nnunet_raw/Dataset004_MainTracts_Multilabel/labelsTs"
DATASET_JSON="${DATASET_ROOT}/nnunet_raw/Dataset004_MainTracts_Multilabel/dataset.json"
PLANS_JSON="${DATASET_ROOT}/results/Dataset004_MainTracts_Multilabel/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"

# Hardware
DEVICE="cuda"                         # cuda | cpu | mps
NUM_GPUS=1                            # set >1 for DDP training

# Preprocessing parallelism
NP_PREPROCESS="4"                     # processes for 3d_fullres preprocessing
PREPROCESS_CONFIGS="${CONFIGURATION}" # configs to preprocess (space-separated)

# =============================================================================
# Environment setup
# =============================================================================

source /home/r948e/dev/nnUNet/nnunet_env.sh
export nnUNet_raw="${DATASET_ROOT}/nnunet_raw"
export nnUNet_preprocessed="${DATASET_ROOT}/nnunet_preprocessed"
export nnUNet_results="${RESULTS_DIR}"
export nnUNet_compile=0              # disable torch.compile (Triton requires matching gcc+python headers)

echo "============================================================"
echo "  nnUNet Multilabel Pipeline"
echo "  Dataset ID  : ${DATASET_ID}"
echo "  Config      : ${CONFIGURATION}"
echo "  Fold        : ${FOLD}"
echo "  Results dir : ${RESULTS_DIR}"
echo "============================================================"

STEP="${1:-all}"

# =============================================================================
# STEP 1 — Plan & Preprocess
# =============================================================================
run_preprocess() {
    echo ""
    echo ">>> [1/4] Plan & Preprocess (--multilabel)"
    echo "    Configs : ${PREPROCESS_CONFIGS}"
    echo "    Processes: ${NP_PREPROCESS}"
    nnUNetv2_plan_and_preprocess \
        -d "${DATASET_ID}" \
        -c ${PREPROCESS_CONFIGS} \
        -np "${NP_PREPROCESS}" \
        --multilabel \
        --verify_dataset_integrity
    echo "    Done. 'multilabel': true written to raw dataset.json."
}

# =============================================================================
# STEP 2 — Train
# =============================================================================
run_train() {
    echo ""
    echo ">>> [2/4] Train"
    echo "    Trainer : ${TRAINER}  Plans: ${PLANS}  Fold: ${FOLD}"
    nnUNetv2_train \
        "${DATASET_ID}" \
        "${CONFIGURATION}" \
        "${FOLD}" \
        -tr "${TRAINER}" \
        -p "${PLANS}" \
        -num_gpus "${NUM_GPUS}" \
        -device "${DEVICE}" \
        --multilabel
    echo "    Training done."
}

# =============================================================================
# STEP 3 — Predict
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
        -d "${DATASET_ID}" \
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
# STEP 4 — Evaluate
# =============================================================================
run_evaluate() {
    echo ""
    echo ">>> [4/4] Evaluate"
    if [[ -z "${LABELS_TS}" || -z "${PRED_DIR}" || -z "${DATASET_JSON}" || -z "${PLANS_JSON}" ]]; then
        echo "    ERROR: Set LABELS_TS, PRED_DIR, DATASET_JSON, and PLANS_JSON in the CONFIG section."
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
