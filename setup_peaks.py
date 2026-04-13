"""
Overview of all nnUNet changes for diffusion MRI peak (fiber orientation) data.

Run to print a summary:
    python setup_peaks.py

Background
----------
Standard nnUNet is designed for scalar MRI images (FA, T1, DWI b0 ...).
Peaks are 3D unit vectors stored per voxel: 3 fiber directions × (x, y, z) = 9 channels.
They require specific handling at every stage of the pipeline:

  Preprocessing   no normalisation (vectors must not be shifted/scaled),
                  linear resampling (cubic interpolation distorts unit vectors)

  Augmentation    rotation MUST also rotate the stored vector components (R @ v),
                  no intensity augmentations (noise, blur, gamma are meaningless),
                  mirroring MUST negate the flipped vector component (sign correction)

  Architecture    --peaks and --multilabel are INDEPENDENT flags.
                  --peaks  = vector-aware preprocessing + trainer (NoNorm, linear resample,
                             PeakSpatialTransform, PeakMirrorTransform)
                  --multilabel = sigmoid + BCE/Dice loss, 4D NIfTI label I/O
                  Use both together for the typical tractography use case.

This module documents every file touched and every design decision made.
"""


# =============================================================================
# FILES CHANGED
# =============================================================================

CHANGED_FILES = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  FILES CHANGED FOR PEAKS SUPPORT                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

── NEW FILES ────────────────────────────────────────────────────────────────

nnunetv2/training/data_augmentation/peak_transforms.py
    Contains two batchgeneratorsv2-compatible transforms:

    PeakSpatialTransform(BasicTransform)
        Replaces the standard SpatialTransform for peak data.
        • Samples random rotation angles and scale factors each iteration.
        • Builds a grid over patch_size in output pixel space.
        • Applies the full affine A = R @ S to the grid (maps output coords
          back to the inflated initial patch in the input).
        • Calls F.grid_sample(mode='bilinear') — correct linear interpolation
          for the warped image.
        • After warping, applies the PURE rotation matrix R (no scaling)
          to the stored peak vector channels: out = R @ vec for each peak.
          This keeps vector directions geometrically consistent with the
          rotated image grid.
        • Elastic deformation supported (disabled by default, p=0): correcting
          vectors for elastic deformation requires the per-voxel Jacobian of
          the deformation field, which is expensive and not implemented.
        • Rotation range: ±30° (π/6 rad) on all three axes — same as the
          standard nnUNet trainer. Initial patch is inflated by the parent
          class to avoid black borders.

    PeakMirrorTransform(MirrorTransform)
        Drop-in replacement for MirrorTransform.
        • Flips spatial axes (same as parent).
        • For each flipped axis i, negates peak channels i, i+3, i+6, ...
          (i.e. component i of every stored peak direction).
        • Example: left–right flip (axis 0) → negate p0_x, p1_x, p2_x.

    Helper functions (ported from nnssl/spatial_transforms_peaks.py):
        _rotation_matrix_3d(angles)  — Euler angles → 3×3 rotation matrix
        _apply_rotation_to_peaks()   — applies R to each (3,) peak group
        _identity_grid()             — centered pixel-space grid for grid_sample
        _sample_scalar()             — samples float from (a,b) or fixed value

nnunetv2/training/nnUNetTrainer/nnUNetTrainerPeaks.py
    Custom trainer class. Inherits nnUNetTrainer.

    configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        Delegates to super() — rotation is now properly supported, so the
        initial patch is inflated as usual by get_patch_size().
        Previously this was overridden to disable rotation entirely.

    get_training_transforms() [static]
        • Uses PeakSpatialTransform (rotation + scaling, vector-corrected)
          instead of SpatialTransform.
        • Uses PeakMirrorTransform instead of MirrorTransform.
        • Removes ALL intensity augmentations:
            - GaussianNoiseTransform
            - GaussianBlurTransform
            - BrightnessMultiplicativeTransform
            - ContrastAugmentationTransform
            - SimulateLowResolutionTransform
            - GammaCorrectionTransform
          These are meaningless/harmful for unit vector data.
        • Raises RuntimeError if do_dummy_2d_data_aug=True (peaks data
          should be approximately isotropic; anisotropic DWI not supported).

run_peaks_pipeline.sh
    End-to-end bash pipeline script for the peaks dataset.
    Mirrors run_multilabel_pipeline.sh but uses --peaks and -tr nnUNetTrainerPeaks.

── MODIFIED FILES ────────────────────────────────────────────────────────────

nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py
    determine_resampling() [line ~122]
        Added: if ALL channel names are "nonorm" (case-insensitive), use
        data_order=1 (linear interpolation) instead of data_order=3 (cubic).
        This is critical for peak data — cubic interpolation can produce
        vector magnitudes outside [0, 1] and distorts direction information.

        Before:
            data_order = 3

        After:
            channel_names = self.dataset_json.get('channel_names', ...)
            all_nonorm = all(n.casefold() == 'nonorm' for n in channel_names.values())
            data_order = 1 if all_nonorm else 3

nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py
    Added helper function _apply_peaks_flags(dataset_ids):
        Sets all entries in channel_names to "nonorm" in the raw dataset.json
        on disk. Runs before fingerprint extraction so the planner sees the
        correct channel names. Does NOT touch the multilabel flag.

    preprocess_entry() and plan_and_preprocess_entry():
        Added --peaks flag (action='store_true').
        Both --peaks and --multilabel are evaluated independently (not elif)
        so you can combine, use separately, or omit either flag.

nnunetv2/run/run_training.py
    get_trainer_from_args():
        Added peaks: bool = False parameter.
        When peaks=True and trainer_name == 'nnUNetTrainer' (default),
        automatically switches to 'nnUNetTrainerPeaks'.
        The multilabel flag is handled separately and independently.

    run_training():
        Added peaks: bool = False, passed through to get_trainer_from_args().

    run_training_entry():
        Added --peaks flag, passed to run_training().

── UNCHANGED (no peaks-specific modification needed) ────────────────────────

nnunetv2/preprocessing/normalization/map_channel_name_to_normalization.py
    Already maps 'nonorm' → NoNormalization (case-insensitive).
    No change needed — setting channel names to "nonorm" is sufficient.

nnunetv2/inference/predict_from_raw_data.py
    Uses the model's stored dataset.json for inference.
    If dataset.json already has multilabel=True and channel_names=nonorm
    (written during plan_and_preprocess --peaks), no CLI override is needed.
    The --multilabel flag can still be used as a manual override.

nnunetv2/evaluation/evaluate_predictions.py
    Uses --multilabel flag (already implemented) — no peaks-specific change.

nnunetv2/imageio/simpleitk_reader_writer.py
    4D NIfTI I/O is shared with the multilabel pipeline. No change needed.
"""


# =============================================================================
# AUGMENTATION COMPARISON
# =============================================================================

AUGMENTATION_TABLE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  AUGMENTATION COMPARISON                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Transform              │ Standard nnUNet │ nnUNetTrainerPeaks              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Rotation               │ ✓ (grid only)   │ ✓ PeakSpatialTransform         ║
║                         │                 │   grid + R applied to vectors   ║
║  Scaling                │ ✓               │ ✓ grid only (dirs unchanged)    ║
║  Elastic deformation    │ ✓ (p=0.2)       │ ✗ disabled (p=0)               ║
║                         │                 │   (Jacobian correction needed)  ║
║  Mirroring              │ ✓ MirrorTransf. │ ✓ PeakMirrorTransform          ║
║                         │                 │   flip + sign-negate component  ║
║  Gaussian noise         │ ✓               │ ✗ removed                      ║
║  Gaussian blur          │ ✓               │ ✗ removed                      ║
║  Brightness/Contrast    │ ✓               │ ✗ removed                      ║
║  Simulated low-res      │ ✓               │ ✗ removed                      ║
║  Gamma correction       │ ✓               │ ✗ removed                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# DATASET.JSON STRUCTURE
# =============================================================================

DATASET_JSON_TEMPLATE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  dataset.json STRUCTURE FOR PEAKS DATA                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

{
  "channel_names": {
    "0": "nonorm",   ← peak 0, x-component  ┐
    "1": "nonorm",   ← peak 0, y-component  │
    "2": "nonorm",   ← peak 0, z-component  ┘ NoNormalization applied
    "3": "nonorm",   ← peak 1, x-component  ┐
    "4": "nonorm",   ← peak 1, y-component  │
    "5": "nonorm",   ← peak 1, z-component  ┘
    "6": "nonorm",   ← peak 2, x-component  ┐
    "7": "nonorm",   ← peak 2, y-component  │
    "8": "nonorm"    ← peak 2, z-component  ┘
  },
  "multilabel": true,          ← enables sigmoid + BCE/Dice, 4D NIfTI I/O
  "labels": {
    "background": 0,
    "AF_left": 1,
    ...
  },
  "numTraining": 85,
  "file_ending": ".nii.gz"
}

⚠ IMPORTANT: The current Dataset004_MainTracts_Multilabel_Peaks/dataset.json
  was copied from the FA dataset and only has "0": "FA". You must add all 9
  channel_names entries BEFORE running plan_and_preprocess, otherwise nnUNet
  only preprocesses channel 0 and ignores the other 8.

  Quick fix (run once):
      python - <<'EOF'
      import json
      path = "<DATASET_ROOT>/nnunet_raw/Dataset005_MainTracts_Multilabel_Peaks/dataset.json"
      with open(path) as f: dj = json.load(f)
      dj["channel_names"] = {str(i): "nonorm" for i in range(9)}
      dj["multilabel"] = True
      with open(path, "w") as f: json.dump(dj, f, indent=2)
      print("Fixed:", path)
      EOF

  Alternatively, run:   nnUNetv2_plan_and_preprocess -d <ID> --peaks
  This sets all existing channel_names to "nonorm", but only if the keys
  0–8 are already present in the dict.

Image files per subject (3 peaks × 3 components):
  subject_0000.nii.gz  peak 0 x
  subject_0001.nii.gz  peak 0 y
  subject_0002.nii.gz  peak 0 z
  subject_0003.nii.gz  peak 1 x
  subject_0004.nii.gz  peak 1 y
  subject_0005.nii.gz  peak 1 z
  subject_0006.nii.gz  peak 2 x
  subject_0007.nii.gz  peak 2 y
  subject_0008.nii.gz  peak 2 z

Ground-truth labels: same 4D NIfTI format as multilabel
  shape (x, y, z, C)  — one binary channel per tract class
"""


# =============================================================================
# DATASET NAMING NOTE
# =============================================================================

NAMING_NOTE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  DATASET NAMING ISSUE                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

The current peaks dataset is named:
    Dataset004_MainTracts_Multilabel_Peaks

The FA dataset is named:
    Dataset004_MainTracts_Multilabel

Both have ID 004.  nnUNet resolves -d 4 by searching nnunet_raw for
Dataset004_*.  With two matching folders, the result is AMBIGUOUS.

Recommended fix: rename the peaks dataset to Dataset005:
    mv Dataset004_MainTracts_Multilabel_Peaks Dataset005_MainTracts_Multilabel_Peaks

Then update run_peaks_pipeline.sh:
    DATASET_NAME="Dataset005_MainTracts_Multilabel_Peaks"

Alternatively, pass the full name to -d to bypass integer lookup:
    nnUNetv2_train Dataset004_MainTracts_Multilabel_Peaks 3d_fullres 0 --peaks
"""


# =============================================================================
# CLI USAGE
# =============================================================================

CLI_USAGE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  CLI USAGE (after dataset.json is fixed and dataset renamed to 005)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

# Source env (do this in every new terminal)
source /home/r948e/dev/nnUNet/nnunet_env.sh
export nnUNet_raw=".../Robins_runs/nnunet_raw"
export nnUNet_preprocessed=".../Robins_runs/nnunet_preprocessed"
export nnUNet_results=".../Robins_runs/nnunet_results"
export nnUNet_compile=0

# 1. Plan & preprocess
#    --peaks  → sets channel_names→nonorm (NoNorm + linear resampling)
#    --multilabel → sets multilabel=true in dataset.json (4D labels, sigmoid loss)
#    Both flags are independent — combine as needed:
#      peaks only (single-label targets): --peaks
#      peaks + multilabel (tractography):  --peaks --multilabel
nnUNetv2_plan_and_preprocess \\
    -d 5 -c 3d_fullres -np 4 \\
    --peaks --multilabel --verify_dataset_integrity

# 2. Train
#    --peaks auto-selects nnUNetTrainerPeaks if no -tr is given
#    --multilabel overrides dataset_json['multilabel'] in memory (optional
#    if already written to disk in step 1)
nnUNetv2_train 5 3d_fullres 0 --peaks --multilabel

# 3. Predict
#    Dataset.json in the model folder already has multilabel=True from step 1
#    so --multilabel is optional (use as override if needed)
nnUNetv2_predict \\
    -i imagesTs/ -o predictions/ \\
    -d 5 -c 3d_fullres -tr nnUNetTrainerPeaks -f 0 \\
    -chk checkpoint_best.pth --multilabel

# 4. Evaluate
nnUNetv2_evaluate_folder \\
    labelsTs/ predictions/ \\
    -djfile dataset.json -pfile plans.json --multilabel

# Or use the pipeline script:
./run_peaks_pipeline.sh preprocess
./run_peaks_pipeline.sh train
./run_peaks_pipeline.sh predict
./run_peaks_pipeline.sh evaluate
./run_peaks_pipeline.sh            # runs all four steps
"""


# =============================================================================
# DESIGN DECISIONS & CAVEATS
# =============================================================================

CAVEATS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  DESIGN DECISIONS & CAVEATS                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. Why NoNormalization for peaks?
   Peaks are unit vectors: background voxels have value 0 and fiber voxels
   have components in [-1, 1]. Z-score normalization would destroy this
   structure (shift the zero baseline, change the scale). NoNormalization
   leaves values unchanged so the network can learn from raw vector components.

2. Why linear resampling (order=1)?
   Cubic interpolation (order=3) between adjacent peak vectors can produce
   results with magnitude > 1 (overshoot) or change direction unpredictably
   near sharp orientation boundaries (e.g. crossing fibers). Linear
   interpolation is a conservative choice: it may slightly blur but doesn't
   overshoot and preserves the zero background correctly.

3. Why apply R to vectors but not scale S?
   A rotation R changes the orientation of a vector: if the image is rotated
   by R, a fiber pointing in direction v must now be represented as R @ v.
   Scaling S changes voxel spacing but NOT the orientation of structures:
   a fiber still points in the same anatomical direction regardless of
   image zoom. Therefore only R is propagated to the vector channels.

4. Why is elastic deformation disabled?
   For a smooth elastic deformation field φ, the correct vector update at
   each voxel is v_new = J_φ @ v, where J_φ is the Jacobian of the
   deformation at that voxel. Computing a per-voxel Jacobian is expensive
   and not implemented. Applying elastic deformation to the grid without
   vector correction introduces small inconsistencies near high-curvature
   regions of the deformation. This is acceptable for normal images but
   problematic for unit vector fields. Disabled by default (p=0); can be
   enabled for mild deformations where approximation error is acceptable.

5. Why PeakMirrorTransform instead of just disabling mirroring?
   Mirroring (reflection) is the most powerful augmentation for brain data
   because it exploits left–right symmetry. We keep it — but for a left–right
   flip (axis 0), the x-component of every peak must change sign (p0_x → -p0_x)
   while y and z components remain unchanged. The same logic applies to any axis.

6. Rotation range ±30° (standard nnUNet default):
   This is the same range used by the standard trainer. The initial patch is
   inflated by the parent configure_rotation_dummyDA_mirroring_and_inital_patch_size
   to cover the full rotated extent, so no black borders appear at the patch edges.
"""


if __name__ == "__main__":
    print(CHANGED_FILES)
    print(AUGMENTATION_TABLE)
    print(DATASET_JSON_TEMPLATE)
    print(NAMING_NOTE)
    print(CLI_USAGE)
    print(CAVEATS)
