"""
nnUNetTrainerPeaks — nnUNet trainer for diffusion MRI fiber orientation peak data.

Key differences from the standard trainer:

1. Augmentation pipeline is adapted for 9-channel peak vector fields:
   - Rotation is ENABLED via PeakSpatialTransform: the rotation matrix R is applied
     both to the voxel sampling grid AND to the stored vector component channels,
     so peak directions stay geometrically consistent with the rotated image.
   - Scaling is applied to the voxel grid only (scaling does not change directions).
   - Elastic deformation is disabled by default (correcting vectors would require
     the full per-voxel Jacobian of the deformation field).
   - All intensity-based augmentations are REMOVED (noise, blur, brightness,
     contrast, simulated low-resolution, gamma) — these have no geometric
     meaning for directional data.
   - MirrorTransform is replaced by PeakMirrorTransform, which negates the
     appropriate vector component channels whenever a spatial axis is flipped.

2. configure_rotation_dummyDA_mirroring_and_inital_patch_size delegates to
   super() so the initial patch size is correctly inflated to accommodate rotation
   (standard nnUNet behaviour, no longer suppressed).

Usage:
    nnUNetv2_train <dataset> 3d_fullres 0 -tr nnUNetTrainerPeaks --peaks --multilabel

Or set automatically via the --peaks CLI flag on nnUNetv2_train.
"""

from typing import Tuple, List, Union

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.training.data_augmentation.peak_transforms import PeakMirrorTransform, PeakSpatialTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerPeaks(nnUNetTrainer):
    """
    Trainer adapted for 9-channel DWI peak vector input (3 peaks × 3 components).

    Assumes channel layout: [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z]
    """

    # Number of peaks and spatial components — change if your data differs.
    NUM_PEAKS = 3
    NUM_COMPONENTS = 3  # x, y, z

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Delegate to parent so the initial patch size is correctly inflated for
        rotation (standard nnUNet behaviour). PeakSpatialTransform will handle
        vector-consistent rotation.
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        self.print_to_log_file(
            'nnUNetTrainerPeaks: rotation ENABLED via PeakSpatialTransform '
            '(peak vectors rotated consistently with the image grid)'
        )
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        """
        Peaks-adapted augmentation pipeline:
        - PeakSpatialTransform: rotation (with vector correction) + scaling
        - PeakMirrorTransform: spatial flip + vector component sign correction
        - All intensity augmentations removed (noise, blur, brightness, contrast,
          simulated low-resolution, gamma are meaningless for vector fields)
        """
        if do_dummy_2d_data_aug:
            raise RuntimeError(
                'nnUNetTrainerPeaks does not support dummy 2D data augmentation '
                '(triggered for highly anisotropic patches). Peak data should be '
                'approximately isotropic. If this is unexpected, check your dataset '
                'voxel spacing and patch size.'
            )

        transforms = [
            PeakSpatialTransform(
                patch_size=patch_size,
                p_rotation=0.5,
                rotation=rotation_for_DA,    # range set by planner, typically ±30°
                p_scaling=0.2,
                scaling=(0.85, 1.25),
                p_sync_scale=1.0,
                p_elastic_deform=0.0,        # disabled: elastic deform can't be vector-corrected
                num_peaks=nnUNetTrainerPeaks.NUM_PEAKS,
                num_components=nnUNetTrainerPeaks.NUM_COMPONENTS,
            )
        ]

        # Peak-aware mirroring (replaces standard MirrorTransform)
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                PeakMirrorTransform(
                    allowed_axes=mirror_axes,
                    num_peaks=nnUNetTrainerPeaks.NUM_PEAKS,
                    num_components=nnUNetTrainerPeaks.NUM_COMPONENTS,
                )
            )

        # No intensity augmentations: noise, blur, brightness, contrast,
        # simulated low-resolution, gamma are all meaningless for vector fields.

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True,
                )
            )
            transforms.append(RandomTransform(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    strel_size=(1, 8),
                    p_per_label=1,
                ), apply_probability=0.4
            ))
            transforms.append(RandomTransform(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                    p_per_label=1,
                ), apply_probability=0.2
            ))

        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0,
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
