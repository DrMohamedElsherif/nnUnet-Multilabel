"""
nnUNetTrainerPeaks — nnUNet trainer for diffusion MRI fiber orientation peak data.

Key differences from the standard trainer:

1. Augmentation pipeline is adapted for 9-channel peak vector fields:
   - Rotation is DISABLED: rotating the image would require rotating the stored
     vector components too, which SpatialTransform does not do.
   - Scaling is kept: voxel rescaling does not change vector directions.
   - All intensity-based augmentations are REMOVED (noise, blur, brightness,
     contrast, simulated low-resolution, gamma) — these have no geometric
     meaning for directional data.
   - MirrorTransform is replaced by PeakMirrorTransform, which negates the
     appropriate vector component channels whenever a spatial axis is flipped.

2. configure_rotation_dummyDA_mirroring sets rotation_for_DA = (0, 0) so that
   the initial (padded) patch size is not inflated for rotation.

Usage:
    nnUNetv2_train <dataset> 3d_fullres 0 -tr nnUNetTrainerPeaks --peaks --multilabel

Or set automatically via the --peaks CLI flag on nnUNetv2_train.
"""

from typing import Tuple, List, Union

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
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

from nnunetv2.training.data_augmentation.peak_transforms import PeakMirrorTransform
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
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
        Disable rotation (vectors not transformed) and compute a non-inflated
        initial patch size based on scaling only.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        if dim == 2:
            do_dummy_2d_data_aug = False
            mirror_axes = (0, 1)
        else:
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import ANISO_THRESHOLD
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            mirror_axes = (0, 1, 2)

        # rotation_for_DA = (0, 0) → no rotation, patch padding only for scaling
        rotation_for_DA = (0., 0.)
        initial_patch_size = get_patch_size(
            patch_size[-dim:],
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.85, 1.25),       # same scaling range as default trainer
        )
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.print_to_log_file('nnUNetTrainerPeaks: rotation DISABLED, using PeakMirrorTransform')
        self.inference_allowed_mirroring_axes = mirror_axes

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
        - SpatialTransform: scaling only (p_rotation=0), no elastic deformation
        - PeakMirrorTransform: spatial flip + vector component sign correction
        - All intensity augmentations removed
        """
        transforms = []

        # Spatial: scaling only, rotation disabled (p_rotation=0)
        if do_dummy_2d_data_aug:
            from batchgeneratorsv2.transforms.spatial.spatial import Convert3DTo2DTransform, Convert2DTo3DTransform
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size

        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0,               # disabled — rotation would mis-orient vectors
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.85, 1.25),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
            )
        )

        if do_dummy_2d_data_aug:
            from batchgeneratorsv2.transforms.spatial.spatial import Convert2DTo3DTransform
            transforms.append(Convert2DTo3DTransform())

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
