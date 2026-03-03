"""
Peak-aware spatial transforms for diffusion MRI fiber orientation data.

Peaks are 3D unit vectors stored as 9 channels per voxel (3 peaks × 3 components):
    channels = [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z]

Standard MirrorTransform flips voxel positions but leaves channel values unchanged.
For peaks this is wrong: flipping spatial axis i must also negate the corresponding
vector component in channels [i, i+num_components, i+2*num_components, ...].

Example — left–right (axis 0) flip:
    Before: voxel at x=10 has peak component p0_x = +0.8 (pointing right)
    After flip: that voxel is now at x=N-10, but still says p0_x = +0.8 (pointing right)
                → wrong, it should now say p0_x = -0.8 (pointing left)
"""

from typing import Tuple

import torch
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform


class PeakMirrorTransform(MirrorTransform):
    """
    Drop-in replacement for MirrorTransform that additionally negates the peak
    vector components corresponding to any flipped spatial axis.

    Args:
        allowed_axes:   Spatial axes eligible for random flipping (0=x, 1=y, 2=z).
        num_peaks:      Number of fiber directions per voxel (default 3).
        num_components: Spatial components per peak (default 3 for x/y/z).
    """

    def __init__(
        self,
        allowed_axes: Tuple[int, ...] = (0, 1, 2),
        num_peaks: int = 3,
        num_components: int = 3,
    ):
        super().__init__(allowed_axes=allowed_axes)
        self.num_peaks = num_peaks
        self.num_components = num_components

    # get_parameters() is inherited — randomly picks a subset of allowed_axes per sample.

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        """
        img shape: (C, X, Y, Z)  — note: no batch dimension here, BasicTransform
        applies per-sample before batching.

        1. Flip spatial axes as usual.
        2. For each flipped axis i, negate peak channels [i, i+3, i+6, ...].
        """
        axes = params['axes']
        if not axes:
            return img

        # --- spatial flip (same as parent) ---
        spatial_axes = [i + 1 for i in axes]   # +1 to skip channel dim
        img = torch.flip(img, spatial_axes)

        # --- sign correction for directional channels ---
        # Channel layout: [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z]
        # Flipping spatial axis i → negate component i of every peak.
        for axis in axes:
            channels_to_negate = [
                axis + peak * self.num_components
                for peak in range(self.num_peaks)
                if axis + peak * self.num_components < img.shape[0]
            ]
            if channels_to_negate:
                img[channels_to_negate] *= -1

        return img

    # _apply_to_segmentation: spatial flip only (inherited behaviour is correct).
