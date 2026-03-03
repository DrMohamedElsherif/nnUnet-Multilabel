"""
Peak-aware spatial transforms for diffusion MRI fiber orientation data.

Peaks are 3D unit vectors stored as 9 channels per voxel (3 peaks × 3 components):
    channels = [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z]

Transforms provided
-------------------
PeakMirrorTransform
    Drop-in for MirrorTransform. Flips spatial axes AND negates the corresponding
    vector component channels so peak directions remain geometrically correct.

PeakSpatialTransform
    Replaces SpatialTransform for peak data. Applies rotation + scaling + optional
    smooth elastic deformation to the voxel grid, and also applies the same rotation
    matrix R to the stored peak vectors so directions remain consistent with the
    warped image.

    Scaling does not change vector directions and is applied to the spatial grid only.

    Elastic deformation is supported for the grid but NOT for vector correction
    (correct treatment requires the full per-voxel Jacobian of the deformation field,
    which is expensive). Keep p_elastic_deform=0 (default) for strict correctness.

    The input tensor may be larger than patch_size (nnUNet inflates the initial patch
    to accommodate rotation via get_patch_size). The affine grid samples from the
    full inflated input so rotations fill the output patch without black borders.
"""

from __future__ import annotations

from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import fourier_gaussian
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform

ScalarLike = Union[float, int, Tuple[float, float]]


# ── scalar sampling ────────────────────────────────────────────────────────────

def _sample_scalar(s: ScalarLike, rng: np.random.RandomState) -> float:
    if isinstance(s, (int, float, np.floating, np.integer)):
        return float(s)
    if isinstance(s, (tuple, list)) and len(s) == 2:
        return float(rng.uniform(float(s[0]), float(s[1])))
    raise TypeError(f"Unsupported scalar spec: {type(s)}")


# ── grid helpers ───────────────────────────────────────────────────────────────

def _identity_grid(size: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """(*size, ndim) grid of pixel coordinates centered at 0."""
    axes = [
        torch.linspace((1 - s) / 2, (s - 1) / 2, s, device=device, dtype=torch.float32)
        for s in size
    ]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)


# ── rotation / affine helpers ──────────────────────────────────────────────────

def _rotation_matrix_3d(angles: List[float]) -> np.ndarray:
    """3×3 rotation matrix from Euler angles [ax, ay, az]; convention: Rz @ Ry @ Rx."""
    ax, ay, az = angles
    Rx = np.array([[1,          0,           0         ],
                   [0,  np.cos(ax), -np.sin(ax)],
                   [0,  np.sin(ax),  np.cos(ax)]])
    Ry = np.array([[ np.cos(ay), 0, np.sin(ay)],
                   [ 0,          1, 0         ],
                   [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0,           0,          1]])
    return Rz @ Ry @ Rx


def _apply_rotation_to_peaks(
        img: torch.Tensor,
        R: torch.Tensor,
        num_peaks: int,
        num_components: int,
) -> torch.Tensor:
    """
    Rotate stored peak vectors by rotation matrix R.

    img:  (C, *spatial)  where C = num_peaks * num_components
    R:    (num_components, num_components) rotation matrix (torch)

    Each contiguous group of num_components channels encodes one peak direction.
    Only rotation is applied — scaling does not affect vector directions.
    """
    spatial = img.shape[1:]
    out = torch.empty_like(img)
    for i in range(num_peaks):
        ch = slice(i * num_components, (i + 1) * num_components)
        vec = img[ch].reshape(num_components, -1)          # (3, N)
        out[ch] = (R @ vec).reshape(num_components, *spatial)
    return out


# ── PeakMirrorTransform ────────────────────────────────────────────────────────

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
        img shape: (C, X, Y, Z)  — no batch dimension (BasicTransform applies per-sample).

        1. Flip spatial axes as usual.
        2. For each flipped axis i, negate peak channels [i, i+3, i+6, ...].
        """
        axes = params['axes']
        if not axes:
            return img

        # spatial flip
        img = torch.flip(img, [a + 1 for a in axes])   # +1 to skip channel dim

        # sign correction: flipping axis i → negate component i of every peak
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


# ── PeakSpatialTransform ───────────────────────────────────────────────────────

class PeakSpatialTransform(BasicTransform):
    """
    Spatial augmentation for peak vector fields.

    What it does per sample
    -----------------------
    1. Samples random rotation angles and scale factors.
    2. Builds an affine matrix A = R @ S (rotation × scaling).
    3. Constructs a sampling grid over *patch_size* in output pixel space,
       applies A to map those positions back to the (potentially larger) input,
       normalises to [-1,1] relative to the input shape, and calls F.grid_sample
       with bilinear interpolation for images and nearest for segmentations.
    4. After warping, applies the pure rotation matrix R (no scaling) to the stored
       peak vector channels so directions stay consistent with the rotated grid.
    5. Optionally adds smooth elastic deformation to the grid (Fourier-space noise).
       Elastic deformation is NOT propagated to vectors (would require Jacobian);
       keep p_elastic_deform=0 (default) for strict geometric correctness.

    Args
    ----
    patch_size:               Output patch size (D, H, W).
    p_rotation:               Per-sample probability of rotation (default 0.5).
    rotation:                 Rotation angle range in radians (default ±π/6 = ±30°).
    p_scaling:                Per-sample probability of scaling (default 0.2).
    scaling:                  Scale factor range (default 0.85–1.25).
    p_sync_scale:             Probability of using the same scale for all axes (default 1).
    p_elastic_deform:         Probability of elastic deformation (default 0 = disabled).
    elastic_deform_scale:     Smoothness of elastic deformation relative to patch size.
    elastic_deform_magnitude: Amplitude of elastic deformation in pixels.
    num_peaks:                Number of fiber directions per voxel (default 3).
    num_components:           Spatial components per peak — 3 for (x, y, z).
    """

    def __init__(
            self,
            patch_size: Tuple[int, ...],
            p_rotation: float = 0.5,
            rotation: ScalarLike = (-np.pi / 6, np.pi / 6),
            p_scaling: float = 0.2,
            scaling: ScalarLike = (0.85, 1.25),
            p_sync_scale: float = 1.0,
            p_elastic_deform: float = 0.0,
            elastic_deform_scale: ScalarLike = (0.0, 0.2),
            elastic_deform_magnitude: ScalarLike = (0.0, 0.2),
            num_peaks: int = 3,
            num_components: int = 3,
    ):
        super().__init__()
        self.patch_size = tuple(int(x) for x in patch_size)
        self.p_rotation = p_rotation
        self.rotation = rotation
        self.p_scaling = p_scaling
        self.scaling = scaling
        self.p_sync_scale = p_sync_scale
        self.p_elastic_deform = p_elastic_deform
        self.elastic_deform_scale = elastic_deform_scale
        self.elastic_deform_magnitude = elastic_deform_magnitude
        self.num_peaks = num_peaks
        self.num_components = num_components

    def get_parameters(self, **data_dict) -> dict:
        rng = np.random.RandomState()

        do_rot   = rng.uniform() < self.p_rotation
        do_scale = rng.uniform() < self.p_scaling
        do_elas  = rng.uniform() < self.p_elastic_deform

        # --- rotation ---
        angles = (
            [_sample_scalar(self.rotation, rng) for _ in range(3)]
            if do_rot else [0., 0., 0.]
        )

        # --- scaling ---
        if do_scale:
            if rng.uniform() < self.p_sync_scale:
                s = _sample_scalar(self.scaling, rng)
                scales = [s, s, s]
            else:
                scales = [_sample_scalar(self.scaling, rng) for _ in range(3)]
        else:
            scales = [1., 1., 1.]

        # Pure rotation matrix R  → applied to peak vector channels
        R_np = _rotation_matrix_3d(angles)
        # Full affine A = R @ S  → applied to spatial sampling grid
        affine_np = R_np @ np.diag(scales)

        # --- elastic deformation (optional) ---
        offsets = None
        if do_elas:
            dim = 3
            sync = rng.uniform() < 0.5
            elas_scales = (
                [_sample_scalar(self.elastic_deform_scale, rng)] * dim if sync
                else [_sample_scalar(self.elastic_deform_scale, rng) for _ in range(dim)]
            )
            sigmas = [es * p for es, p in zip(elas_scales, self.patch_size)]
            mags   = [_sample_scalar(self.elastic_deform_magnitude, rng) for _ in range(dim)]

            offsets = torch.normal(0., 1., size=(dim, *self.patch_size))
            for d in range(dim):
                tmp = np.fft.fftn(offsets[d].numpy())
                tmp = fourier_gaussian(tmp, sigmas[d])
                offsets[d] = torch.from_numpy(np.fft.ifftn(tmp).real).float()
                mx = torch.max(torch.abs(offsets[d])).item()
                if mx > 1e-8:
                    offsets[d] *= mags[d] / mx

            offsets = offsets.permute(1, 2, 3, 0)  # (*patch_size, 3)

        return {
            'affine': affine_np if (do_rot or do_scale) else None,
            'R':      R_np      if do_rot               else None,
            'elastic_offsets': offsets,
        }

    def _warp(self, x: torch.Tensor, params: dict, mode: str) -> torch.Tensor:
        """
        Spatially warp x using affine + elastic offsets.
        x: (C, *input_spatial) — may be larger than patch_size.
        Returns: (C, *patch_size).
        """
        device = x.device
        input_shape = tuple(x.shape[1:])   # (D, H, W) of the actual input

        # Identity grid over output patch_size, pixel coords centred at 0
        grid = _identity_grid(self.patch_size, device=device)  # (*patch, 3)

        # Apply affine A: maps output pixel coords → input pixel coords
        # grid @ A.T  is equivalent to  A @ p  for each position vector p
        A_np = params['affine']
        if A_np is not None:
            A = torch.from_numpy(A_np).to(device=device, dtype=torch.float32)
            grid = grid @ A.T

        # Add elastic displacement in pixel space
        offsets = params['elastic_offsets']
        if offsets is not None:
            grid = grid + offsets.to(device=device, dtype=torch.float32)

        # Normalise to [-1, 1] relative to INPUT shape for F.grid_sample
        for d in range(len(input_shape)):
            grid[..., d] = grid[..., d] / (input_shape[d] / 2.0)

        # Reverse dimension order: (d0, d1, d2) → (d2, d1, d0) as grid_sample expects (W, H, D)
        grid = torch.flip(grid, (-1,))

        return F.grid_sample(
            x.unsqueeze(0),       # (1, C, *input_spatial)
            grid.unsqueeze(0),    # (1, *patch_size, 3)
            mode=mode,
            padding_mode='zeros',
            align_corners=True,
        ).squeeze(0)              # (C, *patch_size)

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        """
        img: (C, *spatial) — no batch dimension (BasicTransform applies per-sample).
        1. Warp voxel positions with the full affine (rotation + scaling).
        2. Rotate peak vector channels with the pure rotation matrix R.
        """
        img = self._warp(img, params, mode='bilinear')

        R_np = params['R']
        if R_np is not None:
            R = torch.from_numpy(R_np).to(device=img.device, dtype=img.dtype)
            img = _apply_rotation_to_peaks(img, R, self.num_peaks, self.num_components)

        return img

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        """Spatial warp only — segmentation labels carry no vector information."""
        return self._warp(seg.float(), params, mode='nearest').to(seg.dtype)
