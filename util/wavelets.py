# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

SQRT_HALF = 0.7071067811865476

# 95th percentile of |LF| coefficients over the training set (ImageNet, Haar level 1)
LF_SCALE = 1.94


def haar_forward(x):
    """Orthonormal 2D Haar DWT: (B, C, H, W) -> LF (B, C, H/2, W/2), HF (B, 3C, H/2, W/2) as [LH, HL, HH]."""
    L = (x[..., 0::2] + x[..., 1::2]) * SQRT_HALF
    H = (x[..., 0::2] - x[..., 1::2]) * SQRT_HALF
    LL = (L[..., 0::2, :] + L[..., 1::2, :]) * SQRT_HALF
    LH = (L[..., 0::2, :] - L[..., 1::2, :]) * SQRT_HALF
    HL = (H[..., 0::2, :] + H[..., 1::2, :]) * SQRT_HALF
    HH = (H[..., 0::2, :] - H[..., 1::2, :]) * SQRT_HALF
    return LL, torch.cat([LH, HL, HH], dim=1)


def haar_inverse(ll, hf):
    """Inverse of haar_forward."""
    LH, HL, HH = hf.chunk(3, dim=1)
    L = torch.stack([(ll + LH) * SQRT_HALF, (ll - LH) * SQRT_HALF], dim=3).flatten(2, 3)
    H = torch.stack([(HL + HH) * SQRT_HALF, (HL - HH) * SQRT_HALF], dim=3).flatten(2, 3)
    return torch.stack([(L + H) * SQRT_HALF, (L - H) * SQRT_HALF], dim=4).flatten(3, 4)
