# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is derived from JiT (https://github.com/LTH14/JiT), Copyright (c) 2025 Tianhong Li,
# licensed under the MIT license found in the LICENSE-JiT file in the root directory of this source tree.

import numpy as np
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def extract_5_crops(arr, crop_size=299):
    """Four corners + center crop of an (H, W, C) array, at native resolution (for 5-crop FID)."""
    h, w = arr.shape[:2]
    top, left = (h - crop_size) // 2, (w - crop_size) // 2
    return [arr[:crop_size, :crop_size], arr[:crop_size, w - crop_size:],
            arr[h - crop_size:, :crop_size], arr[h - crop_size:, w - crop_size:],
            arr[top:top + crop_size, left:left + crop_size]]
