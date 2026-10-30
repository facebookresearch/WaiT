# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is derived from JiT (https://github.com/LTH14/JiT), Copyright (c) 2025 Tianhong Li,
# licensed under the MIT license found in the LICENSE-JiT file in the root directory of this source tree.

"""Reference statistics for 5-crop FID and hFWD.

Writes in{img_size}_{split}_{5crop,hfwd}_stats.npz (keys: mu, sigma), using the same preprocessing as training
(ADM center crop) and the same features as util/fwd.py.

    python prepare_ref.py --data_path ${IMAGENET_PATH} --img_size 512 --output_path fid_stats
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from util.crop import center_crop_arr
from util.fwd import Moments, crop_features, packet_dim, packet_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='ImageNet root (contains train/ and val/)')
    parser.add_argument('--output_path', type=str, default='fid_stats', help='Output folder')
    parser.add_argument('--img_size', type=int, default=512, help='Resolution to center-crop to')
    parser.add_argument('--split', type=str, default='train', help='ImageNet split folder (train or val)')
    parser.add_argument('--num_images', type=int, default=0, help='Use only the first N images (0 = all)')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(os.path.join(args.data_path, args.split), transform=transform)
    if args.num_images:
        dataset = Subset(dataset, range(min(args.num_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=256, num_workers=32, shuffle=False)

    device = 'cuda'
    crop5 = Moments(1, 2048, device)
    hfwd = Moments(256, packet_dim(args.img_size), device)
    with torch.no_grad():
        for images, _ in tqdm(loader):
            # ToTensor divides uint8 pixels by 255, so this recovers the exact pixels
            images_uint8 = (images * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
            crop5.update(crop_features(list(images_uint8), device))
            hfwd.update(packet_features(images.to(device)))

    os.makedirs(args.output_path, exist_ok=True)
    for name, acc in (('5crop', crop5), ('hfwd', hfwd)):
        mu, sigma = acc.finalize()
        mu, sigma = mu.cpu().numpy(), sigma.cpu().numpy()
        if name == '5crop':
            mu, sigma = mu[0], sigma[0]
        out = os.path.join(args.output_path, 'in{}_{}_{}_stats.npz'.format(args.img_size, args.split, name))
        np.savez(out, mu=mu, sigma=sigma, num_images=acc.n)
        print('saved', out, '({} images)'.format(acc.n))


if __name__ == '__main__':
    main()
