# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Feature statistics for the evaluation metrics.

5-crop FID  FID on Inception-v3 features of five 299x299 crops (four corners and center) taken from each image at
            native resolution, so that local detail is not lost to downsampling.
hFWD        High-frequency Frechet Wavelet Distance. Each image is decomposed by a level-4 Haar wavelet packet
            transform into 256 frequency bands (packets). The coefficients of each packet are flattened into one
            vector (packets larger than 32x32 are first average-pooled to 32x32, giving 3 x 32 x 32 = 3072 values at
            512x512), a Frechet distance is computed per packet, and hFWD is the mean over all packets except
            packet 0, the lowest-frequency band.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from util.crop import extract_5_crops
from util.wavelets import haar_forward

LEVEL = 4  # 4^4 = 256 packets
POOL = 32  # packets larger than POOL x POOL are average-pooled down to it

_inception = None
_reference_cache = {}


class Moments:
    """Running sums of (B, G, D) features -> per-group mean (G, D) and covariance (G, D, D), summed over ranks."""

    def __init__(self, groups, dim, device):
        self.n = 0
        self.s1 = torch.zeros(groups, dim, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(groups, dim, dim, dtype=torch.float64, device=device)

    def update(self, f):
        f = f.double()
        self.n += f.shape[0]
        self.s1 += f.sum(0)
        self.s2.baddbmm_(f.permute(1, 2, 0), f.permute(1, 0, 2))

    def finalize(self):
        n = torch.tensor([float(self.n)], dtype=torch.float64, device=self.s1.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for t in (n, self.s1, self.s2):
                torch.distributed.all_reduce(t)
        n = n.item()
        mu = self.s1 / n
        sigma = self.s2.baddbmm_(mu.unsqueeze(2), mu.unsqueeze(1), alpha=-n).div_(n - 1)
        return mu, sigma


def packet_dim(img_size):
    side = min(img_size // 2 ** LEVEL, POOL)
    return 3 * side * side


def packet_features(x):
    """x: (B, 3, H, W) in [0, 1] -> (B, 256, D) flattened (and, above POOL, pooled) packet coefficients."""
    # level-4 Haar wavelet packet transform: split every band into LL, LH, HL, HH, four times (packet 0 is LL...L)
    bands = [x]
    for _ in range(LEVEL):
        split = []
        for band in bands:
            ll, hf = haar_forward(band)
            split += [ll, *hf.chunk(3, dim=1)]
        bands = split
    packets = torch.stack(bands, dim=1)  # (B, 256, 3, h, w)
    B, P, C, h, w = packets.shape
    if h > POOL or w > POOL:
        packets = F.adaptive_avg_pool2d(packets.reshape(B * P, C, h, w), (POOL, POOL)).reshape(B, P, C, POOL, POOL)
    return packets.reshape(B, P, -1)


def crop_features(images_hwc, device):
    """List of (H, W, 3) uint8 arrays -> (5 * B, 1, 2048) InceptionV3 features of their 5 crops."""
    global _inception
    if _inception is None:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        _inception = FeatureExtractorInceptionV3(name='inception-v3-compat', features_list=['2048']).to(device).eval()
    crops = np.stack([c for img in images_hwc for c in extract_5_crops(img)])
    x = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous().to(device)
    return _inception.convert_features_tuple_to_dict(_inception(x))['2048'].unsqueeze(1)


@torch.no_grad()
def image_moments(paths, img_size, crop5=True, packets=True, batch_size=32, device='cuda'):
    """5-crop Inception and/or packet moments of the given images, summed over all ranks.

    Collective when distributed: every rank must call it (each with its own share of the images)."""
    acc = {}
    if crop5:
        acc['crop5'] = Moments(1, 2048, device)
    if packets:
        acc['hfwd'] = Moments(256, packet_dim(img_size), device)
    for i in range(0, len(paths), batch_size):
        imgs = [np.asarray(Image.open(p).convert('RGB')) for p in paths[i:i + batch_size]]
        if crop5:
            acc['crop5'].update(crop_features(imgs, device))
        if packets:
            x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255).to(device)
            acc['hfwd'].update(packet_features(x))
    return {k: m.finalize() for k, m in acc.items()}


def load_stats(path):
    """(mu, sigma) of a reference .npz, kept in memory for repeated evaluations (the hFWD one is ~19 GB)."""
    if path not in _reference_cache:
        ref = np.load(path)
        _reference_cache[path] = (ref['mu'], ref['sigma'])
    return _reference_cache[path]


def frechet_distance(mu1, sigma1, mu2, sigma2, device='cuda', eps=1e-6):
    """Frechet distance via eigendecomposition (float64 on GPU); never negative."""
    mu1, sigma1, mu2, sigma2 = (torch.as_tensor(a, dtype=torch.float64, device=device) for a in (mu1, sigma1, mu2, sigma2))
    evals, evecs = torch.linalg.eigh(sigma1)
    sqrt1 = evecs @ torch.diag(evals.clamp_min(eps).sqrt()) @ evecs.T
    m = torch.linalg.eigvalsh(sqrt1 @ sigma2 @ sqrt1).clamp_min(0)
    d = mu1 - mu2
    return float((d @ d + torch.trace(sigma1) + torch.trace(sigma2) - 2 * m.sqrt().sum()).clamp_min(0))


def calculate_hfwd(gen_stats, ref_path, device='cuda'):
    """hFWD of generated packet moments (mu (256, D), sigma (256, D, D)) against a reference .npz."""
    mu_r, sigma_r = load_stats(ref_path)
    mu_g, sigma_g = gen_stats
    dists = [frechet_distance(mu_r[p], sigma_r[p], mu_g[p], sigma_g[p], device) for p in range(mu_r.shape[0])]
    return float(np.mean(dists[1:]))


def calculate_fid_5crop(gen_stats, ref_path, device='cuda'):
    mu_r, sigma_r = load_stats(ref_path)
    mu_g, sigma_g = gen_stats
    return frechet_distance(mu_r, sigma_r, mu_g[0], sigma_g[0], device)
