# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is derived from JiT (https://github.com/LTH14/JiT), Copyright (c) 2025 Tianhong Li,
# licensed under the MIT license found in the LICENSE-JiT file in the root directory of this source tree.

import torch
import torch.nn as nn
from model_wait import WaiT_models
from util.wavelets import haar_forward, haar_inverse, LF_SCALE


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = WaiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps

        # wavelet bands: LF is normalized by LF_SCALE
        self.t_star = args.t_star
        self.lf_scale = LF_SCALE
        self.noise_scale_lf = args.noise_scale_lf
        self.noise_scale_hf = args.noise_scale_hf

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.rho = args.rho

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def v_loss(self, x, x_pred, t):
        loss = ((x - x_pred) / (1 - t).clamp_min(self.t_eps)) ** 2
        return loss.mean(dim=(1, 2, 3)).mean()

    def forward(self, x, labels, phase):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        x_lf, x_hf = haar_forward(x)
        x_lf = x_lf / self.lf_scale
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))

        if phase == 0:
            # coarse: normalized LF band only, t in [0, 1]
            z = t * x_lf + (1 - t) * torch.randn_like(x_lf) * self.noise_scale_lf
            x_pred = self.net(z, t.flatten(), labels_dropped, res_idx=0)
            return self.v_loss(x_lf, x_pred, t)

        # fine: full image for t in [t*, 1]
        t = self.t_star + (1 - self.t_star) * t
        t_hf = (t - self.t_star) / (1 - self.t_star)
        z_lf = t * x_lf + (1 - t) * torch.randn_like(x_lf) * self.noise_scale_lf
        z_hf = t_hf * x_hf + (1 - t_hf) * torch.randn_like(x_hf) * self.noise_scale_hf
        z = haar_inverse(z_lf * self.lf_scale, z_hf)

        # per-band v-loss: LF on the coarse clock t, HF on the delayed clock t_hf
        x_pred = self.net(z, t.flatten(), labels_dropped, res_idx=1)
        x_pred_lf, x_pred_hf = haar_forward(x_pred)
        loss_lf = self.v_loss(x_lf, x_pred_lf / self.lf_scale, t)
        loss_hf = self.v_loss(x_hf, x_pred_hf, t_hf)
        return loss_lf + loss_hf

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)

        # rho % of steps on [0, t*] (coarse), the rest on [t*, 1] (fine), uniformly spaced within each phase
        steps0 = min(self.steps - 1, max(1, round(self.rho * self.steps)))
        steps1 = self.steps - steps0
        timesteps0 = torch.linspace(0.0, self.t_star, steps0 + 1, device=device).view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)
        timesteps1 = torch.linspace(self.t_star, 1.0, steps1 + 1, device=device).view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # phase 0: ode on normalized LF band
        z = self.noise_scale_lf * torch.randn(bsz, 3, self.img_size // 2, self.img_size // 2, device=device)
        for i in range(steps0):
            z = stepper(z, timesteps0[i], timesteps0[i + 1], labels, 0)

        # transition at t*: un-normalize LF, add fresh HF noise, go to target space
        z_hf = self.noise_scale_hf * torch.randn(bsz, 9, self.img_size // 2, self.img_size // 2, device=device)
        z = haar_inverse(z * self.lf_scale, z_hf)

        # phase 1: ode on the full image
        for i in range(steps1 - 1):
            z = stepper(z, timesteps1[i], timesteps1[i + 1], labels, 1)
        # last step euler
        z = self._euler_step(z, timesteps1[-2], timesteps1[-1], labels, 1)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, res_idx):
        # conditional
        x_cond = self.net(z, t.flatten(), labels, res_idx)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes), res_idx)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, res_idx):
        v_pred = self._forward_sample(z, t, labels, res_idx)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, res_idx):
        v_pred_t = self._forward_sample(z, t, labels, res_idx)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, res_idx)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
