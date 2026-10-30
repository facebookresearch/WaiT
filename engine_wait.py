# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is derived from JiT (https://github.com/LTH14/JiT), Copyright (c) 2025 Tianhong Li,
# licensed under the MIT license found in the LICENSE-JiT file in the root directory of this source tree.

import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
from util.fwd import image_moments, calculate_fid_5crop, calculate_hfwd
import torch_fidelity
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        # alternate between coarse (0) and fine (1) batches
        phase = data_iter_step % 2

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels, phase)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    # evaluate every EMA that is tracked (ema_params1, ema_params2, ...)
    ema_idx = 1
    while getattr(model_without_ddp, 'ema_params{}'.format(ema_idx), None) is not None:
        evaluate_ema(model_without_ddp, args, epoch, ema_idx, batch_size=batch_size, log_writer=log_writer)
        ema_idx += 1


def evaluate_ema(model_without_ddp, args, epoch, ema_idx, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "ema{}-{}-steps{}-rho{}-cfg{}-interval{}-{}-image{}-res{}".format(
            ema_idx, model_without_ddp.method, model_without_ddp.steps, model_without_ddp.rho, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    use_crops = args.img_size >= 299 and bool(args.ref_5crop_path)
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)

    # switch to ema params
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = getattr(model_without_ddp, 'ema_params{}'.format(ema_idx))[i]
    print("Switch to ema{}".format(ema_idx))
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # 5-crop / wavelet-packet moments: every rank processes its own share of the images (summed over ranks)
    gen_stats = {}
    if use_crops or args.ref_path:
        files = sorted(os.path.join(save_folder, f) for f in os.listdir(save_folder) if f.endswith('.png'))
        gen_stats = image_moments(files[local_rank::world_size], args.img_size, crop5=use_crops, packets=bool(args.ref_path))

    # compute FID, IS, 5-crop FID and hFWD
    if log_writer is not None:
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=args.ref_fid_path,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_ema{}_cfg{}_res{}".format(ema_idx, model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("[ema{}] FID: {:.4f}, Inception Score: {:.4f}".format(ema_idx, fid, inception_score))

        if use_crops:
            fid_5crop = calculate_fid_5crop(gen_stats['crop5'], args.ref_5crop_path)
            log_writer.add_scalar('fid_5crop{}'.format(postfix), fid_5crop, epoch)
            print("[ema{}] 5-crop FID: {:.4f}".format(ema_idx, fid_5crop))

        if args.ref_path:
            hfwd = calculate_hfwd(gen_stats['hfwd'], args.ref_path)
            log_writer.add_scalar('hfwd{}'.format(postfix), hfwd, epoch)
            print("[ema{}] hFWD: {:.4f}".format(ema_idx, hfwd))

        shutil.rmtree(save_folder)
    del gen_stats

    torch.distributed.barrier()
