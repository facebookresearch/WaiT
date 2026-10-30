# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is derived from JiT (https://github.com/LTH14/JiT), Copyright (c) 2025 Tianhong Li,
# licensed under the MIT license found in the LICENSE-JiT file in the root directory of this source tree.

def adjust_learning_rate(optimizer, epoch, args):
    """Linear warmup, then constant learning rate."""
    lr = args.lr * min(1.0, epoch / args.warmup_epochs) if args.warmup_epochs > 0 else args.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
