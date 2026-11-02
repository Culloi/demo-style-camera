#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ukiyoe / sketch CycleGAN 权重 -> ONNX 导出 (编译机 320 重转用).

ukiyoe: Pim66070135/CycleGan_style_ukiyoe latest_net_G.pth
        (junyanz pytorch-CycleGAN-and-pix2pix 原版 ResnetGenerator)
sketch: fracta1l/cyclegan_photo2sketch generator_a_to_b.safetensors
        (ngf32 轻量, 3 残差块, space hw5_gan 原版结构)
统一导出 256x256 (全卷积, convert_style.py --size 320 改写输入尺寸).
输入输出均为 [-1,1] (tanh), 板端用 --norm anime / out_range pm1.
"""
import functools
import json
import os

import torch
import torch.nn as nn

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "onnx")


# ------------------------------------------------------------------
# ukiyoe: junyanz 原版结构 (与 Pim space app.py 逐行一致)
# ------------------------------------------------------------------
def _instance_norm(ngf):
    return nn.InstanceNorm2d(ngf, affine=False, track_running_stats=True)


# 必须是 functools.partial: ResnetGenerator 靠它判断 use_bias (InstanceNorm 带
# track_running_stats 时 conv 保留 bias, 与 .pth 一致)
_instance_norm = functools.partial(nn.InstanceNorm2d, affine=False,
                                   track_running_stats=True)


class RunningNorm(nn.Module):
    """InstanceNorm(eval, track_running_stats=True) 的可导出等价实现.

    torch 2.1 对 track_running_stats=True 的 InstanceNorm 符号导出报错
    ("instance_norm training for unknown batch size"); eval 语义就是用
    running_mean/var 做固定归一化, 直接写成常量 Sub/Div 原始算子, RKNN
    转换无影响. buffer 名与 InstanceNorm 一致, 权重可直接加载
    (仅 num_batches_tracked 多余, strict=False 跳过)."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps

    def forward(self, x):
        shape = [1, -1] + [1] * (x.dim() - 2)
        return (x - self.running_mean.reshape(shape)) / torch.sqrt(
            self.running_var.reshape(shape) + self.eps)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self._build(dim, padding_type, norm_layer,
                                      use_dropout, use_bias)

    def _build(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias),
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=None,
                 use_dropout=False, n_blocks=9, padding_type='reflect',
                 use_bias=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = _instance_norm
        if use_bias is None:
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------------------
# sketch: fracta1l hw5_gan space 原版结构
# ------------------------------------------------------------------
class SketchResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class SketchGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32,
                 n_residual_blocks=3):
        super().__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels, hidden_channels, kernel_size=7),
                  nn.InstanceNorm2d(hidden_channels),
                  nn.ReLU(inplace=True)]
        curr = hidden_channels
        for _ in range(2):                                   # 下采样 x2
            layers += [nn.Conv2d(curr, curr * 2, kernel_size=3, stride=2,
                                 padding=1),
                       nn.InstanceNorm2d(curr * 2),
                       nn.ReLU(inplace=True)]
            curr *= 2
        for _ in range(n_residual_blocks):
            layers += [SketchResidualBlock(curr)]
        for _ in range(2):                                   # 上采样 x2
            layers += [nn.ConvTranspose2d(curr, curr // 2, kernel_size=3,
                                          stride=2, padding=1,
                                          output_padding=1),
                       nn.InstanceNorm2d(curr // 2),
                       nn.ReLU(inplace=True)]
            curr //= 2
        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(curr, out_channels, kernel_size=7),
                   nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def export(net, sd, out, tag, strict=True):
    missing, unexpected = net.load_state_dict(sd, strict=strict)
    if strict:
        missing = unexpected = ()
    else:
        # RunningNorm 只允许 num_batches_tracked 对不上 (buffer 未注册), 其余必须齐
        assert all(k.endswith("num_batches_tracked") for k in unexpected), \
            f"多余权重: {[k for k in unexpected if not k.endswith('num_batches_tracked')][:5]}"
        assert all(k.endswith("num_batches_tracked") for k in missing), \
            f"缺失权重: {[k for k in missing if not k.endswith('num_batches_tracked')][:5]}"
    net.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = net(x)
    print(f"[{tag}] load OK, 前向 out shape={tuple(y.shape)} "
          f"range=[{y.min():.3f}, {y.max():.3f}]")
    torch.onnx.export(net, x, out, opset_version=11,
                      input_names=["input"], output_names=["output"],
                      do_constant_folding=True)
    print(f"[{tag}] -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    # ukiyoe (RunningNorm: torch 2.1 无法直接导出 track_running_stats InstanceNorm)
    net = ResnetGenerator(norm_layer=RunningNorm, use_bias=True)
    sd = torch.load(os.path.join(D, "ukiyoe_net_G.pth"), map_location="cpu")
    export(net, sd, os.path.join(D, "ukiyoe.onnx"), "ukiyoe", strict=False)

    # sketch
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(D, "sketch_config.json")))
    g = cfg["generator"]
    net = SketchGenerator(g["in_channels"], g["out_channels"],
                          g["hidden_channels"], g["n_residual_blocks"])
    sd = load_file(os.path.join(D, "sketch_generator_a2b.safetensors"))
    export(net, sd, os.path.join(D, "sketch.onnx"), "sketch")


if __name__ == "__main__":
    main()
