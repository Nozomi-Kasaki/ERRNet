# Add your custom network here
from .default import DRNet, RDAUNet, ERRNetGatedAdapter, ERRNetPP
import torch.nn as nn


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)


def rdaunet(in_channels, out_channels, **kwargs):
    return RDAUNet(in_channels, out_channels, base_channels=64, growth_channels=32)


def errnet_adapter(in_channels, out_channels, **kwargs):
    return ERRNetGatedAdapter(in_channels, out_channels, adapter_channels=48)


def errnet_pp(in_channels, out_channels, **kwargs):
    return ERRNetPP(in_channels, out_channels, base_channels=128, growth_channels=32)
