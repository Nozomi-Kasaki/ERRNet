# Define network components here
import torch
from torch import nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y        
     

class DRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d, 
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False):
        super(DRNet, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        
        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        # Residual layers
        dilation_config = [1] * n_resblocks

        self.res_module = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=dilation_config[i], norm=norm, act=act, 
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])

        # Upsampling Layers
        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)

        if not pyramid:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        else:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4,8,16,32), ct_channels=n_feats//4)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_module(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)
        x = self.deconv3(x)

        return x


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        maxv = F.adaptive_max_pool2d(x, 1)
        return x * self.sigmoid(self.mlp(avg) + self.mlp(maxv))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.max(x, dim=1, keepdim=True)[0]
        return x * self.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super(CBAM, self).__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


class ResidualDenseAttentionBlock(nn.Module):
    def __init__(self, channels, growth_channels=32, num_layers=4, res_scale=0.2):
        super(ResidualDenseAttentionBlock, self).__init__()
        self.res_scale = res_scale
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels + i * growth_channels, growth_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        self.local_fusion = nn.Conv2d(channels + num_layers * growth_channels, channels, kernel_size=1)
        self.attention = CBAM(channels, reduction=8)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        out = self.local_fusion(torch.cat(features, dim=1))
        out = self.attention(out)
        return x + out * self.res_scale


class RDABlockStack(nn.Sequential):
    def __init__(self, channels, num_blocks, growth_channels=32):
        super(RDABlockStack, self).__init__(
            *[ResidualDenseAttentionBlock(channels, growth_channels=growth_channels) for _ in range(num_blocks)])


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.body(x)


class UpsampleFuseBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=1, growth_channels=32):
        super(UpsampleFuseBlock, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            RDABlockStack(out_channels, num_blocks, growth_channels=growth_channels)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.pre(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class ASPPContext(nn.Module):
    def __init__(self, channels, dilations=(1, 2, 4, 8)):
        super(ASPPContext, self).__init__()
        branch_channels = channels // 2
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * (len(dilations) + 1), channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(channels, reduction=8)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        global_feature = F.interpolate(self.global_branch(x), size=(h, w), mode='bilinear', align_corners=False)
        features.append(global_feature)
        return self.fuse(torch.cat(features, dim=1))


class SobelEdgeExtractor(nn.Module):
    def __init__(self):
        super(SobelEdgeExtractor, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, rgb):
        rgb = torch.clamp(rgb, 0, 1)
        gray = rgb.mean(dim=1, keepdim=True)
        gray = F.pad(gray, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(gray, self.sobel_x)
        gy = F.conv2d(gray, self.sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge_max = edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return edge / edge_max


class RefinementResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RefinementResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            CBAM(channels, reduction=8)
        )

    def forward(self, x):
        return x + self.body(x) * 0.1


class ReflectionGatedRefinement(nn.Module):
    """Small residual adapter that edits a pretrained ERRNet output conservatively."""
    def __init__(self, channels=48, num_blocks=4, max_delta=0.12, residual_scale=0.08):
        super(ReflectionGatedRefinement, self).__init__()
        self.max_delta = max_delta
        self.adapter_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.edge = SobelEdgeExtractor()
        self.stem = nn.Sequential(
            nn.Conv2d(11, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.body = nn.Sequential(*[RefinementResidualBlock(channels) for _ in range(num_blocks)])
        self.delta_head = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self._init_heads()

    def _init_heads(self):
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.delta_head.bias, 0.0)
        nn.init.normal_(self.mask_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.mask_head.bias, -2.0)

    def forward(self, input_rgb, base_output):
        input_rgb = torch.clamp(input_rgb, 0, 1)
        base_rgb = torch.clamp(base_output, 0, 1)
        reflection_hint = input_rgb - base_rgb
        features = torch.cat([
            input_rgb,
            base_rgb,
            reflection_hint,
            self.edge(input_rgb),
            self.edge(base_rgb)
        ], dim=1)
        hidden = self.body(self.stem(features))
        mask = torch.sigmoid(self.mask_head(hidden))
        delta = torch.tanh(self.delta_head(hidden)) * self.max_delta
        correction = self.adapter_scale * mask * delta
        output = torch.clamp(base_output + correction, 0, 1)
        return output, correction, mask


class ERRNetGatedAdapter(nn.Module):
    """Pretrained ERRNet backbone plus a gated residual refinement adapter.

    Old ERRNet checkpoints can be loaded directly: their weights are mapped into
    the internal backbone and the adapter remains near identity.
    """
    def __init__(self, in_channels, out_channels=3, adapter_channels=48):
        super(ERRNetGatedAdapter, self).__init__()
        self.backbone = DRNet(
            in_channels, out_channels, 256, 13, norm=None, res_scale=0.1,
            se_reduction=8, bottom_kernel_size=1, pyramid=True)
        self.refiner = ReflectionGatedRefinement(channels=adapter_channels)
        self.last_backbone = None
        self.last_correction = None
        self.last_mask = None

    def _match_input_size(self, tensor, input_size):
        h, w = input_size
        th, tw = tensor.shape[-2:]
        if (th, tw) == (h, w):
            return tensor
        if th >= h and tw >= w:
            return tensor[..., :h, :w]
        return F.interpolate(tensor, size=(h, w), mode='bilinear', align_corners=False)

    def forward(self, x):
        base_output = self.backbone(x)
        base_output = self._match_input_size(base_output, x.shape[-2:])
        output, correction, mask = self.refiner(x[:, :3], base_output)
        self.last_backbone = base_output
        self.last_correction = correction
        self.last_mask = mask
        return output

    def set_backbone_trainable(self, trainable):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def optimizer_param_groups(self, base_lr, backbone_lr_scale=1.0, adapter_lr_scale=1.0):
        return [
            {
                'params': list(self.backbone.parameters()),
                'lr': base_lr * backbone_lr_scale,
                'lr_scale': backbone_lr_scale,
                'name': 'backbone',
            },
            {
                'params': list(self.refiner.parameters()),
                'lr': base_lr * adapter_lr_scale,
                'lr_scale': adapter_lr_scale,
                'name': 'adapter',
            },
        ]

    def load_state_dict(self, state_dict, strict=True):
        if state_dict and not any(key.startswith(('backbone.', 'refiner.')) for key in state_dict.keys()):
            print('[i] loading legacy ERRNet weights into gated-adapter backbone')
            return self.backbone.load_state_dict(state_dict, strict=strict)
        return super(ERRNetGatedAdapter, self).load_state_dict(state_dict, strict=strict)


class RDAUNet(nn.Module):
    """Residual Dense Attention U-Net with edge guidance.

    The network predicts a logit-space correction over the input RGB image.
    This identity-biased output keeps early training PSNR stable while the
    U-Net path learns reflection suppression.
    """
    def __init__(self, in_channels, out_channels=3, base_channels=64, growth_channels=32):
        super(RDAUNet, self).__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        )

        self.enc1 = RDABlockStack(c1, num_blocks=2, growth_channels=growth_channels)
        self.down1 = DownsampleBlock(c1, c2)
        self.enc2 = RDABlockStack(c2, num_blocks=2, growth_channels=growth_channels)
        self.down2 = DownsampleBlock(c2, c3)

        self.bottleneck = nn.Sequential(
            RDABlockStack(c3, num_blocks=4, growth_channels=growth_channels),
            ASPPContext(c3),
            RDABlockStack(c3, num_blocks=2, growth_channels=growth_channels)
        )

        self.up2 = UpsampleFuseBlock(c3, c2, c2, num_blocks=2, growth_channels=growth_channels)
        self.up1 = UpsampleFuseBlock(c2, c1, c1, num_blocks=2, growth_channels=growth_channels)
        self.refine = RDABlockStack(c1, num_blocks=2, growth_channels=growth_channels)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self._init_identity_bias()

    def _init_identity_bias(self):
        nn.init.normal_(self.out_conv.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.out_conv.bias, 0.0)

    def _edge_map(self, x):
        rgb = torch.clamp(x[:, :3], 0, 1)
        gray = rgb.mean(dim=1, keepdim=True)
        gray = F.pad(gray, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(gray, self.sobel_x)
        gy = F.conv2d(gray, self.sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge_max = edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return edge / edge_max

    def forward(self, x):
        base_rgb = torch.clamp(x[:, :3], 1e-4, 1 - 1e-4)
        base_logit = torch.log(base_rgb / (1 - base_rgb))

        stem = self.stem(x) + self.edge_proj(self._edge_map(x))
        e1 = self.enc1(stem)
        e2 = self.enc2(self.down1(e1))
        bottleneck = self.bottleneck(self.down2(e2))
        d2 = self.up2(bottleneck, e2)
        d1 = self.up1(d2, e1)
        correction = self.out_conv(self.refine(d1))
        return torch.sigmoid(base_logit + correction)
