import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import ResLayerDG
from ..utils.mask import Mask_s, Mask_c


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 eta=8,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None, **kwargs):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None

        # mask
        self.mask_s  = Mask_s( inplanes, eta, **kwargs)  
        self.mask_c1 = Mask_c( inplanes, planes, **kwargs)
        self.mask_c2 = Mask_c( planes, planes, **kwargs)

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # mask flops
        flops_mask_c1 = self.mask_c1.get_flops()
        flops_mask_c2 = self.mask_c2.get_flops()
        self.flops_mask = torch.Tensor([flops_mask_c1 + flops_mask_c2])


    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(input):
            x, norm_1, norm_2, flops = input

            identity = x
            _,_,h1,w1 = x.shape
            # spatial mask
            mask_s_m, norm_s = self.mask_s(x) # [N, 1, h, w]
            mask_c1, norm_c1, norm_c1_t = self.mask_c1(x)
            # conv 1
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            mask_s1 = nn.functional.upsample_nearest(mask_s_m, size=(h1,w1)) # [N, 1, H1, W1]
            out = out * mask_c1 * mask_s1 if not self.training else out * mask_c1
            # conv 2
            mask_c2, norm_c2, norm_c2_t = self.mask_c2(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            _,_,h2,w2 = out.shape
            mask_s = nn.functional.upsample_nearest(mask_s_m, size=(h2,w2)) # [N, 1, H2, W2]
            out = out * mask_c2 * mask_s if not self.training else out * mask_c2
            # conv 3
            out = self.conv3(out)
            out = self.norm3(out)
            out = out * mask_s
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
            # norm
            norm_1 = torch.cat((norm_1, torch.cat((norm_s, torch.Tensor([mask_s_m[0,].numel()]).to(x.device))).unsqueeze(0)))
            norm_2 = torch.cat((norm_2, torch.cat((norm_c1, norm_c1_t)).unsqueeze(0)))
            norm_2 = torch.cat((norm_2, torch.cat((norm_c2, norm_c2_t)).unsqueeze(0)))
            # flops
            flops_blk = self.get_flops(mask_s_m, mask_s, mask_s1, mask_c1, mask_c2, h1, w1, h2, w2)
            flops = torch.cat((flops, flops_blk))
            return (out, norm_1, norm_2, flops)

        out = _inner_forward(x)
        return out

    def get_flops(self, mask_s_m, mask_s, mask_s1, mask_c1, mask_c2, h1, w1, h2, w2):
        s_sum = mask_s.sum((1,2,3))
        c1_sum, c2_sum = mask_c1.sum((1,2,3)), mask_c2.sum((1,2,3))
        # conv
        s_sum_1 = mask_s1.sum((1,2,3))
        flops_conv1 = s_sum_1 * c1_sum * self.inplanes
        flops_conv2 = 9 * s_sum * c2_sum * c1_sum
        flops_conv3 = s_sum * self.planes * self.expansion * c2_sum
        # mask_s
        flops_mask_s  = self.mask_s.get_flops(mask_s_m)
        # total
        flops = flops_conv1+flops_conv2+flops_conv3
        # total flops
        flops_conv1_full = torch.Tensor([h1 * w1 * self.planes * self.inplanes])
        flops_conv2_full = torch.Tensor([9 * h2 * w2 * self.planes * self.planes])
        flops_conv3_full = torch.Tensor([h2 * w2 * self.planes * self.planes*self.expansion])
        flops_full = (flops_conv1_full+flops_conv2_full+flops_conv3_full).to(flops.device)
        if self.downsample is not None:
            flops_dw = torch.Tensor([h2 * w2*self.planes*self.expansion*self.inplanes]).to(flops.device)
            flops += flops_dw
            flops_full +=  flops_dw
        return torch.cat((flops, flops_full)).unsqueeze(0)


@BACKBONES.register_module()
class ResNetDG(nn.Module):
    """ResNetDG backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 den_tar=0.5,
                 lbda=5,
                 gamma=1,
                 eta=8,
                 bias=-1):
        super(ResNetDG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.den_tar = den_tar
        self.lbda  = lbda
        self.gamma = gamma
        self.eta = eta

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            self.eta = int(self.eta/2) if stride==2 else self.eta
            dcn = self.dcn if self.stage_with_dcn[i] else None
            stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                eta=self.eta,
                bias=bias,
                plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)


    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayerDG(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            pretrained_dict = checkpoint['state_dict']
            missing_keys = [k for k in model_dict if k not in pretrained_dict]
            unexpected_keys = [k for k in pretrained_dict if k not in model_dict and 'eta' not in k]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            if unexpected_keys:
                print(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}\n')
            if missing_keys:
                print(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        batch_num, _, _, _ = x.shape
        # residual modules
        norm1 = torch.zeros(1, batch_num+1).to(x.device)
        norm2 = torch.zeros(1, batch_num+1).to(x.device)
        flops = torch.zeros(1, batch_num+1).to(x.device)
        # conv
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        _, _, h, w = x.shape        
        x = self.maxpool(x)
        outs = []
        x = (x, norm1, norm2, flops)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x[0])
        x, norm1, norm2, flops = x
        # norm and flops
        # norm_s = norm1[1:, 0:batch_num].permute(1, 0).contiguous()
        # norm_c = norm2[1:, 0:batch_num].permute(1, 0).contiguous()
        # norm_s_t = norm1[1:, -1].unsqueeze(0)
        # norm_c_t = norm2[1:, -1].unsqueeze(0)
        flops_conv1 = torch.Tensor([49*h*w*self.stem_channels*3]).to(x.device)
        flops_real = flops[1:, 0:batch_num].permute(1, 0).contiguous()
        flops_ori  = flops[1:, -1].unsqueeze(0)
        # get loss
        loss_spar = self.spar_loss(flops_real, flops_ori, flops_conv1,
                                   self.den_tar, self.lbda)
        if self.training:
            return tuple(outs), loss_spar
        else:
            flops_conv = flops_real.mean(0).sum()
            flops_ori = flops_ori.mean(0).sum() + flops_conv1
            flops_real = flops_conv + flops_conv1
            return tuple(outs), loss_spar, flops_real, flops_ori
    
    def spar_loss(self, flops_real, flops_ori, flops_conv1, den_target, lbda):
        # block flops
        flops_conv = flops_real.mean(0).sum()
        flops_ori = flops_ori.mean(0).sum() + flops_conv1
        flops_real = flops_conv + flops_conv1
        # loss
        rloss = lbda * (flops_real / flops_ori - den_target)**2
        return rloss

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNetDG, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
