import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from .CoordAttention import mixSACA
# from utils import GRN

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x



class Block(nn.Module):
    r""" Conv2NeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, dim_power, drop_path=0., ):
        super().__init__()
        inner_dim = int(dim_power * dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, padding="same")  # depthwise conv
        self.bn0 = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, inner_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(inner_dim)
        self.pwconv2 = nn.Linear(inner_dim, dim)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, padding="same")  # depthwise conv
        self.bn1 = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # dw1
        x = self.dwconv(x)
        x = self.bn0(x)
        inner = x

        # 1x1 -> 1x1
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)

        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = inner + x

        # dw2
        x = self.dwconv2(x)
        x = self.bn1(x)

        x = input + self.drop_path(x)
        return x


class Conv2NeXt(nn.Module):
    r""" Conv2NeXt

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 10
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [128, 128, 256, 256]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=10, patch=1, dim_power=1.0,
                 depths=[3, 3, 9, 3], dims=[128, 128, 256, 256], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., input_size=64
                 ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        if input_size <= 32:
            patch = 1
        else:
            assert input_size % 32 == 0
            patch = input_size // 32
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch, stride=patch),
            # GRNL(dims[0]),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            if i == 1:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first"),
                )
            else:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first"),
                    mixSACA(dims[i + 1], kernel_size=3, reduction=2)
                )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], dim_power=dim_power, drop_path=dp_rates[cur + j]
                        ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

@register_model
def proposed_tiny(pretrained=False, **kwargs):
    model = Conv2NeXt(depths=[3, 3, 6, 3], dims=[96, 96, 192, 192], **kwargs)
    return model


@register_model
def proposed_base(pretrained=False, **kwargs):
    model = Conv2NeXt(depths=[3, 3, 9, 3], dims=[128, 128, 256, 256], dim_power=4, **kwargs)
    return model

