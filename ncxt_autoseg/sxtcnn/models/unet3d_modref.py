from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

if __name__ == "__main__" or not __package__:
    from torchsummary import summary
else:
    from .torchsummary import summary

# adapted from https://github.com/jaxony/unet-pytorch

# TODO: argument for batchnorm and dropout


def conv3x3x3(in_channels,
              out_channels,
              stride=1,
              padding=0,
              bias=True,
              groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1x1(in_channels, out_channels))


def conv1x1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=None,
                 batchnorm=None,
                 padding=1,
                 pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.padding = padding
        self.dropout3d = None
        self.batch_norm = None

        self.conv1 = conv3x3x3(
            self.in_channels, self.out_channels, padding=self.padding)
        self.conv2 = conv3x3x3(
            self.out_channels, self.out_channels, padding=self.padding)

        if dropout is not None:
            self.dropout3d = nn.Dropout3d(p=dropout)

        if batchnorm is not None:
            self.batch_norm = nn.BatchNorm3d(self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        orig = x.shape
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.dropout3d is not None:
            x = self.dropout3d(x)

        before_pool = x
        if self.pooling:
            x = self.pool(x)

        if DEBUG: print(f'DownConv {orig} -> {x.shape}')
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 batchnorm=None,
                 merge_mode='concat',
                 up_mode='transpose',
                 padding=1):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.padding = padding
        self.batch_norm = None

        self.upconv = upconv2x2x2(
            self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3x3(
                2 * self.out_channels, self.out_channels, padding=self.padding)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3x3(
                self.out_channels, self.out_channels, padding=self.padding)
        self.conv2 = conv3x3x3(
            self.out_channels, self.out_channels, padding=self.padding)
        if batchnorm is not None:
            self.batch_norm = nn.BatchNorm3d(self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up_orig = from_up.shape
        from_up = self.upconv(from_up)

        if self.padding == 0:
            l_from_down = from_up.shape[-1]
            l_from_up = from_down.shape[-1]
            l_crop = (int)((l_from_up - l_from_down) / 2)
            from_down = from_down[:, :, l_crop:-l_crop, l_crop:-l_crop, l_crop:
                                  -l_crop]
            if DEBUG:
                print(
                    f'UpConv: conc1: from_down {l_from_down} from_up {l_from_up} crop = {l_crop}'
                )

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if DEBUG:
            print(
                f'UpConv from_down {from_down.shape} from_up {from_up_orig} -> {x.shape}'
            )
        return x


class UNet3D(nn.Module):
    """ `UNet3D` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).


    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self,
                 num_classes,
                 in_channels=3,
                 skip_channels=0,
                 depth=5,
                 start_filts=64,
                 dropout=None,
                 batchnorm=None,
                 padding=1,
                 up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet3D, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.start_filts = start_filts
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.depth = depth
        self.padding = padding

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(
                ins,
                outs,
                pooling=pooling,
                padding=self.padding,
                batchnorm=self.batchnorm,
                dropout=self.dropout)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(
                ins,
                outs,
                up_mode=up_mode,
                merge_mode=merge_mode,
                padding=self.padding,
                batchnorm=self.batchnorm)
            self.up_convs.append(up_conv)

        if self.skip_channels > 0:
            self.feature_pool = conv1x1x1(outs + skip_channels, outs)

        self.conv_final = conv1x1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):

        encoder_outs = []
        if self.skip_channels > 0:
            x_skip = x[:, self.in_channels:, :]
            x = x[:, :self.in_channels, :]
            if DEBUG:
                print(f'Input_forward {x.shape} Input_skip {x_skip.shape}')

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.

        # concatenate with skipped inout data
        if self.skip_channels > 0:
            x = torch.cat((x, x_skip), 1)
            x_orig = x.shape
            x = self.feature_pool(x)
            if DEBUG: print(f'feature pool form  {x_orig} -> {x.shape}')

        x_orig = x.shape
        x = self.conv_final(x)

        if DEBUG: print(f'UNet final {x_orig} -> {x.shape}')
        return x

    def features(self, x):
        encoder_outs = []

        if self.skip_channels > 0:
            x_skip = x[:, self.in_channels:, :]
            x = x[:, :self.in_channels, :]

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        if self.skip_channels > 0:
            x = torch.cat((x, x_skip), 1)
            x = self.feature_pool(x)

        return x

    def summary(self, shape):
        return summary(self, shape, device='cpu')


DEBUG = False
if __name__ == "__main__":
    DEBUG = True
    print(f'Testing debug {DEBUG}')
    """
    testing
    """
    in_channels = 2
    out_channels = 3  #TODO:fix
    start_filts = 8
    depth = 3
    model = UNet3D(
        out_channels,
        in_channels=in_channels,
        skip_channels=start_filts,
        depth=depth,
        start_filts=start_filts,
        batchnorm=True)

    if DEBUG:
        x = Variable(
            torch.FloatTensor(
                np.random.random((1, in_channels + start_filts, 16, 16, 16))))
        out = model(x)
    else:
        model.summary((1, 32, 32, 32))
