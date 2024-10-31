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


def conv3x3x3(
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    padding_mode="zeros",
    bias=True,
    groups=1,
):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        bias=bias,
        groups=groups,
    )


def upconv2x2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            conv1x1x1(in_channels, out_channels),
        )


def conv1x1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=None,
        instancenorm=True,
        padding_mode="zeros",
    ):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = None
        self.instance_norm = None
        self.conv = conv3x3x3(
            self.in_channels, self.out_channels, padding_mode=padding_mode
        )

        if dropout:
            self.dropout = nn.Dropout3d(p=dropout)

        if instancenorm:
            self.instance_norm = nn.InstanceNorm3d(self.out_channels)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        return F.relu(x)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=None,
        instancenorm=True,
        padding_mode="zeros",
        pooling=True,
    ):
        super(DownConv, self).__init__()
        self.pooling = pooling
        self.block1 = ConvBlock(
            in_channels, out_channels, dropout, instancenorm, padding_mode=padding_mode
        )
        self.block2 = ConvBlock(
            out_channels, out_channels, dropout, instancenorm, padding_mode=padding_mode
        )
        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        orig = x.shape
        x = self.block1(x)
        if DEBUG:
            print(f"DownConv {orig} -> {x.shape}")
        orig = x.shape
        x = self.block2(x)
        if DEBUG:
            print(f"DownConv {orig} -> {x.shape}")

        before_pool = x
        if self.pooling:
            x = self.pool(x)
        if DEBUG:
            print(f"DownConv {orig} -> {x.shape}")
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=None,
        instancenorm=True,
        padding_mode="zeros",
        up_mode="transpose",
    ):
        super(UpConv, self).__init__()

        self.upconv = upconv2x2x2(in_channels, out_channels, mode=up_mode)
        self.block1 = ConvBlock(
            2 * out_channels,
            out_channels,
            dropout,
            instancenorm,
            padding_mode=padding_mode,
        )
        self.block2 = ConvBlock(
            out_channels, out_channels, dropout, instancenorm, padding_mode=padding_mode
        )

    def forward(self, from_down, from_up):
        """Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up_orig = from_up.shape
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = self.block1(x)
        x = self.block2(x)

        if DEBUG:
            print(
                f"UpConv from_down {from_down.shape} from_up {from_up_orig} -> {x.shape}"
            )
        return x


class UNet3D(nn.Module):
    """`UNet3D` class is based on https://arxiv.org/abs/1505.04597

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

    def __init__(
        self,
        num_classes,
        in_channels=1,
        depth=3,
        start_filts=32,
        dropout=0,
        instancenorm=False,
        padding_mode="zeros",
        up_mode="transpose",
    ):
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

        if up_mode in ("transpose", "upsample"):
            self.up_mode = up_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for '
                'upsampling. Only "transpose" and '
                '"upsample" are allowed.'.format(up_mode)
            )

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.dropout = dropout
        self.instancenorm = instancenorm
        self.padding_mode = padding_mode
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(
                ins,
                outs,
                dropout=self.dropout,
                instancenorm=self.instancenorm,
                padding_mode=self.padding_mode,
                pooling=pooling,
            )
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
                dropout=self.dropout,
                instancenorm=self.instancenorm,
                padding_mode=self.padding_mode,
            )
            self.up_convs.append(up_conv)

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
        # print("\tIn Model: input size", x.size() )
        encoder_outs = []

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
        x_orig = x.shape
        x = self.conv_final(x)

        if DEBUG:
            print(f"UNet final {x_orig} -> {x.shape}")
        return x

    def features(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        return x

    def summary(self, shape):
        return summary(self, shape, device="cpu")

    def receptive_fied(self, tensor):
        with torch.no_grad():
            # store params
            base_params = dict(self.named_parameters())
            stored_params = copy.deepcopy(base_params)
            # test weights
            for name_base, param_base in self.named_parameters():
                if "bias" in str(name_base):
                    base_params[name_base].data.copy_(param_base.data * 0)
                if "weight" in str(name_base):
                    base_params[name_base].data.copy_(
                        param_base.data * 0 + 1.0 / param_base.data[0, :].nelement()
                    )
            features = self.features(tensor)
            features_abs = torch.abs(features)
            retval = torch.sum(features_abs, dim=(0, 1))

            # restore params
            for name_base, param_base in stored_params.items():
                base_params[name_base].data.copy_(param_base.data)

            return retval / torch.max(retval)


import copy

DEBUG = False
if __name__ == "__main__":
    DEBUG = False
    print(f"Testing debug {DEBUG}")
    """
    testing
    """
    L = 64
    MODEL = UNet3D(2, in_channels=1, depth=4, start_filts=16)

    if DEBUG:
        TEST_TENSOR = Variable(torch.FloatTensor(np.random.random((1, 1, L, L, L))))
        OUT = MODEL(TEST_TENSOR)
    else:
        # MODEL.summary((1, L, L, L))
        TEST_TENSOR = Variable(torch.FloatTensor(np.random.random((1, 1, L, L, L))))
        rf = MODEL.receptive_fied(TEST_TENSOR)
        print(rf.shape)
        rf = rf.cpu().numpy()
        print(f"rf {np.min(rf)} -- {np.max(rf)}")
