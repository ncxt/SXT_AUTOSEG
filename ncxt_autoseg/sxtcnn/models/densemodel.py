import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

if __name__ == "__main__":
    from torchsummary import summary
else:
    from .torchsummary import summary

# adapted from https://github.com/bfortuner/pytorch_tiramisu


def conv3x3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )


def conv1x1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module("norm", nn.BatchNorm3d(in_channels))
        self.add_module("relu", nn.ReLU(True))
        self.add_module("conv", conv3x3x3(in_channels, growth_rate))
        self.add_module("drop", nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList(
            [
                DenseLayer(in_channels + i * growth_rate, growth_rate)
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        x_size = x.shape
        if self.upsample:
            new_features = []
            # we pass all previous activations into each dense layer normally
            # But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            x = torch.cat(new_features, 1)
            if DEBUG:
                print(
                    f"{self.__class__.__name__: <20} {list(x_size)} to {list(x.shape)}"
                )
            return x
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)  # 1 = channel axis
            if DEBUG:
                print(
                    f"{self.__class__.__name__: <20} {list(x_size)} to {list(x.shape)}"
                )
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module("norm", nn.BatchNorm3d(num_features=in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", conv3x3x3(in_channels, in_channels))
        self.add_module("drop", nn.Dropout3d(0.2))
        self.add_module("maxpool", nn.MaxPool3d(kernel_size=2, stride=2))

    def forward(self, x):
        x_size = x.shape
        x = super().forward(x)
        if DEBUG:
            print(f"{self.__class__.__name__: <20} {list(x_size)} to {list(x.shape)}")
        return x


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x, skip):
        x_size = x.shape
        skip_size = skip.shape
        out = self.convTrans(x)
        out = torch.cat([out, skip], 1)
        if DEBUG:
            print(
                f"{self.__class__.__name__: <20} {list(x_size)}+{list(skip_size)} to {list(out.shape)}"
            )
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module(
            "bottleneck", DenseBlock(in_channels, growth_rate, n_layers, upsample=True)
        )

    def forward(self, x):
        return super().forward(x)


class FCDenseNet3D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5),
        bottleneck_layers=5,
        growth_rate=16,
        out_chans_first_conv=48,
        n_classes=12,
    ):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module("firstconv", conv3x3x3(in_channels, out_chans_first_conv))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i])
            )
            cur_channels_count += growth_rate * down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module(
            "bottleneck", Bottleneck(cur_channels_count, growth_rate, bottleneck_layers)
        )
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(
                TransitionUp(prev_block_channels, prev_block_channels)
            )
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(
                DenseBlock(cur_channels_count, growth_rate, up_blocks[i], upsample=True)
            )
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(
            TransitionUp(prev_block_channels, prev_block_channels)
        )
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(
            DenseBlock(cur_channels_count, growth_rate, up_blocks[-1], upsample=False)
        )
        cur_channels_count += growth_rate * up_blocks[-1]

        ## Softmax ##

        self.finalConv = conv3x3x3(cur_channels_count, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        return out

    def summary(self, shape):
        summary(self, shape, device="cpu")


DEBUG = False
if __name__ == "__main__":
    DEBUG = False
    print(f"Testing debug {DEBUG}")
    """
    testing
    """

    model = FCDenseNet3D(
        in_channels=1,
        down_blocks=(2, 3, 4),
        up_blocks=(4, 3, 2),
        bottleneck_layers=4,
        growth_rate=8,
        out_chans_first_conv=16,
        n_classes=2,
    )

    if DEBUG:
        x = Variable(torch.FloatTensor(np.random.random((1, 1, 32, 32, 32))))
        out = model(x)
    else:
        summary(model, (1, 32, 32, 32), device="cpu")
