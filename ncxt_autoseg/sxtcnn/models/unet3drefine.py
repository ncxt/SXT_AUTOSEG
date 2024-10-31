from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

if __name__ == "__main__" or not __package__:
    from torchsummary import summary
    from unet3d import UNet3D, conv1x1x1
else:
    from .torchsummary import summary
    from .unet3d import UNet3D, conv1x1x1


class RefUNet3D(nn.Module):
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
        super(RefUNet3D, self).__init__()

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

        self.unet = UNet3D(num_classes, in_channels, depth, start_filts,
                           dropout, batchnorm, padding, up_mode, merge_mode)

        self.feature_pool = conv1x1x1(start_filts + skip_channels, start_filts)
        self.conv_final = conv1x1x1(start_filts, self.num_classes)

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

        x_skip = x[:, self.in_channels:, :]
        x = x[:, :self.in_channels, :]
        if DEBUG: print(f'x {x.shape} x_skip {x_skip.shape}')

        x_unet = self.unet.features(x)

        if DEBUG: print(f'x_unet {x_unet.shape}')

        x = torch.cat((x_skip, x_unet), 1)
        x = self.feature_pool(x)
        x = self.conv_final(x)

        return x

    def features(self, x):
        x_skip = x[:, self.in_channels:, :]
        x = x[:, :self.in_channels, :]
        x_unet = self.unet.features(x)

        x = torch.cat((x_skip, x_unet), 1)
        x = self.feature_pool(x)

        return x

    def summary(self, shape):
        return summary(self, shape, device='cpu')


DEBUG = False
if __name__ == "__main__":
    # DEBUG = True
    print(f'Testing debug {DEBUG}')
    """
    testing
    """
    in_channels = 2
    out_channels = 3  #TODO:fix
    start_filts = 8
    depth = 3
    model = RefUNet3D(
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
        model.summary((in_channels + start_filts, 32, 32, 32))
