import torch
import torch.nn as nn
import functools


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, in_dim, out_dim, activation=nn.ReLU(), ksize=3, pad=1, downsample=False):
        """Initialize the Resnet block"""
        super(ResnetBlock, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.model = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=ksize, padding=pad),
            nn.BatchNorm2d(in_dim),
            activation,
            nn.Conv2d(in_dim, out_dim, kernel_size=ksize, padding=pad),
            nn.BatchNorm2d(out_dim),
        )

        self.sc_layer = (in_dim != out_dim) or downsample
        if self.sc_layer:
            self.sc = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_dim),
            )

    def residual(self, x):
        # x = self.activation(x)
        x = self.model(x)
        if self.downsample:
            x = _downsample(x)

        return x
    
    def shortcut(self, x):
        if self.sc_layer:
            x = self.sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

        