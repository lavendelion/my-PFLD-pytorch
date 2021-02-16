import torch.nn as nn
import torch
from PIL import Image, ImageDraw
import random


class Conv2d_withBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True):
        super(Conv2d_withBN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True):
        super(DepthwiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=bias),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, t_factor, padding=0, bias=True):
        super(InvertedBottleneck, self).__init__()
        self.stride = stride
        self.residual_add = True if (self.stride==1) and (in_channels==out_channels) else False
        mid_channels = t_factor*in_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=mid_channels, bias=bias),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs+inputs if self.residual_add else outputs

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        n_samples = x.shape[0]
        x = x.reshape(n_samples, -1)
        return x


if __name__ == '__main__':
    print(random.random())