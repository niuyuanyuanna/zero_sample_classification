#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/28/18 4:23 PM
# @Author  : NYY
# @Site    : www.niuyuanyuanna@github.io
import torch.nn as nn
import math
import numpy as np


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, mid_channels, stride)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_channels, mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        """
        :param in_channels:
        :param mid_channels:
        :param stride:
        :param downsample:
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def get_blocks_layer_params(layers):
    if layers == 18:
        return BasicBlock, [2, 2, 2, 2]
    if layers == 34:
        return BasicBlock, [3, 4, 6, 3]
    if layers == 50:
        return Bottleneck, [3, 4, 6, 3]
    if layers == 101:
        return Bottleneck, [3, 4, 23, 3]
    if layers == 152:
        return Bottleneck, [3, 8, 36, 3]


class ResNet(nn.Module):

    def __init__(self, config, pretrained=True, remove_output=True, num_classes=1000):
        """
        :param config: resnet layers, such as 18, 34, 50, 101, 152
        :param pretrained: weather if the model is pre-trained
        :param remove_output:
        :param num_classes: output channel, standard is 1000
        """
        super(ResNet, self).__init__()
        self.config = config
        self.pretrained = pretrained
        self.remove_output = remove_output
        layers = int(config.network.split('#')[1])

        self.lock, self.layer_nums = get_blocks_layer_params(layers)

    def _make_layer(self, block, block_nums, mid_channels, stride=1):
        downsample = None

        if stride != 1 or self.base_chanels != mid_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.base_channels, mid_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * block.expansion)
            )
        layers = list()
        layers.append(block(self.base_channels, mid_channels, stride, downsample))
        self.base_channels = mid_channels * block.expansion
        for i in range(1, block_nums):
            layers.append(block(self.base_channels, mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if self.remove_output:
            return x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




