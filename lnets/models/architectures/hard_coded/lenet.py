import torch.nn as nn
import torch.nn.functional as F

from lnets.models.layers import *
from lnets.models.activations import MaxMin, GroupSort


class LeNet(nn.Module):
    def __init__(self, in_channels, output_dim, linear_type, activation, dropout_on, config):
        super(LeNet, self).__init__()
        self.config = config

        self.scale = Scale(config.model.l_constant ** 0.25, config.cuda)
        # Select linear layer type.
        if linear_type == "bjorck":
            conv = BjorckConv2d
            linear = BjorckLinear
        elif linear_type == "standard":
            conv = StandardConv2d
            linear = StandardLinear
        else:
            conv = None
            linear = None
            print("Layer type not supported. ")
            exit(-1)

        # Select activation.
        self.act_type = activation
        if activation == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
        elif activation == "maxmin":
            self.act1 = MaxMin(5, axis=1)
            self.act2 = MaxMin(10, axis=1)
            self.act3 = MaxMin(25)
        elif activation == "group_sort":
            self.act1 = GroupSort(1, axis=1)
            self.act2 = GroupSort(1, axis=1)
            self.act3 = GroupSort(1)
        else:
            print("Activation not supported. ")
            exit(-1)

        # Save dropout_on option.
        self.dropout_on = dropout_on

        self.conv1 = conv(in_channels, 10, kernel_size=5, config=self.config)
        self.conv2 = conv(10, 20, kernel_size=5, config=self.config)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = linear(320, 50, config=config)
        self.fc2 = linear(50, output_dim, config=config)

    def forward(self, x):
        # Layer 1.
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.act1(x)
        x = self.scale(x)

        # Layer 2.
        x = self.conv2(x)
        if self.dropout_on:
            x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = self.act2(x)
        x = self.scale(x)

        # Reshape.
        x = x.view(-1, 320)

        # Layer 3.
        x = self.fc1(x)
        x = self.act3(x)
        x = self.scale(x)

        if self.dropout_on:
            x = F.dropout(x, training=self.training)

        # Layer 4.
        x = self.fc2(x)
        x = self.scale(x)

        return x
