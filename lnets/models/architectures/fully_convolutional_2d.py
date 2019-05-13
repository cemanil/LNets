import torch.nn as nn

from lnets.models.layers import *
from lnets.models.utils import *
from lnets.models.architectures.base_architecture import Architecture


class FullyConv2D(Architecture):
    def __init__(self, in_channels, channels, kernels, strides, linear_type, activation, bias=True, config=None):
        super(FullyConv2D, self).__init__()
        self.config = config

        # Process layer sizes and numbers.
        self.in_channels = in_channels
        self.channels = channels.copy()
        self.channels.insert(0, self.in_channels)
        self.num_layers = len(self.channels)

        # Set kernel sizes and strides.
        self.kernels = kernels
        self.strides = strides

        # Lipschitz constant of the whole module. l_correction_constant is used to make sure the Lipschitz constant
        # of the convnet is 1 _without_ the other lipschitz constant.
        l_constant = config.model.l_constant
        l_correction_constant = config.model.l_correction_constant
        l_constant_per_layer = (l_constant * l_correction_constant) ** (1.0 / (self.num_layers - 1))

        # Other parameters of the convolutional network.
        self.conv_parameters = dict(padding=config.model.padding, dilation=config.model.dilation,
                                    groups=config.model.groups, bias=bias, config=config)

        # Select activation function and grouping.
        self.act_func = select_activation_function(activation)

        if "groupings" in self.config.model:
            self.groupings = self.config.model.groupings
            self.groupings.insert(0, -1)  # For easier bookkeeping later on.

        # Select linear layer type.
        self.linear_type = linear_type
        self.use_bias = bias
        self.linear = select_linear_layer(self.linear_type)

        # Construct a sequence of linear + activation function layers. The last layer is linear.
        layers = self._get_sequential_layers(activation, l_constant_per_layer, self.linear)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _get_sequential_layers(self, activation, l_constant_per_layer, linear):
        layers = list()

        # The first linear layer. Note the scaling layer is to have control over the Lipschitz constant of the network.
        layers.append(linear(self.channels[0], self.channels[1], kernel_size=self.kernels[0], stride=self.strides[0],
                             **self.conv_parameters))
        layers.append(Scale(l_constant_per_layer, cuda=self.config.cuda))

        # Series of activation + linear. Control Lipsthitz constant of the network by adding the scaling layers.
        for i in range(1, len(self.channels) - 1):
            # Determine the downsampling that happens after each activation.
            if activation == "maxout":
                downsampling_factor = (1.0 / self.groupings[i])
            elif activation == "maxmin":
                downsampling_factor = (2.0 / self.groupings[i])
            else:
                downsampling_factor = 1.0

            # Add the activation function.
            if activation in ["maxout", "maxmin", "group_sort"]:
                layers.append(self.act_func(self.channels[i] // self.groupings[i], axis=1))
            else:
                layers.append(self.act_func())

            # Add the linear transformations.
            layers.append(linear(int(downsampling_factor * self.channels[i]), self.channels[i+1],
                                 kernel_size=self.kernels[i], stride=self.strides[i], **self.conv_parameters))
            layers.append(Scale(l_constant_per_layer, cuda=self.config.cuda))

        return layers
