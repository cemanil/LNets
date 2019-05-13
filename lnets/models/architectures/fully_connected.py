import torch.nn as nn

from lnets.models.layers import *
from lnets.models.utils import *
from lnets.models.architectures.base_architecture import Architecture


class FCNet(Architecture):
    def __init__(self, layers, input_dim, linear_type, activation, bias=True, config=None, dropout=False):
        super(FCNet, self).__init__()
        self.config = config

        # Bookkeeping related to layer sizes and Lipschitz constant.
        self.input_dim = input_dim
        self.layer_sizes = layers.copy()
        self.layer_sizes.insert(0, self.input_dim)  # For bookkeeping purposes.
        self.l_constant = config.model.l_constant
        self.num_layers = len(self.layer_sizes)

        # Select activation function and grouping.
        self.act_func = select_activation_function(activation)

        if "groupings" in self.config.model:
            self.groupings = self.config.model.groupings
            self.groupings.insert(0, -1)  # For easier bookkeeping later on.

        # Select linear layer type.
        self.linear_type = linear_type
        self.use_bias = bias
        self.linear = select_linear_layer(self.linear_type)

        # Construct a sequence of linear + activation function layers.
        layers = self._get_sequential_layers(activation=activation,
                                             l_constant_per_layer=self.l_constant ** (1.0 / (self.num_layers - 1)),
                                             config=config, dropout=dropout)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        return self.model(x)

    def _get_sequential_layers(self, activation, l_constant_per_layer, config, dropout=False):
        # First linear transformation.
        # Add layerwise output scaling to control the Lipschitz Constant of the whole network.
        layers = list()
        if dropout:
            layers.append(nn.Dropout(0.2))
        layers.append(self.linear(self.layer_sizes[0], self.layer_sizes[1], bias=self.use_bias, config=config))
        layers.append(Scale(l_constant_per_layer, cuda=self.config.cuda))

        for i in range(1, len(self.layer_sizes) - 1):
            # Determine the downsampling that happens after each activation.
            if activation == "maxout":
                downsampling_factor = (1.0 / self.groupings[i])
            elif activation == "maxmin" or activation == "norm_twist":
                downsampling_factor = (2.0 / self.groupings[i])
            else:
                downsampling_factor = 1.0

            # Add the activation function.
            if activation in ["maxout", "maxmin", "group_sort", "norm_twist"]:
                layers.append(self.act_func(self.layer_sizes[i] // self.groupings[i]))
            else:
                layers.append(self.act_func())

            if dropout:
                layers.append(nn.Dropout(0.5))

            # Add the linear transformations.
            layers.append(
                self.linear(int(downsampling_factor * self.layer_sizes[i]), self.layer_sizes[i + 1], bias=self.use_bias,
                            config=config))
            layers.append(Scale(l_constant_per_layer, cuda=self.config.cuda))

        return layers

    def project_network_weights(self, proj_config):
        # Project the weights on the manifold of orthonormal matrices.
        for i, layer in enumerate(self.model):
            if hasattr(self.model[i], 'project_weights'):
                self.model[i].project_weights(proj_config)

    def get_activations(self, x):
        activations = []
        x = x.view(-1, self.input_dim)
        for m in self.model:
            x = m(x)
            if not isinstance(m, DenseLinear) and not isinstance(m, Scale) and not isinstance(m, nn.Dropout):
                activations.append(x.detach().clone())
        return activations
