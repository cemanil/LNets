import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

from lnets.utils.math.projections import *


class DenseLinear(nn.Module):

    def __init__(self):
        super(DenseLinear, self).__init__()

    def _set_network_parameters(self, in_features, out_features, bias=True, cuda=None):
        self.in_features = in_features
        self.out_features = out_features

        # Set weights and biases.
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _set_config(self, config):
        self.config = config

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        raise NotImplementedError

    def project_weights(self, proj_config):
        with torch.no_grad():
            projected_weights = project_weights(self.weight, proj_config, self.config.cuda)
            # Override the previous weights.
            self.weight.data.copy_(projected_weights)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

