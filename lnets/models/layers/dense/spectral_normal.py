import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from lnets.models.layers.dense.base_dense_linear import DenseLinear


class SpectralNormLinear(DenseLinear):
    r"""
    Applies a linear transformation to incoming distrib: :math:`y = Ax + b` such that A has spectral norm close to 1.
    """

    def __init__(self, in_features, out_features, bias=True, config=None):
        super(SpectralNormLinear, self).__init__()
        self._set_config(config)
        self._set_network_parameters(in_features, out_features, bias, cuda=config.cuda)

        with torch.no_grad():
            self.u = Parameter(torch.Tensor(out_features))
            self.v = Parameter(torch.Tensor(in_features))

        self.reset_u()

        self.power_iters = config.model.linear.power_iters

    def forward(self, x):
        self.power_iteration()
        spectral_norm = self.u.dot(self.weight.matmul(self.v))
        normalized_w = self.weight / spectral_norm

        return F.linear(x, normalized_w, self.bias)

    def reset_u(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.u.data.normal_(-stdv, stdv)

    def power_iteration(self):
        with torch.no_grad():
            for _ in range(self.power_iters):
                w_t_u = self.weight.t().matmul(self.u)
                w_t_u.div_(w_t_u.norm())
                self.v.data.copy_(w_t_u.data)
                w_v = self.weight.matmul(self.v)
                w_v.div_(w_v.norm())
                self.u.data.copy_(w_v)
