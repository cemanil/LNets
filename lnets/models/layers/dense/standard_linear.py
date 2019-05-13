import torch.nn.functional as F

from lnets.models.layers.dense.base_dense_linear import DenseLinear


class StandardLinear(DenseLinear):
    r"""
    Applies a linear transformation to the incoming distrib: :math:`y = Ax + b`
    """

    def __init__(self, in_features, out_features, bias=True, config=None):
        super(DenseLinear, self).__init__()
        self._set_config(config)
        self._set_network_parameters(in_features, out_features, bias, cuda=config.cuda)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
