import torch
import torch.nn.functional as F

from lnets.models.layers.dense.base_dense_linear import DenseLinear
from lnets.utils.math.projections import get_weight_signs, get_linf_projection_threshold


class LInfProjectedLinear(DenseLinear):
    r"""
    Applies a linear transformation to the incoming distrib: :math:`y = Ax + b`
    such that the L-infinity norm of A is less than 1 by projecting it to the L1 ball
    """

    def __init__(self, in_features, out_features, bias=True, config=None):
        super(DenseLinear, self).__init__()
        self._set_config(config)
        self._set_network_parameters(in_features, out_features, bias, config.cuda)

    def forward(self, x):
        thresholds = get_linf_projection_threshold(self.weight, self.config.cuda)
        signs = get_weight_signs(self.weight)
        projected_weights = signs * torch.clamp(torch.abs(self.weight) - thresholds.unsqueeze(-1),
                                                min=torch.tensor(0).float())

        return F.linear(x, projected_weights, self.bias)

