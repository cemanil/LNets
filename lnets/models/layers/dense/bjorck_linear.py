import torch.nn.functional as F

from lnets.models.layers.dense.base_dense_linear import DenseLinear
from lnets.utils.math.projections import bjorck_orthonormalize, get_safe_bjorck_scaling


class BjorckLinear(DenseLinear):
    def __init__(self, in_features=1, out_features=1, bias=True, config=None):
        super(BjorckLinear, self).__init__()
        self._set_config(config)
        self._set_network_parameters(in_features, out_features, bias, cuda=config.cuda)

    def forward(self, x):
        # Scale the values of the matrix to make sure the singular values are less than or equal to 1.
        if self.config.model.linear.safe_scaling:
            scaling = get_safe_bjorck_scaling(self.weight, cuda=self.config.cuda)
        else:
            scaling = 1.0

        ortho_w = bjorck_orthonormalize(self.weight.t() / scaling,
                                        beta=self.config.model.linear.bjorck_beta,
                                        iters=self.config.model.linear.bjorck_iter,
                                        order=self.config.model.linear.bjorck_order).t()
        return F.linear(x, ortho_w, self.bias)
