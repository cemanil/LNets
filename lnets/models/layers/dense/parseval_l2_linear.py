import torch.nn.functional as F

from lnets.models.layers.dense.base_dense_linear import DenseLinear


class ParsevalL2Linear(DenseLinear):
    def __init__(self, in_features=1, out_features=1, bias=True, config=None):
        super(ParsevalL2Linear, self).__init__()
        self._set_config(config)
        self._set_network_parameters(in_features, out_features, bias, cuda=config.cuda)

    def forward(self, x):
        self.project_weights(self.config.model.per_update_proj)

        return F.linear(x, self.weight, self.bias)

