import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

from lnets.models.layers.conv.base_conv2d import BaseConv2D
from lnets.utils.math.projections import get_weight_signs, get_linf_projection_threshold


class LInfProjectedConv2D(BaseConv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, config=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BaseConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        self._set_config(config)
        self.original_shape = self.weight.shape
        self.out_channels = out_channels

        if stride == 1 or stride == [1, 1]:
            print("BEWARE: Norm is not being preserved due to stride > 1.  ")

    def forward(self, x):
        # Reshape and put in a matrix form.
        flattened_weights = self.conv_form_to_matrix_form(self.weight, (self.out_channels, -1))

        # Orthonormalize. The scaling makes sure the singular values of the matrix are constrained by 1.
        thresholds = get_linf_projection_threshold(flattened_weights, self.config.cuda)
        signs = get_weight_signs(flattened_weights)
        flattened_projected_weights = signs * torch.clamp(torch.abs(flattened_weights) - thresholds.unsqueeze(-1),
                                                          min=torch.tensor(0).float())

        # Reshape back.
        projected_weights = self.matrix_form_to_conv_form(flattened_projected_weights, self.original_shape)

        return F.conv2d(x, projected_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
