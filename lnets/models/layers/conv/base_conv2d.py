import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from lnets.utils.math.projections import project_weights


class BaseConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BaseConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        self.original_shape = self.weight.shape
        self.out_channels = out_channels

    def _set_config(self, config):
        self.config = config

    def project_weights(self, proj_config):
        with torch.no_grad():
            flattened_weights = self.conv_form_to_matrix_form(self.weight, (self.out_channels, -1))

            flattened_projected_weights = project_weights(flattened_weights, proj_config, cuda=self.config.cuda)

            projected_weights = self.matrix_form_to_conv_form(flattened_projected_weights, self.original_shape)

            self.weight.data.copy_(projected_weights)

    @staticmethod
    def conv_form_to_matrix_form(weight, matrix_form_shape):
        return weight.view(matrix_form_shape)

    @staticmethod
    def matrix_form_to_conv_form(weight, conv_form_shape):
        return weight.view(conv_form_shape)

    def forward(self, x):
        raise NotImplementedError

