import torch.nn as nn

from lnets.models.layers import *
from lnets.models.activations import *


def select_linear_layer(linear_type):
    if linear_type == "standard":
        return StandardLinear
    elif linear_type == "spectral_normal":
        return SpectralNormLinear
    elif linear_type == "bjorck":
        return BjorckLinear
    elif linear_type == "l_inf_projected":
        return LInfProjectedLinear
    elif linear_type == "parseval_l2":
        return ParsevalL2Linear
    elif linear_type == "standard_conv2d":
        return StandardConv2d
    elif linear_type == "bjorck_conv2d":
        return BjorckConv2d
    elif linear_type == "l_inf_projected_conv2d":
        return LInfProjectedConv2D
    else:
        print("The requested dense linear layer is not supported yet. ")
        exit(-1)


def select_activation_function(activation):
    if activation == 'identity':
        act_func = Identity
    elif activation == 'relu':
        act_func = nn.ReLU
    elif activation == "abs":
        act_func = Abs
    elif activation == 'sigmoid':
        act_func = nn.Sigmoid
    elif activation == 'tanh':
        act_func = nn.Tanh
    elif activation == 'maxout':
        act_func = Maxout
    elif activation == 'maxmin':
        act_func = MaxMin
    elif activation == "group_sort":
        act_func = GroupSort
    else:
        act_func = None
        raise Exception('Unexpected activation function. ')
    return act_func
