import torch
import torch.nn as nn


class Scale(nn.Module):
    r"""Scales the input vector by a given scalar.
    """

    def __init__(self, factor, cuda=False):
        super(Scale, self).__init__()
        self.factor = factor

        if cuda:
            self.factor = torch.Tensor([self.factor]).cuda()

    def reset_parameters(self):
        pass

    def forward(self, input):
        if self.factor == 1:  # This is to make sure this operation is not backpropped on, or unnecessarily computed.
            return input
        else:
            return self.factor * input

    def extra_repr(self):
        return 'factor={}'.format(self.factor)
