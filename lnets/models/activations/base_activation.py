import torch.nn as nn


class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()

    def forward(self, x):
        raise NotImplementedError
