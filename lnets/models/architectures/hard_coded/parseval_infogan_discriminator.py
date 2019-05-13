from torch import nn

from lnets.models.layers import *
from lnets.models.activations import *
from lnets.models.architectures.base_architecture import Architecture


class ParsevalInfoGanDiscriminator(Architecture):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    # Note that the BatchNorm layers are also removed.
    def __init__(self, input_dim=1, output_dim=1, input_size=32, parseval=True, config=None):
        super(ParsevalInfoGanDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        model_list = list([
            # Conv.
            StandardConv2d(self.input_dim, 64, 4, 2, 1, config=config),

            # Activ.
            MaxMin(num_units=32, axis=1),

            # Conv
            StandardConv2d(64, 128, 4, 2, 1, config=config),

            # Activ.
            MaxMin(num_units=64, axis=1),

            # Flatten.
            InfoGanFlatten(input_size=input_size),

            # Linear.
            StandardLinear(128 * (self.input_size // 4) * (self.input_size // 4), 1024, config=config),

            # Activ.
            MaxMin(num_units=512),

            # Linear.
            StandardLinear(1024, self.output_dim, config=config),
        ])

        self.model = nn.Sequential(*model_list)

        initialize_weights(self)

    def forward(self, x):
        return self.model(x)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class InfoGanFlatten(nn.Module):
    def __init__(self, input_size):
        super(InfoGanFlatten, self).__init__()
        self.input_size = input_size

    def forward(self, x):
        return x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
