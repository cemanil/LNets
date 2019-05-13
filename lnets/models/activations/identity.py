from lnets.models.activations.base_activation import Activation


class Identity(Activation):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
