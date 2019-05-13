import torch.nn as nn

from lnets.models.activations import Activation


class Architecture(nn.Module):
    def __init__(self):
        super(Architecture, self).__init__()

    def __len__(self):
        return len(self.model)

    def __getitem__(self, idx):
        return self.model[idx]

    def forward(self, x):
        raise NotImplementedError

    def project_network_weights(self, proj_config):
        # Project the weights on the manifold of orthonormal matrices.
        for i, layer in enumerate(self.model):
            try:
                self.model[i].project_weights(proj_config)
            except:
                continue

    def get_activations(self, x):
        activations = []
        x = x.view(-1, self.input_dim)
        for m in self.model:
            x = m(x)
            if not isinstance(m, Activation):
                activations.append(x.detach().clone())
        return activations
