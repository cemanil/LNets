import torch
from torch.autograd import Variable

from lnets.models.model_types.base_model import ExperimentModel


class DualOptimModel(ExperimentModel):
    def _init_meters(self):
        super(DualOptimModel, self)._init_meters()

    def loss(self, sample, test=False):
        # d1 stands for distribution 1.
        # d2 stands for distribution 2.

        samples_from_d1 = Variable(sample[0])
        samples_from_d2 = Variable(sample[1])

        potentials_from_d1 = self.model.forward(samples_from_d1)
        potentials_from_d2 = self.model.forward(samples_from_d2)

        assert potentials_from_d1.shape[1] == 1
        assert potentials_from_d2.shape[1] == 1

        loss = -1 * (torch.mean(potentials_from_d1) - torch.mean(potentials_from_d2))

        return loss, (potentials_from_d1, potentials_from_d2)

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())
