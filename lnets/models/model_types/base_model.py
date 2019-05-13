from collections import OrderedDict

import torch.nn as nn
import torchnet as tnt


class ExperimentModel(nn.Module):
    def __init__(self, model):
        super(ExperimentModel, self).__init__()
        self.model = model
        self._init_meters()

    def forward(self, x):
        return self.model(x)

    def loss(self, sample, test=False):
        raise NotImplementedError

    def input_size(self):
        return self.model.input_size

    def _init_meters(self):
        self.meters = OrderedDict([('loss', tnt.meter.AverageValueMeter())])

    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset()

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].data[0])
