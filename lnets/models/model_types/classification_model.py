import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchnet as tnt

from lnets.models.model_types.base_model import ExperimentModel
from lnets.models.regularization.spec_jac import jac_spectral_norm


class ClassificationModel(ExperimentModel):
    def _init_meters(self):
        super(ClassificationModel, self)._init_meters()
        self.meters['acc'] = tnt.meter.ClassErrorMeter(accuracy=True)

    def loss(self, sample, test=False):
        inputs = Variable(sample[0], volatile=test)
        targets = Variable(sample[1], volatile=test)
        o = torch.squeeze(self.model.forward(inputs))

        return F.cross_entropy(o, targets), {'logits': o}

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())
        self.meters['acc'].add(state['output']['logits'].data, state['sample'][1])


class MarginClassificationModel(ExperimentModel):
    def __init__(self, model, config):
        super(MarginClassificationModel, self).__init__(model)
        self.margin = config.model.margin * config.model.l_constant

    def _init_meters(self):
        super(MarginClassificationModel, self)._init_meters()
        self.meters['acc'] = tnt.meter.ClassErrorMeter(accuracy=True)

    def loss(self, sample, test=False):
        inputs = Variable(sample[0], volatile=test)
        targets = Variable(sample[1], volatile=test)
        o = torch.squeeze(self.model.forward(inputs))
        logits = o.detach().clone()

        # Add margin buffer to all entries except true class in each row
        # Equivalently, subtract the margin from the correct class.
        o[torch.arange(o.size(0)), targets] -= self.margin
        return F.cross_entropy(o, targets), {'logits': logits}

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())
        self.meters['acc'].add(state['output']['logits'].data, state['sample'][1])


class HingeLossClassificationModel(ExperimentModel):
    def __init__(self, model, config):
        super(HingeLossClassificationModel, self).__init__(model)
        self.margin = config.model.margin * config.model.l_constant

    def _init_meters(self):
        super(HingeLossClassificationModel, self)._init_meters()
        self.meters['acc'] = tnt.meter.ClassErrorMeter(accuracy=True)

    def loss(self, sample, test=False):
        inputs = Variable(sample[0], volatile=test)
        targets = Variable(sample[1], volatile=test)
        o = torch.squeeze(self.model.forward(inputs))

        return F.multi_margin_loss(o, targets, margin=self.margin), {'logits': o}

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())
        self.meters['acc'].add(state['output']['logits'].data, state['sample'][1])


class JacSpecClassificationModel(ExperimentModel):
    def __init__(self, model, reg_scale, cuda=True):
        super(JacSpecClassificationModel, self).__init__(model)
        self.reg_scale = reg_scale
        self.u = torch.randn(10)
        if cuda:
            self.u = self.u.cuda()

    def _init_meters(self):
        super(JacSpecClassificationModel, self)._init_meters()
        self.meters['acc'] = tnt.meter.ClassErrorMeter(accuracy=True)
        self.meters['sn'] = tnt.meter.AverageValueMeter()

    def loss(self, sample, test=False):
        inputs = Variable(sample[0], requires_grad=True)
        targets = Variable(sample[1])
        o = self.model.forward(inputs)
        xe = F.cross_entropy(o, targets)
        spec_norm, u = jac_spectral_norm(o, inputs, self.u)
        self.u = u

        return xe + self.reg_scale * spec_norm, {'logits': o, 'sn': spec_norm}

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())
        self.meters['acc'].add(state['output']['logits'].data, state['sample'][1])
        self.meters['sn'].add(state['output']['sn'].item())
