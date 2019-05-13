import torch
from torch.optim.optimizer import Optimizer, required


class AggMo(Optimizer):
    r"""Implements Aggregated Momentum Gradient Descent.
    https://arxiv.org/abs/1804.00325
    """

    def __init__(self, params, lr=required, momentum=[0.0, 0.9, 0.99, 0.999], weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(AggMo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AggMo, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            total_mom = float(len(momentum))

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = {}
                    for beta in momentum:
                        param_state['momentum_buffer'][beta] = torch.zeros_like(p.data)
                for beta in momentum:
                    buf = param_state['momentum_buffer'][beta]
                    buf.mul_(beta).add_(d_p)
                    p.data.sub_(group['lr'] / total_mom , buf)
        return loss

    def zero_momentum_buffers(self):
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                param_state = self.state[p]
                param_state['momentum_buffer'] = {}
                for beta in momentum:
                    param_state['momentum_buffer'][beta] = torch.zeros_like(p.data)

    def set_momentum(self, momentum):
        for group in self.param_groups:
            group['momentum'] = momentum

    def update_hparam(self, name, value):
        for param_group in self.param_groups:
            param_group[name] = value
