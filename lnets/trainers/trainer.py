"""
Based on code from https://github.com/pytorch/tnt/blob/master/torchnet/trainers/trainers.py
"""


class Trainer(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, model, iterator, maxepoch, optimizer):
        # Initialize the state that will fully describe the status of training.
        state = {
            'model': model,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'epoch': 0,
            't': 0,
            'train': True,
            'stop': False
        }

        # On training start.
        model.train()  # Switch to training mode.
        self.hook('on_start', state)

        # Loop over epochs.
        while state['epoch'] < state['maxepoch'] and not state['stop']:
            # On epoch start.
            self.hook('on_start_epoch', state)

            # Loop over samples.
            for sample in state['iterator']:
                # On sample.
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['model'].loss(state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    loss.backward()
                    self.hook('on_forward', state)
                    # To free memory in save_for_backward,
                    # state['output'] = None
                    # state['loss'] = None
                    return loss

                # On update.
                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.hook('on_update', state)

                state['t'] += 1
            state['epoch'] += 1

            # On epoch end.
            self.hook('on_end_epoch', state)

        # On training end.
        self.hook('on_end', state)

        return state

    def test(self, model, iterator):
        # Initialize the state that will fully describe the status of training.
        state = {
            'model': model,
            'iterator': iterator,
            't': 0,
            'train': False,
        }
        model.eval()  # Set the PyTorch model to evaluation mode.

        # On start.
        self.hook('on_start', state)
        self.hook('on_start_val', state)

        # Loop over samples - for one epoch.
        for sample in state['iterator']:
            # On sample.
            state['sample'] = sample
            self.hook('on_sample', state)

            def closure():
                loss, output = state['model'].loss(state['sample'], test=True)
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)
                # To free memory in save_for_backward.
                # state['output'] = None
                # state['loss'] = None

            closure()
            state['t'] += 1

        # On training end.
        self.hook('on_end_val', state)
        self.hook('on_end', state)
        model.train()
        return state
