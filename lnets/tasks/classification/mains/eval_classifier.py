"""
Check the calibration of the model
"""

import json
import os.path
import argparse
from munch import Munch
import matplotlib.pyplot as plt

import torch

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.utils.conversion import convert_model_from_bjorck
from lnets.trainers.trainer import Trainer


def check_logit_margins(model, data):
    logit_margins = []
    for x,y in data:
        x,y = x.cuda(), y.cuda()
        logits = model(x)
        top, indices = logits.topk(2, 1)
        logit_margins.append(torch.abs(top[:, 0] - top[:, 1])[indices[:,0] == y])
    logit_margins = torch.cat(logit_margins)
    return logit_margins


def main(opt):
    exp_dir = opt['model']['exp_path']

    model_path = os.path.join(exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(exp_dir, 'logs', 'config.json'), 'r') as f:
        model_config = Munch.fromDict(json.load(f))

    # Weird required hack to fix groupings (None is added to start during model training)
    if 'groupings' in model_config.model and model_config.model.groupings[0] is -1:
        model_config.model.groupings = model_config.model.groupings[1:]
    model_config.cuda = opt['cuda']
    model_config.data.cuda = opt['cuda']
    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path))

    if opt['cuda']:
        print('Using CUDA')
        model.cuda()

    def on_sample(state):
        if opt['cuda']:
            state['sample'] = [x.cuda() for x in state['sample']]

    def on_forward(state):
        state['model'].add_to_meters(state)

    data = load_data(model_config)

    # Change the model to use ortho layers by copying the base weights
    model = convert_model_from_bjorck(model, model_config)
    # model.model.project_network_weights(Munch.fromDict({'type': 'l_inf_projected'}))

    # Instantiate the trainer.
    trainer = Trainer()

    trainer.hooks['on_sample'] = on_sample
    trainer.hooks['on_forward'] = on_forward
    print('TESTING')
    state = trainer.test(model, data['test'])
    for tag, meter in state['model'].meters.items():
        print(tag, meter.value())
    logit_margins = check_logit_margins(model, data['test'])
    print(logit_margins.min().item(), logit_margins.max().item(), logit_margins.mean().item())
    plt.hist(logit_margins.detach().cpu().numpy())
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained classification network')

    parser.add_argument('--model.exp_path', type=str, metavar='MODELPATH',
                        help="location of pretrained model weights to evaluate")
    parser.add_argument('--cuda', action='store_true', help="run in CUDA mode (default: False)")

    args = vars(parser.parse_args())

    opt = {}
    for k, v in args.items():
        cur = opt
        tokens = k.split('.')
        for token in tokens[:-1]:
            if token not in cur:
                cur[token] = {}
            cur = cur[token]
        cur[tokens[-1]] = v

    main(opt)
