"""
Check the calibration of the model
"""

import json
import os.path
import argparse
from munch import Munch
import numpy as np
import matplotlib.pyplot as plt

import torch

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.utils.conversion import convert_model_from_bjorck


def get_undead_rate(model, data, threshold=0.8, cuda=True):
    undead_rate = []
    for x,_ in data:
        if cuda:
            x = x.cuda()
        activations = model.model.get_activations(x)
        undead_rate.append([])
        for layer_a in activations:
            layer_a = layer_a.cpu().numpy()
            undead_rate[-1].append((layer_a > 0).astype(np.float).mean(0))

    undead_rate = np.array(undead_rate).mean(0)
    return (undead_rate >= threshold).astype(np.float).mean(1)


def main(opt):
    # if not os.path.isdir(opt['output_root']):
    #     os.makedirs(opt['output_root'])

    exp_dir = opt['model']['exp_path']

    model_path = os.path.join(exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(exp_dir, 'logs', 'config.json'), 'r') as f:
        model_config = Munch.fromDict(json.load(f))

    # Weird required hack to fix groupings (None is added to start during model training)
    if 'groupings' in model_config.model and model_config.model.groupings[0] is -1:
        model_config.model.groupings = model_config.model.groupings[1:]
    model_config.model.linear.bjorck_iters = 20
    model_config.cuda = opt['cuda']
    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path))

    if opt['cuda']:
        print('Using CUDA')
        model.cuda()

    data = load_data(model_config)

    # Change the model to use ortho layers by copying the base weights
    model = convert_model_from_bjorck(model, model_config)
    model.eval()
    rates = []
    thresholds = np.linspace(0.0, 1.0, 50, endpoint=True)
    for t in thresholds:
        undead_rate = get_undead_rate(model, data['test'], threshold=t, cuda=opt['cuda'])
        rates.append(undead_rate)
    plt.plot(thresholds, np.array(rates))
    plt.show()
    np.save('undead_rates', rates)
    print(undead_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute undead unit rates per layer')

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
