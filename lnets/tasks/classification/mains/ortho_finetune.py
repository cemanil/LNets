"""
Do finetuning to ensure that the network is actually orthonormal.
"""

import json
import os.path
import argparse
from munch import Munch

import torch

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.utils.conversion import convert_model_to_bjorck
from lnets.models.layers import BjorckLinear
from lnets.tasks.classification.mains.train_classifier import train


def main(opt):
    if not os.path.isdir(opt['output_root']):
        os.makedirs(opt['output_root'])

    exp_dir = opt['model']['exp_path']

    model_path = os.path.join(exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(exp_dir, 'logs', 'config.json'), 'r') as f:
        model_config = Munch.fromDict(json.load(f))

    # Weird required hack to fix groupings (None is added to start during model training)
    if 'groupings' in model_config.model and model_config.model.groupings[0] is -1:
        model_config.model.groupings = model_config.model.groupings[1:]

    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path))

    if opt['data']['cuda']:
        print('Using CUDA')
        model.cuda()

    model_config.data.cuda = opt['data']['cuda']
    data = load_data(model_config)

    # Change the model to use ortho layers by copying the base weights
    bjorck_iters = 50
    model = convert_model_to_bjorck(model, model_config)
    for m in model.modules():
        if isinstance(m, BjorckLinear):
            m.config.model.linear.bjorck_iter = bjorck_iters

    model_config.output_root = opt['output_root']
    model_config.optim.lr_schedule.lr_init = 1e-5
    model_config.optim.epochs = 5
    model = train(model, data, model_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do orthonormal finetuning on classifier')

    parser.add_argument('--model.exp_path', type=str, metavar='MODELPATH',
                        help="location of pretrained model weights to evaluate")
    parser.add_argument('--output_root', type=str, default="./outs/classification/finetune",
                        help='output directory to which results should be saved')
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")

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
