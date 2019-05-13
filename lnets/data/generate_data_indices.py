from lnets.data.load_data import get_datasets
from lnets.data.utils import save_indices

import argparse
import os
from munch import Munch


def main(opt):
    opt.data.transform = Munch(type='none')

    indices_path = os.path.join(opt.data.root, opt.data.name)

    train_data, _, _ = get_datasets(opt)

    save_indices(train_data, indices_path, opt.per_class_count, opt.data.class_count, opt.val_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data indices. ')
    parser.add_argument('--data.name', type=str, metavar='MODELPATH',
                        help="location of pretrained model weights to evaluate")
    parser.add_argument('--data.root', type=str, help='output directory to which results should be saved')
    parser.add_argument('--data.class_count', type=int, help='total number of classes in dataset')
    parser.add_argument('--per_class_count', type=int, help="How many training data points per class")
    parser.add_argument('--val_size', type=int, help="Total number of validation points")
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
    main(Munch.fromDict(opt))
