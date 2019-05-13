import numpy as np


def initialize_best_val(criterion):
    if criterion == 'min':
        return np.inf
    elif criterion == "max":
        return -np.inf
    else:
        print("The optimization criterion must be either 'max' or 'min'. ")


def to_cuda(pytorch_object, cuda=False):
    if cuda:
        return pytorch_object.cuda()
    else:
        return pytorch_object
