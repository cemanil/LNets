import numpy as np
import torch
import random


def set_experiment_seed(seed):
    # Set the seed.
    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
