from lnets.data.small_data import get_small_data_indices

import numpy as np
import os


def save_indices(dataset, indices_path, per_class_count, total_class_count, val_size):
    train_indices, val_indices = get_small_data_indices(dataset, per_class_count, total_class_count, val_size)
    np.savetxt(os.path.join(indices_path, "train_indices_{}.txt".format(per_class_count)), train_indices)
    np.savetxt(os.path.join(indices_path, "val_indices_{}.txt".format(per_class_count)), val_indices)


def load_indices(path, per_class_count):
    train_indices = os.path.join(path, "train_indices_{}.txt".format(per_class_count))
    val_indices = os.path.join(path, "val_indices_{}.txt".format(per_class_count))
    return np.loadtxt(train_indices, dtype=np.int32), np.loadtxt(val_indices, dtype=np.int32)
