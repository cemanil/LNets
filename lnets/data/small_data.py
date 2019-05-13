import numpy as np


def get_small_data_indices(dataset, total_per_class, class_count, val_size):
    total_points = len(dataset)
    if total_per_class * class_count + val_size > total_points:
        raise Exception('More data points requested than is in data')
    random_indices = np.random.permutation(total_points)

    small_data_indices = {}
    val_indices = []
    for c in range(class_count):
        small_data_indices[c] = []

    for idx in random_indices:
        _, y = dataset[idx]
        y = int(y.item())
        if len(small_data_indices[y]) < total_per_class:
            small_data_indices[y].append(idx)
        elif len(val_indices) < val_size:
            val_indices.append(idx)
        if all([len(small_data_indices[c]) == total_per_class for c in range(class_count)]):
            if len(val_indices) == val_size:
                break
    if not all([len(small_data_indices[c]) == total_per_class for c in range(class_count)]):
        raise Warning('Uneven class counts in small data indices')
    return np.array([small_data_indices[c] for c in
                     range(class_count)]).astype(np.int32).flatten(), np.array(val_indices).astype(np.int32)
