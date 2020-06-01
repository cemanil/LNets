import numpy as np
import torch.nn as nn


class GroupSort(nn.Module):

    def __init__(self, num_units, axis=-1):
        super(GroupSort, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        group_sorted = group_sort(x, self.num_units, self.axis)
        assert check_group_sorted(group_sorted, self.num_units, axis=self.axis) == 1, "GroupSort failed. "

        return group_sorted

    def extra_repr(self):
        return 'num_groups: {}'.format(self.num_units)


def process_group_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size


def group_sort(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, _ = grouped_x.sort(dim=sort_dim)
    sorted_x = sorted_grouped_x.view(*list(x.shape))

    return sorted_x


def check_group_sorted(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)

    x_np = x.cpu().data.numpy()
    x_np = x_np.reshape(*size)
    axis = axis if axis == -1 else axis + 1
    x_np_diff = np.diff(x_np, axis=axis)

    # Return 1 iff all elements are increasing.
    if np.sum(x_np_diff < 0) > 0:
        return 0
    else:
        return 1