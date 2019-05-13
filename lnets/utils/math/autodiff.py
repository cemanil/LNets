import torch
from torch.autograd import grad


def compute_jacobian(output, inputs, create_graph=True, retain_graph=True):
    """
    :param output: Batch X Classes
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :return: jacobian: Batch X Size X Classes
    """
    assert inputs.requires_grad

    # num_classes = output.size()[1]

    return torch.stack([grad([output[:, i].sum()], [inputs], retain_graph=retain_graph, create_graph=create_graph)[0]
                        for i in range(output.size(1))], dim=-1)
