import os

import torch
import torch.nn.functional as F
from lnets.tasks.adversarial.mains.utils import save_image

import numpy as np

from lnets.data.load_data import load_data
from lnets.utils.config import process_config
from lnets.utils.saving_and_loading import load_model_from_config
from lnets.utils.misc import to_cuda
from lnets.models.regularization.spec_jac import jac_spectral_norm
from lnets.utils.math.autodiff import compute_jacobian


def get_adv_gradient(model, x, adv_targets):
    # The following is needed to backprop on the inputs.
    x.requires_grad = True

    # Clear the gradient buffers.
    if x.grad is not None:
        x.grad.zero_()
        for p in model.parameters():
            p.grad.zero_()

    #  Take the derivarive of the loss wrt. the inputs.
    out = model(x)
    loss = F.cross_entropy(out, adv_targets)
    loss.backward()
    x_grad = x.grad.data

    return x_grad


def check_grad_norm(model, data, cuda, epochs=3):
    u = to_cuda(torch.randn(10), cuda)
    for _ in range(epochs):
        for x, _ in data:
            model.zero_grad()
            x = to_cuda(x, cuda)
            x.requires_grad = True
            logits = model(x)
            s, u = jac_spectral_norm(logits, x, u)
    return s


def slow_check_grad_norm(model, data, cuda):
    spectral_rads = []
    for x, _ in data:
        model.zero_grad()
        x = to_cuda(x, cuda).view(-1, 784)
        x.requires_grad = True
        logits = model(x)
        jac = compute_jacobian(logits, x)
        for j in jac:
            _, S, _ = torch.svd(j)
            spectral_rads.append(torch.max(S).cpu().detach().item())
    return np.mean(spectral_rads), np.max(spectral_rads)


def check_adv_gradients(config):
    # Create the output directory.
    output_root = config.output_root
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    # Load a pretrained model.
    pretrained_path = config.pretrained_path
    model, pretrained_config = load_model_from_config(pretrained_path)

    # Push model to GPU if available.
    if config.cuda:
        print('Using cuda: Yes')
        model.cuda()

    model.eval()

    # Get data.
    pretrained_config.data.cuda = config.cuda
    pretrained_config.data.batch_size = config.data.batch_size
    data = load_data(pretrained_config)

    # Compute adversarial gradients and save their visualizations.
    for i, (x, _) in enumerate(data['test']):
        x = to_cuda(x, cuda=config.cuda)

        # Save the input image.
        save_path = os.path.join(output_root, 'x{}.png'.format(i))
        save_image(x, save_path)

        # Save the adversarial gradients.
        for j in range(pretrained_config.data.class_count):
            y = j * torch.ones(x.size(0)).type(torch.long)
            y = to_cuda(y, cuda=config.cuda)

            # Compute and save the adversarial gradients.
            x_grad = get_adv_gradient(model, x, y)
            save_image(x_grad, os.path.join(output_root, 'x_{}_grad_{}.png'.format(i, j)), normalize=True,
                       scale_each=True)
        break

    # Produce joint image.
    nrow = config.visualization.num_rows
    x_sub = to_cuda(torch.zeros(nrow, *x.size()[1:]).copy_(x[:nrow]).detach(), config.cuda)
    print("Size of visualization: ", x_sub.size(), "Maximum pixel value: ", x_sub.max())
    tensors = []
    c = 0
    for i, (x, y) in enumerate(data['test']):
        for (k, t) in enumerate(y):
            if t == c:
                c += 1
                tensors.append(x[k])
                if len(tensors) == pretrained_config.data.class_count:
                    break
        if len(tensors) == pretrained_config.data.class_count:
            break

    # Collect tensors from each class
    x_sub = to_cuda(torch.stack(tensors, 0), cuda=config.cuda)

    tensors = [x_sub]
    for j in range(pretrained_config.data.class_count):
        y = j * torch.ones(x_sub.size(0)).type(torch.long)
        y = to_cuda(y, cuda=config.cuda)

        # Compute and visualize the adversarial gradients.
        model.zero_grad()
        x_grad = get_adv_gradient(model, x_sub, y).clone().detach()
        tensors.append(x_grad)

    # Concatenate and visualize.
    joint_tensor = torch.cat(tensors, dim=0)
    save_image(joint_tensor, os.path.join(output_root, 'x_joint.png'), nrow=pretrained_config.data.class_count,
               normalize=True, colormap='seismic')
    # print("Train sigma(J): {}".format(check_grad_norm(model, data['train'], config.cuda)))
    # print("Val sigma(J): {}".format(check_grad_norm(model, data['validation'], config.cuda)))
    # print("Test sigma(J): {}".format(check_grad_norm(model, data['test'], config.cuda)))


if __name__ == '__main__':
    cfg = process_config()

    check_adv_gradients(cfg)
