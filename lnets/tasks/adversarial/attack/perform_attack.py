import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable, grad

from foolbox.adversarial import Adversarial


def cw_loss(logits, y):
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, y.view(-1, 1), 1)
    correct_logit = (logits * one_hot).sum(1)
    worst_wrong_logit = logits[one_hot == 0].view(one_hot.size(0), -1).max(1)[0]

    adv_margin = correct_logit - worst_wrong_logit
    return -F.relu(adv_margin + 50).mean()


def tensor_clamp(x, min_x, max_x):
    x[x < min_x] = min_x[x < min_x]
    x[x > max_x] = max_x[x > max_x]
    return x


def manual_fgs(model, x, y, eps=0.1, clamp=True):
    model.zero_grad()
    x.requires_grad = True
    pred = model(x)
    loss = cw_loss(pred, y)

    g = torch.sign(grad(loss, x)[0])
    adv_x = x + g * eps
    if clamp:
        adv_x.clamp_(0, 1)
    return adv_x, model(adv_x).argmax(1), pred.argmax(1)


def manual_pgd(model, x, y, stepsize=0.01, eps=0.1, iters=100, rand_start=True, clamp=True):
    model.zero_grad()
    x_adv = torch.zeros_like(x)
    x_adv.copy_(x)

    x_min = (x.detach() - eps)
    x_max = (x.detach() + eps)
    if clamp:
        x_min.clamp_(0, 1)
        x_max.clamp_(0, 1)

    if rand_start:
        rand = torch.zeros_like(x)
        rand.uniform_(-eps, eps)
        x_adv = x_adv + rand
    x_adv.requires_grad = True
    for i in range(iters):
        model.zero_grad()
        if x_adv.grad:
            x_adv.grad.zero_()
        pred = model(x_adv)
        # loss = F.cross_entropy(pred, y)
        loss = cw_loss(pred, y)
        g = torch.sign(grad(loss, x_adv)[0])
        x_adv = x_adv + g * stepsize
        x_adv = tensor_clamp(x_adv, x_min, x_max)
    return x_adv, model(x_adv).argmax(1), model(x).argmax(1)


def perform_attack(attack, model, input_adv, cuda=True, **attack_kwargs):
    adversarial_np = attack(input_adv, **attack_kwargs)
    image = torch.Tensor(input_adv.original_image)
    label = input_adv.original_class

    if adversarial_np is None:
        # Attack failed.
        return adversarial_np, 0.0, 0.0, 0.0
    else:
        # Check if attack was successful.
        adversarial = Variable(torch.Tensor(adversarial_np))
        if cuda:
            adversarial = adversarial.cuda()
            image = image.cuda()

        pred = lambda x: F.softmax(model.forward(x.unsqueeze(0)), dim=1).max(1)[1].data[0]

        # Compute adversarial MSE.
        adv_mse = torch.pow(adversarial - Variable(image), 2).mean().item()
        adv_inf = torch.max(torch.abs(adversarial - image))
        adv_inf = adv_inf.item()

        if pred(adversarial).item() != label and adv_inf > 0.0:
            success = 1.0
        else:
            success = 0.0

        return adversarial_np, success, adv_mse, adv_inf


def batch_attack(attack, model, criterion, x, y, attack_config={}, distance=None):
    adv_ex = []
    adv_targets = []
    ret_success = []
    ret_adv_mse = []
    ret_adv_inf = []

    for i in range(x.shape[0]):
        is_cuda = x.is_cuda
        input_adv = Adversarial(model, criterion, x[i].cpu().numpy(), y[i], distance=distance)
        adv, success, adv_mse, adv_inf = perform_attack(attack, model._model, input_adv, cuda=is_cuda, **attack_config)
        if adv is not None:
            adv_targets.append(y[i])
            adv_ex.append(adv)
        ret_success.append(success)
        ret_adv_mse.append(adv_mse)
        ret_adv_inf.append(adv_inf)
    return torch.Tensor(np.array(adv_ex)), torch.LongTensor(np.array(adv_targets)), torch.Tensor(
        ret_success), torch.Tensor(ret_adv_mse), torch.Tensor(ret_adv_inf)
