"""
Evaluate adversarial robustness of a given classifier.
"""
import os
import json
from itertools import islice
import math
from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from lnets.data.load_data import load_data
from lnets.utils.saving_and_loading import load_model_from_config
from lnets.utils.misc import to_cuda
from lnets.tasks.adversarial.attack.perform_attack import manual_fgs, manual_pgd

from munch import Munch


def loader_accuracy(model, loader):
    acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        preds = model(x)
        acc += (preds.argmax(1) == y).type(torch.float).mean()
        n += 1
    return (acc / n).item()


def accuracy(model, x, y):
    preds = []
    for ex in x:
        preds.append(model(ex).argmax())
    preds = torch.stack(preds)
    return (preds == y).type(torch.float).mean()


def theoretical_adversary(model, x, y, eps_range):
    """
    Computes the theoretical lower bound on adversarial accuracy
    """
    logits = model(x)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, y.view(-1, 1), 1)
    correct_logit = (logits * one_hot).sum(1)
    worst_wrong_logit = logits[one_hot == 0].view(one_hot.size(0), -1).max(1)[0]

    adv_margin = F.relu(correct_logit - worst_wrong_logit)

    eps_acc = []
    for e in eps_range:
        eps_acc.append(np.expand_dims((adv_margin > 2 * e).float().detach().cpu().numpy(), 0))
    return np.concatenate(eps_acc, 0).T


def get_theoretic_lowerbound(model, eps_range, config, pretrained_config, output_root):
    n_examples = config['num_examples']
    n_batches = int(math.ceil((n_examples * 1.0) / pretrained_config.optim.batch_size))

    pretrained_config.cuda = pretrained_config.cuda
    data = load_data(pretrained_config)

    # Perform the attack.
    eps_acc = []
    for sample in tqdm(islice(data['validation'], n_batches), total=n_batches):
        x = to_cuda(sample[0], cuda=pretrained_config.cuda)
        y = to_cuda(sample[1].type(torch.LongTensor), cuda=pretrained_config.cuda)
        eps_acc.append(theoretical_adversary(model, x, y, pretrained_config.model.l_constant * eps_range))
    avg_eps_acc = np.concatenate(eps_acc, 0).mean(0)

    results = {
        'eps': eps_range.tolist(),
        'acc': avg_eps_acc.tolist()
    }

    print(results)
    with open(os.path.join(output_root, 'results.json'), 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


def generate_examples(model, config, pretrained_config, output_root):
    adv_example_filepath = os.path.join(output_root, 'examples')
    adv_targets_filepath = os.path.join(output_root, 'adv_targets')
    targets_filepath = os.path.join(output_root, 'targets')

    n_examples = config['num_examples']
    n_batches = int(math.ceil((n_examples * 1.0) / pretrained_config.optim.batch_size))

    pretrained_config.cuda = pretrained_config.cuda
    data = load_data(pretrained_config)

    # Save the results of the computations in the following variable.
    adv_ex = torch.Tensor()
    adv_targets = torch.LongTensor()
    true_targets = torch.LongTensor()
    adv_mse = torch.Tensor()
    adv_inf = torch.Tensor()
    success = torch.Tensor()
    margins = torch.Tensor()

    # Perform the attack.
    i = 0
    for sample in tqdm(islice(data['validation'], n_batches), total=n_batches):
        i += 1
        model.zero_grad()
        x = to_cuda(sample[0], cuda=pretrained_config.cuda)
        y = to_cuda(sample[1].type(torch.LongTensor), cuda=pretrained_config.cuda)
        true_targets = torch.cat([true_targets, y.detach().cpu()], 0)

        if config.fgs:
            adv, adv_t, original_pred = manual_fgs(model, x, y, config.eps, clamp=False)
        elif config.pgd:
            adv, adv_t, original_pred = manual_pgd(model, x, y, config.eps, config.eps, rand_start=False, clamp=False)
        adv_ex = torch.cat([adv_ex, adv.cpu().detach()], 0)
        adv_targets = torch.cat([adv_targets, adv_t.cpu().detach()], 0)

        # import pdb; pdb.set_trace()
        original_top_2 = model(x).topk(2, 1)[0]
        original_margin = original_top_2[:, 0] - original_top_2[:, 1]
        margins = torch.cat([margins, original_margin.cpu().detach()], 0)

        batch_success = ((original_pred == y) & (adv_t != y)).float()
        success = torch.cat([success, batch_success.cpu().detach()], 0)
        adv_mse = torch.cat(
            [adv_mse, ((adv.view(adv.size(0), -1) - x.view(adv.size(0), -1)) ** 2).mean(-1).cpu().detach()], 0)
        adv_inf = torch.cat(
            [adv_inf, (adv.view(adv.size(0), -1) - x.view(adv.size(0), -1)).abs().max(-1)[0].cpu().detach()], 0)

    total_accuracy = accuracy(model, to_cuda(adv_ex, cuda=config.cuda), to_cuda(true_targets, cuda=config.cuda)).item()

    # Summarize the results.
    results = {
        "eps": config.eps,
        "success_rate": success.mean().item(),
        "defense_rate": 1 - success.mean().item(),
        "total_acc": total_accuracy,
        "all_margins_mean": margins.mean().item(),
        "successful_margins": ((margins * success).sum() / success.sum()).item(),
        "mean_mse": ((adv_mse * success).sum() / success.sum()).item(),
        "mean_inf": ((adv_inf * success).sum() / success.sum()).item(),
        "mse_quartiles": list(np.percentile(adv_mse[success == 1.0].numpy(), [0, 25, 50, 75, 100])) \
            if len(adv_mse[success == 1.0]) > 0 else [0, 0, 0, 0, 0]
    }

    results["median_mse"] = results["mse_quartiles"][2]

    print("success rate: {}".format(results["success_rate"]))
    print("defense rate: {}".format(results["defense_rate"]))
    print("total accuracy: {}".format(results["total_acc"]))
    print("Avg Margin: {}".format(results['all_margins_mean']))
    print("Avg Success Margin: {}".format(results['successful_margins']))
    print("mean MSE for successful attacks: {}".format(results["mean_mse"]))
    print("mean L_inf for successful attacks: {}".format(results["mean_inf"]))
    print("MSE quartiles for successful attacks: {}".format(results["mse_quartiles"]))

    with open(os.path.join(output_root, 'results.json'), 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)

    np.save(adv_example_filepath, adv_ex)
    np.save(adv_targets_filepath, adv_targets)
    np.save(targets_filepath, true_targets)

    return adv_ex


def eval_on_examples(model, output_root, cuda=True):
    adv_examples = np.load(os.path.join(output_root, 'examples.npy'))
    adv_targets = np.load(os.path.join(output_root, 'targets.npy'))

    print(adv_examples.shape)
    adv_ex_t = torch.Tensor(adv_examples)
    save_image(adv_ex_t, 'test.png')

    adv_examples = to_cuda(torch.Tensor(adv_examples), cuda)
    adv_targets = to_cuda(torch.LongTensor(adv_targets), cuda)
    print("Adv Accuracy: {}".format(accuracy(model, adv_examples, adv_targets).item()))


def load_model_hack(pretrained_root):
    model_path = os.path.join(pretrained_root, 'checkpoints', 'best', 'best_model.pt')
    json_path = os.path.join(pretrained_root, 'logs', 'config.json')

    with open(json_path, 'r') as f:
        model_config = Munch.fromDict(json.load(f))

    if 'groupings' in model_config.model and model_config.model.groupings[0] is -1:
        model_config.model.groupings = model_config.model.groupings[1:]

    from lnets.models import get_model
    model = get_model(model_config)
    state_dict = torch.load(model_path)['state_dict'][0]

    for (a, b) in zip(model.model.model.state_dict(), state_dict):
        model.model.model.state_dict()[a].copy_(state_dict[b])
    return model, model_config


def main(config):
    print(config)
    # Create the output directory.
    output_root = config.output_root
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    # Load pretrained model
    pretrained_path = config.model.exp_path
    model, pretrained_config = load_model_from_config(pretrained_path)
    # model, pretrained_config = load_model_hack(pretrained_path)

    # Push model to GPU if available.
    if config.cuda:
        print('Using cuda: {}'.format("Yes"))
        to_cuda(model, cuda=config.cuda)

    model.eval()
    eps_range = np.linspace(0.01, 0.5, 20)

    # exp_root = os.path.join(output_root, 'theory')
    # if not os.path.isdir(exp_root):
    #     os.makedirs(exp_root)
    # get_theoretic_lowerbound(model, eps_range, config, pretrained_config, exp_root)

    config.fgs = True
    for e in eps_range:
        config.eps = e
        exp_root = os.path.join(output_root, 'fgs', str(e))
        if not os.path.isdir(exp_root):
            os.makedirs(exp_root)
        generate_examples(model, config, pretrained_config, exp_root)

    config.fgs = False
    config.pgd = True
    for e in eps_range:
        config.eps = e
        exp_root = os.path.join(output_root, 'pgd', str(e))
        if not os.path.isdir(exp_root):
            os.makedirs(exp_root)
        generate_examples(model, config, pretrained_config, exp_root)

    # eval_on_examples(model, output_root, config.cuda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate adversarial robustness of classification network')

    parser.add_argument('--model.exp_path', type=str, metavar='MODELPATH',
                        help="location of pretrained model weights to evaluate")
    parser.add_argument('--output_root', type=str, metavar='OUTPUT_ROOT',
                        help="output root for experiment")
    parser.add_argument('--num_examples', type=int, metavar='NUM_EXAMPLES',
                        help="number of examples to test", default=1000)
    parser.add_argument('--cuda', action='store_true', default=False, help="run in CUDA mode (default: False)")

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
