"""
Evaluate adversarial robustness of a given classifier.
"""
import os
import json
from itertools import islice
import math
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import TensorDataset
from torchvision.utils import save_image

import foolbox.attacks
from foolbox.criteria import Misclassification
from foolbox.models import PyTorchModel

from lnets.data.load_data import load_data
from lnets.utils.config import process_config
from lnets.utils.saving_and_loading import load_model_from_config
from lnets.utils.misc import to_cuda
from lnets.tasks.adversarial.attack.perform_attack import batch_attack
from lnets.tasks.adversarial.mains.check_adv_gradients import slow_check_grad_norm


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


def evaluate_adv_grad_norms(model, adv_ex, adv_t, cuda):
    dataset = TensorDataset(adv_ex, adv_t)
    s_avg, s_max = slow_check_grad_norm(model, dataset, cuda)
    print(s_avg, s_max)


def generate_examples(model, config, pretrained_config, output_root):
    adv_example_filepath = os.path.join(output_root, 'examples')
    adv_targets_filepath = os.path.join(output_root, 'targets')

    # Set up adversarial attack.
    adv_model = PyTorchModel(model, (0, 1), pretrained_config.data.class_count, cuda=config.cuda)
    criterion = Misclassification()
    attack = getattr(foolbox.attacks, config.name)(adv_model, criterion)

    # Get data.
    pretrained_config.cuda = config.cuda
    pretrained_config.optim.batch_size = config.data.batch_size
    data = load_data(pretrained_config)
    # print('Test Accuracy:{}'.format(loader_accuracy(model, data['test'])))

    n_examples = config['num_examples']
    n_batches = int(math.ceil((n_examples * 1.0) / pretrained_config.optim.batch_size))

    # Save the results of the computations in the following variable.
    adv_ex = torch.Tensor()
    adv_targets = torch.LongTensor()
    adv_mse = torch.Tensor()
    adv_inf = torch.Tensor()
    success = torch.Tensor()

    # Set up distance for the adversarial attack.
    distance_name = config.get('distance')
    distance = getattr(foolbox.distances, distance_name) if distance_name is not None \
        else foolbox.distances.MeanSquaredDistance

    # Perform the attack.
    for sample in tqdm(islice(data['validation'], n_batches), total=n_batches):
        x = sample[0]
        y = sample[1].type(torch.LongTensor)
        x = to_cuda(x, cuda=config.cuda)

        adv, adv_t, batch_success, batch_adv_mse, batch_adv_inf = batch_attack(attack, adv_model, criterion, x,
                                                                               y.cpu().numpy(),
                                                                               config['attack_kwargs'], distance)
        adv_ex = torch.cat([adv_ex, adv], 0)
        adv_targets = torch.cat([adv_targets, adv_t], 0)
        success = torch.cat([success, batch_success], 0)
        adv_mse = torch.cat([adv_mse, batch_adv_mse], 0)
        adv_inf = torch.cat([adv_inf, batch_adv_inf], 0)

    # evaluate_adv_grad_norms(model, adv_ex, adv_targets, config.cuda)
    # Summarize the results.
    results = {
        "success_rate": success.mean().item(),
        "defense_rate": 1 - success.mean().item(),
        "mean_mse": ((adv_mse * success).sum() / success.sum()).item(),
        "mean_inf": ((adv_inf * success).sum() / success.sum()).item(),
        "mse_quartiles": list(np.percentile(adv_mse[success == 1.0].numpy(), [0, 25, 50, 75, 100]))
    }

    results["median_mse"] = results["mse_quartiles"][2]

    print("success rate: {}".format(results["success_rate"]))
    print("defense rate: {}".format(results["defense_rate"]))
    print("mean MSE for successful attacks: {}".format(results["mean_mse"]))
    print("mean L_inf for successful attacks: {}".format(results["mean_inf"]))
    print("MSE quartiles for successful attacks: {}".format(results["mse_quartiles"]))

    with open(os.path.join(config['output_root'], 'results.json'), 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)

    np.save(adv_example_filepath, adv_ex)
    np.save(adv_targets_filepath, adv_targets)

    print(accuracy(model, to_cuda(adv_ex, cuda=config.cuda), to_cuda(adv_targets, cuda=config.cuda)))


def eval_on_examples(model, output_root, cuda=True):
    adv_examples = np.load(os.path.join(output_root, 'examples.npy'))
    adv_targets = np.load(os.path.join(output_root, 'targets.npy'))

    print(adv_examples.shape)
    adv_ex_t = torch.Tensor(adv_examples)
    save_image(adv_ex_t, 'test.png')

    adv_examples = to_cuda(torch.Tensor(adv_examples), cuda)
    adv_targets = to_cuda(torch.LongTensor(adv_targets), cuda)
    print("Adv Accuracy: {}".format(accuracy(model, adv_examples, adv_targets).item()))


def main(config):
    # Create the output directory.
    output_root = config.output_root
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    # Load pretrained model
    pretrained_path = config.pretrained_path
    model, pretrained_config = load_model_from_config(pretrained_path)

    # Push model to GPU if available.
    if config.cuda:
        print('Using cuda: {}'.format("Yes"))
        to_cuda(model, cuda=config.cuda)

    model.eval()

    # model.model.project_network_weights(Munch.fromDict({'type': 'l_inf_projected'}))
    generate_examples(model, config, pretrained_config, output_root)

    # eval_on_examples(model, output_root, config.cuda)


if __name__ == '__main__':
    cfg = process_config()

    main(cfg)
