import errno
import json
from torchvision.utils import save_image
from munch import Munch

from lnets.tasks.dualnets.visualize.visualize_dualnet import *
from lnets.models import get_model


def save_imgs(tensor, fname, save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    save_image(tensor, os.path.join(save_dir, fname))


def save_model(model, save_path):
    try:
        os.makedirs(os.path.dirname(save_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    torch.save(model.state_dict(), save_path)


def save_optimizer(optimizer, save_path):
    try:
        os.makedirs(os.path.dirname(save_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    torch.save(optimizer.state_dict(), save_path)


def save_best_model_and_optimizer(state, best_value, best_path, config):
    """
    Save model that performs the best on the validation set.
    """
    criterion = config['optim']['criterion']
    new_best = False
    for tag, meter in state['model'].meters.items():
        if tag == criterion['tag']:
            new_val = meter.value()[0]
            if criterion['minmax'] == 'min':
                if new_val < best_value:
                    best_value = new_val
                    new_best = True
            else:
                if new_val > best_value:
                    best_value = new_val
                    new_best = True
            break
    if new_best:
        best_model_path = os.path.join(best_path, "best_model.pt")
        best_optimizer_path = os.path.join(best_path, "best_optimizer.pt")
        print('Saving new best model at {}. '.format(best_path))
        save_model(state['model'], best_model_path)
        save_optimizer(state['optimizer'], best_optimizer_path)

    return best_value, new_best


def save_current_model_and_optimizer(model, optimizer, model_dir, epoch):
    # Save model state.
    save_model_path = os.path.join(model_dir, "model_{}.pt".format(epoch))
    save_model(model, save_model_path)

    # Save optimizer state.
    save_optimizer_path = os.path.join(model_dir, "optimizer_{}.pt".format(epoch))
    save_optimizer(optimizer, save_optimizer_path)


def load_model(model, load_path):
    model.reset_meters()
    print("Reading model from: {}".format(load_path))
    model.load_state_dict(torch.load(load_path))


def load_optimizer(optimizer, load_path):
    print("Reading optimizer from: {}".format(load_path))
    optimizer.load_state_dict(torch.load(load_path))


def load_best_model_and_optimizer(model, optimizer, best_path):
    best_model_path = os.path.join(best_path, "best_model.pt")
    load_model(model, best_model_path)

    best_optimizer_path = os.path.join(best_path, "best_optimizer.pt")
    load_optimizer(optimizer, best_optimizer_path)


def save_1_or_2_dim_dualnet_visualizations(model, figures_dir, config, epoch=None, loss=None,
                                           after_training=False):

    dim = config.distrib1.dim
    if not after_training:
        if dim == 2:
            save_2d_dualnet_visualizations(model, figures_dir, config, epoch, loss)
        if dim == 1:
            save_1d_dualnet_visualizations(model, figures_dir, config, epoch, loss)
    else:
        if dim == 2:
            save_2d_dualnet_visualizations(model, figures_dir, config, after_training=True)
        if dim == 1:
            save_1d_dualnet_visualizations(model, figures_dir, config, after_training=True)


def load_model_from_config(pretrained_root):
    model_path = os.path.join(pretrained_root, 'checkpoints', 'best', 'best_model.pt')
    json_path = os.path.join(pretrained_root, 'logs', 'config.json')

    with open(json_path, 'r') as f:
        model_config = Munch.fromDict(json.load(f))

    # Weird required hack to fix groupings (None is added to start during model training).
    if 'groupings' in model_config.model and model_config.model.groupings[0] is -1:
        model_config.model.groupings = model_config.model.groupings[1:]

    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path))

    return model, model_config


