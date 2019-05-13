import os
import torch
import datetime
import json
import matplotlib as mpl
mpl.use("Agg")

from lnets.tasks.gan.models.WGAN import WGAN
from lnets.tasks.gan.models.WGAN_GP import WGAN_GP
from lnets.tasks.gan.models.LWGAN import LWGAN
from lnets.utils.config import process_config
from lnets.utils.seeding import set_experiment_seed


def get_exp_name(config):
    # Get experiment name.
    now = datetime.datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

    base_exp_name = config.exp_name
    task_name = config.task
    data_name = config.dataset
    gan_type = config.gan_type

    exp_name = "{}_{}_{}_{}_{}".format(task_name, base_exp_name, data_name, gan_type, now_str)

    return exp_name


def create_dirs(config):
    # Get experiment name.
    exp_name = get_exp_name(config)

    # Construct names of related directories.
    exp_dir = os.path.join(config.output_root, exp_name)
    log_dir = os.path.join(exp_dir, 'logs')
    model_dir = os.path.join(exp_dir, 'checkpoints')
    figures_dir = os.path.join(exp_dir, 'figures')
    hparams_dir = os.path.join(exp_dir, "hparams")
    data_root = config.data_root

    # Create non-existing directories.
    for dr in [exp_dir, log_dir, model_dir, figures_dir, hparams_dir, data_root]:
        if not os.path.exists(dr):
            print(dr)
            os.makedirs(dr)

    # Add to config dictionary.
    config.model_dir = model_dir
    config.log_dir = log_dir
    config.figures_dir = figures_dir
    config.hparams_dir = hparams_dir
    config.data_root = data_root

    return config


def save_hparams(config):
    hparams_string = json.dumps(config)

    hparams_path = os.path.join(config.hparams_dir, "hparams.json")
    with open(hparams_path, "w") as hparam_file:
        hparam_file.write(hparams_string)


def main():
    # Parse config json.
    cfg = process_config()

    # Set the seed.
    set_experiment_seed(cfg.seed)

    # Create directories to be used in the experiments.
    cfg = create_dirs(cfg)

    if cfg.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    # Declare instance for GAN.
    if cfg.gan_type == 'WGAN':
        gan = WGAN(cfg)
    elif cfg.gan_type == 'LWGAN':
        gan = LWGAN(cfg)
    elif cfg.gan_type == 'WGAN_GP':
        gan = WGAN_GP(cfg)
    else:
        raise Exception("[!] There is no option for " + cfg.gan_type)

    # Save the hyperparameter json.
    save_hparams(cfg)

    # Launch the graph in a session.
    gan.train()
    print(" [*] Training finished!")

    # Visualize learned generator.
    gan.visualize_results(cfg.epoch)
    print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
