import os
import datetime
import torch
import torch.optim.lr_scheduler as lr_scheduler
from munch import Munch

from lnets.optimizers.aggmo import AggMo


def get_optimizer(config, params, momentum=None, betas=None):
    optim_name = config.optim.optimizer.lower()
    lr = config.optim.lr_schedule.lr_init

    if momentum is None:
        momentum = config.optim.momentum

    if betas is None:
        betas = config.optim.betas

    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr, momentum=momentum,
                                    weight_decay=config.optim.wdecay)
    elif optim_name == 'nesterov':
        optimizer = torch.optim.SGD(params, lr, momentum=momentum, nesterov=True,
                                    weight_decay=config.optim.wdecay)
    elif optim_name == 'aggmo':
        optimizer = AggMo(params, lr, momentum=betas, weight_decay=config.optim.wdecay)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(params, lr, betas=(momentum, 0.999), weight_decay=config.optim.wdecay)
    else:
        raise ValueError("The requested optimizer type is not supported. ")

    return optimizer


def get_scheduler(config, optimizer):
    lr_schedule_config = config.optim.lr_schedule
    if lr_schedule_config.name == 'exp':
        return lr_scheduler.ExponentialLR(optimizer, lr_schedule_config.lr_decay, lr_schedule_config.last_epoch)
    elif lr_schedule_config.name == 'step':
        return lr_scheduler.MultiStepLR(optimizer, lr_schedule_config.milestones, lr_schedule_config.lr_decay)


def get_model_repr(config):
    if "fc" in config.model.name:
        model_repr = "{}_linear_{}_act_{}_depth_{}_width_{}_grouping_{}".format(config.model.name,
                                                                                config.model.linear.type,
                                                                                config.model.activation,
                                                                                len(config.model.layers),
                                                                                max(config.model.layers),
                                                                                max(config.model.groupings))
    elif "fully_conv" in config.model.name:
        model_repr = "{}_act_{}_depth_{}_channels_{}_grouping_{}".format(config.model.name,
                                                                         config.model.activation,
                                                                         len(config.model.channels),
                                                                         max(config.model.channels),
                                                                         max(config.model.groupings))
    elif "lenet" in config.model.name:
        model_repr = "lenet"
    elif "alexnet" in config.model.name:
        model_repr = "alexnet"
    elif config.model.name == "lipschitz_infogan_discriminator":
        model_repr = "lipschitz_infogan_discriminator"
    elif config.model.name == "parseval_infogan_discriminator":
        model_repr = "parseval_infogan_discriminator"
    else:
        model_repr = None
        print("Write a new model repr for this architecture. ")
        exit(-1)

    return model_repr


def get_optimizer_repr(config):
    return "{}_{}".format(config.optim.optimizer, config.optim.lr_schedule.lr_init)


def get_experiment_name(config):
    now = datetime.datetime.now()

    base_exp_name = config.exp_name
    task_name = config.task

    try:
        data_name = config.data.name
    except:
        data_name = 'none'
        print("No dataset seems to be used for the training. ")
        try:
            data_name = config.distrib1.name + '_and_' + config.distrib2.name
        except:
            print("No distribution seem to be used for the training. ")

    optim_name = get_optimizer_repr(config)

    model_name = get_model_repr(config)

    exp_name = "{}_{}_{}_{}_{}_{}".format(task_name, base_exp_name, data_name, optim_name, model_name,
                                          now.strftime("%Y_%m_%d_%H_%M_%S_%f"))
    return exp_name


def get_training_dirs(config):
    exp_dir = os.path.join(config.output_root, get_experiment_name(config))
    log_dir = os.path.join(exp_dir, 'logs')
    model_dir = os.path.join(exp_dir, 'checkpoints')
    figures_dir = os.path.join(exp_dir, 'figures')
    best_path = os.path.join(model_dir, "best")

    print("Experiment dir: {}".format(exp_dir))
    print("Log dir: {}".format(log_dir))
    print("Model dir: {}".format(model_dir))
    print("Figures dis: {}".format(figures_dir))
    print("Best model dir: {}".format(best_path))

    dirs = dict(exp_dir=exp_dir, log_dir=log_dir, model_dir=model_dir, figures_dir=figures_dir, best_path=best_path)

    for dir_key in dirs:
        if not os.path.exists(dirs[dir_key]):
            os.makedirs(dirs[dir_key])

    return Munch.fromDict(dirs)
