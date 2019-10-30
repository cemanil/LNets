from functools import partial
from tqdm import tqdm

#mpl.use('Agg')
import matplotlib.pyplot as plt
plt.interactive(False)

from lnets.utils.config import process_config
from lnets.tasks.dualnets.distrib.load_distrib import load_distrib
from lnets.trainers.trainer import Trainer
from lnets.utils.logging import Logger
from lnets.utils.training_getters import get_optimizer, get_scheduler
from lnets.utils.saving_and_loading import *
from lnets.utils.seeding import set_experiment_seed
from lnets.utils.misc import *
from lnets.utils.training_getters import get_training_dirs
from lnets.tasks.dualnets.visualize.visualize_dualnet import *


def train_dualnet(model, loaders, config):
    # Set the seed.
    set_experiment_seed(config.seed)

    # Get relevant paths.
    dirs = get_training_dirs(config)

    # Get optimizer and learning rate scheduler.
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)

    # Load pretrained model and the state of the optimizer when it was saved.
    if config.model.pretrained_best_path:
        load_best_model_and_optimizer(model, optimizer, config.model.pretrained_best_path)

    # Push model to GPU if available.
    if config.cuda:
        print('Using cuda: {}'.format("Yes"))
        model.cuda()

    # Get logger, and log the config.
    logger = Logger(dirs.log_dir)
    logger.log_config(config)

    # Instantiate the trainer.
    trainer = Trainer()

    # Initialize "best performance" statistic, to be used when saving best model.
    best_val = initialize_best_val(config.optim.criterion.minmax)

    # Define hooks.
    def on_sample(state):
        if config.cuda:
            state['sample'] = [x.cuda() for x in state['sample']]

    def on_forward(state):
        state['model'].add_to_meters(state)

        # Clip gradients.
        torch.nn.utils.clip_grad_norm_(state['model'].parameters(), config.optim.max_grad_norm)

        # Save the most recent loss.
        state['recent_losses'].append(state['loss'].item())

    def on_update(state):
        if config.model.per_update_proj.turned_on:
            state['model'].model.project_network_weights(config.model.per_update_proj)

    def on_start(state):
        state['loader'] = state['iterator']
        state['scheduler'] = scheduler

        # Keep track of the max, mean and min singular values of the second layer weights.
        state["max_singular"] = list()
        state["mean_singular"] = list()
        state["min_singular"] = list()
        state["singulars"] = list()

    def on_start_val(state):
        # Initialize a list that is to store all of the losses encountered in the recent epoch.
        state['recent_losses'] = list()

    def on_start_epoch(state):
        state['model'].reset_meters()
        state['iterator'] = tqdm(state['loader'], desc='Epoch {}'.format(state['epoch']))

        # Initialize a list that is to store all of the losses encountered in the recent epoch.
        state['recent_losses'] = list()

        # Project the weights on the orthonormal matrix manifold if the layer type is suitable to do so.
        if config.model.per_epoch_proj.turned_on:
            if state['epoch'] % config.model.per_epoch_proj.every_n_epochs == 0 and state['epoch'] != 0:
                state['model'].model.project_network_weights(config.model.per_epoch_proj)
                # Reset optimizer is necessary. Especially useful for stateful optimizers.
                if config.model.per_epoch_proj.reset_optimizer:
                    state['optimizer'] = get_optimizer(config, model.parameters())

    def on_end_epoch(hook_state, state):
        scheduler.step()

        print("\t\t\tTraining loss: {:.4f}".format(state['model'].meters['loss'].value()[0]))
        logger.log_meters('train', state)

        if state['epoch'] % config.logging.report_freq == 0:
            if config.logging.save_model:
                save_current_model_and_optimizer(model, optimizer, model_dir=dirs.model_dir, epoch=state['epoch'])

            # Visualize the learned critic landscape.
            if config.visualize:
                save_1_or_2_dim_dualnet_visualizations(model, dirs.figures_dir, config,
                                                       state['epoch'], state['loss'])

        # Check if this is the best model.
        if config.logging.save_best:
            hook_state['best_val'], new_best = save_best_model_and_optimizer(state, hook_state['best_val'],
                                                                             dirs.best_path, config)

        # Validate the model.
        if loaders['validation'] is not None:
            valid_state = trainer.test(model, loaders['validation'])
            logger.log_meters('validation', valid_state)

    def on_end_val(state):
        print("Averaged validation loss: {}".format(np.array(state['recent_losses']).mean()))

    trainer.hooks['on_start'] = on_start
    trainer.hooks['on_start_val'] = on_start_val
    trainer.hooks['on_sample'] = on_sample
    trainer.hooks['on_forward'] = on_forward
    trainer.hooks['on_update'] = on_update
    trainer.hooks['on_start_epoch'] = on_start_epoch
    trainer.hooks['on_end_epoch'] = partial(on_end_epoch, {'best_val': best_val, 'wait': 0})
    trainer.hooks['on_end_val'] = on_end_val

    # Enter the training loop.
    training_state = trainer.train(model, loaders['train'], maxepoch=config.optim.epochs, optimizer=optimizer)

    # Save the singular value statistics.
    singulars = dict()
    singulars['max_singulars'] = training_state['max_singular']
    singulars['mean_singulars'] = training_state['mean_singular']
    singulars['min_singulars'] = training_state['min_singular']
    singulars['singulars'] = training_state['singulars']

    import pickle
    pickle.dump(singulars, open(os.path.join(dirs.log_dir, "singular_vals_dict.pkl"), "wb"))

    # Pick the best model according to validation score and test it.
    model.reset_meters()
    best_model_path = os.path.join(dirs.best_path, "best_model.pt")
    if os.path.exists(dirs.best_path):
        model.load_state_dict(torch.load(best_model_path))
    if loaders['test'] is not None:
        print("Testing best model. ")
        test_state = trainer.test(model, loaders['test'])
        logger.log_meters('test', test_state)
    else:
        raise RuntimeError("The trained models must be tested with a testing distribution. ")

    # Visualize the learned critic landscape.
    if config.visualize:
        save_1_or_2_dim_dualnet_visualizations(model, dirs.figures_dir, config,
                                               after_training=False)

    return test_state


if __name__ == '__main__':
    print("test")
    # Get the config, initialize the model and construct the distribution loader.
    cfg = process_config()
    dual_model = get_model(cfg)
    print(dual_model)
    distrib_loaders = load_distrib(cfg)

    # Train.
    final_state = train_dualnet(dual_model, distrib_loaders, cfg)
