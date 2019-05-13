import os
from torch.utils.data import Subset, DataLoader
import torchvision.datasets as datasets

from lnets.data.data_transforms import get_data_transforms
from lnets.data.utils import load_indices


def get_datasets(config):
    data_name = config['data']['name'].lower()
    path = os.path.join(config['data']['root'], data_name)

    train_transform, test_transform = get_data_transforms(config)

    train_data_args = dict(download=True, transform=train_transform)
    val_data_args = dict(download=True, transform=test_transform)
    test_data_args = dict(train=False, download=True, transform=test_transform)

    if data_name == 'mnist':
        train_data = datasets.MNIST(path, **train_data_args)
        val_data = datasets.MNIST(path, **val_data_args)
        test_data = datasets.MNIST(path, **test_data_args)
    elif data_name == 'cifar10':
        train_data = datasets.CIFAR10(path, **train_data_args)
        val_data = datasets.CIFAR10(path, **val_data_args)
        test_data = datasets.CIFAR10(path, **test_data_args)
    elif data_name == 'cifar100':
        train_data = datasets.CIFAR100(path, **train_data_args)
        val_data = datasets.CIFAR100(path, **val_data_args)
        test_data = datasets.CIFAR100(path, **test_data_args)
    elif data_name == 'fashion-mnist':
        train_data = datasets.FashionMNIST(path, **train_data_args)
        val_data = datasets.FashionMNIST(path, **val_data_args)
        test_data = datasets.FashionMNIST(path, **test_data_args)
    elif data_name == 'imagenet-torchvision':
        train_data = datasets.ImageFolder(os.path.join(path, 'train'), transform=train_transform)
        val_data = datasets.ImageFolder(os.path.join(path, 'valid'), transform=test_transform)
        # Currently not loaded.
        test_data = None
    else:
        raise NotImplementedError('Data name %s not supported' % data_name)

    return train_data, val_data, test_data


def build_loaders(config, train_data, val_data, test_data):
    data_name = config['data']['name'].lower()
    batch_size = config['optim']['batch_size']
    num_workers = config['data']['num_workers']

    if config['data']['indices_path'] is not None:
        train_indices, val_indices = load_indices(config['data']['indices_path'], config['data']['per_class_count'])
        train_data = Subset(train_data, train_indices)
        val_data = Subset(val_data, val_indices)
    elif data_name != 'imagenet-torchvision':
        # Manually readjust train/val size for memory saving.
        data_size = len(train_data)
        train_size = int(data_size * config['data']['train_size'])

        train_data.train_data = train_data.train_data[:train_size]
        train_data.train_labels = train_data.train_labels[:train_size]

        if config['data']['train_size'] != 1:
            val_data.train_data = val_data.train_data[train_size:]
            val_data.train_labels = val_data.train_labels[train_size:]
        else:
            val_data = None

    loaders = {
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'validation': DataLoader(val_data, batch_size=batch_size, num_workers=num_workers),
        'test': DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    }

    return loaders


def load_data(config):
    train_data, val_data, test_data = get_datasets(config)
    return build_loaders(config, train_data, val_data, test_data)
