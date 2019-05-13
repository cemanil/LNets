import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def dataloader(dataset, input_size, batch_size, data_root="data", split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        data_path = os.path.join(data_root, "mnist")
        data_loader = DataLoader(
            datasets.MNIST(data_path, train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_path = os.path.join(data_root, "fashion-mnist")
        data_loader = DataLoader(
            datasets.FashionMNIST(data_path, train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_path = os.path.join(data_root, "cifar10")
        data_loader = DataLoader(
            datasets.CIFAR10(data_path, train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_path = os.path.join(data_root, "svhn")
        data_loader = DataLoader(
            datasets.SVHN(data_path, split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_path = os.path.join(data_root, "stl10")
        data_loader = DataLoader(
            datasets.STL10(data_path, split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_path = os.path.join(data_root, "lsun")
        data_loader = DataLoader(
            datasets.LSUN(data_path, classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)

    return data_loader
