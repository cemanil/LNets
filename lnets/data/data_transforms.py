import torchvision.transforms as transforms


def get_data_transforms(config):
    # train_transform = None
    test_transform = None

    if config.data.transform.type == 'cifar':
        train_transform, test_transform = get_cifar_transform(config)
    elif config.data.transform.type == 'imagenet':
        train_transform, test_transform = get_imagenet_transform(config)
    else:
        train_transform = transforms.ToTensor()

    # Make sure to turn the input images into PyTorch tensors.
    if test_transform is None:
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_cifar_transform(config):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return train_transform, test_transform


def get_imagenet_transform(config):
    normalize = transforms.Normalize(mean=config.data.transform.norm_mean,
                                     std=config.data.transform.norm_std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform
