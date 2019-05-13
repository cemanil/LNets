import torch

from lnets.utils.dynamic_importer import dynamic_import


def load_distrib(config):
    distrib_loaders = dict()

    distrib_loaders['train'] = DistribLoader(config, mode="train")
    distrib_loaders['validation'] = DistribLoader(config, mode="test")
    distrib_loaders['test'] = DistribLoader(config, mode="test")

    return distrib_loaders


class DistribLoader(object):
    def __init__(self, config, mode="train"):
        assert mode == "train" or mode == "test", "Mode must be either 'train' or 'test'."
        self.distrib1 = construct_distrib_instance(config.distrib1)
        self.distrib2 = construct_distrib_instance(config.distrib2)
        self.config = config
        self.mode = mode

    def __iter__(self):
        self.sampled_so_far = 0
        return self

    def __next__(self):
        if self.sampled_so_far < self.config.optim.epoch_len:
            self.sampled_so_far += 1

            if self.mode == "train":
                distrib1_samples = self.distrib1(self.config.distrib1.sample_size)
                distrib2_samples = self.distrib2(self.config.distrib2.sample_size)

            elif self.mode == "test":
                distrib1_samples = self.distrib1(self.config.distrib1.test_sample_size)
                distrib2_samples = self.distrib2(self.config.distrib2.test_sample_size)

            # if the samples are already PyTorch tensors, don't touch them.
            if not isinstance(distrib1_samples, torch.Tensor):
                distrib1_samples = torch.from_numpy(distrib1_samples).float()

            if not isinstance(distrib2_samples, torch.Tensor):
                distrib2_samples = torch.from_numpy(distrib2_samples).float()

            return (distrib1_samples.float(),
                    distrib2_samples.float())
        else:
            raise StopIteration


def construct_distrib_instance(distrib_config):
    assert type(distrib_config.filepath) == str, "distrib_config should have a string field called 'filepath'. "
    assert type(distrib_config.name) == str, "distrib_config should have a string field called 'name'. "
    assert distrib_config.filepath.endswith(".py"), "distrib_config.filename has to be a python file. "

    name = distrib_config.name
    filepath = distrib_config.filepath

    # Import the distribution. Here, the distrib_config.filepath is the script in which the distribution is
    # implemented and distrib_config.class_name is the (string) name of the distribution class you are trying to import.
    distrib_class = dynamic_import(filepath, name)

    # Create an instance.
    distrib = distrib_class(distrib_config)

    return distrib
