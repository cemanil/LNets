import json
import pprint
from munch import Munch
pp = pprint.PrettyPrinter()

from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib
from lnets.tasks.gan.models.WGAN import WGAN


class GANSampler(BaseDistrib):
    def __init__(self, config):
        super(GANSampler, self).__init__(config)

        self.config = config

        # Load GAN hyperparameters from GAN training json.
        self.gan_config_json_path = config.gan_config_json_path
        self.gan_config = Munch(json.load(open(self.gan_config_json_path)))
        print('-------- GAN Training Config --------')
        pp.pprint(self.gan_config)
        print('------------------------')

        # Instantiate the GAN model class.
        self.gan = self.instantiate_gan()

        # Load weights.
        self.gan.load()

        # Whether we want to sample real of generated images.
        self.generate_type = self.config.generate_type
        assert self.generate_type == "real" or self.generate_type == "generated", \
            "Must be one of 'generated', or 'real'. "

    def __call__(self, size):
        assert size == self.gan_config.batch_size

        if self.generate_type == "generated":
            samples = self.gan.get_generated(size)
        elif self.generate_type == "real":
            samples = self.gan.get_real(size)

        return samples
            
    def instantiate_gan(self):
        if self.gan_config.gan_type == 'WGAN':
            gan = WGAN(self.gan_config)
        else:
            raise Exception("[!] There is no option for " + self.gan_config.gan_type)

        return gan
