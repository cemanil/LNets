import numpy as np

from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib


class MultiSphericalShell(BaseDistrib):
    def __init__(self, config):
        super(MultiSphericalShell, self).__init__(config)

        self.dim = config.dim
        self.empty_dim = config.empty_dim

        self.center_x = config.center_x
        self.radius = config.radius
        self.reshape_to_grid = config.reshape_to_grid

        assert self.dim > 0, "Dimensionality must be larger than 0. "
        assert self.empty_dim >= 0, "The number of empty dimensions must be at least 0. "
        assert self.radius > 0, "Radius of the sphere must be strictly larger than 0. "

    def __call__(self, size):
        # First, sample origin-centered shells.
        if self.dim > 1:
            # Sample from an isotropic normal distribution and normalize the obtained samples.
            samples = np.random.multivariate_normal(mean=np.zeros(shape=self.dim), cov=np.eye(self.dim),
                                                    size=size)
            # Origin centered sample.
            samples = samples * (self.radius / np.linalg.norm(samples, axis=1)[..., None])

        else:
            # A "circle" in 1D are simply two points equidistant from the origin.
            samples = self.radius * 2 * (np.random.binomial(1, 0.5, size=size) - 0.5)

            samples = samples[..., None]

        # Compute the centers.
        candidate_centers_x = np.array(self.center_x)
        idx = np.random.randint(candidate_centers_x.shape[0], size=size)
        centers_x = candidate_centers_x[idx][..., None]

        centers = np.zeros(shape=(size, self.dim))
        centers[:, 0:1] = centers_x

        # Move to the new centers.
        samples = centers + samples

        # Add empty dimensions. This makes it possible to code up cones embedded in higher dimensions.
        if self.empty_dim != 0:
            samples = np.concatenate((samples, np.zeros(shape=(size, self.empty_dim))), axis=1)

        # Reshape the samples into 2d grids if requested. Useful for testing convolutional neural net implementations.
        if self.reshape_to_grid:
            assert self.reshape_to_grid[0] * self.reshape_to_grid[1] == samples.shape[1]

            new_shape = (samples.shape[0], 1, self.reshape_to_grid[0], self.reshape_to_grid[1])
            samples = samples.reshape(new_shape)

        return samples
