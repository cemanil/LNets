"""
Spectral norm regularization works as follows:

1. Initialize a vector u randomly (which does not require gradients)
At each update:
    1. Apply one-step power iteration method by computing Jvp
    2. Compute the spectral norm (u^T)Jv (where u and Jv is computed in step 1)
    3. Use the spectralnorm as a regularization term requiring only one additional backward pass
"""

import torch.nn.functional as F
from torch.autograd import grad


def jac_spectral_norm(output, x, u):
    """
    Returns updated estimates of spectral norm and u

    (Might need to average over batch before-hand for
    correct stochastic computation)


    RETURNS: s, u
             spectral norm, leading singular vector
    """
    # First we compute the update for u.
    u = u.clone()
    u.requires_grad = True
    vjp = grad(output.mean(0), x, u, create_graph=True)[0].view(x.size(0), -1)
    v = F.normalize(vjp).detach()

    # A new trick for calculating Jacobian vector products.
    # https://www.reddit.com/r/MachineLearning/comments/6gvv0o/r_first_blog_post_a_new_trick_for_calculating/
    jvp = grad(vjp, u, v, create_graph=True)[0]
    u = F.normalize(jvp, dim=0).detach()
    spectral_norm = u.dot(jvp)
    return spectral_norm, u
