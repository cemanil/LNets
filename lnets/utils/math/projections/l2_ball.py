import numpy as np
import torch

from lnets.utils.misc import to_cuda


def bjorck_orthonormalize(w, beta=0.5, iters=20, order=1):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """
    # TODO: Make sure the higher order terms can be implemented more efficiently.
    if order == 1:
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w = (1 + beta) * w - beta * w.mm(w_t_w)

    elif order == 2:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w = (+ (15 / 8) * w
                 - (5 / 4) * w.mm(w_t_w)
                 + (3 / 8) * w.mm(w_t_w_w_t_w))

    elif order == 3:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w)

            w = (+ (35 / 16) * w
                 - (35 / 16) * w.mm(w_t_w)
                 + (21 / 16) * w.mm(w_t_w_w_t_w)
                 - (5 / 16) * w.mm(w_t_w_w_t_w_w_t_w))

    elif order == 4:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)

        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w)
            w_t_w_w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w_w_t_w)

            w = (+ (315 / 128) * w
                 - (105 / 32) * w.mm(w_t_w)
                 + (189 / 64) * w.mm(w_t_w_w_t_w)
                 - (45 / 32) * w.mm(w_t_w_w_t_w_w_t_w)
                 + (35 / 128) * w.mm(w_t_w_w_t_w_w_t_w_w_t_w))

    else:
        print("The requested order for orthonormalization is not supported. ")
        exit(-1)

    return w


def get_safe_bjorck_scaling(weight, cuda=True):
    bjorck_scaling = torch.tensor([np.sqrt(weight.shape[0] * weight.shape[1])]).float()
    bjorck_scaling = to_cuda(bjorck_scaling, cuda=cuda)

    return bjorck_scaling


def project_on_l2_ball(weight, bjorck_iter, bjorck_order, bjorck_beta=0.5, cuda=True):
    with torch.no_grad():
        # Run Bjorck orthonormalization procedure to project the matrices on the orthonormal matrices manifold.
        ortho_weights = bjorck_orthonormalize(weight.t(),
                                              beta=bjorck_beta,
                                              iters=bjorck_iter,
                                              order=bjorck_order).t()

        return ortho_weights
