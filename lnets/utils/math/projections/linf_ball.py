import torch


def get_linf_projection_threshold(weight, cuda):
    with torch.no_grad():
        if not cuda:
            # Sort the weights.
            sorted_weights, _ = torch.abs(weight).sort(dim=1, descending=True)
            sorted_weights.float()

            # Find the threshold as described in Algorithm 1, Laurent Condat, Fast Projection onto the Simplex
            # and the L1 Ball.
            partial_sums = torch.cumsum(sorted_weights, dim=1)
            indices = torch.arange(end=partial_sums.shape[1]).float()
            candidate_ks = (partial_sums < torch.tensor(1).float() +
                            (indices + torch.tensor(1).float()) * sorted_weights)
            candidate_ks = (candidate_ks.float() +
                            (1.0 / (2 * partial_sums.shape[1])) * (indices + torch.tensor(1).float()).float())
            _, ks = torch.max(candidate_ks.float(), dim=1)
            ks = ks.float()
            index_ks = torch.cat((torch.arange(end=weight.shape[0]).unsqueeze(-1).float(),
                                  ks.unsqueeze(1)), dim=1).long()

            thresholds = (partial_sums[index_ks[:, 0], index_ks[:, 1]] - torch.tensor(1).float()) / (
                    ks + torch.tensor(1).float())

        else:
            # Sort the weights.
            sorted_weights, _ = torch.abs(weight).sort(dim=1, descending=True)

            # Find the threshold as described in Algorithm 1, Laurent Condat, Fast Projection onto the Simplex
            # and the L1 Ball.
            partial_sums = torch.cumsum(sorted_weights, dim=1)
            indices = torch.arange(end=partial_sums.shape[1]).float().cuda()
            candidate_ks = (partial_sums < torch.tensor(1).float().cuda() +
                            (indices + torch.tensor(1).float().cuda()) * sorted_weights)
            candidate_ks = (candidate_ks.float().cuda() +
                            (1.0 / (2 * partial_sums.shape[1])) * (indices +
                                                                   torch.tensor(1).float().cuda()).float())
            _, ks = torch.max(candidate_ks.float(), dim=1)
            ks = ks.float().cuda()
            index_ks = torch.cat((torch.arange(end=weight.shape[0]).unsqueeze(-1).float().cuda(),
                                  ks.unsqueeze(1)), dim=1).long()

            thresholds = (partial_sums[index_ks[:, 0], index_ks[:, 1]] - torch.tensor(1).float().cuda()) / (
                    ks + torch.tensor(1).float().cuda())
    return thresholds


def get_weight_signs(weight):
    with torch.no_grad():
        return torch.sign(weight)


def project_on_linf_ball(weight, cuda):
    with torch.no_grad():
        thresholds = get_linf_projection_threshold(weight, cuda)
        signs = get_weight_signs(weight)
        signs[signs == 0] = 1
        projected_weights = signs * torch.clamp(torch.abs(weight) - thresholds.unsqueeze(-1),
                                                min=torch.tensor(0).float())

        return projected_weights


def get_l_inf_row_normalization_factors(weight, scale_all=True, cuda=False):
    with torch.no_grad():
        row_sums = torch.sum(torch.abs(weight), dim=1)

        if not scale_all:
            if cuda:
                clipped_row_sums = torch.max(torch.tensor(1).cuda().float(), row_sums)
            else:
                clipped_row_sums = torch.max(torch.tensor(1).float(), row_sums)

            return clipped_row_sums

    return row_sums


def scale_on_linf_ball(weight, scale_all=True, cuda=False):
    with torch.no_grad():
        row_scaling_factors = get_l_inf_row_normalization_factors(weight, scale_all, cuda)
        l_inf_weight = weight / row_scaling_factors.unsqueeze(-1)

        return l_inf_weight
