import torch
import torch.nn as nn

def gmm_kl_distance(pred_mu, pred_sigma, mus, sigma):
    """Pytorch implementation of the Kullback–Leibler divergence.
    :param: pred_mu, extracted attribute vector with shape [N, d]
    :param: mus, mean tensor with shape [N, d]
    """
    #sigma = torch.tensor(0.25).to(device) if sigma is None else sigma
    return (0.5 * (torch.log(sigma/pred_sigma) + (pred_sigma + (pred_mu - mus)**2)/sigma - 1.0)).sum(dim=1).mean()


def gmm_kl_distance_sp(pred_mus, pred_sigma, mus, sigma):
    """Pytorch implementation of the Kullback–Leibler divergence.
      :param: pred_mu, extracted attribute vector with shape [N, d]
      :param: mus, mean tensor with shape [N, d]
    """
    #sigma = torch.tensor(0.25).to(device) if sigma is None else sigma
    kl_loss = 0.0
    for i, pred_mu in enumerate(pred_mus):
        kl_loss += (0.5 * (torch.log(sigma/pred_sigma[i].exp()) + (pred_sigma[i].exp() + (pred_mu - mus[:,i:i+1])**2)/sigma - 1.0)).sum(dim=1).mean()
    return kl_loss


def gmm_earth_mover_distance(pred_mus, mus):
    """Pytorch implementation of the Earth Mover.
    :param: pred, extracted attribute vector with shape [N, d*V]
    :param: mus, mean tensor with shape [N, d]
    """
    return torch.abs(pred_mus - mus).sum(dim=1).mean() 


def gmm_earth_mover_distance_sp(pred_mus, mus):
    """Pytorch implementation of the Earth Mover.
    :param: pred, extracted attribute vector with shape [N, d*V]
    :param: mus, mean tensor with shape [N, d]
    """
    em_loss = 0.0
    for i, pred_mu in enumerate(pred_mus):
        em_loss += torch.abs(pred_mu - mus[:,i:i+1]).sum(dim=1).mean() 
    return em_loss