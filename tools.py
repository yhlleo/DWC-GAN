import torch
import numpy as np
import torch.distributions as tdist
from torch.distributions.multivariate_normal import MultivariateNormal

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None, device=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = 1 - c_trg[:, i] # Reverse attribute value.
        elif dataset == 'RaFD':
            c_trg = label2onehot(torch.ones(c_org.size(0))*i, c_dim)

        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def asign_label(label, c_dim=None, mode='CelebA', normalize=True):
    if mode in ['CelebA', 'CUB200', 'Clevr']:
        asigned_label = label.clone()
    else: # mode in ['RaFD', 'A2B']
        asigned_label = label2onehot(label, c_dim)
    if normalize:
        asigned_label = asigned_label*2.0-1.0
    return asigned_label

def distribution_sampling(mu, v_dim, stddev=0.5, device=None):
    # standard deviation of the distribution (often referred to as sigma)
    stddev = torch.ones(mu.size()).to(device) * stddev
    norm = tdist.Normal(mu, stddev) 
    sampling = norm.sample((1, v_dim))
    z_random = sampling.transpose(2,1).transpose(3,2).contiguous().view(mu.size(0),-1)
    return z_random
    """
    z_random = []
    cov = torch.eye(mu.size(1)).to(device) * stddev**2
    for i in range(mu.size(0)):
        m = MultivariateNormal(mu[[i]], cov)
        z_random.append(m.sample())
    return torch.cat(z_random,dim=0)
    """

def dist_sampling_split(mu, c_dim=8, stddev=0.5, device=None):
    cov = torch.ones(mu.size()).to(device) * stddev
    norm = tdist.Normal(mu, cov)
    sampling = norm.sample((1, c_dim))
    z_random = sampling.transpose(2,1).transpose(3,2).contiguous().view(mu.size(0),-1)
    return z_random
    """
    z_random = []
    cov = torch.eye(mu.size(1)).to(device) * stddev**2
    for i in range(mu.size(0)):
        m = MultivariateNormal(mu[[i]], cov)
        z = m.sample((1,c_dim)).transpose(2,1).transpose(3,2).contiguous().view(1, -1)
        z_random.append(z)
    return torch.cat(z_random,dim=0)
    """
