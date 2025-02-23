import torch

from models import generator_dict, discriminator_dict
from torch import optim
import torch.utils.data as utils

import numpy as np


# k_value = the maximum value of K (in case K changes during the process)
def get_models(model_type, conditioning, k_value, d_act_dim, device):
    G = generator_dict[model_type]
    D = discriminator_dict[model_type]
    generator = G(conditioning, k_value=k_value, device=device)
    discriminator = D(conditioning, k_value=k_value, device=device, act_dim=d_act_dim)

    generator.to(device)
    discriminator.to(device)

    return generator, discriminator


def get_optimizers(generator, discriminator, lr=1e-4, beta1=0.8, beta2=0.999):
    g_optimizer = optim.Adam(generator.parameters(),
                             lr=lr,
                             betas=(beta1, beta2))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=lr,
                             betas=(beta1, beta2))
    return g_optimizer, d_optimizer


def get_test(get_data, batch_size, variance, k_value, device, ftrs_dim):
    x_test, y_test = get_data(batch_size, ftrs_dim, var=variance)
    x_test, y_test = torch.from_numpy(x_test).float().to(
        device), torch.from_numpy(y_test).float().to(device)
    return x_test, y_test


def get_dataset(get_data, batch_size, npts, variance, k_value, ftrs_dim):
    samples, labels = get_data(npts, ftrs_dim, var=variance)
    tensor_samples = torch.stack([torch.Tensor(x) for x in samples])
    tensor_labels = torch.stack([torch.tensor(x) for x in labels])
    dataset = utils.TensorDataset(tensor_samples, tensor_labels)
    train_loader = utils.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=None,
                                    drop_last=True)

    return train_loader

def sample_new_x_test_y_test(dataloader, test_batch_size, train_batch_size):
    # get x_test and y_test in the size of "test_batch_size", from train_loader:
    dataloader_iterator = iter(dataloader)
    x_test_all = []
    y_test_all = []
    for i in range(test_batch_size // train_batch_size):  
        x_test_batch, y_test_batch = next(dataloader_iterator)
        x_test_all.append(x_test_batch)
        y_test_all.append(y_test_batch)

    x_test = torch.cat(x_test_all, dim=0)
    y_test = torch.cat(y_test_all, dim=0)

    return x_test.cuda(), y_test.cuda()


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    for x_next, y_next in data_loader:
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
        if n > N:
            break
    x = torch.cat(x, dim=0)[:N]
    y = torch.cat(y, dim=0)[:N]
    return x, y