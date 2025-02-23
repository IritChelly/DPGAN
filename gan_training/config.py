import yaml
from torch import optim
from os import path
from gan_training.models import generator_dict, discriminator_dict, encoder_dict
from gan_training.train import toggle_grad
from clusterers import clusterer_dict


# General config
def load_config(path, default_path):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        # Add item if not yet in dict1
        if k not in dict1:
            dict1[k] = None
        # Update
        if isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def get_clusterer(config):
    return clusterer_dict[config['clusterer']['name']]


def build_models(config, device):
    # Get classes
    Generator = generator_dict[config['generator']['name']]
    Discriminator = discriminator_dict[config['discriminator']['name']]
    E_on_G = encoder_dict[config['encoder']['name']]

    # Build models
    generator = Generator(z_dim=config['z_dist']['dim'],
                          nlabels=config['generator']['nlabels'],
                          device=device,
                          size=config['data']['img_size'],
                          conditioning=config['generator']['conditioning'],
                          nc=config['generator']['nc'],
                          **config['generator']['kwargs'])
    discriminator = Discriminator(
        nlabels=config['discriminator']['nlabels'],
        conditioning=config['discriminator']['conditioning'],
        device=device,
        size=config['data']['img_size'],
        nc=config['discriminator']['nc'],
        **config['discriminator']['kwargs'])
    encoder = E_on_G(device=device,
                          encoder_args=config['encoder_args'],
                          nc=config['generator']['nc'],
                          dataset_name=config['data']['type'],
                          size=config['data']['img_size'])

    return generator, discriminator, encoder


def build_optimizers(generator, discriminator, encoder, config):
    optimizer = config['training']['optimizer']
    lr_g = config['training']['lr_g']
    lr_encoder = config['training']['lr_e']
    lr_d = config['training']['lr_d']
    
    toggle_grad(generator, True)
    toggle_grad(discriminator, True)
    toggle_grad(encoder, True)

    g_params = generator.parameters()
    encoder_params = encoder.parameters()
    d_params = discriminator.parameters()

    if optimizer == 'rmsprop':
        g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
        d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)
        encoder_optimizer = optim.RMSprop(encoder_params, lr=lr_encoder, alpha=0.99, eps=1e-8)
    elif optimizer == 'adam':
        beta1 = config['training']['beta1']
        beta2 = config['training']['beta2']
        g_optimizer = optim.Adam(g_params, lr=lr_g, betas=(beta1, beta2), eps=1e-8)
        d_optimizer = optim.Adam(d_params, lr=lr_d, betas=(beta1, beta2), eps=1e-8)
        encoder_optimizer = optim.Adam(encoder_params, lr=lr_encoder, weight_decay=1e-4)
    elif optimizer == 'sgd':
        g_optimizer = optim.SGD(g_params, lr=lr_g, momentum=0.)
        d_optimizer = optim.SGD(d_params, lr=lr_d, momentum=0.)
        encoder_optimizer = optim.SGD(encoder_params, lr=lr_encoder, momentum=0.)

    return g_optimizer, d_optimizer, encoder_optimizer


# Some utility functions
def get_parameter_groups(parameters, gradient_scales, base_lr):
    param_groups = []
    for p in parameters:
        c = gradient_scales.get(p, 1.)
        param_groups.append({'params': [p], 'lr': c * base_lr})
    return param_groups
