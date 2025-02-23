from gan_training.models import (dcgan_deep, dcgan_shallow, resnet2)
from gan_training.models.E_models import (encoder_on_G)

generator_dict = {
    'resnet2': resnet2.Generator,
    'dcgan_deep': dcgan_deep.Generator,
    'dcgan_shallow': dcgan_shallow.Generator
}

discriminator_dict = {
    'resnet2': resnet2.Discriminator,
    'dcgan_deep': dcgan_deep.Discriminator,
    'dcgan_shallow': dcgan_shallow.Discriminator
}

encoder_dict = {
    #'resnet2': resnet2.Autoencoder,
    #'dcgan_deep': dcgan_deep.Autoencoder,
    'E_on_G': encoder_on_G.E_on_G
}
