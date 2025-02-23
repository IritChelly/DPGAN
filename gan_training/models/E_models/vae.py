import torch.nn as nn
import torch
from gan_training.models.vae_models.ae_models.decoders_encoders import *
from gan_training.models.vae_models.ae_models.densenet import *
from gan_training.models.vae_models.ae_models.vgg import VGGM
from gan_training.models.vae_models.ae_models.wide_resnet import WideResNet
from gan_training.models.vae_models.ae_models.resnet_encoders import resnet18_decoder, resnet18_encoder, resnet50_decoder, resnet50_encoder



class VAE(nn.Module):
    def __init__(self, vae_args, vae_input_dim, nc, img_sz):

        super(VAE, self).__init__()

        vae_type = vae_args['vae_type']
        first_conv = vae_args['resnet']['first_conv']
        maxpool1 = vae_args['resnet']['maxpool1']
        hidden_dims = vae_args['fc']['hidden_dims']
        self.enc_output_dim = vae_args['encoder_output_dim']  # the dimension of the layer before the layer that computes mu and log_var
        self.latent_dim = vae_args['latent_dim']  # the dimension of z, mu and log_var
        self.nc = nc
        self.img_sz = img_sz
        self.c = 0

        vae_options = {
            'resnet18': {'enc': resnet18_encoder, 'dec': resnet18_decoder},
            'resnet50': {'enc': resnet50_encoder, 'dec': resnet50_decoder},
            'mnist-fc':{'enc':MnistEncoder, 'dec':MnistDecoder},
            'fc':{'enc':FCEncoder, 'dec':FCDecoder},
            'conv':{'enc':ConvEncoder, 'dec':ConvDecoder},
            'cifar-conv':{'enc':ConvEncoder, 'dec':ConvDecoder2}
            #'densenet':{'enc':DenseNet3,'dec':DenseNet3},
            #'vgg':{'enc':VGGM,'dec':VGGM},
            #'wresnet':{'enc':WideResNet,'dec':WideResNet}
        }

        # Define encoder and decoder:
        if vae_type == 'resnet18' or vae_type == 'resnet50':
            self.encoder = vae_options[vae_type]['enc'](self.nc, self.enc_output_dim, first_conv, maxpool1)
            self.decoder = vae_options[vae_type]['dec'](self.latent_dim, self.img_sz, self.nc, first_conv, maxpool1)
        else:
            self.encoder = vae_options[vae_type]['enc'](self.nc, self.img_sz, self.enc_output_dim, hidden_dims)
            self.decoder = vae_options[vae_type]['dec'](self.nc, self.img_sz, self.latent_dim, self.enc_output_dim, hidden_dims)

        # Define the layer that computes mu and log var:
        self.fc_mu = nn.Linear(self.enc_output_dim , self.latent_dim)
        self.fc_var = nn.Linear(self.enc_output_dim , self.latent_dim)


    def sample(self, mu, log_var):
        var = torch.exp(log_var / 2)
        
        # # DEBUG
        # print('--var--:')
        # print(torch.max(log_var), torch.min(log_var), torch.mean(log_var), torch.max(var), torch.min(var), torch.mean(var))
        # # END DEBUG

        # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, var)
        z = q.rsample()
        return z

    def forward(self, X):
        # X shape: (batch_size, 3, img_sz, img_sz)
        # mu, log_var and z are of the same shape: VAE_latent_dim

        # Encoder + sampling z:
        emb = self.encoder(X)          # (batch_sz, enc_output_dim)
        mu = self.fc_mu(emb)           # (batch_sz, latent_dim)
        log_var = self.fc_var(emb)     # (batch_sz, latent_dim)
        z = self.sample(mu, log_var)   # (batch_sz, latent_dim)
        latent_ftrs = z.clone()        # (batch_sz, latent_dim)

        # Decoder:
        X_rec = self.decoder(z)        # (batch_sz, nc, img_sz, img_sz)

        return latent_ftrs, X_rec, mu, log_var
