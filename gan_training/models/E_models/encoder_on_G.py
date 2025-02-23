from gan_training.models.E_models.augmentations import Augment
from gan_training.models.E_models.encoder import Encoder
from torch import nn


class E_on_G(nn.Module):
    def __init__(self,
                 device,
                 encoder_args,
                 nc,
                 dataset_name,
                 size=32):
        super(E_on_G, self).__init__()

        self.img_sz = size   # in mnist img_sz = 32
        self.nc = nc
        self.device = device

        self.augment = Augment((self.img_sz, self.img_sz), nc, device, dataset_name)

        self.encoder = Encoder(encoder_args, nc, self.img_sz).to(self.device)

    def forward(self, x, get_features=False, is_augment=True):
        # x: (batch_size, 3, img_sz, img_sz)

        if get_features or not is_augment:  # Returns the latent features from the encoder:
            x_emb = self.encoder(x)
            return x_emb

        # Perform augmenation on input, we will get x_aug1, x_aug2, each in the shape of (batch_size, 3, img_sz, img_sz):
        x_aug1, x_aug2 = self.augment(x)   # each output is in the shape: (batch_size, 3, img_sz, img_sz)

        # Feed augmented images through encoder separately:
        x_emb_1 = self.encoder(x_aug1)  # x_emb_1 in shape (batch_size, encoder_latent_dim)
        x_emb_2 = self.encoder(x_aug2)
  
        return x_emb_1, x_emb_2
