import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, encoder_args, nc, img_sz):

        super(Encoder, self).__init__()
        encoder_type = encoder_args['encoder_type']  # options: {'resnet18', 'resnet50'}
        self.enc_output_dim = encoder_args['encoder_output_dim']  # the dimension of the layer before the layer that computes mu and log_var
        self.nc = nc
        self.img_sz = img_sz
        self.c = 0

        self.model_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=self.enc_output_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=self.enc_output_dim)}

        self.backbone = self._get_basemodel(encoder_type)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        # model_name: {'resnet18', 'resnet50'}

        model = self.model_dict[model_name]
        return model


    def forward(self, x):
        # x shape: (batch_size, 3, img_sz, img_sz)

        x_emb = self.backbone(x)  # (batch_sz, enc_output_dim)
        return x_emb

