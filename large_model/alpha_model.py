import torch
import torch.nn as nn

from   large_model import encoders, decoders


class AlphaModel(nn.Module):
    def __init__(self):

        super(AlphaModel, self).__init__()
        self.encoder = encoders.resnet_gca_encoder_29()
        self.decoder = decoders.res_gca_decoder_22()

    def forward(self, image, trimap):
        inp = torch.cat((image, trimap), dim=1)
        embedding, mid_fea = self.encoder(inp)
        alpha, info_dict = self.decoder(embedding, mid_fea)

        return alpha, info_dict
