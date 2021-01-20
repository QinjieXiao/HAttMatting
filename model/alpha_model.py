import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from . import encoders
from . import decoders

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(
            brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class AlphaModel(nn.Module):
    def __init__(self):

        super(AlphaModel, self).__init__()
        self.encoder = encoders.resnet_gca_encoder_29()
        self.decoder = decoders.res_gca_decoder_22()

    def forward(self, image, trimap):
        trimap = trimap.argmax(dim=1, keepdim=True)
        trimap = F.one_hot(trimap.permute(0, 2, 3, 1),
                           num_classes=3).squeeze(3).permute(0, 3, 1, 2)
        inp = torch.cat((image, trimap), dim=1)
        embedding, mid_fea = self.encoder(inp)
        alpha, info_dict = self.decoder(embedding, mid_fea)

        return alpha

    def transform(self, x):
        x = data_transforms['valid'](x)
        return x.unsqueeze(0)

    def migrate(self, state_dict):
        with torch.no_grad():
            for name, p in self.state_dict().items():
                if name in state_dict:
                    if p.data.shape == state_dict[name].shape:
                        # print(name)
                        p.copy_(state_dict[name])
