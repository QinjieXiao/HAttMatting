import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import pytorch_lightning as pl

from .downpath import DownPath
from .conv_batchnorm_relu import ConvBatchNormRelu
from .uppath import UpPath

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(
            brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class TrimapModel(pl.LightningModule):
    def __init__(self):
        super(TrimapModel, self).__init__()

        self.down1 = DownPath(2, 3, 64, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down2 = DownPath(2, 64, 128, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down3 = DownPath(3, 128, 256, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down4 = DownPath(3, 256, 512, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down5 = DownPath(3, 512, 512, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down = ConvBatchNormRelu(
            512, 512, kernel_size=3, padding=1, bias=True)

        self.trimap_5 = UpPath(512, 512, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.trimap_4 = UpPath(512, 512, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.trimap_3 = UpPath(512, 256, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.trimap_2 = UpPath(256, 128, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.trimap_1 = UpPath(128, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.trimap_conv1 = ConvBatchNormRelu(
            64, 64, kernel_size=5, padding=2, bias=True)
        self.trimap_conv2 = ConvBatchNormRelu(
            64, 3, kernel_size=5, padding=2, bias=True)

        self.refine_1 = ConvBatchNormRelu(
            7, 64, kernel_size=3, padding=1, bias=True)
        self.refine_2 = ConvBatchNormRelu(
            64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_3 = ConvBatchNormRelu(
            64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_pred = ConvBatchNormRelu(
            64, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        l1, s1, x1, i1 = self.down1(x)
        l2, s2, x2, i2 = self.down2(x1)
        l3, s3, x3, i3 = self.down3(x2)
        l4, s4, x4, i4 = self.down4(x3)
        l5, s5, x5, i5 = self.down5(x4)
        x6 = self.down(x5)

        t5, x6t = self.trimap_5(x6, x5, i5, s5, return_conv_result=True)
        t5 = torch.add(t5, l5)
        t4 = torch.add(self.trimap_4(t5, x4, i4, s4), l4)
        t3 = torch.add(self.trimap_3(t4, x3, i3, s3), l3)
        t2 = torch.add(self.trimap_2(t3, x2, i2, s2), l2)
        t1 = torch.add(self.trimap_1(t2, x1, i1, s1), l1)
        raw_trimap = self.trimap_conv2(self.trimap_conv1(t1))
        print(raw_trimap.shape)
        raw_trimap = raw_trimap.argmax(dim=1, keepdim=True)

        return raw_trimap

    def transform(self, x):
        x = data_transforms['valid'](x)
        return x.unsqueeze(0)

    def migrate(self, state_dict):
        with torch.no_grad():
            for i, (name, p) in enumerate(self.state_dict().items()):
                # if name in state_dict:
                # if p.data.shape == state_dict[name].shape:
                print(i, name)
                p.copy_(state_dict[name])


if __name__ == "__main__":

    model = Model()
    trainer = pl.Trainer(weighs_summary='full')
    # summary(model, (3, 800, 600))
