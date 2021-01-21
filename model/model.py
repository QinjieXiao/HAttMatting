import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import pytorch_lightning as pl

from .trimap_model import TrimapModel
from .alpha_model import AlphaModel
from .net import VGG16
from .T_Net import RD_FPNnet as TNet


class Model(pl.LightningModule):
    def __init__(self, stage, lr=0.001, weight_decay=0, momentum=0.9):
        super(Model, self).__init__()
        self.stage = stage
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.trimap = TrimapModel()
        self.alpha = VGG16()
        self.example_input_array = torch.zeros(1, 3, 800, 576)
        if self.stage == 'train_trimap':
            self.freeze_alpha_path()

    def forward(self, x):
        # trimap_input = self.trimap.transform(x)
        # alpha_input = self.alpha.transform(x)
        trimap = self.trimap(x)
        alpha = self.alpha(x, trimap)

        return trimap, alpha

    def migrate(self, state_dict):
        with torch.no_grad():
            for i, (name, p) in enumerate(self.state_dict().items()):
                if name in state_dict:
                    if p.data.shape == state_dict[name].shape:
                        print(i, name)
                        p.copy_(state_dict[name])

    def freeze_alpha_path(self):
        for name, p in self.named_parameters():
            if 'alpha' in name:
                p.requires_grad = False
