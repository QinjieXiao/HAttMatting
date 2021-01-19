import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

import pytorch_lightning as pl

from .trimap_model import TrimapModel
from .alpha_model import AlphaModel

class Model(pl.LightningModule):
    def __init__(self, stage, lr=0.001, weight_decay=0, momentum=0.9):
        super(Model, self).__init__()
        self.stage = stage
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.trimap = TrimapModel()
        self.alpha = AlphaModel()
        if self.stage == 'train_trimap':
            self.freeze_alpha_path()
    
    def forward(self, x):
        trimap = self.trimap(x)
        trimap_argmax = trimap.argmax(dim=1, keepdim=True)
        trimap_argmax = F.one_hot(trimap_argmax.permute(0, 2, 3, 1), num_classes=3).squeeze(3).permute(0, 3, 1, 2)
        alpha, _ = self.alpha(x, trimap_argmax)
        return trimap, alpha
    
    def migrate(self, state_dict):
        for name, p in state_dict.items():
            if p.data.shape == state_dict[name].shape:
                p.data = state_dict[name]
    
    def freeze_alpha_path(self):
        for name, p in self.named_parameters():
            if 'alpha' in name:
                p.requires_grad = False