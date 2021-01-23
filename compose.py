import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from model import Model
from data_human import HADataset
from loss import LossFunction


def train_dataloader(self):
    return DataLoader(HADataset('train'), batch_size=8, shuffle=True, pin_memory=True, num_workers=8)


def val_dataloader(self):
    return DataLoader(HADataset('valid'), batch_size=1, shuffle=False, pin_memory=True, num_workers=4)


def training_step(self, batch, batch_idx):
    _, img, alpha_label, trimap_label, _ = batch
    # img = img.type(torch.FloatTensor, device=self.device)
    # alpha_label = alpha_label.type(torch.FloatTensor, device=self.device)

    trimap_out, alpha_out = self(img)

    # Calculate loss
    if self.stage == 'train_trimap':
        loss = self.criterion(trimap_out, trimap_label)
    elif self.stage == 'train_alpha':
        loss = self.criterion(img, alpha_out, alpha_label, trimap_out, trimap_label)

    self.log('train_loss', loss)

    return loss


def validation_step(self, batch, batch_idx):
    _, img, alpha_label, trimap_label, _ = batch
    # img = img
    # alpha_label = alpha_label

    trimap_out, alpha_out = self(img)

    # Calculate loss
    if self.stage == 'train_trimap':
        loss = self.val_criterion(trimap_out, trimap_label)
    elif self.stage == 'train_alpha':
        loss = self.val_criterion(img, alpha_out, alpha_label, trimap_out, trimap_label)

    self.log('val_loss', loss)

    return loss


def build_loss(self):
    criterion = LossFunction(self.stage)('focal')
    setattr(self, 'criterion', criterion)
    val_criterion = LossFunction(self.stage)('ce')
    setattr(self, 'val_criterion', val_criterion)


def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.parameters(
    ), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: 0.95)
    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
    #     optimizer, lr_lambda=lambda epoch: 0.95 if epoch % 20 == 0 and epoch > 0 else 1)
    return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def compose(stage, attention):
    setattr(Model, 'train_dataloader', train_dataloader)
    setattr(Model, 'val_dataloader', val_dataloader)
    # setattr(Model, 'test_dataloader', test_dataloader)
    setattr(Model, 'build_loss', build_loss)
    setattr(Model, 'configure_optimizers', configure_optimizers)
    setattr(Model, 'training_step', training_step)
    setattr(Model, 'validation_step', validation_step)

    model = Model(stage, attention=attention)
    model.build_loss()
    return model
