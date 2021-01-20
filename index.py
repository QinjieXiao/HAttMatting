import math
import cv2
from tqdm.auto import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from model import Model
from torchsummary import summary
from compose import compose
from utils import parse_args

global args
args = parse_args()


if __name__ == "__main__":
    pl.seed_everything(42)
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='fusion_logs',
        log_graph=True
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = compose(args.stage)
    trainer = pl.Trainer(
        # precision=16,
        gpus=0,
        benchmark=True,
        accumulate_grad_batches=1,
        progress_bar_refresh_rate=200,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        fast_dev_run=True
        #  resume_from_checkpoint
    )
    trainer.fit(model)
