'''
Author: BigCiLeng && bigcileng@outlook.com
Date: 2023-07-25 11:53:45
LastEditors: BigCiLeng && bigcileng@outlook.com
LastEditTime: 2023-08-01 01:15:11
FilePath: /SQN_pl/sqn_system.py
Description: 

Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
'''
# torch
from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torch.utils.data

# torch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# lib
import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# mine
from utils.helper_tool import ConfigS3DIS as cfg
from models.SQN import Network, compute_loss, compute_acc, IoUCalculator
from dataset.s3dis_dataset import S3DIS, S3DISSampler

def evaluate(model, points):
    model.eval()
    with torch.no_grad():
        scores = model(points)
    return scores

class SQN_System(pl.LightningModule):

    def __init__(self, hparams):
        super(SQN_System, self).__init__()
        self.save_hyperparameters()
        cfg.retrain = self.hparams['hparams'].retrain
        self.model = Network(cfg)
        self.iou_calc = IoUCalculator(cfg) # 初始化IOU计算器

        if self.hparams['hparams'].dataset_name == 'S3DIS' or self.hparams['hparams'].dataset_name == 'Semantic3D':
            self.loss_type = 'wce'  # sqrt, lovas
        else:
            self.loss_type = 'sqrt'  # wce, lovas
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, input, is_training):
        return self.model(input, is_training)

    def decode_batch(self, batch):
        return batch

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def setup(self, stage):
        if self.hparams['hparams'].work_type == 'train':
            if self.hparams['hparams'].dataset_name == 'S3DIS':
                dataset = S3DIS(5)
                self.training_dataset = S3DISSampler(dataset, 'training')
                self.validation_dataset = S3DISSampler(dataset, 'validation')
        elif self.hparams['hparams'].work_type == 'test':
            if self.hparams['haparams'].dataset_name == 'S3DIS':
                dataset = S3DIS(5)
                self.test_dataset = S3DISSampler(dataset, 'validation')
        else:
            raise Exception('There are no work_type in hparams')
    def train_dataloader(self):
        return DataLoader(self.training_dataset, 
                          batch_size=cfg.batch_size,
                          shuffle=True, 
                          collate_fn=self.training_dataset.collate_fn,
                          num_workers=self.hparams['hparams'].num_workers,
                          pin_memory=True
                          )

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, 
                          batch_size=cfg.batch_size, 
                          shuffle=True, 
                          collate_fn=self.validation_dataset.collate_fn,
                          num_workers=self.hparams['hparams'].num_workers,
                          pin_memory=True
                          )
    
    def test_dataloader(self):
        return  DataLoader(self.test_dataset, 
                           batch_size=cfg.batch_size, 
                           shuffle=True, 
                           collate_fn=self.test_dataset.collate_fn,
                           num_workers=self.hparams['hparams'].num_workers,
                           pin_memory=True
                           )


    def configure_optimizers(self):
        # Load the Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['hparams'].adam_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.hparams['hparams'].scheduler_gamma)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):    
        # Forward pass
        input = self.decode_batch(batch)
        end_points = self(batch, is_training=True)
        end_points['batch_anno_labels'] = torch.cat([input['batch_anno_labels'], input['batch_anno_labels']], dim=0)
        loss, end_points = compute_loss(end_points, cfg, self.loss_type)

        acc, end_points = compute_acc(end_points)
        self.iou_calc.add_data(end_points)               # 保存训练结果，用于计算iou

        self.log('train/loss', loss.cpu().detach().item(), sync_dist=True)
        self.log('train/acc', acc.cpu().detach().item(), sync_dist=True)
        preds = {'loss': loss.cpu().detach(), 'acc':acc.cpu().detach()}
        self.training_step_outputs.append(preds)
        return loss

    def on_training_epoch_end(self):
        outputs = self.training_step_outputs
        loss = torch.stack([x['loss'] for x in outputs])
        acc = torch.stack([x['acc'] for x in outputs])

        mean_loss = torch.mean(loss)
        mean_accuracy = torch.mean(acc)
        mean_iou, iou_list = self.iou_calc.compute_iou()
        self.log('train/mean_loss', mean_loss.cpu().detach().item(), sync_dist=True)
        self.log('train/mean_acc', mean_accuracy.cpu().detach().item(), sync_dist=True)
        self.log('train/mean_iou', mean_iou.item(), sync_dist=True)
        
        # clear
        self.iou_calc.clear() # 初始化IOU计算器
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            end_points = self(batch, is_training=False)

        loss, end_points = compute_loss(end_points, cfg, self.loss_type)

        acc, end_points = compute_acc(end_points)
        self.iou_calc.add_data(end_points)               # 保存训练结果，用于计算iou

        self.log('val/loss', loss.cpu().detach().item(), sync_dist=True)
        self.log('val/acc', acc.cpu().detach().item(), sync_dist=True)
        preds = {'loss': loss.cpu().detach(), 'acc':acc.cpu().detach()}
        self.validation_step_outputs.append(preds)
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = torch.stack([x['loss'] for x in outputs])
        acc = torch.stack([x['acc'] for x in outputs])

        mean_loss = torch.mean(loss)
        mean_accuracy = torch.mean(acc)
        mean_iou, iou_list = self.iou_calc.compute_iou()
        self.log('val/mean_loss', mean_loss.cpu().detach().item(), sync_dist=True)
        self.log('val/mean_acc', mean_accuracy.cpu().detach().item(), sync_dist=True)
        self.log('val/mean_iou', mean_iou.item(), sync_dist=True)
        
        # clear
        self.iou_calc.clear() # 初始化IOU计算器
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_nb):
        points, labels = self.decode_batch(batch)
        with torch.no_grad():
            scores = self(points)
        predictions = torch.max(scores, dim=-2).indices
        accuracy = (predictions == labels).float().mean()
        print('Accuracy:', accuracy.item())
        predictions = predictions.numpy()
        cloud = points.squeeze(0)[:,:3]
        # write_ply('MiniDijon9.ply', [cloud, predictions], ['x', 'y', 'z', 'class'])


def train(args):
    if args.load == '':
        checkpoint = None
    else:
        checkpoint = ModelCheckpoint(dirpath=os.path.join(os.path.abspath(f'./ckpts/{args.name}')),
                                    filename='best',
                                    monitor='train/mean_iou',
                                    mode='max',
                                    save_top_k=5,
                                    )
        
    wandb_logger = WandbLogger(project="SQN", name=args.name)
    system = SQN_System(hparams=args)
    trainer = pl.Trainer(
                         logger=wandb_logger,
                         max_epochs=cfg.max_epoch,
                         callbacks=[checkpoint] if checkpoint is not None else None,
                         accelerator=args.device,
                         devices=args.device_nums,
                         strategy=DDPStrategy(),
                         check_val_every_n_epoch=5,
                         num_sanity_val_steps=1,
                         benchmark=True,
                         log_every_n_steps=1,
                         )
    trainer.fit(system)

def test(args):
    system = SQN_System.load_from_checkpoint(args.load)
    trainer = pl.Trainer()
    trainer.test(system)

if __name__ == '__main__':

    """Parse program arguments"""
    parser = argparse.ArgumentParser(
        prog='SQN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    base = parser.add_argument_group('Base options')
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')
    base.add_argument('--dataset_name', type=str, help='name of the dataset',
                        default='S3DIS')
    
    base.add_argument('--work_type', type=str, help='train, val, test', default='train')

    expr.add_argument('--load', type=str, help='model to load',
                        default='')
    expr.add_argument('--retrain', type=bool, help='retrain the model with predicted pseudo labels',
                        default=False)

    param.add_argument('--adam_lr', type=float, help='learning rate of the optimizer',
                        default=1e-2)

    param.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler',
                        default=0.95)

    misc.add_argument('--device', type=str, help='cpu/gpu',
                        default='gpu')
    misc.add_argument('--device_nums', type=int, help='nums of device(gpu) to use',
                        default=1)
    misc.add_argument('--name', type=str, help='name of the experiment',
                        default=None)
    misc.add_argument('--num_workers', type=int, help='number of threads for loading data',
                        default=4)
    # misc.add_argument('--save_freq', type=int, help='frequency of saving checkpoints',
    #                     default=10)

    args = parser.parse_args()


    if args.name is None:
        if args.load:
            args.name = args.load
        else:
            args.name = datetime.now().strftime('%Y-%m-%d_%H:%M')
            args.name = args.name + '_' + args.dataset_name
    if args.work_type == 'train':
        train(args)
    elif args.work_type == 'test':
        test(args)
    
