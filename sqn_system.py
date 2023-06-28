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
from dataset.dataset import data_loaders
from models.model import SQN
from utils.tools import DataProcessing
from utils.metrics import accuracy, intersection_over_union
from utils.ply import read_ply, write_ply
# from tf_interpolate import three_nn, three_interpolate
def evaluate(model, points):
    model.eval()
    with torch.no_grad():
        scores = model(points)
    return scores

class SQN_System(pl.LightningModule):

    def __init__(self, hparams):
        super(SQN_System, self).__init__()
        self.save_hyperparameters()
        self.num_classes = hparams.num_classes

        if self.hparams['hparams'].dataset_name in ['S3DIS', 'Semantic3D']:
            self.loss_type = 'wce'
        else:
            self.loss_type = 'sqrt'
        self.ignored_labels = self.hparams['hparams'].ignored_labels
        self.weights = DataProcessing.get_class_weights(self.hparams['hparams'].dataset_name, self.loss_type)
        self.loss = self.get_loss

        self.training_step_outputs = []
        self.validation_step_outputs = []
    def forward(self, input):
        return self.model(input)

    def decode_batch(self, batch):
        points, queried_pc_labels, queried_idx, cloud_idx, xyz_with_anno, labels_with_anno = batch
        return {'points': points, 'queried_pc_labels': queried_pc_labels,
                'xyz_with_anno': xyz_with_anno, 'labels_with_anno': labels_with_anno}

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def setup(self, stage):
        if self.hparams['hparams'].work_type == 'train':
            self.train_loader, self.val_loader = data_loaders(
                Path(os.path.join(self.hparams['hparams'].dataset, self.hparams['hparams'].train_dir)),
                self.hparams['hparams'],
                self.hparams['hparams'].dataset_sampling,
                batch_size=self.hparams['hparams'].batch_size,
                num_workers=self.hparams['hparams'].num_workers,
                pin_memory=True
            )
            tem = next(iter(self.train_loader))
            d_in = tem[0].size(-1)
            self.model = SQN(
                d_in,
                self.num_classes,
                True,
                self.hparams['hparams'].num_points,
                num_neighbors=self.hparams['hparams'].neighbors,
                decimation=self.hparams['hparams'].decimation,
            )
        elif self.hparams['hparams'].work_type == 'test':
            print('Loading data...')
            self.test_loader, _ = data_loaders(
                Path(os.path.join(self.hparams['hparams'].dataset,self.hparams['hparams'].test_dir)),   
                self.hparams['hparams'],             
                self.hparams['hparams'].dataset_sampling,
                batch_size=self.hparams['hparams'].batch_size,
                num_workers=self.hparams['hparams'].num_workers,
                pin_memory=True
            )
            d_in = 6
            num_classes = self.hparams['hparams'].num_classes
            self.model = SQN(d_in, num_classes,False,self.hparams['hparams'].num_points, 16, 4)
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        # Load the Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['hparams'].adam_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.hparams['hparams'].scheduler_gamma)
        return [self.optimizer], [self.scheduler]
    
    def get_loss(self, logits, labels, pre_cal_weights, num_classes, loss_type='sqrt'):
        logits = logits.reshape(-1, self.num_classes)
        labels = labels.reshape(-1)
        # 创建一个布尔张量，表示哪些标签需要被忽略
        ignored_bool = torch.zeros_like(labels, dtype=torch.bool)
        ignored_bool = ignored_bool | (labels == self.ignored_labels)

        # 收集不被忽略的logits和labels
        valid_idx = torch.nonzero(~ignored_bool, as_tuple=True)
        valid_logits = logits[valid_idx]
        # valid_labels_init = labels[valid_idx]
        valid_labels = labels[valid_idx]

        # Reduce label values in the range of logit shape
        # reducing_list = torch.arange(self.num_classes, dtype=torch.int32)
        # inserted_value = torch.zeros(1, dtype=torch.int32)
        # reducing_list = torch.cat([reducing_list[:self.ignored_labels], inserted_value, reducing_list[self.ignored_labels:]], 0)
        # valid_labels = reducing_list[valid_labels_init]

        # 计算loss
        output_loss = DataProcessing.get_loss(valid_logits, valid_labels, self.weights, self.num_classes, self.loss_type)
        return output_loss
    def get_accuracy(self, logits, labels):
        logits = logits.reshape(-1, self.num_classes)
        labels = labels.reshape(-1)
        # 创建一个布尔张量，表示哪些标签需要被忽略
        ignored_bool = torch.zeros_like(labels, dtype=torch.bool)
        ignored_bool = ignored_bool | (labels == self.ignored_labels)

        # 收集不被忽略的logits和labels
        valid_idx = torch.nonzero(~ignored_bool, as_tuple=True)
        valid_logits = logits[valid_idx]
        # valid_labels_init = labels[valid_idx]
        valid_labels = labels[valid_idx]
        valid_logits = valid_logits.permute(1,0)
        acc = accuracy(valid_logits, valid_labels)
        return acc
    def get_intersection_over_union(self, logits, labels):
        logits = logits.reshape(-1, self.num_classes)
        labels = labels.reshape(-1)
        # 创建一个布尔张量，表示哪些标签需要被忽略
        ignored_bool = torch.zeros_like(labels, dtype=torch.bool)
        ignored_bool = ignored_bool | (labels == self.ignored_labels)

        # 收集不被忽略的logits和labels
        valid_idx = torch.nonzero(~ignored_bool, as_tuple=True)
        valid_logits = logits[valid_idx]
        # valid_labels_init = labels[valid_idx]
        valid_labels = labels[valid_idx]
        valid_logits = valid_logits.permute(1,0)
        iou = intersection_over_union(valid_logits, valid_labels)
        return iou
    def training_step(self, batch, batch_idx):
        input = self.decode_batch(batch)
        points = input['points']
        labels = input['queried_pc_labels']
        labels = torch.cat([labels, labels], dim=0)
        scores = self(input)

        logp = scores

        loss = self.loss(logp, labels, self.weights, self.num_classes, self.loss_type)
        acc = self.get_accuracy(logp, labels)
        iou = self.get_intersection_over_union(logp, labels)

        acc_mean = torch.tensor(acc[-1], dtype=torch.float32)
        iou_mean = torch.tensor(iou[-1], dtype=torch.float32)

        log= {'lr': self.get_lr(self.optimizer), 'train/loss': loss, 'train/accuracy': acc_mean, 'train/iou': iou_mean}

        preds = {'loss': loss, 'accuracy': acc_mean,'iou': iou_mean, 'log': log}

        self.log('train/loss', loss.clone().detach(), sync_dist=True)
        self.log('train/accuracy', acc_mean.clone().detach(), sync_dist=True)
        self.log('train/iou', iou_mean.clone().detach(), sync_dist=True)
        self.training_step_outputs.append(preds)

        return loss

    def on_training_epoch_end(self):
        outputs = self.training_step_outputs
        loss = torch.stack([x['loss'] for x in outputs])
        acc = torch.stack([x['accuracy'] for x in outputs])
        iou = torch.stack([x['iou'] for x in outputs])

        mean_loss = torch.mean(loss)
        mean_accuracy = torch.mean(acc)
        mean_iou = torch.mean(iou)
        self.log('mean_train_loss', mean_loss.clone().detach(), sync_dist=True)
        self.log('mean_train_accuracy', mean_accuracy.clone().detach(), sync_dist=True)
        self.log('mean_train_iou', mean_iou.clone().detach(), sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        input = self.decode_batch(batch)
        points = input['points']
        labels = input['queried_pc_labels']
        labels = torch.cat([labels, labels], dim=0)
        with torch.no_grad():
            scores = self(input)
        logp = scores

        loss = self.loss(logp, labels, self.weights, self.num_classes, self.loss_type)
        acc = self.get_accuracy(logp, labels)
        iou = self.get_intersection_over_union(logp, labels)

        acc_mean = torch.tensor(acc[-1], dtype=torch.float32)
        iou_mean = torch.tensor(iou[-1], dtype=torch.float32)

        log = {'val/loss': loss, 'val/accuracy': acc_mean, 'val/iou': iou_mean}
        pred = {'loss': loss, 'accuracy': acc_mean,'iou': iou_mean, 'log': log}
        self.validation_step_outputs.append(pred)

        return pred

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = torch.stack([x['loss'] for x in outputs])
        acc = torch.stack([x['accuracy'] for x in outputs])
        iou = torch.stack([x['iou'] for x in outputs])


        mean_loss = torch.mean(loss)
        mean_accuracy = torch.mean(acc)
        mean_iou = torch.mean(iou)

        self.log('mean_val_loss', mean_loss.clone().detach(), sync_dist=True)
        self.log('mean_val_accuracy', mean_accuracy.clone().detach(), sync_dist=True)
        self.log('mean_val_iou', mean_iou.clone().detach(), sync_dist=True)
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
        write_ply('MiniDijon9.ply', [cloud, predictions], ['x', 'y', 'z', 'class'])


def train(args):
    if args.load == '':
        checkpoint = None
    else:
        checkpoint = ModelCheckpoint(dirpath=os.path.join(os.path.abspath(f'./ckpts/{args.name}')),
                                    filename='best',
                                    monitor='mean_train_loss',
                                    mode='min',
                                    save_top_k=5,
                                    )
        
    wandb_logger = WandbLogger(project="SQN", name=args.name)
    system = SQN_System(hparams=args)
    trainer = pl.Trainer(
                         logger=wandb_logger,
                         max_epochs=args.epochs,
                         callbacks=[checkpoint] if checkpoint is not None else None,
                         accelerator=args.device,
                         devices=args.gpu,
                         strategy=DDPStrategy(find_unused_parameters=True),
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
        prog='RandLA-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    base = parser.add_argument_group('Base options')
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')
    base.add_argument('--dataset_name', type=str, help='name of the dataset',
                        default='S3DIS')
    base.add_argument('--dataset', type=str, help='location of the dataset',
                        default='/share/dataset/S3DIS')
    
    base.add_argument('--work_type', type=str, help='train, val, test', default='train')

    base.add_argument('--num_classes', type=int, help='nums of label type', default=13)

    expr.add_argument('--epochs', type=int, help='number of epochs',
                        default=50)
    expr.add_argument('--load', type=str, help='model to load',
                        default='')
    expr.add_argument('--num_points', type=int, help='Number of input points', default=40960)
    expr.add_argument('--sub_grid_size', type=float, help='preprocess_parameter', default=0.06)
    expr.add_argument('--train_steps', type=int, help='Number of steps per epochs', default=200)
    expr.add_argument('--val_steps', type=int, help='Number of validation steps per epoch', default=100)

    param.add_argument('--adam_lr', type=float, help='learning rate of the optimizer',
                        default=1e-2)
    param.add_argument('--batch_size', type=int, help='batch size',
                        default=5)
    param.add_argument('--decimation', type=int, help='ratio the point cloud is divided by at each layer',
                        default=4)
    param.add_argument('--dataset_sampling', type=str, help='how dataset is sampled',
                        default='active_learning', choices=['active_learning', 'random'])
    param.add_argument('--neighbors', type=int, help='number of neighbors considered by k-NN',
                        default=16)
    param.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler',
                        default=0.95)
    param.add_argument('--labeled_point', type=str, help='0.1%/1%/10%/100%',
                        default='0.1%')
    param.add_argument('--retrain', type=bool, help='',
                        default=1)
    param.add_argument('--ignored_labels', type=int, help='useless label', default=12)
    
    dirs.add_argument('--test_dir', type=str, help='location of the test set in the dataset dir',
                        default='test')
    dirs.add_argument('--train_dir', type=str, help='location of the training set in the dataset dir',
                        default='train')
    dirs.add_argument('--val_dir', type=str, help='location of the validation set in the dataset dir',
                        default='val')
    dirs.add_argument('--logs_dir', type=Path, help='path to tensorboard logs',
                        default='runs')
    misc.add_argument('--device', type=str, help='cpu/gpu',
                        default='gpu')
    misc.add_argument('--gpu', type=int, help='which GPU to use (-1 for CPU)',
                        default=2)
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
    

