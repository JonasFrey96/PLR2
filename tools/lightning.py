import copy
import datetime
import sys
import os
import time



# misc
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from math import pi
import coloredlogs
import datetime

# src modules
from src.helper import pad
from src.loaders import ConfigLoader

import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

import pytorch_lightning as pl
coloredlogs.install()

# network dense fusion
from lib.utils import setup_logger
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.network import PoseNet, PoseRefineNet

# dataset
from loaders_v2 import GenericDataset


class TrackNet6D(LightningModule):
    def __init__(self, , **kwargs):

        super().__init__()

        self.estimator = PoseNet(
          num_points=exp_cfg['d_train']['num_points'],
          num_obj=exp_cfg['d_train']['objects'],
          ff_cfg=exp_cfg['net']['ff_cfg'])

        self.refiner = PoseRefineNet(
          num_points=exp_cfg['d_train']['num_points'],
          num_obj=exp_cfg['d_train']['objects'])

        self.criterion = Loss(num_poi, self.sym_list)
        self.criterion_refine = Loss_refine(num_poi, self.sym_list)

        self.refine = False

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        # unpack the batch and apply forward pass
        x, y = batch
        y_hat = self(x)

        #chosse correct loss function initalized earlier
        loss = F.cross_entropy(y_hat, y)

        #choose correct loss here
        
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        dataset_train = GenericDataset(
            cfg_d=exp_cfg['d_train'],
            cfg_env=env_cfg)
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train,
                                                            batch_size=exp_cfg['loader']['batch_size'],
                                                            shuffle=exp_cfg['loader']['shuffle'],
                                                            num_workers=exp_cfg['loader']['workers'])
        return dataloader_train

    def test_dataloader(self):
        dataset_test = GenericDataset(
            cfg_d=exp_cfg['d_test'],
            cfg_env=env_cfg)
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_train,
                                                            batch_size=exp_cfg['loader']['batch_size'],
                                                            shuffle=exp_cfg['loader']['shuffle'],
                                                            num_workers=exp_cfg['loader']['workers'])
        return dataloader_test


if __name__ == "__main__":

        # logging.info("GC COUNT: %s"%str(gc.get_count()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=file_path, default='yaml/exp/exp_ws.yml',  # required=True,
                        help='The main experiment yaml file.')
    parser.add_argument('--env', type=file_path, default='yaml/env/env_jonas_natrix.yml',
                        help='The environment yaml file.')
    args = parser.parse_args()

    exp_cfg = ConfigLoader().from_file(exp_cfg_path)
    env_cfg = ConfigLoader().from_file(exp_cfg_path)
    # keep this in mind 
    """
    Trainer args (gpus, num_nodes, etc…) && Program arguments (data_path, cluster_email, etc…)
    Model specific arguments (layer_dim, num_layers, learning_rate, etc…)
    """
    
    # copy config files to model path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        logger.info((pad("Generating network run folder")))
    else:
        logger.warning((pad("Network run folder already exits")))

    logger.info(pad(f'Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}'))
    shutil.copy(exp_cfg_path, f'{model_path}/{exp_cfg_fn}')
    shutil.copy(env_cfg_path, f'{model_path}/{env_cfg_fn}')

    model = TrackNet6D()

    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=1, num_nodes=1)
    trainer.fit(model, fast_dev_run=True)
