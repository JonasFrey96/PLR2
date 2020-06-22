import warnings
warnings.simplefilter("ignore", UserWarning)

import copy
import datetime
import sys
import os
import time
import shutil
import argparse
import logging

# misc
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from math import pi
import coloredlogs
import datetime

sys.path.insert(0, os.getcwd())
print(os.path.join(os.getcwd()+'/src'))
sys.path.append( os.path.join(os.getcwd()+'/src'))
sys.path.append( os.path.join(os.getcwd()+'/lib'))
# src modules
from helper import pad
from loaders_v2 import ConfigLoader

import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping
from pytorch_lightning.callbacks import ModelCheckpoint

coloredlogs.install()

# network dense fusion
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.network import PoseNet, PoseRefineNet

# dataset
from loaders_v2 import GenericDataset

class TrackNet6D(LightningModule):
    def __init__(self, exp, env):
        super().__init__()
        self.hparams = {'exp': exp, 'env': env}

        self.env = env
        self.exp = exp
  
        self.estimator = PoseNet(
           num_points=exp['d_train']['num_points'],
           num_obj=exp['d_train']['objects'] )

        # self.refiner = PoseRefineNet(
        #   num_points=exp['d_train']['num_points'],
        #   num_obj=exp['d_train']['objects'])

        num_poi = exp['d_train']['num_pt_mesh_small']
        self.criterion = Loss(num_poi, exp['d_train']['obj_list_sym'])
        num_poi = exp['d_train']['num_pt_mesh_large']
        self.criterion_refine = Loss_refine(num_poi, exp['d_train']['obj_list_sym'] )

        self.refine = False
        self.w = exp['w_normal']

    def forward(self, img, points, choose, idx):

        pred_r, pred_t, pred_c, emb = self.estimator(
                img, points, choose, idx)

        return pred_r, pred_t, pred_c, emb 

    def training_step(self, batch, batch_idx):
        total_loss = 0 
        for frame in batch: 
          
          # unpack the batch and apply forward pass
          if frame[0].dtype == torch.bool:
            continue
            #return {'loss': loss}


          points, choose, img, target, model_points, idx = frame[0:6]
          ff_trans, ff_rot, depth_img, img_orig, cam = frame[6:11]
          gt_rot_wxyz, gt_trans, unique_desig = frame[11:14] 


          pred_r, pred_t, pred_c, emb = self(img, points, choose, idx)

          loss, dis, new_points, new_target = self.criterion(
              pred_r, pred_t, pred_c, target, model_points, idx, points, self.w, self.refine)  # wxy
          total_loss += loss
        #choose correct loss here

        tensorboard_logs = {'train_loss': total_loss/len(batch) }
        return {'loss':  total_loss/len(batch), 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        total_loss = 0
        for frame in batch: 

          if frame[0].dtype == torch.bool:
            continue

          # unpack the batch and apply forward pass
          points, choose, img, target, model_points, idx = frame[0:6]
          ff_trans, ff_rot, depth_img, img_orig, cam = frame[6:11]
          gt_rot_wxyz, gt_trans, unique_desig = frame[11:14] 

          pred_r, pred_t, pred_c, emb = self(img, points, choose, idx)

          loss, dis, new_points, new_target = self.criterion(
              pred_r, pred_t, pred_c, target, model_points, idx, points, self.w, self.refine)  # wxy
          total_loss += loss
 
        tensorboard_logs = {'train_loss': total_loss/len(batch) }
        return {'loss':  total_loss/len(batch), 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.estimator.parameters(), lr=0.0001)

    def train_dataloader(self):
        dataset_train = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                            batch_size=self.exp['loader']['batch_size'],
                                                            shuffle=True,
                                                            num_workers=self.exp['loader']['workers'],
                                                            pin_memory=True)
        return dataloader_train

    def test_dataloader(self):
        dataset_test = GenericDataset(
            cfg_d=self.exp['d_test'],
            cfg_env=self.env)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                            batch_size=self.exp['loader']['batch_size'],
                                                            shuffle=False,
                                                            num_workers=self.exp['loader']['workers'],
                                                            pin_memory=True)
        return dataloader_test

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=file_path, default='yaml/exp/exp_ws.yml',  # required=True,
                        help='The main experiment yaml file.')
    parser.add_argument('--env', type=file_path, default='yaml/env/env_natrix_jonas.yml',
                        help='The environment yaml file.')
    args = parser.parse_args()
    exp_cfg_path = args.exp
    env_cfg_path = args.env

    exp = ConfigLoader().from_file(exp_cfg_path)
    env = ConfigLoader().from_file(env_cfg_path)
    # keep this in mind 
    """
    Trainer args (gpus, num_nodes, etc…) && Program arguments (data_path, cluster_email, etc…)
    Model specific arguments (layer_dim, num_layers, learning_rate, etc…)
    """

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    p = exp['model_path'].split('/')
    p.append(str(timestamp) + '_' + p.pop())
    new_path = '/'.join(p)
    exp['model_path'] = new_path
    model_path = exp['model_path'] 


    logger = logging.getLogger('TrackNet')


    # copy config files to model path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        logger.info((pad("Generating network run folder")))
    else:
        logger.warning((pad("Network run folder already exits")))

    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    env_cfg_fn = os.path.split(env_cfg_path)[-1]

    logger.info(pad(f'Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}'))
    shutil.copy(exp_cfg_path, f'{model_path}/{exp_cfg_fn}')
    shutil.copy(env_cfg_path, f'{model_path}/{env_cfg_fn}')

    dic = {'exp': exp, 'env': env}
    model = TrackNet6D( **dic)

    #ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch}-{val_loss:.2f}'))
    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=1, num_nodes=1, accumulate_grad_batches=exp['accumulate_grad_batches'], default_root_dir= model_path, fast_dev_run=False, limit_test_batches=0.01, limit_train_batches=0.01)
    # early_stop_callback = EarlyStopping(
    #    monitor='val_accuracy',
    #    min_delta=0.00,
    #    patience=3,
    #    verbose=False,
    #    mode='max'
    # )

    trainer.fit(model)
